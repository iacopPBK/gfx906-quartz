// inkling2_loop: the C API model (ink2_ctx). Prefill runs batched in chunks
// (ink2_pf.h) from the deepest prefix-cache snapshot
// (quartz/core/prefix_cache.h); decode is one step() per token.
#include "ink2_prelude.h"
#include "core/prefix_cache.h"

#include <atomic>
#include <algorithm>
#include <deque>

namespace ink2 {

#include "ink2_hparams.h"
#include "ink2_tp2.h"
#include "ink2_kernels.h"
#include "ink2_cycle.h"
#include "ink2_engine.h"
#include "ink2_pf.h"
#include "ink2_dspark.h"

struct ink2_ctx : qz_ctx {
    std::string name;
    ink2_engine E;
    ink2_pf EPF;
    std::atomic<bool> abort_flag{false};
    // prefix cache: the conv history is the snapshot (KV rows are positional);
    // hi = the highest position written since the last reset, so a snapshot at s
    // is restorable while hi < s - WIN + 1 + PF_RING (its SWA window not overwritten)
    qz_pcache pc;
    float * shist[2] = {};
    int hi = -1;
    // DSpark drafter (QZ_DSPINE=<drafter gguf>): speculative greedy rounds (the
    // anchor and the 7 drafts are the verify kernel's 8 rows, ink2_verify.h)
    ink2_dspark DS;
    bool ds_on = false;
    static constexpr int tail = 64;

    void info(qz_model_info * out) const override {
        memset(out, 0, sizeof(*out));
        snprintf(out->arch, sizeof(out->arch), "inkling2_loop");
        snprintf(out->name, sizeof(out->name), "%s", name.c_str());
        out->n_vocab = VOCAB_REAL;
        out->max_ctx = E.ctx;
        out->img_pad = -1;
    }
    int max_ctx() const override { return E.ctx; }
    void abort() override { abort_flag = true; }
    void sync2() {
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamSynchronize(E.C[c].st)); }
        ICHECK(hipSetDevice(qz_dev_of(0)));
    }

    int generate(const int32_t * prompt, int n_prompt, const int *, int n_images, int max_new,
                 const int32_t * stop, int n_stop, qz_emit_cb cb, void * user,
                 qz_stats * out, const qz_sampling_c * smp) override {
        using clk = std::chrono::steady_clock;
        qz_stats st{};
        st.n_prompt = n_prompt;
        auto fin = [&](int rc) { if (out) *out = st; abort_flag = false; return rc; };
        if (!prompt || n_prompt < 1 || n_images > 0) { st.stop = -1; return fin(-1); }
        if (n_prompt + max_new > E.ctx) { fprintf(stderr, "inkling2: %d + %d tokens exceed the context %d\n", n_prompt, max_new, E.ctx); st.stop = -1; return fin(-1); }
        qz_sampling ms;
        qz_sampler sam;
        if (smp) {
            if (!qz_sampling_from_c(smp, ms)) { st.stop = -1; return fin(-1); }
            sam.set(&ms);
        }
        auto pick = [&]() {
            if (!sam.smp) return E.greedy();
            sam.tk[0] = E.tkh[0];
            sam.tk[1] = E.tkh[1];
            return sam.sample_row(0);
        };
        const auto t0 = clk::now();
        EPF.abort = &abort_flag;
        qz_pcache::hooks h;
        h.reset = [&] { E.reset(); hi = -1; };
        h.restored = [&] { sync2(); };
        h.chunk = [&](int a, int b, int mid) {
            hi = std::max(hi, b - 1);
            if (mid < 0) return EPF.prefill(prompt, a, b, false) ? 0 : -1;
            if (!EPF.prefill(prompt, a, a + mid, false)) return -1;
            for (int c = 0; c < 2; c++) {
                ICHECK(hipSetDevice(qz_dev_of(c)));
                ICHECK(hipMemcpyAsync(shist[c], E.C[c].hist, pc.blobs[0].bytes, hipMemcpyDeviceToDevice, E.C[c].st));
            }
            return EPF.prefill(prompt, a + mid, b, false) ? 0 : -1;
        };
        h.sync = [&] { sync2(); };
        h.valid = [&](int s) { return hi < s - WIN + 1 + PF_RING; };
        std::vector<uint64_t> keys(prompt, prompt + n_prompt);
        const int start = pc.prefill(prompt, n_prompt, keys, END_MESSAGE, PF_M, tail, h);
        if (start < 0) { st.stop = abort_flag ? 2 : -1; return fin(0); }
        st.n_cached = start;
        hi = std::max(hi, n_prompt - 1);
        // the last row (and the logits) through the same batched path
        if (!EPF.prefill(prompt, n_prompt - 1, n_prompt, true)) { pc.drop(); st.stop = 2; return fin(0); }
        st.prefill_cycles = n_prompt - start;
        const auto t1 = clk::now();
        st.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int pos = n_prompt;
        const bool dsr = ds_on && !sam.smp;
        int dr[DS_BS - 1];
        const long r0 = DS.rounds, a0 = DS.accepted;
        // one token: 0 = the generation goes on, 1 = it ends here (the token was emitted)
        auto emit = [&](int32_t t) {
            for (int k = 0; k < n_stop; k++) if (stop[k] == t) { st.stop = 1; return 1; }
            if (cb && !cb(user, &t, 1)) { st.n_gen++; st.stop = 3; return 1; }
            st.n_gen++;
            if (st.n_gen >= max_new) return 1;
            if (abort_flag) { st.stop = 2; return 1; }
            return 0;
        };
        // DSpark greedy rounds: the anchor t0 at pos is verify row 0, the drafter's 7
        // drafts rows 1 .. 7; n leading drafts match the target's own pick, so rows
        // 0 .. n commit, d[0 .. n-1] are emitted, and the row-n pick g[n] at pos + n + 1
        // is the next anchor (emitted at the top of the next round). Without the drafter (or when the 8 rows do
        // not fit) this is one step per token, as before.
        bool anchor = false;
        int32_t anc = 0;
        // the drafter switch (as qwen35's; QZ_NO_SWITCH=1: off): the speculative rounds' rate
        // over a window of the last sw_win tokens (wall time per loop iteration) against the
        // plain step's rate; below it decode goes plain, then back to the rounds after
        // SW_PROBE tokens (doubled while the rounds fail again within 4 windows, up to 2048).
        // The plain step is timed by a few plain steps whenever no timing is within 2048
        // positions (kept across requests); without one, the window's tok / round against
        // sw_lo. Both paths give the same tokens and the plain path fills the drafter's
        // context, so a switch costs nothing.
        static const bool sw_on = [] { const char * v = getenv("QZ_NO_SWITCH"); return !(v && *v && strcmp(v, "0") != 0); }();
        static const float sw_lo = getenv("QZ_SW_LO") ? (float) atof(getenv("QZ_SW_LO")) : 2.7f;
        static const int sw_win = getenv("QZ_SW_WIN") ? atoi(getenv("QZ_SW_WIN")) : 32;
        static double step_ms = 0.0;                            // the plain step's time (EMA) at step_pos
        static int step_pos = -(1 << 30);
        constexpr int SW_MEAS = 6, SW_REMEAS = 2048, SW_PROBE = 256;
        bool spec = true, meas = false;
        int plain_left = 0, probe = SW_PROBE, seg = 0, nsw = 0, ntok_plain = 0;
        std::deque<std::pair<int, double>> wq;                  // (tokens, ms) per round in the window
        int wtok = 0; double wms = 0.0;
        int last_kind = 0, last_n = 0;                          // the previous iteration: 1 plain, 2 a round of last_n tokens
        auto tlast = clk::now();
        while (st.n_gen < max_new) {
            if (sw_on && dsr && last_kind) {                    // the previous iteration's wall time
                const double ms = std::chrono::duration<double, std::milli>(clk::now() - tlast).count();
                if (last_kind == 1) { step_ms = step_ms > 0 ? 0.8 * step_ms + 0.2 * ms : ms; step_pos = pos; }
                else {
                    wq.push_back({last_n, ms}); wtok += last_n; wms += ms; seg += last_n;
                    while (wtok - wq.front().first >= sw_win) { wtok -= wq.front().first; wms -= wq.front().second; wq.pop_front(); }
                    if (seg >= sw_win) {
                        const double tpr = (double) wtok / (double) wq.size();
                        const bool slow = step_ms > 0 ? wtok / wms < 1.0 / step_ms : tpr < sw_lo;
                        if (slow) {
                            probe = nsw && seg < 4 * sw_win ? std::min(2 * probe, 2048) : SW_PROBE;
                            plain_left = probe; spec = false; meas = false; nsw++;
                        }
                    }
                }
            }
            if (sw_on && dsr && spec && std::abs(pos - step_pos) > SW_REMEAS && pos + VR <= E.ctx) {
                spec = false; meas = true; plain_left = SW_MEAS;    // time the plain step here
            }
            tlast = clk::now();
            last_kind = 0;
            if (!anchor) anc = pick();
            anchor = false;
            if (emit(anc)) break;
            if (!dsr || pos + VR > E.ctx || !spec) {
                hi = std::max(hi, pos);
                E.step(anc, pos++, true);
                if (ds_on) DS.fill(DS.tap, 4 * D, pos - 1, 1);
                st.decode_cycles++;
                if (dsr && !spec) {
                    last_kind = 1; ntok_plain++;
                    if (--plain_left <= 0) {                    // back to the rounds (after a timing: the same window)
                        if (!meas) { seg = 0; wq.clear(); wtok = 0; wms = 0.0; nsw++; }
                        spec = true; meas = false;
                    }
                }
                continue;
            }
            DS.draft(anc, pos, dr);
            int toks[VR];
            toks[0] = anc;
            for (int r = 0; r < DS_BS - 1; r++) toks[r + 1] = dr[r];
            E.verify(toks, pos);
            int g[VR], n = 0;
            for (int r = 0; r < VR; r++) g[r] = E.greedy_row(r);
            while (n < DS_BS - 1 && dr[n] == g[n]) n++;
            DS.rounds++;
            DS.accepted += n;
            hi = std::max(hi, pos + VR - 1);          // the SWA ring rows the verify wrote
            st.decode_cycles++;
            bool end = false;
            for (int i = 0; i < n && !end; i++) end = emit(dr[i]) != 0;
            if (end) break;
            E.commit(n + 1);
            DS.fill(E.vtap, 4 * D, pos, n + 1);
            pos += n + 1;
            anc = g[n];
            anchor = true;
            last_kind = 2; last_n = n + 1;
        }
        if (dsr && DS.rounds > r0)
            fprintf(stderr, "dspark: %ld rounds, %.2f tok/rnd (%.2f drafts accepted of %d)\n", DS.rounds - r0,
                    (double) (DS.accepted - a0 + DS.rounds - r0) / (DS.rounds - r0), (double) (DS.accepted - a0) / (DS.rounds - r0), DS_BS - 1);
        if (dsr && sw_on && ntok_plain)
            fprintf(stderr, "switch: %d, %d tokens plain (%.2f ms/step)\n", nsw, ntok_plain, step_ms);
        st.decode_tok = st.decode_cycles;
        st.decode_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
        return fin(0);
    }
};

static qz_ctx * create(const char * path, const qz_create_opts & o) {
    qz_gguf g;
    if (!g.open(path)) { fprintf(stderr, "inkling2: cannot open %s\n", path); return nullptr; }
    hparams hp;
    if (!parse(g, hp) || !check_tensors(g)) { fprintf(stderr, "inkling2: %s is not a supported Inkling-2-Mini GGUF\n", path); return nullptr; }
    if (qz_p2p_check()) return nullptr;   // the peer link both ways, before the weights
    ink2_ctx * c = new ink2_ctx();
    c->name = g.str("general.name");
    c->E.hp = hp;
    c->EPF.E = &c->E;
    c->E.ctx = std::min(o.max_ctx > 0 ? o.max_ctx : 32768, 262144);
    qz_mem_baseline(2);
    if (!qz_peer_enable()) { delete c; return nullptr; }
    for (int i = 0; i < 2; i++) {
        ink2_card & K = c->E.C[i];
        {
            qz_mem_scope mem("weights");
            K.ld.slice = ink2_tp2_spec;
            if (!K.ld.load(path, g, qz_dev_of(i), i)) { fprintf(stderr, "inkling2: card %d load failed\n", i); delete c; return nullptr; }
        }
        printf("inkling2: card %d weights %.2f GB in %.1f s, %.2f GB free\n", i, K.ld.need_bytes / 1e9, K.ld.load_seconds, K.ld.free_after / 1e9);
        ICHECK(hipSetDevice(qz_dev_of(i)));
        ICHECK(hipStreamCreateWithFlags(&K.st, hipStreamNonBlocking));
        c->E.bind(i);
        if (!c->E.alloc(i) || !c->EPF.alloc(i)) { delete c; return nullptr; }
    }
    {   // prefix-cache snapshots: 16 slots of the conv history in pinned host memory
        qz_mem_scope mem("snapshot staging");
        qz_blob hb;
        hb.bytes = (size_t) NAPP * 3 * CHL * 4;
        for (int i = 0; i < 2; i++) {
            ICHECK(hipSetDevice(qz_dev_of(i)));
            ICHECK(qz_dmalloc((void **) &c->shist[i], hb.bytes));
            hb.live[i] = c->E.C[i].hist; hb.stage[i] = c->shist[i];
        }
        c->pc.init({ hb }, 16, c->E.C[0].st, c->E.C[1].st);
    }
    if (const char * dp = getenv("QZ_DSPINE")) {   // the server's drafter flag: a DSpark GGUF here
        c->DS.E = &c->E;
        if (!c->DS.load(dp, c->E.ctx)) { delete c; return nullptr; }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        ICHECK(qz_dmalloc((void **) &c->EPF.dtaps, (size_t) PF_M * 4 * D * 4, "dspark"));
        c->EPF.on_chunk = [c](int p0, int m) { c->DS.fill(c->EPF.dtaps, 4 * D, p0, m); };
        c->ds_on = true;
        printf("inkling2: DSpark drafter %s\n", dp);
    }
    ICHECK(hipSetDevice(qz_dev_of(0)));
    qz_mem_report(stdout, 2);
    return c;
}

}   // namespace ink2

qz_ctx * qz_inkling2_create(const char * path, const qz_create_opts & o) { return ink2::create(path, o); }
