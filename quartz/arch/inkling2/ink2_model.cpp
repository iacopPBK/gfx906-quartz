// inkling2_loop: the C API model (ink2_ctx). Prefill runs batched in chunks
// (ink2_pf.h) from the deepest prefix-cache snapshot
// (quartz/core/prefix_cache.h); decode is one step() per token.
#include "ink2_prelude.h"
#include "core/prefix_cache.h"

#include <atomic>
#include <algorithm>

namespace ink2 {

#include "ink2_hparams.h"
#include "ink2_tp2.h"
#include "ink2_kernels.h"
#include "ink2_cycle.h"
#include "ink2_engine.h"
#include "ink2_pf.h"

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
        while (st.n_gen < max_new) {
            const int32_t t = pick();
            bool hit = false;
            for (int k = 0; k < n_stop; k++) if (stop[k] == t) hit = true;
            if (hit) { st.stop = 1; break; }
            if (cb && !cb(user, &t, 1)) { st.n_gen++; st.stop = 3; break; }
            st.n_gen++;
            if (st.n_gen >= max_new) break;
            if (abort_flag) { st.stop = 2; break; }
            hi = std::max(hi, pos);
            E.step(t, pos++, true);
            st.decode_cycles++;
        }
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
    ICHECK(hipSetDevice(qz_dev_of(0)));
    qz_mem_report(stdout, 2);
    return c;
}

}   // namespace ink2

qz_ctx * qz_inkling2_create(const char * path, const qz_create_opts & o) { return ink2::create(path, o); }
