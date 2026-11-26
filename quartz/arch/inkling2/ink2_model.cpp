// inkling2_loop, the C API model (ink2_ctx). Prefill runs in batched chunks
// (ink2_pf.h) from the deepest prefix-cache snapshot (quartz/core/prefix_cache.h).
// Decode is one step() per token.
#include "ink2_prelude.h"
#include "core/prefix_cache.h"
#include "core/spec_accept.h"

#include <atomic>
#include <algorithm>
#include <deque>

namespace ink2 {

#include "ink2_hparams.h"
#include "ink2_tp.h"
#include "ink2_kernels.h"
#include "ink2_cycle.h"
#include "ink2_engine.h"
#include "ink2_pf.h"
#include "ink2_dspark.h"
#include "ink2_preflight.h"

struct ink2_ctx : qz_ctx {
    std::string name;
    ink2_engine E;
    ink2_pf EPF;
    std::atomic<bool> abort_flag{false};
    // Prefix cache: the snapshot is the conv history, KV rows are positional in the
    // arena. hi is the highest position written since the last reset, so the snapshot
    // at s is restorable while hi < s - WIN + 1 + PF_RING (its SWA window is intact).
    qz_pcache pc;
    float * shist[QZ_NC] = {}, * ehist[QZ_NC] = {};
    int hi = -1;
    // DSpark drafter (QZ_DSPINE=<drafter gguf>) for speculative rounds. The anchor and
    // the 7 drafts are the 8 rows of the verify kernel (ink2_verify.h).
    ink2_dspark DS;
    bool ds_on = false;
    static constexpr int tail = 64;
    // Images (--mmproj, inkling2v): encoded rows cached by content on the host, plus
    // each card's copy of the current prompt's rows.
    struct img { uint64_t hash = 0; int id = 0, nx = 0, ny = 0; unsigned long use = 0; std::vector<float> rows; };
    inkling2v::qz_vision * vis = nullptr;
    std::vector<img> imgs;
    unsigned long img_clock = 0;
    int img_next = 0;
    float * img_dev[QZ_NC] = {};
    size_t img_cap = 0;

    ~ink2_ctx() override { if (vis) inkling2v::qz_vision_free(vis); }   // device memory is freed at exit
    int vision_attach(const char * mmproj) override {
        if (vis) return -1;
        if (QZ_NC != 2) { fprintf(stderr, "inkling2: images are two cards only (the encoder is split in two), not QZ_TP=%d\n", QZ_NC); return -1; }
        if (!inkling2v::qz_vision_is(mmproj)) { fprintf(stderr, "inkling2: %s is not an inkling2v projector\n", mmproj); return -1; }
        vis = inkling2v::qz_vision_load(mmproj);
        ICHECK(hipSetDevice(qz_dev_of(0)));
        if (!vis) return -1;
        if (inkling2v::qz_vision_n_embd(vis) != D) {
            fprintf(stderr, "inkling2: projector width %d != model width %d\n", inkling2v::qz_vision_n_embd(vis), D);
            inkling2v::qz_vision_free(vis); vis = nullptr; return -1;
        }
        return 0;
    }
    static uint64_t fnv(const void * b, size_t n) {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; i++) { h ^= ((const unsigned char *) b)[i]; h *= 1099511628211ull; }
        return h;
    }
    int image_add(const void * bytes, size_t n, int * nx, int * ny) override {
        if (!vis) return -1;
        const uint64_t h = fnv(bytes, n) ^ (uint64_t) n;
        for (auto & im : imgs)
            if (im.hash == h) {
                im.use = ++img_clock;
                if (nx) *nx = im.nx;
                if (ny) *ny = im.ny;
                return im.id;
            }
        int gx = 0, gy = 0;
        const int nt = inkling2v::qz_vision_encode(vis, bytes, n, &gx, &gy);
        ICHECK(hipSetDevice(qz_dev_of(0)));
        if (nt <= 0) return -1;
        if ((int) imgs.size() >= 8) imgs.erase(std::min_element(imgs.begin(), imgs.end(), [](const img & a, const img & b) { return a.use < b.use; }));
        img im;
        im.hash = h; im.id = img_next++; im.nx = gx; im.ny = gy; im.use = ++img_clock;
        im.rows.resize((size_t) nt * D);
        if (inkling2v::qz_vision_copy_rows(vis, im.rows.data()) != nt) return -1;
        imgs.push_back(std::move(im));
        if (nx) *nx = gx;
        if (ny) *ny = gy;
        return imgs.back().id;
    }
    double image_last_ms() const override { return vis ? inkling2v::qz_vision_last_ms(vis) : 0.0; }
    // The prompt's image rows. Each IMG_PAD run takes the next image, nx * ny long.
    // irow is the image row per position (-1 for a token). keys is the prefix-cache key
    // per position, for an image row its content hash mixed with the row. Uploads the
    // rows to both cards; false when a run does not match the images.
    bool image_rows(const int32_t * p, int P, const int * ids, int n_images, std::vector<int> & irow, std::vector<uint64_t> & keys) {
        std::vector<const img *> im(n_images);
        size_t R = 0;
        for (int k = 0; k < n_images; k++) {
            for (auto & e : imgs) if (e.id == ids[k]) im[k] = &e;
            if (!im[k]) { fprintf(stderr, "inkling2: image %d not in the cache\n", ids[k]); return false; }
            R += (size_t) im[k]->nx * im[k]->ny;
        }
        if (p[P - 1] == IMG_PAD) return false;
        irow.assign(P, -1);
        int k = 0, row = 0;
        for (int q = 0; q < P;) {
            if (p[q] != IMG_PAD) { q++; continue; }
            int e = q;
            while (e < P && p[e] == IMG_PAD) e++;
            if (k >= n_images || e - q != im[k]->nx * im[k]->ny) {
                fprintf(stderr, "inkling2: image run of %d tokens at %d does not match image %d\n", e - q, q, k);
                return false;
            }
            for (int i = 0; i < e - q; i++) {
                irow[q + i] = row + i;
                uint64_t x = im[k]->hash + 0x9e3779b97f4a7c15ull * (uint64_t) (i + 1);
                x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29;
                keys[q + i] = x | (1ull << 63);           // never a token id
            }
            row += e - q;
            q = e;
            k++;
        }
        if (k != n_images) { fprintf(stderr, "inkling2: %d images for %d image runs\n", n_images, k); return false; }
        for (int c = 0; c < QZ_NC; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            if (R > img_cap) {
                if (img_dev[c]) (void) qz_dfree(img_dev[c]);
                img_dev[c] = nullptr;
                qz_mem_scope mem("images");
                if (qz_dmalloc((void **) &img_dev[c], R * D * sizeof(float)) != hipSuccess) { img_cap = 0; return false; }
            }
            size_t off = 0;
            for (int j = 0; j < n_images; j++) {
                const size_t b = (size_t) im[j]->nx * im[j]->ny * D;
                ICHECK(hipMemcpy(img_dev[c] + off, im[j]->rows.data(), b * sizeof(float), hipMemcpyHostToDevice));
                off += b;
            }
        }
        if (R > img_cap) img_cap = R;
        ICHECK(hipSetDevice(qz_dev_of(0)));
        return true;
    }

    void info(qz_model_info * out) const override {
        memset(out, 0, sizeof(*out));
        snprintf(out->arch, sizeof(out->arch), "inkling2_loop");
        snprintf(out->name, sizeof(out->name), "%s", name.c_str());
        out->n_vocab = VOCAB_REAL;
        out->max_ctx = E.ctx;
        out->img_pad = IMG_PAD;
        out->vision = vis != nullptr;
        out->dspine = ds_on;
    }
    int max_ctx() const override { return E.ctx; }
    void abort() override { abort_flag = true; }
    void sync2() {
        for (int c = 0; c < QZ_NC; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamSynchronize(E.C[c].st)); }
        ICHECK(hipSetDevice(qz_dev_of(0)));
    }

    int generate(const int32_t * prompt, int n_prompt, const int * images, int n_images, int max_new,
                 const int32_t * stop, int n_stop, qz_emit_cb cb, void * user,
                 qz_stats * out, const qz_sampling_c * smp) override {
        using clk = std::chrono::steady_clock;
        qz_stats st{};
        st.n_prompt = n_prompt;
        auto fin = [&](int rc) { if (out) *out = st; abort_flag = false; QZ_XCK(qz_xck_poll(QZ_NC)); return rc; };
        if (!prompt || n_prompt < 1 || n_images < 0 || (n_images && !images)) { st.stop = -1; return fin(-1); }
        if (n_prompt + max_new > E.ctx) { fprintf(stderr, "inkling2: %d + %d tokens exceed the context %d\n", n_prompt, max_new, E.ctx); st.stop = -1; return fin(-1); }
        qz_sampling ms;
        qz_sampler sam;
        sam.ncard = QZ_NC;
        if (smp) {
            if (!qz_sampling_from_c(smp, ms)) { st.stop = -1; return fin(-1); }
            sam.set(&ms);
        }
        auto pick = [&]() {
            if (!sam.smp) return E.greedy();
            for (int c = 0; c < QZ_NC; c++) sam.tk[c] = E.tkh[c];
            return sam.sample_row(0);
        };
        // Speculative sampling of the drafts (core/spec_accept.h): with a sampler above
        // temperature 0 the drafter draws its drafts from q = softmax(logit / T) and each
        // verify row accepts its draft with probability min(1, p/q) or takes the residual.
        // At temperature 0 the row's pick is an argmax, so an exact match is the test.
        const bool dspec = ds_on && sam.smp && sam.smp->temp > 0.f;
        // q's logits go through the target's final softcap, as p's do
        dsamp dsa = { 1.f / ms.temp, ms.seed, 0, nullptr, E.hp.softcap };
        if (dspec) DS.rnd = 0;
        const auto t0 = clk::now();
        EPF.abort = &abort_flag;
        qz_pcache::hooks h;
        h.reset = [&] { E.reset(); hi = -1; };
        h.restored = [&] { sync2(); };
        bool did_last = false;
        h.chunk = [&](int a, int b, int mid) {
            if (b == n_prompt - 1 && n_prompt - a <= PF_M) {
                hi = std::max(hi, n_prompt - 1);
                did_last = true;
                return EPF.prefill(prompt, a, n_prompt, true, mid, b - a) ? 0 : -1;
            }
            hi = std::max(hi, b - 1);
            return EPF.prefill(prompt, a, b, false, mid) ? 0 : -1;
        };
        h.sync = [&] { sync2(); };
        h.valid = [&](int s) { return hi < s - WIN + 1 + PF_RING; };
        std::vector<uint64_t> keys(prompt, prompt + n_prompt);
        std::vector<int> irow;
        if (n_images || std::find(prompt, prompt + n_prompt, IMG_PAD) != prompt + n_prompt) {
            if (!image_rows(prompt, n_prompt, images, n_images, irow, keys)) { st.stop = -1; return fin(-1); }
            EPF.irow = irow.data(); for (int c = 0; c < QZ_NC; c++) EPF.img[c] = img_dev[c];
        }
        struct unset { ink2_pf & f; ~unset() { f.irow = nullptr; } } un{EPF};
        const int start = pc.prefill(prompt, n_prompt, keys, END_MESSAGE, PF_M, tail, h);
        if (start < 0) { st.stop = abort_flag ? 2 : -1; return fin(0); }
        st.n_cached = start;
        hi = std::max(hi, n_prompt - 1);
        if (did_last) {
            // Restore the full prompt's conv history after pc.prefill() queued its P - 1 snapshot.
            for (int c = 0; c < QZ_NC; c++) {
                ICHECK(hipSetDevice(qz_dev_of(c)));
                ICHECK(hipMemcpyAsync(E.C[c].hist, ehist[c], pc.blobs[0].bytes, hipMemcpyDeviceToDevice, E.C[c].st));
            }
        } else {
            // The last row and its logits go through the same batched path when it did not fit the final chunk.
            if (!EPF.prefill(prompt, n_prompt - 1, n_prompt, true)) { pc.drop(); st.stop = 2; return fin(0); }
        }
        st.prefill_cycles = n_prompt - start;
        const auto t1 = clk::now();
        st.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int pos = n_prompt;
        const bool dsr = ds_on;
        int dr[DS_BS - 1];
        float lgt[DS_BS * DS_QID], lse[DS_BS];    // the candidates' drafter logits, the rows' logsumexp
        const long r0 = DS.rounds, a0 = DS.accepted;
        // One token. Returns 0 to go on, 1 to end here (the token was emitted).
        auto emit = [&](int32_t t) {
            for (int k = 0; k < n_stop; k++) if (stop[k] == t) { st.stop = 1; return 1; }
            if (cb && !cb(user, &t, 1)) { st.n_gen++; st.stop = 3; return 1; }
            st.n_gen++;
            if (st.n_gen >= max_new) return 1;
            if (abort_flag) { st.stop = 2; return 1; }
            return 0;
        };
        // DSpark rounds. The anchor anc at pos is verify row 0, the drafter's 7 greedy
        // drafts are rows 1 .. 7. Each row's pick is the target's own, greedy or one draw
        // from the row with the request's sampler as a plain step would sample, so the
        // output matches plain decode. If the n leading drafts are their rows' picks, rows
        // 0 .. n commit, dr[0 .. n-1] are emitted, and the row-n pick g[n] at pos + n + 1
        // becomes the next anchor (emitted at the top of the next round). Without the
        // drafter, or when the 8 rows do not fit, decode is one step per token.
        bool anchor = false;
        int32_t anc = 0;
        // Drafter switch, off with QZ_NO_SWITCH=1. The rounds' rate over the last sw_win
        // tokens is compared with the plain step's rate. Below SW_MARGIN of it, decode
        // runs SW_PROBE plain steps, then tries the rounds again; the probe doubles on
        // a relapse within 4 windows, up to SW_CAP. The plain step is timed by SW_MEAS
        // steps when no timing is within SW_REMEAS positions; the timing is kept across
        // requests. Without one, rounds under sw_lo tokens count as slow. Both paths
        // give the same tokens, and the plain path fills the drafter's context.
        static const bool sw_on = [] { const char * v = getenv("QZ_NO_SWITCH"); return !(v && *v && strcmp(v, "0") != 0); }();
        static const float sw_lo = getenv("QZ_SW_LO") ? (float) atof(getenv("QZ_SW_LO")) : 2.2f;
        static const int sw_win = getenv("QZ_SW_WIN") ? atoi(getenv("QZ_SW_WIN")) : 24;
        static double step_ms = 0.0;                            // plain step time (EMA) at step_pos
        static int step_pos = -(1 << 30);
        constexpr int SW_MEAS = 6, SW_REMEAS = 2048, SW_PROBE = 64, SW_CAP = 2048;
        constexpr double SW_MARGIN = 0.82;                      // plain mode below 82% of the plain step's rate
        bool spec = true, meas = false;
        int plain_left = 0, probe = SW_PROBE, seg = 0, nsw = 0, ntok_plain = 0;
        double meas_buf[SW_MEAS] = {};
        int meas_n = 0;
        // Adaptive verify width. A round of n rows (the anchor and n - 1 drafts) emits
        // E(n) = (1 - a^n) / (1 - a) tokens for a per-draft acceptance a, in the round time
        // t(n), so the policy takes the n with the best E(n) / t(n). a is the accepted share
        // of the last 32 rounds' tested drafts (a rejection is one more test, the drafts
        // after it are untested), clamped to [0.05, 0.98]. t(n) is a per-n EMA of the round's
        // wall time, seeded from the measured piecewise ratio table RT_SEED (matching the
        // T=2 / T=4 / T=8 verify kernel tiers) until n has samples; the table is kept across
        // requests and reset when pos moves more than SW_REMEAS. Every 64th round probes an
        // adjacent verify tier to refresh its estimate. QZ_DS_ROWS=2..8 fixes n (read per call).
        const char * e_rows = getenv("QZ_DS_ROWS");
        const int rows_fixed = e_rows && *e_rows ? atoi(e_rows) : 0;
        static constexpr double RT_SEED[VR + 1] = { 0.0, 0.0, 0.51, 0.61, 0.66, 0.84, 0.90, 0.95, 1.00 };
        static double rt_ms[VR + 1] = {0};               // per n round time (ms), across requests
        static int rt_n[VR + 1] = {0};                   // its samples
        static int rt_pos = -(1 << 30);                  // position of the current rt_ms table
        if (std::abs(pos - rt_pos) > SW_REMEAS) {
            for (int n = 0; n <= VR; n++) { rt_ms[n] = 0.0; rt_n[n] = 0; }
            rt_pos = pos;
        }
        int rq_a[32], rq_t[32];                          // the last rounds' accepted / tested drafts
        int rq_len = 0, rq_h = 0, rq_sa = 0, rq_st = 0;
        int last_nr = 4, nr_hist[VR + 1] = {0};
        double last_ms = 0.0;
        long nround = 0, probe_at = 64;
        auto a_est = [&]() -> double {
            if (rq_st <= 0) return 0.6;                  // nothing tested yet
            const double a = (double) rq_sa / (double) rq_st;
            return a < 0.05 ? 0.05 : a > 0.98 ? 0.98 : a;
        };
        auto t_est = [&](int n) -> double {
            if (rt_n[n]) return rt_ms[n];
            double t8 = 0.0;
            if (rt_n[last_nr]) t8 = rt_ms[last_nr] / RT_SEED[last_nr];
            else if (rt_n[4])  t8 = rt_ms[4] / RT_SEED[4];
            else if (rt_n[VR]) t8 = rt_ms[VR];
            else               t8 = last_ms / RT_SEED[last_nr];
            return t8 * RT_SEED[n];
        };
        auto pick_rows = [&]() -> int {
            if (rows_fixed >= 2) return rows_fixed > VR ? VR : rows_fixed;
            if (last_ms <= 0.0 && !rt_n[4] && !rt_n[VR]) return 4; // first round: measure T=4 tier
            const double a = a_est(), d = 1.0 - a;
            int bn = 2;
            double bv = -1.0;
            for (int n = 2; n <= VR; n++) {
                const double v = (1.0 - std::pow(a, n)) / d / t_est(n);
                if (v > bv) { bv = v; bn = n; }
            }
            if (nround >= probe_at) {
                if (bn == 2) return 4;
                if (bn <= 4) return a >= 0.70 ? VR : 2;
                return bn < VR ? VR : 4;
            }
            return bn;
        };
        auto note_round = [&](int nr, int acc, int tst, double ms) {
            if (rt_n[nr] == 0)      rt_ms[nr] = ms;
            else if (rt_n[nr] == 1) rt_ms[nr] = ms;      // drop any cold first-round spike
            else                    rt_ms[nr] = 0.75 * rt_ms[nr] + 0.25 * std::min(ms, 1.5 * rt_ms[nr]);
            rt_n[nr]++;
            rt_pos = pos;
            last_ms = ms; last_nr = nr;
            nr_hist[nr]++;
            if (rq_len == 32) { rq_sa -= rq_a[rq_h]; rq_st -= rq_t[rq_h]; } else rq_len++;
            rq_a[rq_h] = acc; rq_t[rq_h] = tst; rq_sa += acc; rq_st += tst;
            rq_h = (rq_h + 1) & 31;
            if (nround++ >= probe_at) probe_at = nround + 64;
        };
        std::deque<std::pair<int, double>> wq;                  // (tokens, ms) per round in the window
        int wtok = 0; double wms = 0.0;
        int last_kind = 0, last_n = 0;                          // previous iter: 1 plain, 2 round, 3 meas plain
        auto tlast = clk::now();
        while (st.n_gen < max_new) {
            if (sw_on && dsr && last_kind) {                    // the previous iteration's wall time
                const double ms = std::chrono::duration<double, std::milli>(clk::now() - tlast).count();
                if (last_kind == 3) {
                    if (meas_n < SW_MEAS) meas_buf[meas_n++] = ms;
                    if (meas_n == SW_MEAS) {
                        std::sort(meas_buf + 1, meas_buf + SW_MEAS);
                        step_ms = meas_buf[1 + (SW_MEAS - 1) / 2];
                        step_pos = pos;
                        meas_n = 0;
                    }
                } else if (last_kind == 1) {
                    step_ms = step_ms > 0 ? 0.8 * step_ms + 0.2 * std::min(ms, 1.5 * step_ms) : ms;
                    step_pos = pos;
                } else {
                    wq.push_back({last_n, ms}); wtok += last_n; wms += ms; seg += last_n;
                    while (wtok - wq.front().first >= sw_win) { wtok -= wq.front().first; wms -= wq.front().second; wq.pop_front(); }
                    if (seg >= sw_win) {
                        const double tpr = (double) wtok / (double) wq.size();
                        const bool slow = step_ms > 0 ? wtok / wms < SW_MARGIN / step_ms : tpr < sw_lo;
                        if (slow) {
                            probe = nsw && seg < 4 * sw_win ? std::min(2 * probe, SW_CAP) : SW_PROBE;
                            plain_left = probe; spec = false; meas = false; nsw++;
                        }
                    }
                }
            }
            if (sw_on && dsr && spec && std::abs(pos - step_pos) > SW_REMEAS && pos + VR <= E.ctx) {
                spec = false; meas = true; plain_left = SW_MEAS; meas_n = 0; // time the plain step here
            }
            tlast = clk::now();
            last_kind = 0;
            if (!anchor) anc = pick();
            anchor = false;
            if (emit(anc)) break;
            const int nr = pick_rows(), nd = nr - 1;
            if (!dsr || pos + nr > E.ctx || !spec) {
                hi = std::max(hi, pos);
                E.step(anc, pos++, true);
                if (ds_on) DS.fill(DS.tap, 4 * D, pos - 1, 1);
                st.decode_cycles++; st.decode_tok++;
                if (dsr && !spec) {
                    last_kind = meas ? 3 : 1; ntok_plain++;
                    if (--plain_left <= 0) {                    // back to the rounds; a timing keeps the window
                        if (!meas) { seg = 0; wq.clear(); wtok = 0; wms = 0.0; nsw++; }
                        spec = true; meas = false;
                    }
                }
                continue;
            }
            const auto tr0 = clk::now();
            dsa.rnd = DS.rnd++;
            DS.draft(anc, pos, dr, dspec ? &dsa : nullptr, nd);
            int toks[VR];
            toks[0] = anc;
            for (int r = 0; r < nd; r++) toks[r + 1] = dr[r];
            E.verify(toks, pos, nr);
            int g[VR], n = 0;
            // Without a sampler, or at temperature 0, the row's pick is an argmax, so the
            // drafts match it exactly or not at all.
            if (!sam.smp || ms.temp <= 0.f) {
                for (int r = 0; r < nr; r++) g[r] = E.greedy_row(r);
                while (n < nd && dr[n] == g[n]) n++;
            } else {                                  // sampled: speculative sampling
                const size_t rb = (size_t) QZ_TOPK_NB * QZ_TOPK;
                for (int c = 0; c < QZ_NC; c++) sam.tk[c].assign(E.vtkp[c], E.vtkp[c] + (size_t) nr * rb);
                // Each row's p is the sampler's own chain, q comes from the drafter: the
                // round's rows' candidate ids (plus each row's draft, when p does not list
                // it) go to the cards in one launch, then row r accepts dr[r] or the
                // residual ends the round. A residual in g[n] anchors the next round like
                // any other row pick, so the round still emits n + 1 tokens.
                std::vector<float> w;
                int rid[DS_BS][DS_QID], rn[DS_BS], dslot[DS_BS];
                float rw[DS_BS][DS_QID];
                double qall[DS_BS][DS_QID];
                for (int r = 0; r < DS_BS; r++)
                    for (int j = 0; j < DS_QID; j++) { rid[r][j] = -1; rw[r][j] = 0.f; }
                for (int r = 0; r < nd; r++) {
                    const int nc = sam.row_chain(r, w);
                    int m = 0;
                    dslot[r] = -1;
                    for (int i = 0; i < nc; i++) {
                        const int id = sam.cand[i].second;
                        rid[r][m] = id;
                        rw[r][m] = nc == 1 ? 1.f : w[i];
                        if (id == dr[r]) dslot[r] = m;
                        m++;
                    }
                    if (dslot[r] < 0) { rid[r][m] = dr[r]; dslot[r] = m; m++; }   // p(d) = 0
                    rn[r] = m;
                }
                DS.qlog(&rid[0][0], &lgt[0], lse, nd);
                for (int r = 0; r < nd; r++)                     // q of every candidate of every row
                    for (int i = 0; i < rn[r]; i++) {
                        double x = lgt[r * DS_QID + i];
                        if (dsa.cap > 0.f) x = (double) dsa.cap * tanh(x / dsa.cap);
                        qall[r][i] = exp(x * dsa.invTq - lse[r]);
                    }
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                for (int r = 0; r < nd; r++) {
                    const qz_spec_out o = qz_spec_row(rid[r], rw[r], rn[r], dr[r], qall[r][dslot[r]], qall[r],
                                                      u01(sam.rng), u01(sam.rng));
                    if (!o.accepted) { g[n] = o.id; break; }          // residual: the round ends
                    g[n] = dr[r];
                    if (++n == nd) { g[nd] = sam.sample_row(nd); break; }
                }
            }
            note_round(nr, n, n + (n < nd ? 1 : 0), std::chrono::duration<double, std::milli>(clk::now() - tr0).count());
            DS.rounds++;
            DS.accepted += n;
            hi = std::max(hi, pos + nr - 1);          // the SWA ring rows the verify wrote
            st.decode_cycles++; st.decode_tok += n + 1;
            st.dspine = 1;
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
        if (dsr && DS.rounds > r0) {
            char b[160];                              // the rounds per width, as n<width>=<rounds>
            int o = 0;
            for (int n = 2; n <= VR; n++)
                if (nr_hist[n]) o += snprintf(b + o, sizeof(b) - (size_t) o, " n%d=%d", n, nr_hist[n]);
            fprintf(stderr, "dspark: %ld rounds, %.2f tok/rnd (%.2f drafts accepted of %d) rows%s\n", DS.rounds - r0,
                    (double) (DS.accepted - a0 + DS.rounds - r0) / (DS.rounds - r0), (double) (DS.accepted - a0) / (DS.rounds - r0), DS_BS - 1, b);
        }
        if (dsr && sw_on && ntok_plain)
            fprintf(stderr, "switch: %d, %d tokens plain (%.2f ms/step)\n", nsw, ntok_plain, step_ms);
        st.decode_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
        return fin(0);
    }
};

static qz_ctx * create(const char * path, const qz_create_opts & o) {
    qz_gguf g;
    if (!g.open(path)) { fprintf(stderr, "inkling2: cannot open %s\n", path); return nullptr; }
    hparams hp;
    if (!parse(g, hp) || !check_tensors(g)) { fprintf(stderr, "inkling2: %s is not a supported Inkling-2-Mini GGUF\n", path); return nullptr; }
    const int ctx = std::min(o.max_ctx > 0 ? o.max_ctx : 32768, 262144);
    // Check memory and the peer link both ways before the weights.
    if (ink2_vram_check(g, ctx, getenv("QZ_DSPINE")) || qz_link_check()) return nullptr;
    ink2_ctx * c = new ink2_ctx();
    c->name = g.str("general.name");
    c->E.hp = hp;
    c->EPF.E = &c->E;
    c->E.ctx = ctx;
    qz_mem_baseline(QZ_NC);
    if (qz_ncard != QZ_NC) { fprintf(stderr, "inkling2: built for %d cards, QZ_TP=%d\n", QZ_NC, qz_ncard); delete c; return nullptr; }
    if (!qz_peer_enable()) { delete c; return nullptr; }
    for (int i = 0; i < QZ_NC; i++) {
        ink2_card & K = c->E.C[i];
        {
            qz_mem_scope mem("weights");
            K.ld.slice = ink2_tp_slice;
            if (!K.ld.load(path, g, qz_dev_of(i), i)) { fprintf(stderr, "inkling2: card %d load failed\n", i); delete c; return nullptr; }
        }
        fprintf(stderr, "inkling2: card %d weights %.2f GB in %.1f s, %.2f GB free\n", i, K.ld.need_bytes / 1e9, K.ld.load_seconds, K.ld.free_after / 1e9);
        ICHECK(hipSetDevice(qz_dev_of(i)));
        ICHECK(qz_card_stream(i, &K.st));
        c->E.bind(i);
        if (!c->E.alloc(i) || !c->EPF.alloc(i)) { delete c; return nullptr; }
    }
    {   // prefix-cache snapshots: 16 slots of the conv history in pinned host memory
        qz_mem_scope mem("snapshot staging");
        qz_blob hb;
        hb.bytes = (size_t) NAPP * 3 * CHL * 4;
        for (int i = 0; i < QZ_NC; i++) {
            ICHECK(hipSetDevice(qz_dev_of(i)));
            ICHECK(qz_dmalloc((void **) &c->shist[i], hb.bytes));
            ICHECK(qz_dmalloc((void **) &c->ehist[i], hb.bytes));
            hb.live[i] = c->E.C[i].hist; hb.stage[i] = c->shist[i];
            c->EPF.shist[i] = c->shist[i];
            c->EPF.ehist[i] = c->ehist[i];
        }
        hipStream_t ps[QZ_NC];
        for (int i = 0; i < QZ_NC; i++) ps[i] = c->E.C[i].st;
        c->pc.init({ hb }, 16, ps, QZ_NC);
    }
    if (const char * dp = getenv("QZ_DSPINE")) {   // QZ_DSPINE from the server, a DSpark GGUF
        c->DS.E = &c->E;
        if (!c->DS.load(dp, c->E.ctx)) { delete c; return nullptr; }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        ICHECK(qz_dmalloc((void **) &c->EPF.dtaps, (size_t) PF_M * 4 * D * 4, "dspark"));
        c->EPF.on_chunk = [c](int p0, int m) { c->DS.fill(c->EPF.dtaps, 4 * D, p0, m); };
        c->ds_on = true;
        fprintf(stderr, "inkling2: DSpark drafter %s\n", dp);
    }
    ICHECK(hipSetDevice(qz_dev_of(0)));
    qz_mem_report(stderr, QZ_NC);
    return c;
}

}   // namespace ink2

qz_ctx * qz_inkling2_create(const char * path, const qz_create_opts & o) { return ink2::create(path, o); }
