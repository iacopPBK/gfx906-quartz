// DSpine decode: the context fill, the draft chain, sampled verify and the round loop.
#pragma once
#include <chrono>

#include "core/devices.h"
#include "core/link.h"
#include <random>
#include <vector>

// context fill input: a chunk's 5 feature layers per row, in pf_gemm's grouped layout
__global__ void __launch_bounds__(256) ds_feat_quant(const float * __restrict__ src, size_t lstr, int8_t * Ag, float * sag, int Mp) {
    const int m = blockIdx.x;
    for (int k = threadIdx.x; k < 5 * QZ_D; k += 256)
        pf_quant_unit(src[(size_t) (k / QZ_D) * lstr + (size_t) m * QZ_D + k % QZ_D], k, m, Mp, Ag, sag, threadIdx.x & 31);
}

struct qz_dsrun {
    bool on = false;
    // mv8 and mv12 share a 12-row replay ring, so either replays the last round's rows
    static constexpr int MVW = 12;
    mv8::mv_ctx mv[QZ_NC];
    mv12::mv_ctx mw[QZ_NC];
    qz_ds_ctx cx[QZ_NC];
    qz_ds_block blk[QZ_NC];
    float * feat[QZ_NC] = {};       // mv12's per-round capture [12][5][QZ_D], per card
    float * feat8[QZ_NC] = {};      // mv8's [8][5][QZ_D]
    // prefill context fill, GEMMs over up to DS_FR rows, per card: quantized features,
    // fc out, its quantized norm, k and v out
    static constexpr int DS_FR = 2048;
    int8_t * fa[QZ_NC] = {}, * fca[QZ_NC] = {};
    float * fas[QZ_NC] = {}, * fcas[QZ_NC] = {}, * fco[QZ_NC] = {}, * fkv[QZ_NC] = {};
    hipEvent_t fev = nullptr;
    float * bas_d = nullptr;
    int * drafts_d = nullptr;
    float2 * topk_d[QZ_NC] = {};    // sampled verify: [MVW][QZ_TOPK_NB][QZ_TOPK] per card
    float * u_d = nullptr, * q_d = nullptr;   // sampled drafts: [DS_T-1] uniforms, [DS_T-1][DS_TOPK] q
    std::mt19937 qrng;
    float qd_h[(DS_T - 1) * DS_TOPK];
    int qc_h[(DS_T - 1) * DS_TOPK];
    unsigned int xtok = 0;

    int init(qz_engine & eng, const char * draft) {
        qz_mem_scope mem("dspine run");
        for (int c = 0; c < QZ_NC; c++)
            if (mv12::mv_init(mw[c], eng, c)) return 1;
        for (int c = 0; c < QZ_NC; c++)
            if (mv8::mv_init(mv[c], eng, c, mw[c].ring8)) return 1;
        mv12::mv_wire_peers(mw);
        mv8::mv_wire_peers(mv);
        feat[0] = mv12::mv_feat_cap(eng);
        feat8[0] = mv8::mv_feat_cap(eng);
        for (int c = 1; c < QZ_NC; c++) { feat[c] = mv12::mv_feat_cap_c(c); feat8[c] = mv8::mv_feat_cap_c(c); }
        // the drafter embeds with the engine's token_embd and reads its half of the head
        // on each card; TG and TSV are built per vocab half too
        auto emb = [&](int c) { return &eng.C[c].ld.ts[eng.C[c].ld.find("token_embd.weight")]; };
        auto hh = [&](int c) { return &eng.C[c].ld.ts[eng.C[c].ld.find("output.weight")]; };
        for (int c = 0; c < QZ_NC; c++) {
            if (qz_ds_ctx_load(cx[c], draft, qz_dev_of(c), c * DS_NKV / QZ_NC, DS_NKV / QZ_NC, c)) return 1;
            if (qz_ds_block_load(blk[c], cx[c], draft, emb(c), hh(c), c)) return 1;
        }
        for (int c = 0; c < QZ_NC; c++) {
            DS_CHECK(hipSetDevice(qz_dev_of(c)));
            DS_CHECK(qz_dmalloc(&fa[c], (size_t) DS_FR * 5 * QZ_D));
            DS_CHECK(qz_dmalloc(&fas[c], (size_t) DS_FR * 5 * QZ_D / 32 * 4));
            DS_CHECK(qz_dmalloc(&fco[c], (size_t) DS_FR * QZ_D * 4));
            DS_CHECK(qz_dmalloc(&fca[c], (size_t) DS_FR * QZ_D));
            DS_CHECK(qz_dmalloc(&fcas[c], (size_t) DS_FR * QZ_D / 32 * 4));
            DS_CHECK(qz_dmalloc(&fkv[c], (size_t) DS_FR * (DS_NKV / QZ_NC) * DS_HD * 4));
            // rows past n are read by the GEMMs, outputs dropped, so keep them finite
            DS_CHECK(hipMemset(fa[c], 0, (size_t) DS_FR * 5 * QZ_D));
            DS_CHECK(hipMemset(fas[c], 0, (size_t) DS_FR * 5 * QZ_D / 32 * 4));
            DS_CHECK(hipMemset(fca[c], 0, (size_t) DS_FR * QZ_D));
            DS_CHECK(hipMemset(fcas[c], 0, (size_t) DS_FR * QZ_D / 32 * 4));
            DS_CHECK(qz_dmalloc(&topk_d[c], (size_t) MVW * QZ_TOPK_NB * QZ_TOPK * sizeof(float2)));
            // the exchange flags: p2p memsets the card's VRAM, host mode writes the pinned
            // mailbox itself (no kernel is in flight yet)
            DS_CHECK(qz_xzero(mv[c].xflag, (size_t) (QZ_NC - 1) * QZ_XFS * 4));
            DS_CHECK(qz_xzero(mw[c].xflag, (size_t) (QZ_NC - 1) * QZ_XFS * 4));
        }
        DS_CHECK(hipSetDevice(qz_dev_of(0)));
        DS_CHECK(hipEventCreateWithFlags(&fev, hipEventDisableTiming));
        DS_CHECK(qz_dmalloc(&bas_d, (DS_T - 1) * DS_TOPK * 4));
        DS_CHECK(qz_dmalloc(&drafts_d, DS_T * 4));
        DS_CHECK(qz_dmalloc(&u_d, DS_T * 4));
        DS_CHECK(qz_dmalloc(&q_d, (size_t) DS_T * DS_TOPK * 4));
        on = true;
        return 0;
    }

    // a prefill chunk's features, [5][Mmax][QZ_D] on card 0 with row r at position
    // p0 + r, into every card's context rings
    void set_rope_shift(int s) { for (auto & m : mv) m.Ah.rope_shift = s; for (auto & m : mw) m.Ah.rope_shift = s; }   // the target's rope is the KV row + s, which images shift
    // batched qz_ds_ctx_rows: card 0 quantizes and the other cards get a copy, then each card
    // runs fc, norm and the k / v GEMMs of its kv heads
    void fill_prefill(const float * pf_feat, int Mmax, int p0, int n) {
        constexpr int KF = 5 * QZ_D;
        for (int b0 = 0; b0 < n; b0 += DS_FR) {
            const int nb = std::min(DS_FR, n - b0), Mp = (nb + PF_MT - 1) / PF_MT * PF_MT, nty = Mp / PF_MT;
            DS_CHECK(hipSetDevice(qz_dev_of(0)));
            hipLaunchKernelGGL(ds_feat_quant, dim3(nb), dim3(256), 0, cx[0].st, pf_feat + (size_t) b0 * QZ_D, (size_t) Mmax * QZ_D,
                               fa[0], fas[0], Mp);
            DS_CHECK(hipEventRecord(fev, cx[0].st));
            for (int c = 1; c < QZ_NC; c++) {
                DS_CHECK(hipSetDevice(qz_dev_of(c)));
                DS_CHECK(hipStreamWaitEvent(cx[c].st, fev, 0));
                // the quantized features card 0 just wrote, into card c's copies: the source
                // is card 0's, so the copy is queued on its stream and card c's stream waits
                // for it (both modes: under p2p the copy no longer runs on card c's stream, so
                // the qz_xcopy event is what orders the GEMMs below).
                DS_CHECK(qz_xcopy(fa[c], c, cx[c].st, fa[0], 0, cx[0].st, (size_t) Mp * KF));
                DS_CHECK(qz_xcopy(fas[c], c, cx[c].st, fas[0], 0, cx[0].st, (size_t) Mp * KF / 32 * 4));
            }
            for (int c = 0; c < QZ_NC; c++) {
                DS_CHECK(hipSetDevice(qz_dev_of(c)));
                qz_ds_ctx & d = cx[c];
                const int nr = d.hn * DS_HD;
                hipLaunchKernelGGL(pf_gemm<0>, dim3(QZ_D / 64, nty), dim3(256), 0, d.st, fa[c], fas[c], Mp,
                                   (const char *) d.fc_v, (const __half *) d.fc_s, KF, KF, fco[c], QZ_D, QZ_D, 0, 0);
                ds_rms<<<(nb + 3) / 4, 256, 0, d.st>>>(fco[c], d.hnorm, fco[c], nb, DS_H, DS_EPS);
                hipLaunchKernelGGL(pf_quant, dim3(nb), dim3(256), 0, d.st, fco[c], QZ_D, QZ_D, fca[c], fcas[c], Mp, 0);
                for (int l = 0; l < DS_FL; l++) {
                    hipLaunchKernelGGL(pf_gemm<0>, dim3((nr + 63) / 64, nty), dim3(256), 0, d.st, fca[c], fcas[c], Mp,
                                       (const char *) d.k_v[l] + (size_t) d.h0 * DS_HD * DS_H,
                                       (const __half *) (d.k_s[l] + (size_t) d.h0 * DS_HD * (DS_H / 32)), DS_H, DS_H, fkv[c], nr, nr, 0, 0);
                    ds_k_head_rows<<<dim3(d.hn, nb), 128, 0, d.st>>>(fkv[c], d.knw[l], d.inv, d.ring_k[l], d.h0, d.hn, p0 + b0);
                    hipLaunchKernelGGL(pf_gemm<0>, dim3((nr + 63) / 64, nty), dim3(256), 0, d.st, fca[c], fcas[c], Mp,
                                       (const char *) d.v_v[l] + (size_t) d.h0 * DS_HD * DS_H,
                                       (const __half *) (d.v_s[l] + (size_t) d.h0 * DS_HD * (DS_H / 32)), DS_H, DS_H, fkv[c], nr, nr, 0, 0);
                    ds_ring_scatter<<<nb, 256, 0, d.st>>>(fkv[c], d.ring_v[l], nr, d.h0 * DS_HD, p0 + b0, nb);
                }
                DS_CHECK(hipGetLastError());
            }
            mark(p0 + b0, nb);
            for (int c = 0; c < QZ_NC; c++) { DS_CHECK(hipSetDevice(qz_dev_of(c))); DS_CHECK(hipStreamSynchronize(cx[c].st)); }
        }
        DS_CHECK(hipSetDevice(qz_dev_of(0)));
    }

    std::vector<int> ring_pos = std::vector<int>(DS_RING, -1);
    void mark(int p, int n) { for (int i = 0; i < n; i++) ring_pos[(p + i) % DS_RING] = p + i; }
    int window_base(int p0) const {
        const int lo = std::max(0, p0 + 1 - DS_W);
        int q = p0 - 1;
        while (q >= lo && ring_pos[q % DS_RING] == q) q--;
        return std::min(q + 1, std::max(lo, p0 - 1));   // keep one row: the kernels' tested range
    }

    // the drafter: block pass on both cards, then the device chain -> drafts[0 .. DS_T-2]
    void draft(int anchor, int p0, int32_t * drafts, float invT = 0.f, int nt = DS_T) {
        const int wbase = window_base(p0);
        qz_ds_block_round2(blk, cx, anchor, p0, wbase, nt);
        DS_CHECK(hipSetDevice(qz_dev_of(0)));
        qz_ds_block & b = blk[0];
        ds_cand_sort<<<1, 64, 0, b.st>>>(b.cand);
        ds_bas_pick<<<((DS_T - 1) * DS_TOPK + 127) / 128, 128, 0, b.st>>>(b.cand, ds_lists_of(b), bas_d);
        if (invT > 0.f) {                    // sampled drafts: host uniforms in, q and the candidates out
            float u[DS_T - 1];
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            for (int i = 0; i < DS_T - 1; i++) u[i] = u01(qrng);
            DS_CHECK(hipMemcpyAsync(u_d, u, sizeof(u), hipMemcpyHostToDevice, b.st));
            ds_walk(b, bas_d, anchor, drafts_d, u_d, invT, q_d);
            DS_CHECK(hipMemcpyAsync(qd_h, q_d, sizeof(qd_h), hipMemcpyDeviceToHost, b.st));
            DS_CHECK(hipMemcpyAsync(qc_h, b.cand, sizeof(qc_h), hipMemcpyDeviceToHost, b.st));
        } else {
            ds_walk(b, bas_d, anchor, drafts_d);
        }
        int d[DS_T - 1];
        DS_CHECK(hipMemcpyAsync(d, drafts_d, (DS_T - 1) * 4, hipMemcpyDeviceToHost, b.st));
        DS_CHECK(hipStreamSynchronize(b.st));
        for (int i = 0; i < DS_T - 1; i++) drafts[i] = d[i];
    }

    // the committed rows' features (mv's capture) into the rings
    void fill_rows(int p, int n, int W) {
        for (int c = 0; c < QZ_NC; c++) { DS_CHECK(hipSetDevice(qz_dev_of(c))); qz_ds_ctx_rows(cx[c], W > 8 ? feat[c] : feat8[c], p, n); }
        mark(p, n);
        for (int c = 0; c < QZ_NC; c++) { DS_CHECK(hipSetDevice(qz_dev_of(c))); DS_CHECK(hipStreamSynchronize(cx[c].st)); }
        DS_CHECK(hipSetDevice(qz_dev_of(0)));
    }

    // the sampled walk over the round's 8 head rows, both cards' halves. out[0 .. n-1]
    // gets the accepted drafts and then the sampled bonus; returns n
    void topk_rows(qz_engine & eng, int W) {
        for (int c = 0; c < QZ_NC; c++) {
            DS_CHECK(hipSetDevice(qz_dev_of(c)));
            hipLaunchKernelGGL(qz_topk_rows, dim3(QZ_TOPK_NB, W), dim3(256), 0, mv[c].st, W > 8 ? mw[c].Ah.y_head : mv[c].Ah.y_head,
                               eng.hrows, c * eng.hrows, topk_d[c]);
            DS_CHECK(hipGetLastError());
        }
        for (int c = 0; c < QZ_NC; c++) {
            DS_CHECK(hipSetDevice(qz_dev_of(c)));
            DS_CHECK(hipStreamSynchronize(mv[c].st));
            eng.sam.tk[c].resize((size_t) W * QZ_TOPK_NB * QZ_TOPK);
            DS_CHECK(hipMemcpy(eng.sam.tk[c].data(), topk_d[c], (size_t) W * QZ_TOPK_NB * QZ_TOPK * sizeof(float2),
                               hipMemcpyDeviceToHost));
        }
        DS_CHECK(hipSetDevice(qz_dev_of(0)));
    }
    int sampled(qz_engine & eng, const int32_t * toks, int * out, int W) {
        topk_rows(eng, W);
        int m = 0;
        for (;; m++) {
            const int x = eng.sam.sample_row(m);
            out[m] = x;
            if (m < W - 1 && x == toks[m + 1]) continue;
            break;
        }
        return m + 1;
    }

    int sampled_q(qz_engine & eng, const int32_t * toks, int * out, int W) {
        topk_rows(eng, W);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::vector<float> w;
        for (int m = 0; m < W - 1; m++) {
            const int d = toks[m + 1];
            const int n = eng.sam.row_chain(m, w);
            // p over the chain's candidates (n == 1: a point mass)
            double S = 0.0;
            for (int i = 0; i < n; i++) S += n == 1 ? 1.0 : w[i];
            auto p_of = [&](int id) {
                for (int i = 0; i < n; i++) if (eng.sam.cand[i].second == id) return (n == 1 ? 1.0 : w[i]) / S;
                return 0.0;
            };
            const float * qr = qd_h + m * DS_TOPK;
            const int * qc = qc_h + m * DS_TOPK;
            auto q_of = [&](int id) {
                for (int j = 0; j < DS_TOPK; j++) if (qc[j] == id) return (double) qr[j];
                return 0.0;
            };
            const double pd = p_of(d), qd = q_of(d);
            if (u01(eng.sam.rng) * qd < pd) { out[m] = d; continue; }
            // reject: the residual max(0, p - q) over p's support (q-only ids have p = 0)
            std::vector<double> r(n);
            double R = 0.0;
            for (int i = 0; i < n; i++) {
                const double pi = (n == 1 ? 1.0 : w[i]) / S;
                r[i] = std::max(0.0, pi - q_of(eng.sam.cand[i].second));
                R += r[i];
            }
            int x = eng.sam.cand[0].second;
            const double tgt = R * u01(eng.sam.rng);
            if (R > 0.0) {
                double run = 0.0;
                x = eng.sam.cand[n - 1].second;
                for (int i = 0; i < n; i++) { run += r[i]; if (run >= tgt && r[i] > 0.0) { x = eng.sam.cand[i].second; break; } }
            }
            out[m] = x;
            return m + 1;
        }
        out[W - 1] = eng.sam.sample_row(W - 1);
        return W;
    }

    // Drafter switch. forced holds the rows already emitted from pos0, the last one the
    // anchor. sw_hook runs after each round with (pos, rows, W, toks, out); true ends
    // generate() with stop 4, sw_pc / sw_pend as in qz_engine.
    std::vector<int32_t> forced;
    std::function<bool(int, int, int, const int32_t *, const int *)> sw_hook;
    int sw_pc = 0;
    std::vector<int32_t> sw_pend;
    int generate(qz_engine & eng, int pos0, int32_t anchor, int max_new, const std::vector<int32_t> & stop,
                 qz_emit_fn emit, void * user, qz_gen_stats * out) {
        qz_gen_stats st;
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        eng.abort_flag = 0;
        auto push = [&](const int32_t * v, int n) {
            int k = 0;
            while (k < n && st.n_gen + k < max_new) {
                bool hit = false;
                for (int32_t s : stop) if (v[k] == s) { hit = true; break; }
                if (hit) { st.stop = 1; break; }
                k++;
            }
            if (k > 0 && emit && !emit(user, v, k)) st.stop = 3;
            st.n_gen += k;
            return st.stop != 0 || st.n_gen >= max_new;
        };
        const bool qmode = eng.sam.smp && eng.sam.smp->temp > 0.f;

        constexpr float qscale = 0.0025f;
        const float invT = qmode ? qscale / eng.sam.smp->temp : 0.f;
        if (qmode) qrng.seed(eng.sam.smp->seed ^ 0x5bd1e995u);
        int pos = pos0, n_rep = 0, cyc = 0;
        int32_t toks[MVW > DS_T ? MVW : DS_T];
        int ctl[QZ_NC][8], out_tok[MVW];
        // verify 12 rows after wk rounds in a row accepted >= wth, else 8 rows. 12-row
        // rounds sum in another order, which can flip near ties.
        constexpr int wth = 8, wk = 3;
        int W = 8, nw12 = 0, streak = 0;
        bool done = max_new == 0;
        bool fin = false;   // a switch's last round: at most 2 rows kept, so the next path redoes them in one go
        while (!done) {
            if (eng.abort_flag) { st.stop = 2; break; }
            if (pos + W > eng.steps) W = 8;
            if (pos + W > eng.steps) { st.stop = 0; break; }        // arena full: end like max_new
            nw12 += W > 8;
            const int nf = (int) forced.size();
            if (nf) {
                int32_t d[DS_T];
                for (int i = 0; i < nf; i++) toks[i] = forced[i];
                draft(forced[nf - 1], pos + nf - 1, d, invT, W);
                for (int i = nf; i < W; i++) toks[i] = d[i - nf];
            } else {
                toks[0] = anchor;
                draft(anchor, pos, toks + 1, invT, W);
            }
            if (W > 8) mv12::mv_round(mw, pos, cyc, n_rep, 1, toks, xtok);
            else mv8::mv_round(mv, pos, cyc, n_rep, 1, toks, xtok);
            for (int c = 0; c < QZ_NC; c++) CHECK(hipStreamSynchronize(mv[c].st));
            bool same = true;
            for (int c = 0; c < QZ_NC; c++) {
                const qz_trunk_args & Ac = W > 8 ? mw[c].Ah : mv[c].Ah;
                CHECK(hipSetDevice(qz_dev_of(c)));
                CHECK(hipMemcpy(ctl[c], Ac.ctl, 8 * 4, hipMemcpyDeviceToHost));
                if (c == 0) CHECK(hipMemcpy(out_tok, Ac.out_tok, W * 4, hipMemcpyDeviceToHost));
                else same = same && !memcmp(ctl[0], ctl[c], 8 * 4);
            }
            CHECK(hipSetDevice(qz_dev_of(0)));
            int n_acc = ctl[0][0];
            if (n_acc < 1 || n_acc > W || ctl[0][2] != pos + n_acc || !same) { st.stop = -1; break; }
            if (nf) {   // forced rows stand; acceptance from the last one (a sampled run keeps a draft only if equal)
                int n = nf - 1;
                if (eng.sam.smp) {
                    topk_rows(eng, W);
                    for (;; n++) {
                        out_tok[n] = eng.sam.sample_row(n);
                        if (n < W - 1 && out_tok[n] == toks[n + 1]) continue;
                        break;
                    }
                } else {
                    while (n < W - 1 && out_tok[n] == toks[n + 1]) n++;
                }
                n_acc = n + 1;
                forced.clear();
                st.decode_cycles++;
                st.decode_tok += n_acc - nf + 1;
                done = push(out_tok + nf - 1, n_acc - nf + 1);
            } else if (qmode) n_acc = sampled_q(eng, toks, out_tok, W);    // out_tok[0 .. n_acc-1] = the emitted tokens
            else if (eng.sam.smp) n_acc = sampled(eng, toks, out_tok, W);
            if (fin) n_acc = std::min(n_acc, 2);
            if (!nf) {
                st.decode_cycles++;
                st.decode_tok += n_acc;
                done = push(out_tok, n_acc);
            }
            fill_rows(pos, n_acc, W);
            if (fin && !done) {
                sw_pc = pos;
                sw_pend.assign(toks, toks + n_acc);
                sw_pend.push_back(out_tok[n_acc - 1]);
                st.stop = 4;
                break;
            }
            fin = !done && sw_hook && sw_hook(pos, n_acc, W, toks, out_tok);
            anchor = out_tok[n_acc - 1];
            n_rep = n_acc;
            pos += n_acc;
            streak = n_acc >= wth ? streak + 1 : 0;
            W = streak >= wk && !fin ? 12 : 8;
            cyc++;
        }
        st.decode_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        st.dspine = 1;
        st.wide_cycles = nw12;
        if (out) *out = st;
        return st.stop < 0 ? -1 : 0;
    }
};
