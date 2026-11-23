// The engine: TP2 load, paired launches, the MTP decode loop and sampling.
#pragma once

#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include "loader.h"
#include "core/devices.h"
#include "qwen35_spec.h"
#include "qwen35_tp2.h"
#include "qwen35_device.hip"
#include "kernels/topk.h"
#include "core/sampler.h"
#include "core/cards.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef CHECK
#error "CHECK already defined"
#endif
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// card geometry (before the short macros below, which clash with its member
// names); matches the hardcoded trunk shapes of qwen35_device.hip
static constexpr int g2_HV = qz_geo<>::HV, g2_CCH = qz_geo<>::CCH, g2_HKV = qz_geo<>::HKV,
                     g2_RING_PAR = qz_geo<>::RING_PAR;

#define D     5120
#define FF    17408
#define G     6144
#define CCH   10240
#define HV    48
#define S     128
#define AQ    12288
#define AKV   1024
#define HQ    24
#define HKV   4
#define HD    256
#define BLKPOS 64

struct qz_card {
    int dev = 0;
    qz_loader ld;
    hipStream_t st = nullptr;
    qz_barrier_state * bst = nullptr, * bst_m = nullptr;
    qz_trunk_args A = {};
    qz_mtp_w * m_d = nullptr;
    float * dn_all = nullptr, * cs_all = nullptr, * ring_all = nullptr;
    int * ctl_d = nullptr;
    std::vector<__half *> kvk, kvv;
    size_t arena_rows = 0;
    float2 * topk_d = nullptr;      // [QZ_TMAX][QZ_TOPK_NB][QZ_TOPK] sampled-verify candidates
    std::vector<qz_trunk_w> wh;     // host copy of the per-layer weight / state pointers (prefill)
    qz_mtp_w mw = {};
    __half * kvk_m = nullptr, * kvv_m = nullptr;
    size_t mtp_arena_rows = 0;
    qz_xbufs x;                     // exchange buffers (core/cards.h)
    // shared scratch: the cycle's and the verify's attention partials and the
    // prefill's buffers, never live together
    char * scr = nullptr;
    size_t scr_bytes = 0;
    // the buffers, 256-byte aligned, from offset 0 of the scratch; false
    // (nothing set) if they do not fit
    bool scr_carve(const std::vector<std::pair<void **, size_t>> & v) {
        size_t o = 0;
        for (const auto & b : v) o += (b.second + 255) / 256 * 256;
        if (!scr || o > scr_bytes) return false;
        o = 0;
        for (const auto & b : v) { *b.first = scr + o; o += (b.second + 255) / 256 * 256; }
        return true;
    }
};
inline size_t qz_scr_size(const std::vector<size_t> & v) {
    size_t o = 0;
    for (size_t b : v) o += (b + 255) / 256 * 256;
    return o;
}

// production generate(): timings and counts of one request
struct qz_gen_stats {
    double prefill_ms = 0, decode_ms = 0;   // wall: forced cycles incl. the boundary one / the rest
    int n_prompt = 0, n_gen = 0;            // prompt tokens, emitted tokens
    int prefill_cycles = 0, decode_cycles = 0, decode_tok = 0;   // decode_tok: accepted, overshoot incl.
    int stop = 0;                           // 0 max_new, 1 stop id, 2 abort, 3 callback, -1 error
    int dspine = 0, wide_cycles = 0;        // decoded on DSpine; its 12-row verify rounds
};
// emit callback: n new tokens; return false to stop the request
typedef bool (*qz_emit_fn)(void * user, const int32_t * toks, int n);

class qz_engine {
public:
    qz_card C[2];
    qz_trunk_args H = {};      // per-call fields; everything else is per card
    bool H_mtp = false;
    long long n_launch = 0;
    qz_hparams hp;
    int n_vocab = 0, hrows = 0, n_layer = 64, steps = 0;
    float eps = 1e-6f;
    int slot_t = 3, adapt_slack = 0;
    size_t scratch_min = 0;    // set before init: the other scratch users' largest need (qz_card::scr)

    // load both cards (weights, exchange buffers), check residency.
    // Returns 0 on success; nonzero (message printed) on failure.
    int init(const char * model, int steps, int slot_t, int adapt_slack);
    void launch_cyc(int T);
    void launch_mtp(int T);
    void sync2();           // both streams; exchange error words must be 0
    void reset_state();     // zero all persistent device state
    // rope position = arena position + s (images move the rope behind the arena)
    void set_rope_shift(int s) { for (auto & K : C) K.A.rope_shift = s; }
    // sampled decode for the next generate() calls (nullptr = greedy); seeds the RNG
    void set_sampling(const qz_sampling * s) { sam.set(s); }
    qz_sampler sam;
    int sampled_verify(int T, bool verify, int pos);   // after launch_cyc(T): ctl[0..2] from samples
    // prompt forced through the cycle, then MTP decode up to max_new tokens or a stop id;
    // pos0 > 0: state up to pos0 already in place (batched prefill). 0, or -1 on error
    int generate(const std::vector<int32_t> & prompt, int max_new,
                 const std::vector<int32_t> & stop, qz_emit_fn emit, void * user,
                 qz_gen_stats * out, int pos0 = 0);
    volatile int abort_flag = 0;          // set from another thread to end generate()
    // drafter switch: after each decode cycle (pos, rows); true ends generate()
    // with stop 4, sw_pc / sw_pend = the uncommitted rows + the next anchor
    std::function<bool(int, int)> sw_hook;
    int sw_pc = 0;
    std::vector<int32_t> sw_pend;

private:
    void put(qz_card & K);
};

// ---- implementation (header-only: qwen35_device.hip is a single-TU include) ----

#include <algorithm>
#include <chrono>


#include <algorithm>
#include <chrono>

// K / V arena pages for `steps` positions (trunk layers, the MTP layer) and the
// bytes both take per card; the preflight sizes the load from the same numbers
inline int qz_kv_pages(int steps) { return (steps + QZ_TMAX + 63) / 64 + 1; }
inline int qz_kv_pages_mtp(int steps) { return (steps + 2 * QZ_TMAX + 63) / 64 + 1; }
inline size_t qz_kv_bytes(int steps, int n_attn = 16) {
    return ((size_t) n_attn * qz_kv_pages(steps) + qz_kv_pages_mtp(steps)) * g2_HKV * BLKPOS * HD * 2 * 2;
}

inline int qz_engine::init(const char * path, int steps_, int slot_t_, int adapt_slack_) {
    slot_t = slot_t_;
    adapt_slack = adapt_slack_;
    steps = steps_;

    qz_gguf g;
    if (!g.open(path)) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    if (!hp.load(g)) { fprintf(stderr, "not a qwen35 model\n"); return 1; }
    if (hp.d_model != D || hp.n_head != HQ || hp.n_kv != HKV || hp.head_dim != HD ||
        hp.q_dim != AQ || hp.n_ff != FF || hp.n_layer != 64 || hp.n_attn != 16 || hp.n_recr != 48 ||
        hp.n_vocab % 64 != 0) {
        fprintf(stderr, "model shape differs from the hardcoded trunk\n");
        return 1;
    }
    n_vocab = hp.n_vocab;
    n_layer = hp.n_layer;
    hrows = n_vocab / 2;
    eps = -1.f;
    {
        const int64_t id = g.key("qwen35.attention.layer_norm_rms_epsilon");
        if (id >= 0) eps = gguf_get_val_f32(g.ctx, id);
    }
    if (eps < 0) eps = 1e-6f;
    printf("engine: eps %g, %d layers, vocab %d (%d per card), steps %d\n",
           eps, n_layer, n_vocab, hrows, steps);

    if (!qz_peer_enable()) return 1;

    const int n_pages = qz_kv_pages(steps);
    const int n_pages_m = qz_kv_pages_mtp(steps);
    for (int c = 0; c < 2; c++) {
        qz_card & K = C[c];
        K.dev = qz_dev_of(c);
        qz_mem_scope mem_target("target");
        K.ld.slice = qz_tp2_spec;
        if (!K.ld.load(path, g, qz_dev_of(c), c)) { fprintf(stderr, "card %d load failed\n", c); return 1; }
        printf("card %d: weights %.2f GB in %.1f s, %.2f GB free\n", c, K.ld.need_bytes / 1e9, K.ld.load_seconds, K.ld.free_after / 1e9);
        CHECK(hipSetDevice(qz_dev_of(c)));
        CHECK(hipStreamCreateWithFlags(&K.st, hipStreamNonBlocking));
        auto find2 = [&](int il, const char * suf) -> const qz_dev_tensor & {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.%s", il, suf);
            const int64_t i = K.ld.find(name);
            if (i < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
            return K.ld.ts[i];
        };
        auto fz = [&](void ** p, size_t b) { CHECK(qz_dmalloc(p, b)); CHECK(hipMemset(*p, 0, b)); };
        qz_mem_scope mem_cycle("cycle");
        CHECK(qz_dmalloc(&K.bst, sizeof(qz_barrier_state)));
        CHECK(qz_dmalloc(&K.bst_m, sizeof(qz_barrier_state)));
        qz_trunk_args & A = K.A;
        fz((void **) &A.x_rows, (size_t) QZ_TMAX * 2 * D * 4);
        fz((void **) &A.y_core, (size_t) QZ_TMAX * D * 4);
        fz((void **) &A.y_g, (size_t) QZ_TMAX * FF * 4);
        fz((void **) &A.y_u, (size_t) QZ_TMAX * FF * 4);
        fz((void **) &A.y_ffn, (size_t) QZ_TMAX * D * 4);
        fz((void **) &A.pl_vals, (size_t) QZ_TMAX * G);
        fz((void **) &A.pl_scal, (size_t) QZ_TMAX * (G / 32) * 4);
        fz((void **) &A.ffn_q, (size_t) QZ_TMAX * FF);
        fz((void **) &A.ffn_sc, (size_t) QZ_TMAX * (FF / 32) * 4);
        fz((void **) &A.nq_vals, (size_t) QZ_TMAX * D);
        fz((void **) &A.nq_scal, (size_t) QZ_TMAX * (D / 32) * 4);
        fz((void **) &A.y_gate, (size_t) QZ_TMAX * G * 4);
        fz((void **) &A.y_q, (size_t) QZ_TMAX * AQ * 4);
        fz((void **) &A.y_k, (size_t) QZ_TMAX * AKV * 4);
        fz((void **) &A.y_v, (size_t) QZ_TMAX * AKV * 4);
        fz((void **) &A.q_f16, (size_t) QZ_TMAX * HQ * HD * 2);
        {
            const size_t pmb = (size_t) QZ_TMAX * HQ * FL9_NS * 4, pab = pmb * HD;   // fl9c partials
            K.scr_bytes = std::max(qz_scr_size({ pmb, pmb, pab }), scratch_min);
            {
                qz_mem_scope mem_scr("scratch");
                fz((void **) &K.scr, K.scr_bytes);
            }
            if (!K.scr_carve({ { (void **) &A.pm, pmb }, { (void **) &A.pl, pmb }, { (void **) &A.pacc, pab } })) {
                fprintf(stderr, "card %d: cycle partials do not fit the scratch\n", c);
                return 1;
            }
        }
        fz((void **) &A.o_preg, (size_t) QZ_TMAX * HQ * HD * 4);
        fz((void **) &A.y_head, (size_t) QZ_TMAX * hrows * 4);
        fz((void **) &A.out_tok, QZ_TMAX * 4);
        fz((void **) &A.h_nextn, (size_t) QZ_TMAX * D * 4);
        fz((void **) &A.mtp_h, (size_t) QZ_TMAX * D * 4);
        fz((void **) &A.cat_vals, (size_t) QZ_TMAX * 2 * D);
        fz((void **) &A.cat_scal, (size_t) QZ_TMAX * (2 * D / 32) * 4);
        fz((void **) &A.mtp_yhead, (size_t) QZ_TMAX * hrows * 4);
        fz((void **) &A.y_part, (size_t) QZ_TMAX * D * 4);
        // persistent state, card-local geometry
        fz((void **) &K.dn_all, (size_t) hp.n_recr * g2_HV * S * S * 4);
        fz((void **) &K.cs_all, (size_t) hp.n_recr * 3 * g2_CCH * 4);
        fz((void **) &K.ring_all, (size_t) hp.n_recr * 2 * g2_RING_PAR * 4);
        fz((void **) &K.ctl_d, 8 * sizeof(int));
        fz((void **) &K.topk_d, (size_t) QZ_TMAX * QZ_TOPK_NB * QZ_TOPK * sizeof(float2));
        K.arena_rows = (size_t) n_pages * g2_HKV * BLKPOS * HD;
        K.kvk.resize(hp.n_attn); K.kvv.resize(hp.n_attn);
        for (int a = 0; a < hp.n_attn; a++) {
            qz_mem_scope mem_kv("kv");
            fz((void **) &K.kvk[a], K.arena_rows * 2);
            fz((void **) &K.kvv[a], K.arena_rows * 2);
        }
        K.mtp_arena_rows = (size_t) n_pages_m * g2_HKV * BLKPOS * HD;
        {
            qz_mem_scope mem_kvm("kv mtp");
            fz((void **) &K.kvk_m, K.mtp_arena_rows * 2);
            fz((void **) &K.kvv_m, K.mtp_arena_rows * 2);
        }
        // exchange: inbox data + checksums, head-merge pairs, flag (own line)
        CHECK(K.x.alloc(QZ_TMAX, D));

        std::vector<qz_trunk_w> wh(n_layer);
        int nr = 0, na = 0;
        for (int il = 0; il < n_layer; il++) {
            qz_trunk_w & W = wh[il];
            W = {};
            W.recr = hp.is_recr(il) ? 1 : 0;
            auto q8 = [&](const char * suf, const char *& v, const __half *& s) {
                const qz_dev_tensor & t = find2(il, suf);
                v = (const char *) t.vals; s = (const __half *) t.scal;
            };
            W.attn_norm = (const float *) find2(il, "attn_norm.weight").vals;
            W.post_norm = (const float *) find2(il, "post_attention_norm.weight").vals;
            q8("ffn_gate.weight", W.fg_vals, W.fg_scal);
            q8("ffn_up.weight", W.fu_vals, W.fu_scal);
            q8("ffn_down.weight", W.fd_vals, W.fd_scal);
            if (W.recr) {
                q8("attn_qkv.weight", W.qkv_vals, W.qkv_scal);
                q8("attn_gate.weight", W.gat_vals, W.gat_scal);
                q8("ssm_beta.weight", W.bet_vals, W.bet_scal);
                q8("ssm_alpha.weight", W.alp_vals, W.alp_scal);
                q8("ssm_out.weight", W.out_vals, W.out_scal);
                W.conv1d = (const float *) find2(il, "ssm_conv1d.weight").vals;
                W.dt_bias = (const float *) find2(il, "ssm_dt.bias").vals;
                W.ssm_a = (const float *) find2(il, "ssm_a").vals;
                W.ssm_norm_w = (const float *) find2(il, "ssm_norm.weight").vals;
                W.dn_state = K.dn_all + (size_t) nr * g2_HV * S * S;
                W.conv_state = K.cs_all + (size_t) nr * 3 * g2_CCH;
                W.ring = K.ring_all + (size_t) nr * 2 * g2_RING_PAR;
                nr++;
            } else {
                q8("attn_q.weight", W.q_vals, W.q_scal);
                q8("attn_k.weight", W.k_vals, W.k_scal);
                q8("attn_v.weight", W.v_vals, W.v_scal);
                q8("attn_output.weight", W.out_vals, W.out_scal);
                W.q_norm_w = (const float *) find2(il, "attn_q_norm.weight").vals;
                W.k_norm_w = (const float *) find2(il, "attn_k_norm.weight").vals;
                W.kv_k = K.kvk[na];
                W.kv_v = K.kvv[na];
                na++;
            }
        }
        if (nr != hp.n_recr || na != hp.n_attn) { fprintf(stderr, "layer kind count mismatch\n"); return 1; }
        qz_trunk_w * w_d;
        CHECK(qz_dmalloc(&w_d, sizeof(qz_trunk_w) * n_layer));
        CHECK(hipMemcpy(w_d, wh.data(), sizeof(qz_trunk_w) * n_layer, hipMemcpyHostToDevice));
        K.wh = wh;

        qz_mtp_w mw = {};
        {
            const int il = n_layer;
            auto q8 = [&](const char * suf, const char *& v, const __half *& s) {
                const qz_dev_tensor & t = find2(il, suf);
                v = (const char *) t.vals; s = (const __half *) t.scal;
            };
            mw.L.recr = 0;
            mw.L.attn_norm = (const float *) find2(il, "attn_norm.weight").vals;
            mw.L.post_norm = (const float *) find2(il, "post_attention_norm.weight").vals;
            q8("ffn_gate.weight", mw.L.fg_vals, mw.L.fg_scal);
            q8("ffn_up.weight", mw.L.fu_vals, mw.L.fu_scal);
            q8("ffn_down.weight", mw.L.fd_vals, mw.L.fd_scal);
            q8("attn_q.weight", mw.L.q_vals, mw.L.q_scal);
            q8("attn_k.weight", mw.L.k_vals, mw.L.k_scal);
            q8("attn_v.weight", mw.L.v_vals, mw.L.v_scal);
            q8("attn_output.weight", mw.L.out_vals, mw.L.out_scal);
            mw.L.q_norm_w = (const float *) find2(il, "attn_q_norm.weight").vals;
            mw.L.k_norm_w = (const float *) find2(il, "attn_k_norm.weight").vals;
            q8("nextn.eh_proj.weight", mw.eh_vals, mw.eh_scal);
            mw.enorm = (const float *) find2(il, "nextn.enorm.weight").vals;
            mw.hnorm = (const float *) find2(il, "nextn.hnorm.weight").vals;
            mw.shnorm = (const float *) find2(il, "nextn.shared_head_norm.weight").vals;
            mw.L.kv_k = K.kvk_m;
            mw.L.kv_v = K.kvv_m;
        }
        CHECK(qz_dmalloc(&K.m_d, sizeof(qz_mtp_w)));
        CHECK(hipMemcpy(K.m_d, &mw, sizeof(qz_mtp_w), hipMemcpyHostToDevice));
        K.mw = mw;

        const int64_t emb_i = K.ld.find("token_embd.weight"), outn_i = K.ld.find("output_norm.weight"),
                      head_i = K.ld.find("output.weight");
        if (emb_i < 0 || outn_i < 0 || head_i < 0) { fprintf(stderr, "missing embedding, output norm or untied head\n"); return 1; }
        if (K.ld.ts[head_i].ne1 != hrows) { fprintf(stderr, "head slice rows %" PRId64 " != %d\n", K.ld.ts[head_i].ne1, hrows); return 1; }
        A.emb_vals = (const int *) K.ld.ts[emb_i].vals;
        A.emb_scal = (const __half *) K.ld.ts[emb_i].scal;
        A.out_norm = (const float *) K.ld.ts[outn_i].vals;
        A.hd_vals = (const char *) K.ld.ts[head_i].vals;
        A.hd_scal = (const __half *) K.ld.ts[head_i].scal;
        A.w = w_d;
        A.ctl = K.ctl_d;
        A.mtp = nullptr;
        A.n_layer = n_layer; A.eps = eps;
        A.tp_card = c; A.hrows = hrows;
        A.xin_mine = K.x.xin; A.xhm_mine = K.x.xhm; A.xflag_mine = K.x.xflag;
        A.xseq = K.x.xseq; A.xerr = K.x.xerr;

        // dynamic LDS opt-in per device
        const int xs_b = QZ_XSTAGE_BYTES;   // TP2 push stage past the cycle LDS
        const int sm1 = (int) qz_cycle_smem<1>() + xs_b, sm2 = (int) qz_cycle_smem<2>() + xs_b, sm3 = (int) qz_cycle_smem<3>() + xs_b, sm4 = (int) qz_cycle_smem<4>() + xs_b;
        CHECK(hipFuncSetAttribute((const void *) qz_cycle<1>, hipFuncAttributeMaxDynamicSharedMemorySize, sm1));
        CHECK(hipFuncSetAttribute((const void *) qz_cycle<2>, hipFuncAttributeMaxDynamicSharedMemorySize, sm2));
        CHECK(hipFuncSetAttribute((const void *) qz_cycle<3>, hipFuncAttributeMaxDynamicSharedMemorySize, sm3));
        CHECK(hipFuncSetAttribute((const void *) qz_cycle<4>, hipFuncAttributeMaxDynamicSharedMemorySize, sm4));
        CHECK(hipFuncSetAttribute((const void *) qz_mtp<3>, hipFuncAttributeMaxDynamicSharedMemorySize, sm3));
        CHECK(hipFuncSetAttribute((const void *) qz_mtp<4>, hipFuncAttributeMaxDynamicSharedMemorySize, sm4));
        // the grid barrier needs all 60 blocks resident: check every specialization
        {
            const qz_resident_kernel ks[] = {
                {(const void *) qz_cycle<1>, sm1, "qz_cycle<1>"}, {(const void *) qz_cycle<2>, sm2, "qz_cycle<2>"},
                {(const void *) qz_cycle<3>, sm3, "qz_cycle<3>"}, {(const void *) qz_cycle<4>, sm4, "qz_cycle<4>"},
                {(const void *) qz_mtp<3>, sm3, "qz_mtp<3>"}, {(const void *) qz_mtp<4>, sm4, "qz_mtp<4>"}};
            if (!qz_resident(qz_dev_of(c), ks, (int) (sizeof(ks) / sizeof(ks[0])), 60, 512)) return 1;
        }
    }
    for (int c = 0; c < 2; c++) {
        C[c].A.xin_peer = C[1 - c].x.xin;
        C[c].A.xhm_peer = C[1 - c].x.xhm;
        C[c].A.xflag_peer = C[1 - c].x.xflag;
    }
    {
        hipFuncAttributes fa;
        CHECK(hipSetDevice(qz_dev_of(0)));
        CHECK(hipFuncGetAttributes(&fa, (const void *) qz_cycle<3>));
        printf("qz_cycle<3> regs %d, scratch %zu B, smem %.1f KB; ", fa.numRegs, fa.localSizeBytes, qz_cycle_smem<3>() / 1024.0);
        CHECK(hipFuncGetAttributes(&fa, (const void *) qz_mtp<3>));
        printf("qz_mtp<3> regs %d, scratch %zu B\n", fa.numRegs, fa.localSizeBytes);
        CHECK(hipFuncGetAttributes(&fa, (const void *) qz_cycle<1>));
        printf("qz_cycle<1> regs %d, scratch %zu B\n", fa.numRegs, fa.localSizeBytes);
    }
    if (60 * QZ_XSLAB * 32 < D) { fprintf(stderr, "push stage too small for 60 blocks\n"); return 1; }
    printf("loaded\n");
    return 0;
}

// ---- paired launch helpers (one host thread, both cards, own streams) ----

inline void qz_engine::put(qz_card & K) {
    qz_trunk_args & A = K.A;
    memcpy(A.toks, H.toks, sizeof(A.toks));
    A.pos = H.pos; A.cycle = H.cycle; A.verify = H.verify; A.mtp_t = H.mtp_t;
    A.mtp = H_mtp ? K.m_d : nullptr;
}

inline void qz_engine::launch_cyc(int T) {
    for (int c = 0; c < 2; c++) {
        qz_card & K = C[c];
        CHECK(hipSetDevice(qz_dev_of(c)));
        put(K);
        CHECK(hipMemsetAsync(K.bst, 0, sizeof(qz_barrier_state), K.st));
        switch (T) {
            case 1: hipLaunchKernelGGL((qz_cycle<1>), dim3(60), dim3(512), qz_cycle_smem<1>() + QZ_XSTAGE_BYTES, K.st, K.bst, K.A); break;
            case 2: hipLaunchKernelGGL((qz_cycle<2>), dim3(60), dim3(512), qz_cycle_smem<2>() + QZ_XSTAGE_BYTES, K.st, K.bst, K.A); break;
            case 3: hipLaunchKernelGGL((qz_cycle<3>), dim3(60), dim3(512), qz_cycle_smem<3>() + QZ_XSTAGE_BYTES, K.st, K.bst, K.A); break;
            default: hipLaunchKernelGGL((qz_cycle<4>), dim3(60), dim3(512), qz_cycle_smem<4>() + QZ_XSTAGE_BYTES, K.st, K.bst, K.A); break;
        }
        CHECK(hipGetLastError());
    }
    n_launch++;
    CHECK(hipSetDevice(qz_dev_of(0)));
}

inline void qz_engine::launch_mtp(int T) {
    for (int c = 0; c < 2; c++) {
        qz_card & K = C[c];
        CHECK(hipSetDevice(qz_dev_of(c)));
        put(K);
        CHECK(hipMemsetAsync(K.bst_m, 0, sizeof(qz_barrier_state), K.st));
        if (T == 4) hipLaunchKernelGGL((qz_mtp<4>), dim3(60), dim3(512), qz_cycle_smem<4>() + QZ_XSTAGE_BYTES, K.st, K.bst_m, K.A);
        else        hipLaunchKernelGGL((qz_mtp<3>), dim3(60), dim3(512), qz_cycle_smem<3>() + QZ_XSTAGE_BYTES, K.st, K.bst_m, K.A);
        CHECK(hipGetLastError());
    }
    n_launch++;
    CHECK(hipSetDevice(qz_dev_of(0)));
}

inline void qz_engine::sync2() {
    for (int c = 0; c < 2; c++) { CHECK(hipSetDevice(qz_dev_of(c))); CHECK(hipStreamSynchronize(C[c].st)); }
    CHECK(hipSetDevice(qz_dev_of(0)));
    for (int c = 0; c < 2; c++) qz_xerr_check(c, C[c].x.xerr, n_launch);
}

inline void qz_engine::reset_state() {
    for (int c = 0; c < 2; c++) {
        qz_card & K = C[c];
        CHECK(hipSetDevice(qz_dev_of(c)));
        CHECK(hipMemset(K.dn_all, 0, (size_t) hp.n_recr * g2_HV * S * S * 4));
        CHECK(hipMemset(K.cs_all, 0, (size_t) hp.n_recr * 3 * g2_CCH * 4));
        CHECK(hipMemset(K.ring_all, 0, (size_t) hp.n_recr * 2 * g2_RING_PAR * 4));
        CHECK(hipMemset(K.ctl_d, 0, 8 * sizeof(int)));
        CHECK(hipMemset(K.kvk_m, 0, K.mtp_arena_rows * 2));
        CHECK(hipMemset(K.kvv_m, 0, K.mtp_arena_rows * 2));
        for (int a = 0; a < hp.n_attn; a++) {
            CHECK(hipMemset(K.kvk[a], 0, K.arena_rows * 2));
            CHECK(hipMemset(K.kvv[a], 0, K.arena_rows * 2));
        }
        CHECK(hipDeviceSynchronize());
    }
    CHECK(hipSetDevice(qz_dev_of(0)));
}

inline int qz_engine::generate(const std::vector<int32_t> & prompt_in, int max_new,
                               const std::vector<int32_t> & stop, qz_emit_fn emit, void * user,
                               qz_gen_stats * out, int pos0) {
    qz_gen_stats st;
    // positions pos0 .. pos0 + size - 1; prompt[pos] below indexes absolute positions
    struct { const std::vector<int32_t> & v; int o; int32_t operator[](int p) const { return v[p - o]; } } prompt{prompt_in, pos0};
    const int P = pos0 + (int) prompt_in.size();
    st.n_prompt = (int) prompt_in.size();
    if (prompt_in.empty() || pos0 < 0 || max_new < 0 || P + max_new + 2 * QZ_TMAX > steps) { st.stop = -1; if (out) *out = st; return -1; }
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const int TF = slot_t ? slot_t : 3;          // forced chunk rows
    int ctl[8];
    if (pos0 == 0) reset_state();
    abort_flag = 0;
    H_mtp = true;
    H.mtp_t = 0;
    int pos = pos0;
    // prefill: forced chunks; the last chunk's ctl[1] is the model's own token
    // after the prompt
    while (pos < P) {
        const int r = std::min(TF, P - pos);
        const bool last = pos + r == P;
        for (int t = 0; t < QZ_TMAX; t++) H.toks[t] = t < r ? prompt[pos + t] : 0;
        H.verify = 0;
        H.pos = pos; H.cycle = st.prefill_cycles;
        // the last chunk's qz_mtp drafts the first decode cycle; earlier chunks
        // only need the MTP K/V catch-up
        H.mtp_t = slot_t ? 0 : (last ? -(r + 8 * adapt_slack) : 3);
        launch_cyc(r);
        if (last && sam.smp) sampled_verify(r, false, pos);   // the first decode token, sampled
        if (!last)
            for (int c = 0; c < 2; c++) {
                CHECK(hipSetDevice(qz_dev_of(c)));
                CHECK(hipMemsetD32Async((hipDeviceptr_t)(C[c].ctl_d + 1), prompt[pos + r], 1, C[c].st));
            }
        CHECK(hipSetDevice(qz_dev_of(0)));
        launch_mtp(slot_t ? slot_t : (last ? 4 : 3));
        sync2();
        CHECK(hipMemcpy(ctl, C[0].ctl_d, 8 * sizeof(int), hipMemcpyDeviceToHost));
        st.prefill_cycles++;
        if (ctl[0] != r || ctl[2] != pos + r) { st.stop = -1; break; }
        pos += r;
    }
    st.prefill_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    if (st.stop < 0) { H_mtp = false; H.mtp_t = 0; if (out) *out = st; return -1; }
    const auto t1 = clk::now();
    // emit up to `room` tokens of v, cut at the first stop id
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
        if (st.stop == 0 && st.n_gen >= max_new) st.stop = 0;
        return st.stop != 0 || st.n_gen >= max_new;
    };
    int TT = slot_t ? slot_t : ctl[7];
    if (TT < 3 || TT > 4) TT = 3;
    // the first generated token: the last prefill chunk's bonus
    bool done = max_new == 0 || push(&ctl[1], 1);
    H.toks[0] = ctl[1];
    for (int i = 1; i < TT; i++) H.toks[i] = ctl[2 + i];
    while (!done) {
        if (abort_flag) { st.stop = 2; break; }
        H.verify = 1;
        H.mtp_t = slot_t ? 0 : -(TT + 8 * adapt_slack);
        H.pos = pos; H.cycle = st.prefill_cycles + st.decode_cycles;
        if (pos + TT + 2 * QZ_TMAX > steps) { st.stop = 0; break; }   // arena full: end like max_new
        launch_cyc(TT);
        if (sam.smp) sampled_verify(TT, true, pos);
        launch_mtp(slot_t ? TT : 4);
        sync2();
        CHECK(hipMemcpy(ctl, C[0].ctl_d, 8 * sizeof(int), hipMemcpyDeviceToHost));
        st.decode_cycles++;
        const int n_acc = ctl[0];
        if (n_acc < 1 || n_acc > TT || ctl[2] != pos + n_acc) { st.stop = -1; break; }
        int32_t acc[QZ_TMAX + 1];
        for (int t = 1; t < n_acc; t++) acc[t - 1] = H.toks[t];
        acc[n_acc - 1] = ctl[1];
        st.decode_tok += n_acc;
        done = push(acc, n_acc);
        if (!done && sw_hook && sw_hook(pos, n_acc)) {
            sw_pc = pos;
            sw_pend.assign(H.toks, H.toks + n_acc);
            sw_pend.push_back(ctl[1]);
            st.stop = 4;
            break;
        }
        const int t_next = slot_t ? TT : ctl[7];
        if (t_next < 3 || t_next > 4) { st.stop = -1; break; }
        H.toks[0] = ctl[1];
        for (int i = 1; i < t_next; i++) H.toks[i] = ctl[2 + i];
        TT = t_next;
        pos += n_acc;
    }
    st.decode_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
    H_mtp = false;
    H.mtp_t = 0;
    if (out) *out = st;
    return st.stop < 0 ? -1 : 0;
}

inline int qz_engine::sampled_verify(int T, bool verify, int pos) {
    // candidates of rows 0 .. T-1 on both cards (the cycle's y_head, stream order)
    for (int c = 0; c < 2; c++) {
        qz_card & K = C[c];
        CHECK(hipSetDevice(qz_dev_of(c)));
        hipLaunchKernelGGL(qz_topk_rows, dim3(QZ_TOPK_NB, T), dim3(256), 0, K.st, K.A.y_head, hrows, c * hrows, K.topk_d);
        CHECK(hipGetLastError());
    }
    for (int c = 0; c < 2; c++) {
        CHECK(hipSetDevice(qz_dev_of(c)));
        CHECK(hipStreamSynchronize(C[c].st));
        sam.tk[c].resize((size_t) QZ_TMAX * QZ_TOPK_NB * QZ_TOPK);
        CHECK(hipMemcpy(sam.tk[c].data(), C[c].topk_d, (size_t) T * QZ_TOPK_NB * QZ_TOPK * sizeof(float2), hipMemcpyDeviceToHost));
    }
    int m = 0, x = 0;
    if (verify) {
        for (;; m++) {
            x = sam.sample_row(m);
            if (m < T - 1 && x == H.toks[m + 1]) continue;
            break;
        }
    } else {
        m = T - 1;
        x = sam.sample_row(m);
    }
    const int c3[3] = { m + 1, x, pos + m + 1 };   // as the greedy verify writes them
    for (int c = 0; c < 2; c++) {
        CHECK(hipSetDevice(qz_dev_of(c)));
        CHECK(hipMemcpy(C[c].ctl_d, c3, sizeof(c3), hipMemcpyHostToDevice));
    }
    CHECK(hipSetDevice(qz_dev_of(0)));
    return m + 1;
}
