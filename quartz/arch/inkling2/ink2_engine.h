// inkling2 engine: weights, buffers and the per-token tables of the forward pass of
// docs/inkling2.md, on both cards. Decode runs in the persistent kernel (ink2_cycle.h),
// one launch per card per token. TP2: each card runs its q / kv heads and half of every
// expert's FF. The attention output and the expert down projection are summed across the
// cards, each card reading the other's partial after a cross-stream event wait: out of its
// VRAM over P2P, or out of a pinned host mailbox both cards address (core/link.h) when
// there is no peer access. The kernels are the same either way.
#pragma once

#include "ink2_verify.h"     // the speculative round's VR-row kernel (after ink2_cycle.h)

#define ICHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "inkling2: HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct qw { const char * v = nullptr; const __half * s = nullptr; int K = 0, N = 0; };

// one physical layer's weights on one card
struct lw {
    qw q, gate, k, v, rel, sink, kcp, vcp, out, gu, dn;
    const float * kconv, * vconv, * pconv, * fconv, * qclip, * rclip, * relb;
    const float * an = nullptr, * qn = nullptr, * kn = nullptr, * vn = nullptr, * resq = nullptr;
    const float * ffq, * ffn, * gin, * pb;
    const float * msgq = nullptr, * cd[NCTRL] = {}, * cg[NCTRL] = {}, * cf[NCTRL] = {};
    const float * cgt[NCTRL] = {}, * cft[NCTRL] = {};          // cg / cf transposed (the cycle's ctrl_up)
};

struct ink2_card {
    qz_loader ld;
    hipStream_t st = nullptr;
    hipEvent_t ev = nullptr;
    lw L[NL];
    qw embd, head;
    const float * embd_norm, * out_norm, * out_q;
    // activations (f32 [D] unless noted)
    float * X, * P, * OUT[2], * BLK[8], * IN[3], * MSG[NCTRL], * CT[NCTRL], * F, * FN, * G;
    float * QR, * GR, * RR, * SK, * KR, * VR, * DK, * DV, * TB, * PART[2], * H, * WTS, * LOG;
    __half * qh;                              // q as f16 * log2(e)/sqrt(HD), what attention reads
    int * IDS;
    unsigned char * aq[4]; float * as[4];      // input planes: Q, K, V, then the ffn / head input
    unsigned char * oq; float * os;           // attention output planes [QL]
    float * apart = nullptr; int * acnt = nullptr;   // attention split partials / counters
    unsigned char * hq; float * hs;           // expert activation planes [NE_USED][FFL]
    float2 * tk;                              // head top-K [QZ_TOPK_NB][QZ_TOPK]
    __half * kc[NAPP], * vc[NAPP];            // per slot: full [ctx][KVL], SWA ring [WIN][KVL]
    float * hist;                             // [NAPP][3][CHL]
    int nvalid = 0, id0 = 0;                  // head rows that are real vocab ids, the first id
    // verify (ink2_verify.h): the VR row replicas of the activations, rs bytes apart
    size_t rs = 0;                            // one activation row's stride (bytes, 256-aligned)
    size_t apsz = 0;                          // attention partials per row (floats)
    float * uh = nullptr;                     // [NAPP][VR][CHL]: each row's conv input
    float * LOGV = nullptr;                   // [VR][nvalid]: the rows' head logits
    float2 * vtk = nullptr;                   // [VR][QZ_TOPK_NB][QZ_TOPK]
    unsigned int * vbst = nullptr;            // the verify grid barrier + chain group flags
    void * vargs_d = nullptr;                 // the verify args
    float * vws = nullptr;                    // the verify attention's wave states
    // persistent decode (ink2_cycle.h)
    qz_xbufs x;                               // TP2 exchange buffers (core/cards.h)
    unsigned int * bst = nullptr;      // the grid barrier flags (CYC_NBMAX)
    int ncu = 0;
    cyc_args ca{};                            // static args, uploaded to ca_d by plan_all
    void * ca_d = nullptr;
};

struct ink2_engine {
    hparams hp;
    int ctx = 0;
    ink2_card C[QZ_NC];
    int nx = 0;                               // exchanges issued (partial double-buffer parity)
    std::vector<float2> tkh[QZ_NC];
    float2 * tkp[QZ_NC] = {};     // pinned: the top-K, then the exchange error words (8 ints)
    std::vector<float2> vtkh[QZ_NC];
    float2 * vtkp[QZ_NC] = {};    // pinned: the verify rows' top-K, then the error words

    // ---- setup
    static qw Qw(const qz_loader & ld, const char * n) {
        const qz_dev_tensor * t = ld.get(n);
        if (!t || t->type != GGML_TYPE_Q8_0) { fprintf(stderr, "inkling2: missing Q8_0 %s\n", n); exit(1); }
        qw r; r.v = (const char *) t->vals; r.s = (const __half *) t->scal; r.K = (int) t->ne0; r.N = (int) t->ne1;
        return r;
    }
    static const float * Fw(const qz_loader & ld, const char * n, bool need = true) {
        const qz_dev_tensor * t = ld.get(n);
        if (!t) { if (need) { fprintf(stderr, "inkling2: missing %s\n", n); exit(1); } return nullptr; }
        return (const float *) t->vals;
    }
    void bind(int c) {
        ink2_card & K = C[c];
        const qz_loader & ld = K.ld;
        char n[96];
        auto nm = [&](int il, const char * s) { snprintf(n, sizeof(n), "blk.%d.%s", il, s); return n; };
        K.embd = Qw(ld, "token_embd.weight");
        K.head = Qw(ld, "output.weight");
        K.embd_norm = Fw(ld, "token_embd_norm.weight");
        K.out_norm = Fw(ld, "output_norm.weight");
        K.out_q = Fw(ld, "output_res_query.weight");
        for (int il = 0; il < NL; il++) {
            lw & L = K.L[il];
            L.q = Qw(ld, nm(il, "attn_q.weight"));       L.gate = Qw(ld, nm(il, "attn_gate.weight"));
            L.k = Qw(ld, nm(il, "attn_k.weight"));       L.v = Qw(ld, nm(il, "attn_v.weight"));
            L.rel = Qw(ld, nm(il, "attn_rel.weight"));   L.sink = Qw(ld, nm(il, "attn_sink_proj.weight"));
            L.out = Qw(ld, nm(il, "attn_output.weight"));
            L.gu = Qw(ld, nm(il, "ffn_gate_up_exps.weight")); L.dn = Qw(ld, nm(il, "ffn_down_exps.weight"));
            L.kconv = Fw(ld, nm(il, "attn_k_conv.weight"));   L.vconv = Fw(ld, nm(il, "attn_v_conv.weight"));
            L.pconv = Fw(ld, nm(il, "attn_post_conv.weight")); L.fconv = Fw(ld, nm(il, "ffn_post_conv.weight"));
            L.qclip = Fw(ld, nm(il, "attn_q_clip.weight"));   L.rclip = Fw(ld, nm(il, "attn_rel_clip.weight"));
            L.relb = Fw(ld, nm(il, "attn_rel_bias.weight"));
            L.ffq = Fw(ld, nm(il, "ffn_res_query.weight"));   L.ffn = Fw(ld, nm(il, "ffn_norm.weight"));
            L.gin = Fw(ld, nm(il, "ffn_gate_inp.weight"));    L.pb = Fw(ld, nm(il, "exp_probs_b.bias"));
            if (il > 0) L.resq = Fw(ld, nm(il, "attn_res_query.weight"));
            if (is_full(il)) {
                L.qn = Fw(ld, nm(il, "attn_q_inp_norm.weight")); L.kn = Fw(ld, nm(il, "attn_k_inp_norm.weight"));
                L.vn = Fw(ld, nm(il, "attn_v_inp_norm.weight"));
                L.kcp = Qw(ld, nm(il, "attn_k_conv_proj.weight")); L.vcp = Qw(ld, nm(il, "attn_v_conv_proj.weight"));
            } else {
                L.an = Fw(ld, nm(il, "attn_norm.weight"));
            }
            if (is_loop(il)) {
                L.msgq = Fw(ld, nm(il, "loop_msg_query.weight"));
                const char * cn[NCTRL] = {"q", "k", "v", "ffn"};
                for (int j = 0; j < NCTRL; j++) {
                    char s[64];
                    snprintf(s, sizeof(s), "loop_%s_ctrl_down.weight", cn[j]);   L.cd[j] = Fw(ld, nm(il, s));
                    snprintf(s, sizeof(s), "loop_%s_ctrl_gain.weight", cn[j]);   L.cg[j] = Fw(ld, nm(il, s));
                    snprintf(s, sizeof(s), "loop_%s_ctrl_forget.weight", cn[j]); L.cf[j] = Fw(ld, nm(il, s));
                    float * t = nullptr;
                    ICHECK(qz_dmalloc(&t, 2 * (size_t) GRANK * D * sizeof(float), "weights"));
                    k_ctrl_t<<<(D + 255) / 256, 256>>>(L.cg[j], t);
                    k_ctrl_t<<<(D + 255) / 256, 256>>>(L.cf[j], t + (size_t) GRANK * D);
                    L.cgt[j] = t; L.cft[j] = t + (size_t) GRANK * D;
                }
            }
        }
        {   // gate_up rows interleaved per 32 (gu_row), so the prefill GEMM's 64-row tiles pair for SITU
            const size_t nv = (size_t) NE * GUL * D, ns = (size_t) NE * GUL * (D / 32);
            char * tv; __half * ts;
            ICHECK(hipMalloc(&tv, nv)); ICHECK(hipMalloc(&ts, ns * 2));
            for (int il = 0; il < NL; il++) {
                qw & W = K.L[il].gu;
                ICHECK(hipMemcpy(tv, W.v, nv, hipMemcpyDeviceToDevice));
                ICHECK(hipMemcpy(ts, W.s, ns * 2, hipMemcpyDeviceToDevice));
                k_gu_interleave<<<dim3(GUL, NE), 256>>>(tv, ts, (char *) W.v, (__half *) W.s);
            }
            ICHECK(hipDeviceSynchronize());
            ICHECK(hipFree(tv)); ICHECK(hipFree(ts));
        }
        const int share = ink2_head_share(VOCAB, QZ_NC);   // whole 32-row blocks per card
        K.id0 = share * c;
        K.nvalid = std::min(share, VOCAB_REAL - K.id0);
    }
    bool alloc(int c) {
        ink2_card & K = C[c];
        ICHECK(hipSetDevice(qz_dev_of(c)));
        qz_mem_scope mem("state");
        size_t need = 0;
        auto f32 = [&](size_t n) { size_t b = (n * 4 + 255) & ~(size_t) 255; need += b; return b; };
        // sizes first, then one allocation carved in the same order
        std::vector<size_t> sz;
        auto add = [&](size_t bytes) { sz.push_back(bytes); };
        const size_t nD = D;
        for (int i = 0; i < 1 + 1 + 2 + 8 + 3 + NCTRL + NCTRL + 2; i++) add(f32(nD));   // X P OUT BLK IN MSG CT F FN
        add(f32(NCTRL * GRANK));                                                          // G
        add(f32(QL)); add(f32(QL)); add(f32(HQL * DREL)); add(f32(64)); add(f32(KVL)); add(f32(KVL));
        add(f32(64)); add(f32(64)); add(f32(QL / 2)); add(f32(HQL * TBW)); add(f32(nD)); add(f32(nD));
        add(f32(NE_USED * FFE)); add(f32(NE_USED)); add(f32(ink2_head_share(VOCAB, QZ_NC))); add(f32(NE_USED));
        for (int i = 0; i < 4; i++) { add(f32(nD / 4)); add(f32(nD / 32)); }              // planes (int8 = n/4 floats)
        add(f32(QL / 4)); add(f32(QL / 32));
        add(f32(NE_USED * FFL / 4)); add(f32(NE_USED * FFL / 32));
        add(f32(QZ_TOPK_NB * QZ_TOPK * 2));
        // the per-token activations above are replicated VR times. Row r of any of them is
        // (char *) p + r * rs, and row 0 is the decode's buffers (ink2_verify.h).
        const size_t aneed = need;
        K.rs = (aneed + 255) & ~(size_t) 255;
        size_t cneed = 0;
        auto addc = [&](size_t bytes) { sz.push_back(bytes); cneed += bytes; };
        for (int s = 0; s < NAPP; s++) {
            const size_t rows = is_full(slot_phys(s)) ? (size_t) ctx : (size_t) PF_RING;
            addc(f32(rows * KVL / 2)); addc(f32(rows * KVL / 2));
        }
        addc(f32((size_t) NAPP * 3 * CHL));
        need = K.rs * VR + cneed;
        char * base = nullptr;
        if (qz_dmalloc((void **) &base, need, "state") != hipSuccess) { fprintf(stderr, "inkling2: card %d: no room for %.2f GB of state\n", c, need / 1e9); return false; }
        ICHECK(hipMemset(base, 0, need));
        size_t o = 0, k = 0;
        auto nx_ = [&]() { char * p = base + o; o += sz[k++]; return p; };
        K.X = (float *) nx_(); K.P = (float *) nx_();
        for (auto & p : K.OUT) p = (float *) nx_();
        for (auto & p : K.BLK) p = (float *) nx_();
        for (auto & p : K.IN) p = (float *) nx_();
        for (auto & p : K.MSG) p = (float *) nx_();
        for (auto & p : K.CT) p = (float *) nx_();
        K.F = (float *) nx_(); K.FN = (float *) nx_();
        K.G = (float *) nx_();
        K.QR = (float *) nx_(); K.GR = (float *) nx_(); K.RR = (float *) nx_(); K.SK = (float *) nx_();
        K.KR = (float *) nx_(); K.VR = (float *) nx_(); K.DK = (float *) nx_(); K.DV = (float *) nx_();
        K.qh = (__half *) nx_(); K.TB = (float *) nx_(); K.PART[0] = (float *) nx_(); K.PART[1] = (float *) nx_();
        K.H = (float *) nx_(); K.WTS = (float *) nx_(); K.LOG = (float *) nx_(); K.IDS = (int *) nx_();
        for (int i = 0; i < 4; i++) { K.aq[i] = (unsigned char *) nx_(); K.as[i] = (float *) nx_(); }
        K.oq = (unsigned char *) nx_(); K.os = (float *) nx_();
        K.hq = (unsigned char *) nx_(); K.hs = (float *) nx_();
        K.tk = (float2 *) nx_();
        o = K.rs * VR;                             // the KV caches / histories are not replicated
        for (int s = 0; s < NAPP; s++) { K.kc[s] = (__half *) nx_(); K.vc[s] = (__half *) nx_(); }
        K.hist = (float *) nx_();
        ICHECK(hipEventCreateWithFlags(&K.ev, hipEventDisableTiming));
        ICHECK(K.x.alloc(VR + 1, D, c, QZ_NC - 1));           // the verify rows' inbox past the decode's 2 D
        K.apsz = (size_t) HKVL * ATT_SPL * GQ * ATT_PW;
        ICHECK(qz_dmalloc(&K.apart, K.apsz * sizeof(float), "state"));   // the decode's (verify: vws)
        ICHECK(qz_dmalloc(&K.acnt, HKVL * sizeof(int), "state"));
        ICHECK(hipMemset(K.acnt, 0, HKVL * sizeof(int)));
        ICHECK(qz_dmalloc((void **) &K.bst, CYC_NBS * sizeof(unsigned int), "state"));
        ICHECK(qz_dmalloc((void **) &K.vbst, VBS_N * sizeof(unsigned int), "state"));
        ICHECK(hipMemset(K.vbst, 0, VBS_N * sizeof(unsigned int)));
        ICHECK(qz_dmalloc((void **) &K.uh, (size_t) NAPP * VR * CHL * sizeof(float), "state"));
        ICHECK(qz_dmalloc((void **) &K.LOGV, (size_t) VR * K.nvalid * sizeof(float), "state"));
        ICHECK(hipMalloc((void **) &K.vtk, (size_t) VR * QZ_TOPK_NB * QZ_TOPK * sizeof(float2)));
        K.ncu = qz_card_ncu(c);                        // one block per CU the card has (CU-masked when virtual)
        if (K.ncu > CYC_NBMAX) K.ncu = CYC_NBMAX;      // the barrier's flag lanes
        if (K.ncu < 2 * CG) { fprintf(stderr, "inkling2: card %d: %d CUs, the persistent decode needs %d\n", c, K.ncu, 2 * CG); return false; }
        {
            ICHECK(hipFuncSetAttribute((const void *) ink2_cycle, hipFuncAttributeMaxDynamicSharedMemorySize, CYC_LDS));
            ICHECK(hipFuncSetAttribute((const void *) ink2_verify, hipFuncAttributeMaxDynamicSharedMemorySize, VER_LDS));
            ICHECK(hipFuncSetAttribute((const void *) k_mix<1>, hipFuncAttributeMaxDynamicSharedMemorySize, MIX_LDS));
            ICHECK(hipFuncSetAttribute((const void *) k_mix<3>, hipFuncAttributeMaxDynamicSharedMemorySize, MIX_LDS));
            ICHECK(hipFuncSetAttribute((const void *) k_mix<4>, hipFuncAttributeMaxDynamicSharedMemorySize, MIX_LDS));
            const qz_resident_kernel ks[] = {{(const void *) ink2_cycle, CYC_LDS, "ink2_cycle"},
                                             {(const void *) ink2_verify, VER_LDS, "ink2_verify"}};
            if (!qz_resident(qz_dev_of(c), ks, 2, K.ncu, 512)) return false;
        }
        cyc_args & A = K.ca;
        A.ev = K.embd.v; A.es = K.embd.s; A.embd_norm = K.embd_norm; A.eps = hp.eps;
        A.X = K.X; A.P = K.P; A.PART0 = K.PART[0]; A.PART1 = K.PART[1];
        A.F = K.F; A.FN = K.FN; A.H = K.H; A.IDS = K.IDS; A.WTS = K.WTS; A.aq3 = K.aq[3]; A.as3 = K.as[3]; A.hq = K.hq; A.hs = K.hs; A.GU = K.LOG;
        ICHECK(qz_dmalloc(&A.gcnt, NE_USED * FFL / 32 * sizeof(int), "state"));
        ICHECK(hipMemset(A.gcnt, 0, NE_USED * FFL / 32 * sizeof(int)));
        A.route_scale = hp.route_scale; A.beta = hp.situ_beta; A.lb = hp.situ_lin_beta;
        A.xin_mine = K.x.xin; A.xflag_mine = K.x.xflag; A.xseq = K.x.xseq; A.xerr = K.x.xerr;
        A.card = c; A.xin_slot = K.x.xin_slot;
        tkh[c].resize((size_t) QZ_TOPK_NB * QZ_TOPK);
        ICHECK(hipHostMalloc((void **) &tkp[c], ((size_t) QZ_TOPK_NB * QZ_TOPK + 4) * sizeof(float2), 0));
        vtkh[c].resize((size_t) VR * QZ_TOPK_NB * QZ_TOPK);
        ICHECK(hipHostMalloc((void **) &vtkp[c], ((size_t) VR * QZ_TOPK_NB * QZ_TOPK + 4) * sizeof(float2), 0));
        return true;
    }
    void reset() {   // a fresh sequence: zero conv history (KV rows are written before they are read)
        for (int c = 0; c < QZ_NC; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ICHECK(hipMemsetAsync(C[c].hist, 0, (size_t) NAPP * 3 * CHL * 4, C[c].st));
        }
    }

    // ---- launch helpers (current device = card c)
    void gemv(int c, std::initializer_list<gjob> js) {
        gjobs J{};
        int g = 0;
        for (const gjob & j : js) { J.j[J.n] = j; g += (j.N + 31) / 32; J.j[J.n].g1 = g; J.n++; }
        k_gemv<<<g, 512, 0, C[c].st>>>(J);
    }
    static gjob job(const qw & w, const unsigned char * aq, const float * as, float * y) {
        gjob j{}; j.wv = w.v; j.ws = w.s; j.av = aq; j.as = as; j.y = y; j.N = w.N; j.K = w.K; return j;
    }
    void mix(int c, const std::vector<const float *> & st, const float * cur, const float * q, int m, float * out) {
        sptrs S{};
        for (const float * p : st) S.p[S.n++] = p;
        S.p[S.n++] = cur;
        if (m == 1) k_mix<1><<<1, 512, MIX_LDS, C[c].st>>>(S, q, out);
        else if (m == 3) k_mix<3><<<1, 512, MIX_LDS, C[c].st>>>(S, q, out);
        else k_mix<4><<<1, 512, MIX_LDS, C[c].st>>>(S, q, out);
    }
    // record each card's partials, then let each wait for every other card's event before
    // reading. Ordering only: no pointer crosses, so host mode needs nothing here (the
    // partials go through the xin mailbox, which qz_xbufs allocates per mode).
    void exchange() {
        for (int c = 0; c < QZ_NC; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipEventRecord(C[c].ev, C[c].st)); }
        for (int c = 0; c < QZ_NC; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            for (int j = 1; j < QZ_NC; j++) ICHECK(hipStreamWaitEvent(C[c].st, C[(c + j) % QZ_NC].ev, 0));
        }
    }

    // ---- the token's 48 layer applications as a table of appk (ink2_cycle.h), built once
    // per card and walked by the persistent kernel. Only pos, tau and the cache row vary.
    struct seq {           // the host-side lists of block buffers (per card the same indices)
        std::vector<int> blocks;      // indices into BLK
        int nblk = 0;                 // BLK buffers used
        int bank = -1;                // pass > 0: BLK index of the previous pass's layer-12 input; OUT parity
        int bank_out = -1;
    };
    std::vector<appk> tab[QZ_NC];
    sptrs head_mix[QZ_NC];
    appk * tab_d[QZ_NC] = {};

    static sptrs sp(const std::vector<const float *> & st, const float * cur) {
        sptrs S{};
        for (const float * p : st) S.p[S.n++] = p;
        S.p[S.n++] = cur;
        return S;
    }
    gjobs gj(std::initializer_list<gjob> js, int & nb) {
        gjobs J{};
        int g = 0;
        for (const gjob & j : js) { J.j[J.n] = j; g += (j.N + 31) / 32; J.j[J.n].g1 = g; J.n++; }
        nb = g;
        return J;
    }
    appk app(int c, int slot, int ip, seq & S, int pre, int pre_ob) {
        ink2_card & K = C[c];
        const lw & L = K.L[ip];
        const bool full = is_full(ip), msg = S.bank >= 0;
        const int push = ip % ARB == 0 ? S.nblk : -1;
        appk e{};
        e.slot = slot; e.ip = ip; e.full = full; e.msg = msg;
        e.pre = pre; e.pre_out = pre > 1 ? K.OUT[pre_ob] : nullptr;
        std::vector<const float *> bl;
        for (int b : S.blocks) bl.push_back(K.BLK[b]);
        const float * in[3] = {K.X, K.X, K.X};
        if (L.resq) {
            e.has_res = 1; e.mres_m = full ? 3 : 1;
            e.mres = sp(bl, K.X); e.qres = L.resq; e.res_out = K.IN[0];
            for (int j = 0; j < 3; j++) in[j] = K.IN[full ? j : 0];
        }
        if (msg) {
            e.mmsg = sp({K.BLK[S.bank]}, K.OUT[S.bank_out]); e.qmsg = L.msgq; e.msg_out = K.MSG[0];
            for (int j = 0; j < 3; j++) e.ca.j[j] = {in[j], K.MSG[j], L.cd[j], L.cg[j], L.cf[j], K.G + j * GRANK, K.CT[j], L.cgt[j], L.cft[j]};
            for (int j = 0; j < 3; j++) in[j] = K.CT[j];
        }
        if (full) {
            e.na.j[0] = {in[0], L.qn, nullptr, K.aq[0], K.as[0]};
            e.na.j[1] = {in[1], L.kn, nullptr, K.aq[1], K.as[1]};
            e.na.j[2] = {in[2], L.vn, nullptr, K.aq[2], K.as[2]};
            e.nna = 3;
        } else if (msg) {
            for (int j = 0; j < 3; j++) e.na.j[j] = {in[j], L.an, nullptr, K.aq[j], K.as[j]};
            e.nna = 3;
        } else {
            e.na.j[0] = {in[0], L.an, nullptr, K.aq[0], K.as[0]};
            e.nna = 1;
        }
        const int ik = e.nna == 3 ? 1 : 0, iv = e.nna == 3 ? 2 : 0;
        if (full)
            e.ga = gj({job(L.q, K.aq[0], K.as[0], K.QR), job(L.gate, K.aq[0], K.as[0], K.GR), job(L.rel, K.aq[0], K.as[0], K.RR),
                       job(L.sink, K.aq[0], K.as[0], K.SK), job(L.k, K.aq[ik], K.as[ik], K.KR), job(L.v, K.aq[iv], K.as[iv], K.VR),
                       job(L.kcp, K.aq[ik], K.as[ik], K.DK), job(L.vcp, K.aq[iv], K.as[iv], K.DV)}, e.nga);
        else
            e.ga = gj({job(L.q, K.aq[0], K.as[0], K.QR), job(L.gate, K.aq[0], K.as[0], K.GR), job(L.rel, K.aq[0], K.as[0], K.RR),
                       job(L.sink, K.aq[0], K.as[0], K.SK), job(L.k, K.aq[ik], K.as[ik], K.KR), job(L.v, K.aq[iv], K.as[iv], K.VR)}, e.nga);
        hpost & a = e.hp;
        a.qr = K.QR; a.rr = K.RR; a.sk = K.SK; a.kr = K.KR; a.vr = K.VR;
        a.dk = full ? K.DK : nullptr; a.dv = full ? K.DV : nullptr;
        a.qclip = L.qclip; a.rclip = L.rclip; a.relb = L.relb; a.kconv = L.kconv; a.vconv = L.vconv;
        a.hist = K.hist + (size_t) slot * 3 * CHL;
        a.qh = K.qh; a.tb = K.TB;
        a.kc = K.kc[slot]; a.vc = K.vc[slot];              // + the row, per token
        a.tau = 1.f; a.qk_clip = hp.qk_clip; a.rel_clip = hp.rel_clip; a.eps = hp.eps;
        attn_args & t = e.at;
        t.qh = K.qh; t.tb = K.TB; t.gate = K.GR; t.kc = K.kc[slot]; t.vc = K.vc[slot]; t.full = full;
        t.oq = K.oq; t.os = K.os; t.part = K.apart; t.cnt = K.acnt;
        e.go = gj({job(L.out, K.oq, K.os, K.PART[0])}, e.ngo);
        e.pconv = L.pconv; e.hsa = K.hist + (size_t) slot * 3 * CHL + H_A; e.blk = push >= 0 ? K.BLK[push] : nullptr;
        if (push >= 0) { S.blocks.push_back(push); S.nblk++; }
        bl.clear();
        for (int b : S.blocks) bl.push_back(K.BLK[b]);
        e.mff = sp(bl, K.X); e.qff = L.ffq;
        const float * f = K.F;
        if (msg) { e.cb.j[0] = {K.F, K.MSG[3], L.cd[3], L.cg[3], L.cf[3], K.G + 3 * GRANK, K.CT[3], L.cgt[3], L.cft[3]}; f = K.CT[3]; }
        e.nb.j[0] = {f, L.ffn, K.FN, K.aq[3], K.as[3]};
        e.gin = L.gin; e.pb = L.pb;
        e.guv = L.gu.v; e.gus = L.gu.s; e.dnv = L.dn.v; e.dns = L.dn.s;
        e.fconv = L.fconv; e.hsf = K.hist + (size_t) slot * 3 * CHL + H_F;
        return e;
    }
    // plan the applications, slots and residual-block lists in the order a step() runs them
    void plan(int c) {
        std::vector<appk> & T = tab[c];
        T.clear();
        seq S;
        for (int il = 0; il < LOOP0; il++) T.push_back(app(c, il, il, S, 0, 0));
        const seq pre = S;
        for (int pass = 0; pass < NPASS; pass++) {
            int ob = 0;
            if (pass > 0) {
                const int bank = S.blocks[pre.blocks.size() + 1];      // the previous pass's layer-12 input
                ob = (pass - 1) & 1;
                const int nblk = S.nblk;
                S = pre;
                S.nblk = nblk;                   // fresh BLK buffers for this pass (the bank stays intact)
                S.bank = bank;
                S.bank_out = ob;
            }
            for (int ip = LOOP0; ip < LOOP1; ip++) {
                const int slot = pass == 0 ? ip : NL + (pass - 1) * (LOOP1 - LOOP0) + (ip - LOOP0);
                T.push_back(app(c, slot, ip, S, ip == LOOP0 ? (pass == 0 ? 1 : 2) : 0, ob));
            }
        }
        S.bank = -1;
        for (int il = LOOP1; il < NL; il++) T.push_back(app(c, il, il, S, 0, 0));
        std::vector<const float *> bl;
        for (int b : S.blocks) bl.push_back(C[c].BLK[b]);
        head_mix[c] = sp(bl, C[c].X);
    }
    bool planned = false;
    // DSpark (set before the first step, card 0): the residual entering the application
    // DSPARK_VISIT[k] goes to dtap + k D, the head's normed input to dtap_head.
    static constexpr int DSPARK_VISIT[3] = {5, 23, 41};
    float * dtap = nullptr, * dtap_head = nullptr;
    void plan_all() {
        if (planned) return;
        for (int c = 0; c < QZ_NC; c++)
            for (int k = 1; k < QZ_NC; k++) {
                const qz_xbufs & P = C[qz_xpeer(QZ_NC, c, k)].x;
                const int sl = qz_xslot_in_peer(QZ_NC, k);
                C[c].ca.xin_px[k - 1] = P.xin + sl * P.xin_slot;
                C[c].ca.xflag_px[k - 1] = P.xflag + sl * P.xflag_slot;
            }
        for (int c = 0; c < QZ_NC; c++) {
            plan(c);
            ICHECK(hipSetDevice(qz_dev_of(c)));
            {
                cyc_args & A = C[c].ca;
                A.hmix = head_mix[c]; A.hq_out = C[c].out_q;
                A.hn = {C[c].F, C[c].out_norm, c == 0 ? dtap_head : nullptr, C[c].aq[3], C[c].as[3]};
                A.hg = job(C[c].head, C[c].aq[3], C[c].as[3], C[c].LOG);
                A.nvalid = C[c].nvalid; A.softcap = hp.softcap;
            }
            if (c == 0 && dtap) for (int k = 0; k < 3; k++) tab[0][DSPARK_VISIT[k]].tap = dtap + (size_t) k * D;
            C[c].ca.napp = (int) tab[c].size();
            if (!tab_d[c]) ICHECK(hipMalloc(&tab_d[c], tab[c].size() * sizeof(appk)));
            ICHECK(hipMemcpy(tab_d[c], tab[c].data(), tab[c].size() * sizeof(appk), hipMemcpyHostToDevice));
            C[c].ca.tab = tab_d[c];
            if (!C[c].ca_d) ICHECK(hipMalloc(&C[c].ca_d, sizeof(cyc_args)));
            ICHECK(hipMemcpy(C[c].ca_d, &C[c].ca, sizeof(cyc_args), hipMemcpyHostToDevice));
        }
        planned = true;
    }
    float tau_at(int pos) const { return 1.f + hp.tau_alpha * logf(fmaxf(1.f, (float) (pos + 1) / (float) hp.tau_floor)); }
    // the persistent path: embed and all applications in one launch per card
    long long n_cyc = 0;
    void run_cycle(int32_t tok, int pos, bool head) {
        for (int c = 0; c < QZ_NC; c++) {
            ink2_card & K = C[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ICHECK(hipMemsetAsync(K.bst, 0, CYC_NBS * sizeof(unsigned int), K.st));
            hipLaunchKernelGGL(ink2_cycle, dim3(K.ncu), dim3(512), CYC_LDS, K.st, K.bst, (const cyc_args *) K.ca_d, tok, pos, tau_at(pos), head ? 1 : 0);
            ICHECK(hipGetLastError());
        }
        n_cyc++;
    }

    // ---- verify (ink2_verify.h): the speculative round's VR rows in one launch per card.
    // The per-row tables are built once from the decode table: every per-token activation
    // pointer plus r * rs, and the attention partials and counters on their own strides.
    std::vector<appk> vtab[QZ_NC];
    appk * vtab_d[QZ_NC] = {};
    float * vtap = nullptr;                        // card 0: the rows' [4][D] drafter features
    bool vplanned = false;
    template <class T> static T * shp(T * p, size_t o) { return p ? (T *) ((char *) p + o) : nullptr; }
    static void sh_sptrs(sptrs & S, size_t o) { for (int i = 0; i < 6; i++) S.p[i] = shp(S.p[i], o); }
    static void sh_cjob(cjob & c, size_t o) {
        c.cur = shp(c.cur, o); c.msg = shp(c.msg, o); c.g = shp(c.g, o); c.out = shp(c.out, o);
    }
    static void sh_njob(njob & j, size_t o) {
        j.x = shp(j.x, o); j.y = shp(j.y, o); j.q = shp(j.q, o); j.qs = shp(j.qs, o);
    }
    static void sh_gjobs(gjobs & J, size_t o) {
        for (int i = 0; i < J.n; i++) { J.j[i].av = shp(J.j[i].av, o); J.j[i].as = shp(J.j[i].as, o); J.j[i].y = shp(J.j[i].y, o); }
    }
    // row r's entry: activations plus r * rs. Weights, KV caches and histories stay shared.
    void shift_appk(appk & e, int r, size_t o) {
        e.pre_out = shp(e.pre_out, o);
        sh_sptrs(e.mres, o); e.res_out = shp(e.res_out, o);
        sh_sptrs(e.mmsg, o); e.msg_out = shp(e.msg_out, o);
        for (int j = 0; j < 3; j++) sh_cjob(e.ca.j[j], o);
        for (int j = 0; j < 3; j++) if (e.na.j[j].x) sh_njob(e.na.j[j], o);
        sh_gjobs(e.ga, o); sh_gjobs(e.go, o);
        hpost & h = e.hp;
        h.qr = shp(h.qr, o); h.rr = shp(h.rr, o); h.sk = shp(h.sk, o); h.kr = shp(h.kr, o); h.vr = shp(h.vr, o);
        h.dk = shp(h.dk, o); h.dv = shp(h.dv, o);
        h.qh = shp(h.qh, o); h.tb = shp(h.tb, o);
        attn_args & t = e.at;
        t.qh = shp(t.qh, o); t.tb = shp(t.tb, o); t.gate = shp(t.gate, o); t.oq = shp(t.oq, o); t.os = shp(t.os, o);
        e.blk = shp(e.blk, o);
        sh_sptrs(e.mff, o);
        sh_cjob(e.cb.j[0], o);
        sh_njob(e.nb.j[0], o);
        if (e.tap) e.tap = vtap + (size_t) r * 4 * D + (e.tap - dtap);
    }
    void vplan() {
        if (vplanned) return;
        plan_all();
        vplanned = true;
        for (int c = 0; c < QZ_NC; c++) {
            const int napp = C[c].ca.napp;                 // the table rows follow the (possibly truncated) count
            const size_t o = C[c].rs;
            vtab[c].resize((size_t) VR * napp);
            for (int r = 0; r < VR; r++)
                for (int a = 0; a < napp; a++) {
                    appk & e = vtab[c][(size_t) r * napp + a];
                    e = tab[c][a];
                    shift_appk(e, r, (size_t) r * o);
                }
            ICHECK(hipSetDevice(qz_dev_of(c)));
            const size_t bytes = vtab[c].size() * sizeof(appk);
            if (!vtab_d[c]) ICHECK(hipMalloc((void **) &vtab_d[c], bytes));
            ICHECK(hipMemcpy(vtab_d[c], vtab[c].data(), bytes, hipMemcpyHostToDevice));
            vargs V;
            memset(&V, 0, sizeof(V));
            V.A = C[c].ca;
            V.A.tab = vtab_d[c];
            V.rs = (int) C[c].rs; V.rsf = (int) (C[c].rs / 4);
            V.uh = C[c].uh; V.LOGV = C[c].LOGV;
            V.vtap = c == 0 ? vtap : nullptr;
            V.hist = C[c].hist;
            if (!C[c].vws) ICHECK(qz_dmalloc((void **) &C[c].vws, VR * VWS * sizeof(float), "state"));
            V.ws = C[c].vws;
            if (!C[c].vargs_d) ICHECK(hipMalloc(&C[c].vargs_d, sizeof(vargs)));
            ICHECK(hipMemcpy(C[c].vargs_d, &V, sizeof(vargs), hipMemcpyHostToDevice));
        }
        ICHECK(hipSetDevice(qz_dev_of(0)));
    }
    // the round: launch both cards, then each card's top-K of the VR logit rows, then sync.
    // nr (2 .. VR) is the round's active rows: toks[0 .. nr-1], the rest of the table is idle.
    long long n_ver = 0;
    void verify(const int * toks, int pos, int nr) {
        vplan();
        nr = nr < 2 ? 2 : nr > VR ? VR : nr;
        vtok tv;
        for (int r = 0; r < VR; r++) { tv.t[r] = r < nr ? toks[r] : 0; tv.tau[r] = tau_at(pos + r); }
        for (int c = 0; c < QZ_NC; c++) {
            ink2_card & K = C[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ICHECK(hipMemsetAsync(K.vbst, 0, VBS_N * sizeof(unsigned int), K.st));
            hipLaunchKernelGGL(ink2_verify, dim3(K.ncu), dim3(512), VER_LDS, K.st, K.vbst, (const vargs *) K.vargs_d, tv, pos, 1, nr);
            ICHECK(hipGetLastError());
            qz_topk_rows<<<dim3(QZ_TOPK_NB, nr), 256, 0, K.st>>>(K.LOGV, K.nvalid, K.id0, K.vtk);
            const size_t n2 = (size_t) VR * QZ_TOPK_NB * QZ_TOPK;
            ICHECK(hipMemcpyAsync(vtkp[c], K.vtk, n2 * sizeof(float2), hipMemcpyDeviceToHost, K.st));
            ICHECK(hipMemcpyAsync(vtkp[c] + n2, K.x.xerr, 8 * sizeof(int), hipMemcpyDeviceToHost, K.st));
        }
        for (int c = 0; c < QZ_NC; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamSynchronize(C[c].st)); }
        for (int c = 0; c < QZ_NC; c++)
            qz_xerr_check_h(c, (const int *) (vtkp[c] + (size_t) VR * QZ_TOPK_NB * QZ_TOPK), n_ver);
        n_ver++;
    }
    // the row's greedy pick, with the same tie rule as greedy(): on a tie the lower id wins
    int greedy_row(int r) const {
        float bv = -INFINITY;
        int bi = -1;
        for (int c = 0; c < QZ_NC; c++) {
            const float2 * b = vtkp[c] + (size_t) r * QZ_TOPK_NB * QZ_TOPK;
            for (int i = 0; i < QZ_TOPK_NB * QZ_TOPK; i++) {
                int id;
                memcpy(&id, &b[i].y, 4);
                if (id >= 0 && (b[i].x > bv || (b[i].x == bv && id < bi))) { bv = b[i].x; bi = id; }
            }
        }
        return bi;
    }
    // the conv histories of the m committed rows (both cards)
    void commit(int m) {
        for (int c = 0; c < QZ_NC; c++) {
            ink2_card & K = C[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            k_vcommit<<<(NAPP * CHL + 255) / 256, 256, 0, K.st>>>(K.hist, K.uh, m);
        }
    }

    // one token's whole forward. With logits, also run the head and copy the top-K out.
    void step(int32_t tok, int pos, bool logits) {
        plan_all();
        run_cycle(tok, pos, logits);
        if (logits) for (int c = 0; c < QZ_NC; c++) {
            ink2_card & K = C[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            qz_topk_rows<<<dim3(QZ_TOPK_NB, 1), 256, 0, K.st>>>(K.LOG, K.nvalid, K.id0, K.tk);
            ICHECK(hipMemcpyAsync(tkp[c], K.tk, tkh[c].size() * sizeof(float2), hipMemcpyDeviceToHost, K.st));
        }
        for (int c = 0; c < QZ_NC; c++) {          // the error words ride the same stream: no extra sync
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ICHECK(hipMemcpyAsync(tkp[c] + QZ_TOPK_NB * QZ_TOPK, C[c].x.xerr, 8 * sizeof(int), hipMemcpyDeviceToHost, C[c].st));
        }
        for (int c = 0; c < QZ_NC; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamSynchronize(C[c].st)); }
        if (logits) for (int c = 0; c < QZ_NC; c++) memcpy(tkh[c].data(), tkp[c], tkh[c].size() * sizeof(float2));
        for (int c = 0; c < QZ_NC; c++) qz_xerr_check_h(c, (const int *) (tkp[c] + QZ_TOPK_NB * QZ_TOPK), n_cyc);
    }
    int greedy() const {
        float bv = -INFINITY;
        int bi = -1;
        for (int c = 0; c < QZ_NC; c++)
            for (const float2 & e : tkh[c]) {
                int id;
                memcpy(&id, &e.y, 4);
                if (id >= 0 && (e.x > bv || (e.x == bv && id < bi))) { bv = e.x; bi = id; }
            }
        return bi;
    }
};
