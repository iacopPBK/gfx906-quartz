// inkling2 engine: the forward pass of docs/inkling2.md as a host-issued
// kernel sequence on both cards, one token per step. TP2: each card runs its
// q / kv heads and its half of every expert's FF; the attention output and
// the expert down projection are summed across the cards (each reads the
// peer's partial over P2P after a cross-stream event wait).
#pragma once

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
    float * QR, * GR, * RR, * SK, * KR, * VR, * DK, * DV, * Q, * TB, * PART[2], * H, * WTS, * LOG;
    int * IDS;
    unsigned char * aq[4]; float * as[4];      // input planes: Q | K | V inputs, ffn / head input
    unsigned char * oq; float * os;           // attention output planes [QL]
    unsigned char * hq; float * hs;           // expert activation planes [NE_USED][FFL]
    float2 * tk;                              // head top-K [QZ_TOPK_NB][QZ_TOPK]
    __half * kc[NAPP], * vc[NAPP];            // per slot: full [ctx][KVL], SWA ring [WIN][KVL]
    float * hist;                             // [NAPP][3][CHL]
    int nvalid = 0, id0 = 0;                  // head rows that are real vocab ids, the first id
};

struct ink2_engine {
    hparams hp;
    int ctx = 0;
    ink2_card C[2];
    int nx = 0;                               // exchanges issued (partial double-buffer parity)
    std::vector<float2> tkh[2];

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
                }
            }
        }
        {   // gate_up rows interleaved per 32 (gu_row): the prefill GEMM's 64-row tiles pair them for SITU
            const size_t nv = (size_t) NE * FFE * D, ns = (size_t) NE * FFE * (D / 32);
            char * tv; __half * ts;
            ICHECK(hipMalloc(&tv, nv)); ICHECK(hipMalloc(&ts, ns * 2));
            for (int il = 0; il < NL; il++) {
                qw & W = K.L[il].gu;
                ICHECK(hipMemcpy(tv, W.v, nv, hipMemcpyDeviceToDevice));
                ICHECK(hipMemcpy(ts, W.s, ns * 2, hipMemcpyDeviceToDevice));
                k_gu_interleave<<<dim3(FFE, NE), 256>>>(tv, ts, (char *) W.v, (__half *) W.s);
            }
            ICHECK(hipDeviceSynchronize());
            ICHECK(hipFree(tv)); ICHECK(hipFree(ts));
        }
        const int half = VOCAB / 2;
        K.id0 = half * c;
        K.nvalid = std::min(half, VOCAB_REAL - K.id0);
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
        add(f32(64)); add(f32(64)); add(f32(QL)); add(f32(HQL * TBW)); add(f32(nD)); add(f32(nD));
        add(f32(NE_USED * FFE)); add(f32(NE_USED)); add(f32(VOCAB / 2)); add(f32(NE_USED));
        for (int i = 0; i < 4; i++) { add(f32(nD / 4)); add(f32(nD / 32)); }              // planes (int8 = n/4 floats)
        add(f32(QL / 4)); add(f32(QL / 32));
        add(f32(NE_USED * FFL / 4)); add(f32(NE_USED * FFL / 32));
        add(f32(QZ_TOPK_NB * QZ_TOPK * 2));
        for (int s = 0; s < NAPP; s++) {
            const size_t rows = is_full(slot_phys(s)) ? (size_t) ctx : (size_t) PF_RING;
            add(f32(rows * KVL / 2)); add(f32(rows * KVL / 2));
        }
        add(f32((size_t) NAPP * 3 * CHL));
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
        K.Q = (float *) nx_(); K.TB = (float *) nx_(); K.PART[0] = (float *) nx_(); K.PART[1] = (float *) nx_();
        K.H = (float *) nx_(); K.WTS = (float *) nx_(); K.LOG = (float *) nx_(); K.IDS = (int *) nx_();
        for (int i = 0; i < 4; i++) { K.aq[i] = (unsigned char *) nx_(); K.as[i] = (float *) nx_(); }
        K.oq = (unsigned char *) nx_(); K.os = (float *) nx_();
        K.hq = (unsigned char *) nx_(); K.hs = (float *) nx_();
        K.tk = (float2 *) nx_();
        for (int s = 0; s < NAPP; s++) { K.kc[s] = (__half *) nx_(); K.vc[s] = (__half *) nx_(); }
        K.hist = (float *) nx_();
        ICHECK(hipEventCreateWithFlags(&K.ev, hipEventDisableTiming));
        tkh[c].resize((size_t) QZ_TOPK_NB * QZ_TOPK);
        return true;
    }
    void reset() {   // a fresh sequence: zero conv history (KV rows are written before they are read)
        for (int c = 0; c < 2; c++) {
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
        k_mix<<<1, 256, 0, C[c].st>>>(S, q, m, out);
    }
    // both cards' partials written: each waits for the other's, then reads it
    void exchange() {
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipEventRecord(C[c].ev, C[c].st)); }
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamWaitEvent(C[c].st, C[1 - c].ev, 0)); }
    }

    // ---- one layer application at state slot `slot` of physical layer ip, for the token at pos
    struct seq {           // the host-side lists of block buffers (per card the same indices)
        std::vector<int> blocks;      // indices into BLK
        int nblk = 0;                 // BLK buffers used
        int bank = -1;                // pass > 0: BLK index of the previous pass's layer-12 input; OUT parity
        int bank_out = -1;
    };
    void layer(int slot, int ip, int pos, seq & S) {
        const bool full = is_full(ip), msg = S.bank >= 0;
        const float tau = full ? 1.f + hp.tau_alpha * logf(fmaxf(1.f, (float) (pos + 1) / (float) hp.tau_floor)) : 1.f;
        const int buf = nx & 1;
        const int push = ip % ARB == 0 ? S.nblk : -1;
        for (int c = 0; c < 2; c++) {
            ink2_card & K = C[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            ICHECK(hipSetDevice(qz_dev_of(c)));
            std::vector<const float *> bl;
            for (int b : S.blocks) bl.push_back(K.BLK[b]);
            const float * in[3] = {K.X, K.X, K.X};
            if (L.resq) {
                mix(c, bl, K.X, L.resq, full ? 3 : 1, K.IN[0]);
                for (int j = 0; j < 3; j++) in[j] = K.IN[full ? j : 0];
            }
            if (msg) {
                mix(c, {K.BLK[S.bank]}, K.OUT[S.bank_out], L.msgq, NCTRL, K.MSG[0]);
                cjobs J{};
                for (int j = 0; j < 3; j++) J.j[j] = {in[j], K.MSG[j], L.cd[j], L.cg[j], L.cf[j], K.G + j * GRANK, K.CT[j]};
                k_ctrl_down<<<dim3(GRANK, 3), 256, 0, st>>>(J);
                k_ctrl_up<<<dim3((D + 255) / 256, 3), 256, 0, st>>>(J);
                for (int j = 0; j < 3; j++) in[j] = K.CT[j];
            }
            njobs N{};
            int nn;
            if (full) {
                N.j[0] = {in[0], L.qn, nullptr, K.aq[0], K.as[0]};
                N.j[1] = {in[1], L.kn, nullptr, K.aq[1], K.as[1]};
                N.j[2] = {in[2], L.vn, nullptr, K.aq[2], K.as[2]};
                nn = 3;
            } else if (msg) {
                for (int j = 0; j < 3; j++) N.j[j] = {in[j], L.an, nullptr, K.aq[j], K.as[j]};
                nn = 3;
            } else {
                N.j[0] = {in[0], L.an, nullptr, K.aq[0], K.as[0]};
                nn = 1;
            }
            k_norm_quant<<<nn, 256, 0, st>>>(N, hp.eps);
            const int ik = nn == 3 ? 1 : 0, iv = nn == 3 ? 2 : 0;
            if (full)
                gemv(c, {job(L.q, K.aq[0], K.as[0], K.QR), job(L.gate, K.aq[0], K.as[0], K.GR), job(L.rel, K.aq[0], K.as[0], K.RR),
                         job(L.sink, K.aq[0], K.as[0], K.SK), job(L.k, K.aq[ik], K.as[ik], K.KR), job(L.v, K.aq[iv], K.as[iv], K.VR),
                         job(L.kcp, K.aq[ik], K.as[ik], K.DK), job(L.vcp, K.aq[iv], K.as[iv], K.DV)});
            else
                gemv(c, {job(L.q, K.aq[0], K.as[0], K.QR), job(L.gate, K.aq[0], K.as[0], K.GR), job(L.rel, K.aq[0], K.as[0], K.RR),
                         job(L.sink, K.aq[0], K.as[0], K.SK), job(L.k, K.aq[ik], K.as[ik], K.KR), job(L.v, K.aq[iv], K.as[iv], K.VR)});
            const int row = full ? pos : pos % PF_RING;
            hpost a{};
            a.qr = K.QR; a.rr = K.RR; a.sk = K.SK; a.kr = K.KR; a.vr = K.VR;
            a.dk = full ? K.DK : nullptr; a.dv = full ? K.DV : nullptr;
            a.qclip = L.qclip; a.rclip = L.rclip; a.relb = L.relb; a.kconv = L.kconv; a.vconv = L.vconv;
            a.hist = K.hist + (size_t) slot * 3 * CHL;
            a.q = K.Q; a.tb = K.TB;
            a.kc = K.kc[slot] + (size_t) row * KVL; a.vc = K.vc[slot] + (size_t) row * KVL;
            a.tau = tau; a.qk_clip = hp.qk_clip; a.rel_clip = hp.rel_clip; a.eps = hp.eps;
            k_head_post<<<HQL + 2 * HKVL, 128, 0, st>>>(a);
            attn_args t{};
            t.q = K.Q; t.tb = K.TB; t.gate = K.GR; t.kc = K.kc[slot]; t.vc = K.vc[slot]; t.pos = pos; t.full = full;
            t.oq = K.oq; t.os = K.os;
            k_attn<<<HQL, 256, 0, st>>>(t);
            gemv(c, {job(L.out, K.oq, K.os, K.PART[buf])});
        }
        exchange();
        nx++;
        for (int c = 0; c < 2; c++) {
            ink2_card & K = C[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            ICHECK(hipSetDevice(qz_dev_of(c)));
            k_xconv<<<(D + 255) / 256, 256, 0, st>>>(K.PART[buf], C[1 - c].PART[buf], L.pconv,
                                                     K.hist + (size_t) slot * 3 * CHL + H_A, K.X, push >= 0 ? K.BLK[push] : nullptr);
        }
        if (push >= 0) { S.blocks.push_back(push); S.nblk++; }
        const int buf2 = nx & 1;
        for (int c = 0; c < 2; c++) {
            ink2_card & K = C[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            ICHECK(hipSetDevice(qz_dev_of(c)));
            std::vector<const float *> bl;
            for (int b : S.blocks) bl.push_back(K.BLK[b]);
            mix(c, bl, K.X, L.ffq, 1, K.F);
            const float * f = K.F;
            if (msg) {
                cjobs J{};
                J.j[0] = {K.F, K.MSG[3], L.cd[3], L.cg[3], L.cf[3], K.G + 3 * GRANK, K.CT[3]};
                k_ctrl_down<<<dim3(GRANK, 1), 256, 0, st>>>(J);
                k_ctrl_up<<<dim3((D + 255) / 256, 1), 256, 0, st>>>(J);
                f = K.CT[3];
            }
            njobs N{};
            N.j[0] = {f, L.ffn, K.FN, K.aq[3], K.as[3]};
            k_norm_quant<<<1, 256, 0, st>>>(N, hp.eps);
            k_router<<<1, 1024, 0, st>>>(K.FN, L.gin, L.pb, hp.route_scale, K.IDS, K.WTS);
            k_moe_up<<<dim3(FFE / 32, NE_USED), 512, 0, st>>>(L.gu.v, L.gu.s, K.IDS, K.aq[3], K.as[3], K.H);
            k_situ<<<NE_USED, FFL, 0, st>>>(K.H, hp.situ_beta, hp.situ_lin_beta, K.hq, K.hs);
            k_moe_down<<<D / 32, 512, 0, st>>>(L.dn.v, L.dn.s, K.IDS, K.WTS, K.hq, K.hs, K.PART[buf2]);
        }
        exchange();
        nx++;
        for (int c = 0; c < 2; c++) {
            ink2_card & K = C[c];
            const lw & L = K.L[ip];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            k_xconv<<<(D + 255) / 256, 256, 0, K.st>>>(K.PART[buf2], C[1 - c].PART[buf2], L.fconv,
                                                       K.hist + (size_t) slot * 3 * CHL + H_F, K.X, nullptr);
        }
    }

    // the whole forward for one token; logits: also run the head and leave the top-K on the host
    void step(int32_t tok, int pos, bool logits) {
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            k_embed<<<1, 256, 0, C[c].st>>>(C[c].embd.v, C[c].embd.s, tok, C[c].embd_norm, hp.eps, C[c].X);
        }
        seq S;
        for (int il = 0; il < LOOP0; il++) layer(il, il, pos, S);
        const seq pre = S;
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipMemcpyAsync(C[c].P, C[c].X, D * 4, hipMemcpyDeviceToDevice, C[c].st)); }
        for (int pass = 0; pass < NPASS; pass++) {
            if (pass > 0) {
                const int bank = S.blocks[pre.blocks.size() + 1];      // the previous pass's layer-12 input
                const int ob = (pass - 1) & 1;
                for (int c = 0; c < 2; c++) {
                    ICHECK(hipSetDevice(qz_dev_of(c)));
                    ICHECK(hipMemcpyAsync(C[c].OUT[ob], C[c].X, D * 4, hipMemcpyDeviceToDevice, C[c].st));
                    ICHECK(hipMemcpyAsync(C[c].X, C[c].P, D * 4, hipMemcpyDeviceToDevice, C[c].st));
                }
                const int nblk = S.nblk;
                S = pre;
                S.nblk = nblk;                   // fresh BLK buffers for this pass (the bank stays intact)
                S.bank = bank;
                S.bank_out = ob;
            }
            for (int ip = LOOP0; ip < LOOP1; ip++) {
                const int slot = pass == 0 ? ip : NL + (pass - 1) * (LOOP1 - LOOP0) + (ip - LOOP0);
                layer(slot, ip, pos, S);
            }
        }
        S.bank = -1;
        for (int il = LOOP1; il < NL; il++) layer(il, il, pos, S);
        if (!logits) return;
        for (int c = 0; c < 2; c++) {
            ink2_card & K = C[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            std::vector<const float *> bl;
            for (int b : S.blocks) bl.push_back(K.BLK[b]);
            mix(c, bl, K.X, K.out_q, 1, K.F);
            njobs N{};
            N.j[0] = {K.F, K.out_norm, nullptr, K.aq[3], K.as[3]};
            k_norm_quant<<<1, 256, 0, K.st>>>(N, hp.eps);
            gemv(c, {job(K.head, K.aq[3], K.as[3], K.LOG)});
            k_softcap<<<(K.nvalid + 255) / 256, 256, 0, K.st>>>(K.LOG, K.nvalid, hp.softcap);
            qz_topk_rows<<<dim3(QZ_TOPK_NB, 1), 256, 0, K.st>>>(K.LOG, K.nvalid, K.id0, K.tk);
            ICHECK(hipMemcpyAsync(tkh[c].data(), K.tk, tkh[c].size() * sizeof(float2), hipMemcpyDeviceToHost, K.st));
        }
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamSynchronize(C[c].st)); }
    }
    int greedy() const {
        float bv = -INFINITY;
        int bi = -1;
        for (int c = 0; c < 2; c++)
            for (const float2 & e : tkh[c]) {
                int id;
                memcpy(&id, &e.y, 4);
                if (id >= 0 && (e.x > bv || (e.x == bv && id < bi))) { bv = e.x; bi = id; }
            }
        return bi;
    }
};
