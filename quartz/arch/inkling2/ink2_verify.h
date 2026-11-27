// inkling2 verify kernel. Runs the VR = 8 rows a speculative round may use (the anchor
// and up to 7 drafts, positions pos .. pos + 7) through the 48 layer applications and the
// head, in one launch per card. Each row runs the decode bodies on its own copy of
// the activations (byte stride RS from row 0), so it is bit-identical to a single
// ink2_cycle step at its position. Row r sees KV rows up to pos + r; its own K and V
// are written before the attention phase.
//
// The host builds the row replicas in ink2_engine::vplan: one table entry per (row,
// application), every activation pointer shifted by the row's stride, so the chain and
// grid bodies run unchanged. Included from ink2_engine.h after ink2_cycle.h.
//
// A round is often shorter than VR rows: nr (2 .. VR, a kernel argument) is the number
// of rows this round runs, and rows nr .. VR - 1 are idle. They get no chain group, no
// head post, no attention unit, no router pick, and since they pick no experts they are
// in no up / down row list. Rows < nr are untouched by this, so their bits do not depend
// on nr. The phases that stream a weight once for all rows keep all VR rows: the qkv /
// out GEMVs, the router, the controller sums and the head GEMV.
#pragma once

// fp contract(on) so the bodies round like ink2_cycle.h; contract(fast) again at the end
#pragma clang fp contract(on)

#define VR_NI __device__ __forceinline__
constexpr int VR = 8;
constexpr int VBS_SL = (CYC_NBMAX + VR * CG + 1) & ~1, VBS_N = VBS_SL + 2 * VR * CP_SZ;   // vbst words: the flags, then the groups' 64-bit slots
#ifndef VG_S
#define VG_S 2
#endif

struct vtok { int t[VR]; float tau[VR]; };        // tau from the host's tau_at

// The verify arguments: the decode args with tab at the VR-row table, aq3 / hq / ... at
// row 0, plus the row-replicated buffers.
struct vargs {
    cyc_args A;
    int rs, rsf;                          // the activation row stride, in bytes and floats
    float * uh;                           // [NAPP][VR][CHL], each row's conv input
    float * LOGV;                         // [VR][nvalid], the head logits per row
    float * vtap;                         // card 0's [VR][4][D] drafter features, else null
    float * hist;                         // base of the histories [NAPP][3][CHL]. Table pointers per slot.
    float * ws;                           // the wave states of v_attn_a and v_attn_b
};

#define VROW(tab, r, ai) ((CK appk *) (tab) + (size_t) (r) * napp + (ai))

// The conv history sequence of a verify row: s[0..2] are the slot's history, oldest
// first, s[3 + k] is row k's conv input u_k. Row r reads h0 = s[r], h1 = s[r + 1] and
// h2 = s[r + 2], so it needs rows r - 3 .. r - 1 at most. Their u is still in memory, so
// no row waits: the raw projection for the K and V convs, the TP2 sums for the post convs.
VR_NI float v_s3(const float * hbase, const float * own0, const qz_xinbox<QZ_NC> & peer0, int i, int k, int rsf) {
    if (k < 3) return hbase[(size_t) k * CHL + i];
    const int q = k - 3;
    return peer0.add(own0[(size_t) q * rsf + i], (size_t) q * D + i);
}

// c_cxconv with the verify window: no history write, row r's u into its uh row
VR_NI void v_cxconv(const float * own0, const qz_xinbox<QZ_NC> & peer0, const float * w, float * hbase, float * x, float * blk, int cw, int tid,
                    int r, int rsf, float * uhrow, float * tap = nullptr) {
    own0 = qz_gp(own0); w = qz_gp(w); hbase = qz_gp(hbase);
    x = qz_gp(x); blk = blk ? qz_gp(blk) : nullptr; tap = tap ? qz_gp(tap) : nullptr; uhrow = qz_gp(uhrow);
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            const int i = ccol(cw, q);
            const float u = peer0.add(own0[(size_t) r * rsf + i], (size_t) r * D + i);
            const float h0 = v_s3(hbase, own0, peer0, i, r, rsf), h1 = v_s3(hbase, own0, peer0, i, r + 1, rsf),
                        h2 = v_s3(hbase, own0, peer0, i, r + 2, rsf);
            const float4 wi = *(const float4 *) (w + (size_t) i * NCONV);
            const float xv = x[i];
            const float o = wi.x * h0 + wi.y * h1 + wi.z * h2 + wi.w * u;
            uhrow[i] = u;
            const float y = (u + o) * (1.f / sqrtf((float) NCONV + 1.f));
            if (blk) { blk[i] = xv; x[i] = y; }
            else x[i] = xv + y;
            if (tap) tap[i] = blk ? y : xv + y;
        }
    }
}

// d_xconv_blk<512> with the verify window, one block per row for the last ffn sum
VR_NI void v_xconv_b0(const float * own0, const qz_xinbox<QZ_NC> & peer0, const float * w, const float * hbase, float * x, float * blk,
                      int tid, int r, int rsf, float * uhrow) {
    constexpr int NT = 512, NEL = (D + NT - 1) / NT;
    own0 = qz_gp(own0); w = qz_gp(w); hbase = qz_gp(hbase); x = qz_gp(x); blk = blk ? qz_gp(blk) : nullptr; uhrow = qz_gp(uhrow);
    float u[NEL], h0[NEL], h1[NEL], h2[NEL], xv[NEL];
    float4 wi[NEL];
    #pragma unroll
    for (int k = 0; k < NEL; k++) {
        const int i = min(tid + NT * k, D - 1);          // clamped: unconditional loads, guarded stores
        u[k] = peer0.add(own0[(size_t) r * rsf + i], (size_t) r * D + i);
        h0[k] = v_s3(hbase, own0, peer0, i, r, rsf);
        h1[k] = v_s3(hbase, own0, peer0, i, r + 1, rsf);
        h2[k] = v_s3(hbase, own0, peer0, i, r + 2, rsf);
        wi[k] = *(const float4 *) (w + (size_t) i * NCONV);
        xv[k] = x[i];
    }
    #pragma unroll
    for (int k = 0; k < NEL; k++) {
        const int i = tid + NT * k;
        if ((k + 1) * NT <= D || i < D) {
            const float o = wi[k].x * h0[k] + wi[k].y * h1[k] + wi[k].z * h2[k] + wi[k].w * u[k];
            uhrow[i] = u[k];
            const float y = (u[k] + o) * (1.f / sqrtf((float) NCONV + 1.f));
            if (blk) { blk[i] = xv[k]; x[i] = y; }
            else x[i] = xv[k] + y;
        }
    }
}

// d_head_post with the verify K / V conv, whose window runs over the rows' raw projections
// and never writes the history. v_gsum2 is bsum<1>(x, red, 2) per 128-thread group, each on
// its own LDS pair, for the block's 4 groups at once.
VR_NI float v_gsum2(float x, float * red, int grp, bool act) {
    x = qz_wsum_dpp(x);
    __syncthreads();
    if (act && (threadIdx.x & 63) == 0) red[grp * 2 + ((threadIdx.x >> 6) & 1)] = x;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < 2; i++) s += red[grp * 2 + i];
    return s;
}
VR_NI void d_head_post_v(const hpost & a, int b, int t, bool act, int dmax, int r, float * uh, int rsf, const float * hb,
                         int grp, int tb_t, int tb_st) {
    __shared__ float red[4 * 2];
    __shared__ float rvs[4][DREL];
    float * rv = rvs[grp];
    if (b < HQL) {
        const int h = b, i = h * HD + t;
        float x = act ? a.qr[i] * a.qclip[i] : 0.f;
        float n2[1] = {v_gsum2(x * x, red, grp, act)};
        const float nr = sqrtf(n2[0]) + 1e-6f;
        x *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        if (act) a.qh[i] = __float2half(x * a.tau * FL_QSC);
        float r_ = (act && t < DREL) ? a.rr[h * DREL + t] * a.rclip[h * DREL + t] : 0.f;
        float r2[1] = {v_gsum2(r_ * r_, red, grp, act)};
        const float rn = sqrtf(r2[0]) + 1e-6f;
        if (act && t < DREL) rv[t] = r_ * fminf(fmaxf(a.rel_clip / rn, 0.f), 1.f) * a.tau;
        __syncthreads();
        const float sink = a.sk[h];
        for (int d = tb_t; d < dmax; d += tb_st) {
            float s = 0.f;
            if (d < REL_EXT)
                #pragma unroll
                for (int k = 0; k < DREL; k++) s += a.relb[d * DREL + k] * rv[k];
            a.tb[h * TBW + d] = s - sink;
        }
        __syncthreads();
        return;
    }
    const bool isk = b < HQL + HKVL;
    const int hk = isk ? b - HQL : b - HQL - HKVL, ch = hk * HD + t;
    const float * raw = isk ? a.kr : a.vr;               // this row's raw projection
    const float * raw0 = raw - (size_t) r * rsf;                    // the same channel in row 0
    const float * dyn = isk ? a.dk : a.dv;
    const float * w = (isk ? a.kconv : a.vconv) + (size_t) ch * NCONV;
    const float * hs = a.hist + (isk ? H_K : H_V);        // the slot's [3][CHL] at this channel group
    const int slot = (int) ((hs - hb) / (3 * CHL));
    float y = 0.f;
    if (act) {
        const float u = raw[ch];
        const int k0 = r;
        const float h0 = k0 < 3 ? hs[(size_t) k0 * CHL + ch] : raw0[(size_t) (k0 - 3) * rsf + ch];
        const float h1 = k0 + 1 < 3 ? hs[(size_t) (k0 + 1) * CHL + ch] : raw0[(size_t) (k0 - 2) * rsf + ch];
        const float h2 = k0 + 2 < 3 ? hs[(size_t) (k0 + 2) * CHL + ch] : raw0[(size_t) (k0 - 1) * rsf + ch];
        float o = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * u;
        if (dyn) {
            const float * tp = dyn + hk * NCONV;
            o += u * tp[0];
            o += h2 * tp[1];
            o += h1 * tp[2];
            o += h0 * tp[3];
        }
        qz_gp(uh)[((size_t) (slot * VR + r)) * CHL + (isk ? H_K : H_V) + ch] = u;
        y = (u + o) * (dyn ? 1.f / sqrtf(2.f * NCONV + 1.f) : 1.f / sqrtf((float) NCONV + 1.f));
    }
    float n2[1] = {v_gsum2(y * y, red, grp, act)};
    if (!act) return;
    if (isk) {
        const float nr = sqrtf(n2[0]) + 1e-6f;
        y *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        a.kc[ch] = __float2half(y);
    } else {
        y *= rsqrtf(n2[0] / HD + a.eps);
        a.vc[ch] = __float2half(y);
    }
}
// c_head_post with the verify window (nper = 1, 2, or 4 units per 512-thread block)
VR_NI void v_head_post(CK hpost * e, int pos, float tau, int v, int tid, int r, float * uh, int rsf, const float * hb, int nper) {
    asm volatile ("" : "+v"(tid));
    hpost h = *e;
    const int row = e->dk ? pos : pos % PF_RING;          // dk: full layers only
    h.qr = qz_gp(h.qr); h.rr = qz_gp(h.rr); h.sk = qz_gp(h.sk); h.kr = qz_gp(h.kr); h.vr = qz_gp(h.vr);
    h.dk = h.dk ? qz_gp(h.dk) : nullptr; h.dv = h.dv ? qz_gp(h.dv) : nullptr;
    h.qclip = qz_gp(h.qclip); h.rclip = qz_gp(h.rclip); h.relb = qz_gp(h.relb); h.kconv = qz_gp(h.kconv); h.vconv = qz_gp(h.vconv);
    h.hist = qz_gp(h.hist); h.qh = qz_gp(h.qh); h.tb = qz_gp(h.tb);
    h.kc = qz_gp(h.kc + (size_t) row * KVL); h.vc = qz_gp(h.vc + (size_t) row * KVL);
    h.tau = e->dk ? tau : 1.f;
    const int tb_st = 512 / nper, grp = tid / tb_st, tb_t = tid % tb_st;
    d_head_post_v(h, v, tb_t & 127, tb_t < 128, min(TBW, pos + 1), r, uh, rsf, hb, grp, tb_t, tb_st);
}


// ---- the wave sums of many values at once. qz_gemv_core's per-row result is the xor
// butterfly over the 64 lanes (QZ_GEMV_WSUM). v_tsum<N, W> runs it for N values over W
// lanes at once: at each level, offset o = W/2 .. 1, a lane keeps the half of its values
// that lane & o selects and adds its partner's copy. Every value gets the same additions
// over the same pairs, so the same bits, in N/2 + N/4 + .. exchanges instead of N per
// level. Value k ends in the lanes where (lane >> (log2 W - log2 N)) % N == k.
template <int O> VR_NI float v_xor(float x) {
    if constexpr (O == 32) return __shfl_xor(x, 32, 64);
    else if constexpr (O == 16) return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x1F | (16 << 10)));
    else if constexpr (O == 8) return qz_dpp<0x128>(x);          // row_ror:8 = xor 8 in a 16-lane row
    else if constexpr (O == 4) return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x1F | (4 << 10)));
    else if constexpr (O == 2) return qz_dpp<0x4E>(x);
    else return qz_dpp<0xB1>(x);
}
template <int N, int O> VR_NI void v_tsum_lv(float (&v)[N], int lane, int n) {
    if (n > 1) {
        const bool hi = lane & O;
        #pragma unroll
        for (int j = 0; j < N / 2; j++)
            if (j < n / 2) {
                const float keep = hi ? v[j + n / 2] : v[j], send = hi ? v[j] : v[j + n / 2];
                v[j] = keep + v_xor<O>(send);
            }
    } else v[0] += v_xor<O>(v[0]);
}
template <int N, int O, int n> VR_NI void v_tsum_r(float (&v)[N], int lane) {
    v_tsum_lv<N, O>(v, lane, n);
    if constexpr (O > 1) v_tsum_r<N, O / 2, (n > 1 ? n / 2 : 1)>(v, lane);
}
template <int N, int W = 64> VR_NI float v_tsum(float (&v)[N], int lane) {
    static_assert((N & (N - 1)) == 0 && N <= W && (W == 16 || W == 32 || W == 64), "a power of two, at most one value per lane");
    v_tsum_r<N, W / 2, N>(v, lane);
    return v[0];
}
template <int N, int W = 64> VR_NI int v_tsum_k(int lane) {
    constexpr int lw = W == 64 ? 6 : W == 32 ? 5 : 4;
    int ln = 0;
    while ((1 << ln) < N) ln++;
    return (lane >> (lw - ln)) & (N - 1);
}
// the per-lane sums of qz_gemv_core<T, S, RS>'s unbuffered walk, without the wave sum
template <int T, int S, int RS = 0> VR_NI void v_core_acc(
        const unsigned char * a_vals, const float * a_scal, int lda, int lsa, int n_u,
        const char * w_vals, const __half * w_scal, int ldw, int lsw, int N, int r0, int lane, float (&acc)[T][S]) {
    #pragma unroll
    for (int t = 0; t < T; t++)
        #pragma unroll
        for (int s = 0; s < S; s++) acc[t][s] = 0.f;
    // 64 < n_u <= 64 + 64 / T: the n_u - 64 tail units (u >= 64) run one row per lane over
    // the whole wave, then return by shuffles. Lane u - 64 adds the product last, as the
    // plain walk's second step does.
    const int nt = n_u - 64;
    const bool tail = nt > 0 && T * nt <= 64;
    for (int u = lane; u < (tail ? 64 : n_u); u += 64) {
        qz_v4u w0[S], w1[S];
        float sw[S];
        #pragma unroll
        for (int s = 0; s < S; s++) {
            const int r = min(r0 + s + (RS && s >= S / 2 ? RS - S / 2 : 0), N - 1);
            const qz_gu4 * wr = (const qz_gu4 *)(w_vals + (size_t) r * ldw + u * 32);
            w0[s] = wr[0];
            w1[s] = wr[1];
            sw[s] = __half2float(__ushort_as_half(((const qz_gh *) w_scal)[(size_t) r * lsw + u]));
        }
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const uint4 a0 = *(const uint4 *)(a_vals + (size_t) t * lda + u * 32);
            const uint4 a1 = *(const uint4 *)(a_vals + (size_t) t * lda + u * 32 + 16);
            const float sa = a_scal[(size_t) t * lsa + u];
            #pragma unroll
            for (int s = 0; s < S; s++) {
                int d = 0;
                DOT4(d, w0[s].x, a0.x);
                DOT4(d, w0[s].y, a0.y);
                DOT4(d, w0[s].z, a0.z);
                DOT4(d, w0[s].w, a0.w);
                DOT4(d, w1[s].x, a1.x);
                DOT4(d, w1[s].y, a1.y);
                DOT4(d, w1[s].z, a1.z);
                DOT4(d, w1[s].w, a1.w);
                acc[t][s] += (float) d * sw[s] * sa;
            }
        }
    }
    if (tail) {                                                             // wave-uniform
        const int tt = lane / nt, u = 64 + lane % nt;
        float q[S], sa = 0.f;                       // d * sw per s and the sa: the owner lane
        #pragma unroll                              // fuses the add as the walk does (contract on)
        for (int s = 0; s < S; s++) q[s] = 0.f;
        if (tt < T) {
            const uint4 a0 = *(const uint4 *)(a_vals + (size_t) tt * lda + u * 32);
            const uint4 a1 = *(const uint4 *)(a_vals + (size_t) tt * lda + u * 32 + 16);
            sa = a_scal[(size_t) tt * lsa + u];
            #pragma unroll
            for (int s = 0; s < S; s++) {
                const int r = min(r0 + s + (RS && s >= S / 2 ? RS - S / 2 : 0), N - 1);
                const qz_gu4 * wr = (const qz_gu4 *)(w_vals + (size_t) r * ldw + u * 32);
                const qz_v4u w0 = wr[0], w1 = wr[1];
                const float sw = __half2float(__ushort_as_half(((const qz_gh *) w_scal)[(size_t) r * lsw + u]));
                int d = 0;
                DOT4(d, w0.x, a0.x); DOT4(d, w0.y, a0.y); DOT4(d, w0.z, a0.z); DOT4(d, w0.w, a0.w);
                DOT4(d, w1.x, a1.x); DOT4(d, w1.y, a1.y); DOT4(d, w1.z, a1.z); DOT4(d, w1.w, a1.w);
                q[s] = (float) d * sw;
            }
        }
        const int src = lane % nt;
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float st = __shfl(sa, t * nt + src, 64);
            #pragma unroll
            for (int s = 0; s < S; s++) {
                const float v = __shfl(q[s], t * nt + src, 64);
                if (lane < nt) acc[t][s] += v * st;
            }
        }
    }
}


// a K = D row pair for all VR rows, with v_core_acc's values. v_wload issues the pair's
// weights, so the next pair loads while v_wdot computes this one. Lane L sums its unit L.
// The 8 tail units 64 .. 71 run one row per 8 lanes and return by shuffles, d * sw and sa
// apart, so the owner lane fuses the add as the plain walk does.
// a K = D row pair for T rows (full double-buffered weight load for both main 64 units and
// 8 tail units). v_wdot_r1 handles single-row experts (cnt == 1) with 2 VGPRs and 1 shuffle.
struct v_wbuf { qz_v4u w0[2], w1[2], t0[2], t1[2]; float sw[2], tw[2]; };
VR_NI void v_wload(v_wbuf & b, const char * wv, const __half * ws, size_t row0, size_t row1, int lane) {
    constexpr int NU = D / 32;
    const int ut = 64 + (lane & 7);
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const size_t row = s ? row1 : row0;
        const qz_gu4 * wr = (const qz_gu4 *) (wv + row * D + lane * 32);
        b.w0[s] = wr[0]; b.w1[s] = wr[1];
        b.sw[s] = __half2float(__ushort_as_half(((const qz_gh *) ws)[row * NU + lane]));
        const qz_gu4 * wt = (const qz_gu4 *) (wv + row * D + ut * 32);
        b.t0[s] = wt[0]; b.t1[s] = wt[1];
        b.tw[s] = __half2float(__ushort_as_half(((const qz_gh *) ws)[row * NU + ut]));
    }
}
template <int T = VR> VR_NI float v_wdot(const v_wbuf & b, const unsigned char * la, const float * ls, int lane, float (&acc)[T][2]) {
    static_assert(D / 32 - 64 == 8 && T <= 8, "the 8 tail units, one row per 8 lanes");
    constexpr int NU = D / 32;
    #pragma unroll
    for (int t = 0; t < T; t++) {
        const uint4 a0 = *(const uint4 *) (la + t * D + lane * 32), a1 = *(const uint4 *) (la + t * D + lane * 32 + 16);
        const float sa = ls[t * NU + lane];
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int d = 0;
            DOT4(d, b.w0[s].x, a0.x); DOT4(d, b.w0[s].y, a0.y); DOT4(d, b.w0[s].z, a0.z); DOT4(d, b.w0[s].w, a0.w);
            DOT4(d, b.w1[s].x, a1.x); DOT4(d, b.w1[s].y, a1.y); DOT4(d, b.w1[s].z, a1.z); DOT4(d, b.w1[s].w, a1.w);
            acc[t][s] = 0.f;
            acc[t][s] += (float) d * b.sw[s] * sa;
        }
    }
    const int tt = min(lane >> 3, T - 1), ut = 64 + (lane & 7);
    const uint4 a0 = *(const uint4 *) (la + tt * D + ut * 32), a1 = *(const uint4 *) (la + tt * D + ut * 32 + 16);
    const float sat = ls[tt * NU + ut];
    float q[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        int d = 0;
        DOT4(d, b.t0[s].x, a0.x); DOT4(d, b.t0[s].y, a0.y); DOT4(d, b.t0[s].z, a0.z); DOT4(d, b.t0[s].w, a0.w);
        DOT4(d, b.t1[s].x, a1.x); DOT4(d, b.t1[s].y, a1.y); DOT4(d, b.t1[s].z, a1.z); DOT4(d, b.t1[s].w, a1.w);
        q[s] = (float) d * b.tw[s];
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        const float st = __shfl(sat, t * 8 + (lane & 7), 64);
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            const float v = __shfl(q[s], t * 8 + (lane & 7), 64);
            if (lane < 8) acc[t][s] += v * st;
        }
    }
    constexpr int NV = T * 2;
    float (&fv)[NV] = *(float (*)[NV]) &acc[0][0];
    return v_tsum<NV>(fv, lane);
}
VR_NI float v_wdot_r4(const v_wbuf & b, const unsigned char * la, const float * ls,
                      int r0, int r1, int r2, int r3, int lane, float (&acc)[4][2]) {
    constexpr int NU = D / 32;
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        const int r = t == 0 ? r0 : t == 1 ? r1 : t == 2 ? r2 : r3;
        const uint4 a0 = *(const uint4 *) (la + r * D + lane * 32), a1 = *(const uint4 *) (la + r * D + lane * 32 + 16);
        const float sa = ls[r * NU + lane];
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int d = 0;
            DOT4(d, b.w0[s].x, a0.x); DOT4(d, b.w0[s].y, a0.y); DOT4(d, b.w0[s].z, a0.z); DOT4(d, b.w0[s].w, a0.w);
            DOT4(d, b.w1[s].x, a1.x); DOT4(d, b.w1[s].y, a1.y); DOT4(d, b.w1[s].z, a1.z); DOT4(d, b.w1[s].w, a1.w);
            acc[t][s] = 0.f;
            acc[t][s] += (float) d * b.sw[s] * sa;
        }
    }
    const int tt = min(lane >> 3, 3), ut = 64 + (lane & 7);
    const int tr = tt == 0 ? r0 : tt == 1 ? r1 : tt == 2 ? r2 : r3;
    const uint4 a0 = *(const uint4 *) (la + tr * D + ut * 32), a1 = *(const uint4 *) (la + tr * D + ut * 32 + 16);
    const float sat = ls[tr * NU + ut];
    float q[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        int d = 0;
        DOT4(d, b.t0[s].x, a0.x); DOT4(d, b.t0[s].y, a0.y); DOT4(d, b.t0[s].z, a0.z); DOT4(d, b.t0[s].w, a0.w);
        DOT4(d, b.t1[s].x, a1.x); DOT4(d, b.t1[s].y, a1.y); DOT4(d, b.t1[s].z, a1.z); DOT4(d, b.t1[s].w, a1.w);
        q[s] = (float) d * b.tw[s];
    }
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        const float st = __shfl(sat, t * 8 + (lane & 7), 64);
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            const float v = __shfl(q[s], t * 8 + (lane & 7), 64);
            if (lane < 8) acc[t][s] += v * st;
        }
    }
    float (&fv)[8] = *(float (*)[8]) &acc[0][0];
    return v_tsum<8>(fv, lane);
}


// d_router for T rows at once: a wave per expert, its weights loaded once. Each row's
// sum follows d_router's order, and v_tsum reduces the row sums on the butterfly's pairs.
template <int T = VR> VR_NI void v_router(const float * F0, const float * wr, const float * bias, float * H0, int rs, int e, int lane) {
    F0 = qz_gp(F0); wr = qz_gp(wr); bias = qz_gp(bias); H0 = qz_gp(H0);
    float s[T];
    #pragma unroll
    for (int t = 0; t < T; t++) s[t] = 0.f;
    #pragma unroll 1
    for (int c0 = 0; c0 < D / 64; c0 += 12) {
        float wv[12];
        #pragma unroll
        for (int k = 0; k < 12; k++) wv[k] = wr[(size_t) e * D + lane + 64 * (c0 + k)];
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float * f = (const float *) ((const char *) F0 + (size_t) t * rs);
            float fv[12];
            #pragma unroll
            for (int k = 0; k < 12; k++) fv[k] = f[lane + 64 * (c0 + k)];
            #pragma unroll
            for (int k = 0; k < 12; k++) s[t] += fv[k] * wv[k];
        }
    }
    const float v = v_tsum<T>(s, lane);                   // row v_tsum_k<T>(lane)
    if ((lane & (64 / T - 1)) == 0) {
        float * ps = (float *) ((char *) H0 + (size_t) v_tsum_k<T>(lane) * rs);
        const float p = sigm(v);
        ps[e] = p; ps[NE + e] = p + bias[e];
    }
}
// v_pick_all by a wave per row: 8 rounds of the wave max of (p + bias, NE - 1 - id).
// That key sorts max first, ties to the lower id, and folds -0 into +0 as == does.
VR_NI unsigned long long v_pkey(float x, int id) {
    unsigned u = __float_as_uint(x + 0.f);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) u << 32) | (unsigned) (NE - 1 - id);
}
VR_NI void v_pick_w(const float * H0, int rsf, float scale, int * sids, float * swts, int tid, int nr) {
    static_assert(VR == 8 && NE == 128, "a wave per row, 2 candidates per lane");
    asm volatile ("" : "+v"(tid));
    const int r = tid >> 6, lane = tid & 63;
    if (r < nr) {
        const float * h = qz_gp(H0) + (size_t) r * rsf;
        const float pa = h[lane], pb = h[lane + 64];
        unsigned long long ka = v_pkey(h[NE + lane], lane), kb = v_pkey(h[NE + lane + 64], lane + 64);
        float lp[NE_USED];
        int ids[NE_USED];
        #pragma unroll
        for (int k = 0; k < NE_USED; k++) {
            unsigned long long m = ka > kb ? ka : kb;
            #pragma unroll
            for (int o = 32; o > 0; o >>= 1) {
                const unsigned lo = __shfl_xor((unsigned) m, o, 64), hi = __shfl_xor((unsigned) (m >> 32), o, 64);
                const unsigned long long q = ((unsigned long long) hi << 32) | lo;
                m = q > m ? q : m;
            }
            const int w = NE - 1 - (int) (unsigned) m;
            if (w == lane) ka = 0; else if (w == lane + 64) kb = 0;
            lp[k] = __shfl(w < 64 ? pa : pb, w & 63, 64);
            ids[k] = w;
        }
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < NE_USED; k++) sum += lp[k];
        sum += 1e-20f;
        #pragma unroll
        for (int k = 0; k < NE_USED; k++)
            if (lane == k) { sids[r * NE_USED + k] = ids[k]; swts[r * NE_USED + k] = lp[k] / sum * scale; }
    }
    __syncthreads();
}

// The experts the active rows picked, grouped by popcount descending (buckets >=4, 3, 2, 1)
// so v_down_dd packs experts with similar row counts into the same wave.
struct vdist { int nd; int dex[VR * NE_USED]; unsigned dmk[VR * NE_USED]; };
VR_NI void v_distinct(const int * sids, vdist & dd, int tid, int nr) {
    __shared__ int bc[4][2], bo[4];
    unsigned m = 0;
    const int lane = tid & 63, wave = tid >> 6;
    if (tid < NE) {
        #pragma unroll
        for (int r = 0; r < VR; r++)
            #pragma unroll
            for (int k = 0; k < NE_USED; k++) m |= (r < nr && sids[r * NE_USED + k] == tid ? 1u : 0u) << r;
    }
    const int pc = __popc(m);
    const int bkt = pc >= 4 ? 0 : (pc == 3 ? 1 : (pc == 2 ? 2 : (pc == 1 ? 3 : -1)));
    unsigned long long bm[4];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        bm[b] = __ballot(tid < NE && bkt == b);
        if (tid < NE && lane == 0) bc[b][wave] = __popcll(bm[b]);
    }
    __syncthreads();
    if (tid == 0) {
        int off = 0;
        #pragma unroll
        for (int b = 0; b < 4; b++) { bo[b] = off; off += bc[b][0] + bc[b][1]; }
        dd.nd = off;
    }
    __syncthreads();
    if (tid < NE && bkt >= 0) {
        const int idx = bo[bkt] + (wave ? bc[bkt][0] : 0) + __popcll(bm[bkt] & ((1ull << lane) - 1));
        dd.dex[idx] = tid; dd.dmk[idx] = m;
    }
    __syncthreads();
}
// row r's slot of expert e, which it picked
VR_NI int v_slot(const int * sids, int r, int e) {
    int k = 0;
    #pragma unroll
    for (int q = 0; q < NE_USED; q++) k = sids[r * NE_USED + q] == e ? q : k;
    return k;
}
// gate | up + SITU + quant over the distinct experts, c_up_situ_w's values for every row
// that picked the expert. A wave owns one (expert, 32-activation quant group) and walks its
// 32 gate and up row pairs for the T rows at once, weights read once, keeps the SITU
// values in its LDS tile, then quantizes each picking row's group into its slot.
template <int T = VR> VR_NI void v_up_dd(const char * wv, const __half * ws, const int * sids, const vdist & dd,
                   const unsigned char * av0, const float * as0, float beta, float lb,
                   unsigned char * hq0, float * hs0, int tid, int rs, int rsf, int nr) {
    asm volatile ("" : "+v"(tid));
    av0 = qz_gp(av0); as0 = qz_gp(as0); hq0 = qz_gp(hq0); hs0 = qz_gp(hs0);
    const int wave = tid >> 6, lane = tid & 63, nw = gridDim.x * 8;
    // the active activation rows (int8 and scales) in dynamic LDS, read by every task
    unsigned char * la = (unsigned char *) cyc_lds;                       // [VR][D]
    float * ls = cyc_lds + VR * D / 4;                                     // [VR][D / 32]
    __syncthreads();
    for (int i = tid; i < nr * D / 16; i += 512) {
        const int r = i / (D / 16), c = i % (D / 16);
        ((uint4 *) la)[i] = *(const uint4 *) (av0 + (size_t) r * rs + c * 16);
    }
    for (int i = tid; i < nr * D / 32; i += 512) {
        const int r = i / (D / 32), c = i % (D / 32);
        ls[i] = as0[(size_t) r * rsf + c];
    }
    __syncthreads();
    float * lq = ls + VR * D / 32 + wave * VR * 32;                        // [VR][32] per wave
    constexpr int NG = FFL / 32;
    #pragma unroll 1
    for (int gw = blockIdx.x * 8 + wave; gw < dd.nd * NG; gw += nw) {
        const int u = gw / NG, g = gw % NG;
        const int e = __builtin_amdgcn_readfirstlane(dd.dex[u]);
        const unsigned mk = __builtin_amdgcn_readfirstlane(dd.dmk[u]);
        const size_t base = (size_t) e * GUL;
        const size_t r0 = base + (size_t) g * 64;                           // gate rows r0 .. r0 + 31, up + 32
        constexpr int LV = 64 / (2 * T);                                   // lanes per value
        const int kv = v_tsum_k<2 * T>(lane), vr = kv >> 1, vs = kv & 1;
        const bool wl = (lane & (LV - 1)) == 0 && vs == 0 && (mk >> vr & 1);
        v_wbuf A, B;
        float acc[T][2];
        v_wload(A, wv, ws, r0, r0 + 32, lane);
        #pragma unroll 1
        for (int a = 0; a < 32; a += 2) {
            v_wload(B, wv, ws, r0 + a + 1, r0 + a + 33, lane);
            float val = v_wdot<T>(A, la, ls, lane, acc), upv = __shfl_down(val, LV, 64);
            if (wl) lq[vr * 32 + a] = situ_v(val, upv, beta, lb);
            // unconditional on purpose: a conditional reload here miscompiles (NaN)
            v_wload(A, wv, ws, r0 + min(a + 2, 31), r0 + min(a + 2, 31) + 32, lane);
            val = v_wdot<T>(B, la, ls, lane, acc); upv = __shfl_down(val, LV, 64);
            if (wl) lq[vr * 32 + a + 1] = situ_v(val, upv, beta, lb);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        // the rows two at a time, one per 32-lane half
        unsigned rem = mk;
        while (rem) {                                                       // wave-uniform
            const int ra = __ffs(rem) - 1;
            rem &= rem - 1;
            const int rb = rem ? __ffs(rem) - 1 : -1;
            if (rb >= 0) rem &= rem - 1;
            const int r = (lane >> 5) ? rb : ra;
            if (r >= 0) {                                                   // half-wave uniform
                const int k = v_slot(sids, r, e);
                quant32(lq[r * 32 + (lane & 31)], hq0 + (size_t) r * rs + k * FFL, hs0 + (size_t) r * rsf + k * (FFL / 32), g * 32 + (lane & 31));
            }
        }
        __builtin_amdgcn_wave_barrier();                                    // the tile is rewritten by the next group
    }
}
// For nr > 4, pack each expert's picking rows in groups of up to 4 and run v_wdot_r4
// instead of computing all 8 rows: >95% of experts have <= 4 picking rows, so they stay
// memory-bandwidth-bound at T=4 cost while the rare >4-row expert runs a second pass.
VR_NI void v_up_dd_r4(const char * wv, const __half * ws, const int * sids, const vdist & dd,
                      const unsigned char * av0, const float * as0, float beta, float lb,
                      unsigned char * hq0, float * hs0, int tid, int rs, int rsf, int nr) {
    asm volatile ("" : "+v"(tid));
    av0 = qz_gp(av0); as0 = qz_gp(as0); hq0 = qz_gp(hq0); hs0 = qz_gp(hs0);
    const int wave = tid >> 6, lane = tid & 63, nw = gridDim.x * 8;
    unsigned char * la = (unsigned char *) cyc_lds;                       // [VR][D]
    float * ls = cyc_lds + VR * D / 4;                                     // [VR][D / 32]
    __syncthreads();
    for (int i = tid; i < nr * D / 16; i += 512) {
        const int r = i / (D / 16), c = i % (D / 16);
        ((uint4 *) la)[i] = *(const uint4 *) (av0 + (size_t) r * rs + c * 16);
    }
    for (int i = tid; i < nr * D / 32; i += 512) {
        const int r = i / (D / 32), c = i % (D / 32);
        ls[i] = as0[(size_t) r * rsf + c];
    }
    __syncthreads();
    float * lq = ls + VR * D / 32 + wave * VR * 32;                        // [VR][32] per wave
    constexpr int NG = FFL / 32, LV = 64 / 8;
    const int kv = v_tsum_k<8>(lane), vr = kv >> 1, vs = kv & 1;
    const bool wl_base = (lane & (LV - 1)) == 0 && vs == 0;
    const int tot = dd.nd * NG;
    #pragma unroll 1
    for (int gw = blockIdx.x * 8 + wave; gw < tot; gw += nw) {
        // Shift task order when tot > nw so multi-pass (pc > 4) experts at u = 0.. land
        // on single-task waves [tot - nw, nw) instead of two-task waves [0, tot - nw).
        const int idx = tot > nw ? (gw + nw) % tot : gw;
        const int u = idx / NG, g = idx % NG;
        const int e = __builtin_amdgcn_readfirstlane(dd.dex[u]);
        const unsigned mk = __builtin_amdgcn_readfirstlane(dd.dmk[u]);
        const size_t r0 = (size_t) e * GUL + (size_t) g * 64;
        unsigned rem_mk = mk;
        #pragma unroll 1
        do {
            int p0 = 0, p1 = 0, p2 = 0, p3 = 0, n1 = 0;
            if (rem_mk) { p0 = __builtin_amdgcn_readfirstlane(__ffs(rem_mk) - 1); rem_mk &= rem_mk - 1; n1 = 1; }
            if (rem_mk) { p1 = __builtin_amdgcn_readfirstlane(__ffs(rem_mk) - 1); rem_mk &= rem_mk - 1; n1 = 2; }
            if (rem_mk) { p2 = __builtin_amdgcn_readfirstlane(__ffs(rem_mk) - 1); rem_mk &= rem_mk - 1; n1 = 3; }
            if (rem_mk) { p3 = __builtin_amdgcn_readfirstlane(__ffs(rem_mk) - 1); rem_mk &= rem_mk - 1; n1 = 4; }
            const bool wl1 = wl_base && vr < n1;
            const int wr1 = vr == 0 ? p0 : vr == 1 ? p1 : vr == 2 ? p2 : p3;
            v_wbuf A, B;
            float acc[4][2];
            v_wload(A, wv, ws, r0, r0 + 32, lane);
            #pragma unroll 1
            for (int a = 0; a < 32; a += 2) {
                v_wload(B, wv, ws, r0 + a + 1, r0 + a + 33, lane);
                float val = v_wdot_r4(A, la, ls, p0, p1, p2, p3, lane, acc), upv = __shfl_down(val, LV, 64);
                if (wl1) lq[wr1 * 32 + a] = situ_v(val, upv, beta, lb);
                v_wload(A, wv, ws, r0 + min(a + 2, 31), r0 + min(a + 2, 31) + 32, lane);
                val = v_wdot_r4(B, la, ls, p0, p1, p2, p3, lane, acc); upv = __shfl_down(val, LV, 64);
                if (wl1) lq[wr1 * 32 + a + 1] = situ_v(val, upv, beta, lb);
            }
        } while (rem_mk);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        unsigned rem = mk;
        while (rem) {
            const int ra = __ffs(rem) - 1;
            rem &= rem - 1;
            const int rb = rem ? __ffs(rem) - 1 : -1;
            if (rb >= 0) rem &= rem - 1;
            const int r = (lane >> 5) ? rb : ra;
            if (r >= 0) {
                const int k = v_slot(sids, r, e);
                quant32(lq[r * 32 + (lane & 31)], hq0 + (size_t) r * rs + k * FFL, hs0 + (size_t) r * rsf + k * (FFL / 32), g * 32 + (lane & 31));
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
}
// the down rows over the distinct experts, c_moe_down_q's quads and per-(row, slot) values.
// Each expert's 4 weight rows load once for the rows that picked it, from its own row list,
// the rows' SITU activations staged in LDS. Then per row the same k-order fold of wk *
// acc. LDS: lacc [VR][MD_Q][NE_USED][4], then the activations (VER_LDS).
constexpr int VER_LDS0 = (VR * MD_Q * NE_USED * 4) * 4 + VR * NE_USED * FFL + VR * NE_USED * (FFL / 32) * 4;
// at TP4 the expert slices halve and the decode's (or the attention's) LDS is the larger
constexpr int VER_LDS = VER_LDS0 > CYC_LDS ? VER_LDS0 : CYC_LDS;
VR_NI void v_down_dd(const char * wv, const __half * ws, const int * sids, const float * swts, const vdist & dd,
                     const unsigned char * aq0, const float * as0, float * y0, const qz_xpush<QZ_NC> & yp0, int tid, int rs, int rsf, float * lacc,
                     int nr) {
    static_assert(FFL / 32 <= 32, "one unit per lane");
    asm volatile ("" : "+v"(tid));
    constexpr int NU = FFL / 32, NQ = D / 4, LR = MD_Q * NE_USED * 4;
    aq0 = qz_gp(aq0); as0 = qz_gp(as0); y0 = qz_gp(y0);
    const int wave = tid >> 6, lane = tid & 63, gq = lane >> 4, l = lane & 15;
    const int q0 = blockIdx.x * NQ / gridDim.x, q1 = (blockIdx.x + 1) * NQ / gridDim.x;
    const int np = (dd.nd + 3) / 4;
    // the active rows' SITU activations, all NE_USED slots, staged in LDS after the partials
    unsigned char * la = (unsigned char *) (lacc + VR * LR);
    float * ls = (float *) (la + VR * NE_USED * FFL);
    for (int i = tid; i < nr * NE_USED * FFL / 16; i += 512) {
        const int r = i / (NE_USED * FFL / 16), o = i % (NE_USED * FFL / 16);
        ((uint4 *) la)[i] = ((const uint4 *) (aq0 + (size_t) r * rs))[o];
    }
    for (int i = tid; i < nr * NE_USED * NU; i += 512) {
        const int r = i / (NE_USED * NU), o = i % (NE_USED * NU);
        ls[i] = as0[(size_t) r * rsf + o];
    }
    __syncthreads();
    for (int c0 = q0; c0 < q1; c0 += MD_Q) {
        const int nq = min(MD_Q, q1 - c0);
        #pragma unroll 1
        for (int task = wave; task < nq * np; task += 8) {
            // 4 experts per wave, 16 lanes each, lane l summing unit l. Units 16 and 17,
            // the 4 rows' tail in lanes l < 8 as row l / 2 and unit 16 + l % 2, go first to
            // lanes 0 and 1: c_moe_down_q's first shfl_down level, a_i + a_(i+16) plus 0
            // above unit 17. Then the 16-lane levels as the xor butterfly.
            const int qd = task / np, ui = 4 * (task % np) + gq;
            const bool live = ui < dd.nd;                                   // 16-lane uniform
            const int e = live ? dd.dex[ui] : dd.dex[0];
            const unsigned mk = live ? dd.dmk[ui] : 0u;
            const int r0 = (c0 + qd) * 4;
            // NU > 16 (TP2's 18): lanes hold units 0..15 plus the tail. NU <= 16 (TP4's 9):
            // lane l holds unit l, zero from NU up, as c_moe_down_q's l < NU
            constexpr bool TAIL = NU > 16;
            const int lu = TAIL ? l : min(l, NU - 1);
            const int st = (l >> 1) & 3, ut = TAIL ? 16 + (l & 1) : 0;
            const char * w = wv + ((size_t) e * D + r0) * FFL;
            const __half * sw = ws + ((size_t) e * D + r0) * NU;
            qz_v4u w0[4], w1[4], t0, t1;
            float wsc[4], tsc;
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                const qz_gu4 * wr = (const qz_gu4 *) (w + (size_t) s * FFL + lu * 32);
                w0[s] = wr[0]; w1[s] = wr[1];
                wsc[s] = __half2float(__ushort_as_half(((const qz_gh *) sw)[(size_t) s * NU + lu]));
            }
            if constexpr (TAIL) {
                const qz_gu4 * wr = (const qz_gu4 *) (w + (size_t) st * FFL + ut * 32);
                t0 = wr[0]; t1 = wr[1];
                tsc = __half2float(__ushort_as_half(((const qz_gh *) sw)[(size_t) st * NU + ut]));
            }
            int cnt = __popc(mk);
            cnt = max(cnt, __shfl_xor(cnt, 16, 64));
            cnt = max(cnt, __shfl_xor(cnt, 32, 64));                        // the wave's longest row list
            unsigned rem = mk;
            #pragma unroll 1
            for (int jr = 0; jr < cnt; jr++) {
                const bool on = rem != 0;
                const int r = on ? __ffs(rem) - 1 : 0;
                rem &= rem - 1;
                const int k = v_slot(sids, r, e);
                const unsigned char * aq = la + (r * NE_USED + k) * FFL;
                const float * as = ls + (r * NE_USED + k) * NU;
                const uint4 a0 = *(const uint4 *) (aq + lu * 32), a1 = *(const uint4 *) (aq + lu * 32 + 16);
                const float sa = as[lu];
                float acc[4];
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    int d = 0;
                    DOT4(d, w0[s].x, a0.x); DOT4(d, w0[s].y, a0.y); DOT4(d, w0[s].z, a0.z); DOT4(d, w0[s].w, a0.w);
                    DOT4(d, w1[s].x, a1.x); DOT4(d, w1[s].y, a1.y); DOT4(d, w1[s].z, a1.z); DOT4(d, w1[s].w, a1.w);
                    acc[s] = TAIL || l < NU ? (float) d * wsc[s] * sa : 0.f;
                }
                if constexpr (TAIL) {
                const uint4 b0 = *(const uint4 *) (aq + ut * 32), b1 = *(const uint4 *) (aq + ut * 32 + 16);
                const float sb = as[ut];
                int dt = 0;
                DOT4(dt, t0.x, b0.x); DOT4(dt, t0.y, b0.y); DOT4(dt, t0.z, b0.z); DOT4(dt, t0.w, b0.w);
                DOT4(dt, t1.x, b1.x); DOT4(dt, t1.y, b1.y); DOT4(dt, t1.z, b1.z); DOT4(dt, t1.w, b1.w);
                const float q = (float) dt * tsc * sb;
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    const float tv = __shfl(q, (lane & ~15) + 2 * s + (l & 1), 64);
                    acc[s] += l < 2 ? tv : 0.f;
                }
                }
                const float val = v_tsum<4, 16>(acc, l);                    // value s in lanes 4 s ..
                if ((l & 3) == 0 && on) lacc[r * LR + (qd * NE_USED + k) * 4 + (l >> 2)] = val;
            }
        }
        __syncthreads();
        for (int i = tid; i < nr * nq * 4; i += 512) {
            const int r = i / (nq * 4), qd = (i >> 2) % nq, s = i & 3, row = (c0 + qd) * 4 + s;
            float tot = 0.f;
            #pragma unroll
            for (int k = 0; k < NE_USED; k++) tot += swts[r * NE_USED + k] * lacc[r * LR + (qd * NE_USED + k) * 4 + s];
            y0[(size_t) r * rsf + row] = tot;
            yp0.put((size_t) r * D + row, tot);
        }
        __syncthreads();
    }
}

// ga GEMVs (K = D = 2304) for T rows: stages each activation plane into LDS once when
// the job's input pointer changes, so waves read activations from LDS instead of global.
template <int S, int T = VR> VR_NI void v_ga_gemv(CK gjobs * J, int nga, int rs, int rsf, int tid) {
    asm volatile ("" : "+v"(tid));
    unsigned char * la = (unsigned char *) cyc_lds;                      // [T][D]
    float * ls = (float *) (la + T * D);                                 // [T][D / 32]
    const unsigned char * cur_av = nullptr;
    const int wave = tid >> 6, lane = tid & 63, nb = gridDim.x;
    constexpr int NV = T * S;
    const int kv = v_tsum_k<NV>(lane), vt = kv / S, vs = kv % S;
    const bool wl = (lane & (64 / NV - 1)) == 0;
    for (int hb = blockIdx.x; hb < (4 / S) * nga; hb += nb) {
        const int b = hb / (4 / S), half = hb % (4 / S);
        int k = 0;
        while (k + 1 < J->n && b >= J->j[k].g1) k++;
        CK gjob * j = &J->j[k];
        if (j->av != cur_av) {
            cur_av = j->av;
            __syncthreads();
            for (int i = tid; i < T * D / 16; i += 512) {
                const int r = i / (D / 16), c = i % (D / 16);
                ((uint4 *) la)[i] = *(const uint4 *) (qz_gp(j->av) + (size_t) r * rs + c * 16);
            }
            for (int i = tid; i < T * (D / 32); i += 512) {
                const int r = i / (D / 32), c = i % (D / 32);
                ls[i] = qz_gp(j->as)[(size_t) r * rsf + c];
            }
            __syncthreads();
        }
        const int g = b - (k ? J->j[k - 1].g1 : 0);
        const int r0 = g * 32 + half * 8 * S + wave * S;
        const int N = j->N;
        float * y = qz_gp(j->y);
        float acc[T][S];
        v_core_acc<T, S>(la, ls, D, D / 32, D / 32, j->wv, j->ws, D, D / 32, N, r0, lane, acc);
        float (&fv)[NV] = *(float (*)[NV]) &acc[0][0];
        const float val = v_tsum<NV>(fv, lane);
        if (wl && r0 + vs < N) y[(size_t) vt * rsf + r0 + vs] = val;
    }
    __syncthreads();
}
// attn_output GEMV for T rows (N = D = 2304, K = QL = 1536): stages T activation rows
// in LDS once per block, balances 576 quads (2 x S=2 pairs per quad, low VGPRs) evenly
// across all blocks, and pushes coalesced float4 stores to both PART0 and the peer inboxes (yp0).
template <int T> VR_NI void v_go_gemv(CK gjob * j, int rs, int rsf, const qz_xpush<QZ_NC> & yp0, int tid) {
    asm volatile ("" : "+v"(tid));
    unsigned char * la = (unsigned char *) cyc_lds;                      // [T][QL]
    float * ls = (float *) (la + T * QL);                                // [T][QL / 32]
    float * tile = ls + T * (QL / 32);                                   // [T][40]
    __syncthreads();
    for (int i = tid; i < T * QL / 16; i += 512) {
        const int r = i / (QL / 16), c = i % (QL / 16);
        ((uint4 *) la)[i] = *(const uint4 *) (qz_gp(j->av) + (size_t) r * rs + c * 16);
    }
    for (int i = tid; i < T * (QL / 32); i += 512) {
        const int r = i / (QL / 32), c = i % (QL / 32);
        ls[i] = qz_gp(j->as)[(size_t) r * rsf + c];
    }
    __syncthreads();
    constexpr int NQ = D / 4;
    const int nb = gridDim.x, wave = tid >> 6, lane = tid & 63;
    const int q0 = blockIdx.x * NQ / nb, q1 = (blockIdx.x + 1) * NQ / nb, nq = q1 - q0;
    constexpr int NV = T * 2;
    const int kv = v_tsum_k<NV>(lane), vt = kv >> 1, vs = kv & 1;
    const bool wl = (lane & (64 / NV - 1)) == 0;
    for (int p = wave; p < 2 * nq; p += 8) {
        float acc[T][2];
        v_core_acc<T, 2>(la, ls, QL, QL / 32, QL / 32, j->wv, j->ws, QL, QL / 32, D, q0 * 4 + p * 2, lane, acc);
        float (&fv)[NV] = *(float (*)[NV]) &acc[0][0];
        const float val = v_tsum<NV>(fv, lane);
        if (wl) tile[vt * nq * 4 + p * 2 + vs] = val;
    }
    __syncthreads();
    for (int i = tid; i < T * nq; i += 512) {
        const int t = i / nq, q = i % nq;
        const float4 v = *(const float4 *) (tile + t * nq * 4 + q * 4);
        const int r = (q0 + q) * 4;
        *(float4 *) (qz_gp(j->y) + (size_t) t * rsf + r) = v;
        yp0.put4((size_t) t * D + r, v);
    }
    __syncthreads();
}
// c_gemv2 over T rows at once: activation row t at av + t rs, output at y + t rsf,
// yp0 the peers' inbox rows at stride D. Weights are read once, and each row's value is
// c_gemv2's, since qz_gemv_core's per-row order does not depend on T.
template <int S, int T = VR> VR_NI void v_gemv(CK gjobs * J, int hb, int rs, int rsf, const qz_xpush<QZ_NC> & yp0) {
    static_assert(S == 2 || S == 4, "8 S rows per unit, 32 per job block");
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int b = hb / (4 / S), half = hb % (4 / S);
    int k = 0;
    while (k + 1 < J->n && b >= J->j[k].g1) k++;
    CK gjob * j = &J->j[k];
    const int g = b - (k ? J->j[k - 1].g1 : 0);
    const int r0 = g * 32 + half * 8 * S + wave * S;
    const int N = j->N, K = j->K;
    float * y = qz_gp(j->y);
    float acc[T][S];
    v_core_acc<T, S>(qz_gp(j->av), qz_gp(j->as), rs, rsf, K / 32, j->wv, j->ws, K, K / 32, N, r0, lane, acc);
    constexpr int NV = T * S;
    float (&fv)[NV] = *(float (*)[NV]) &acc[0][0];
    const float val = v_tsum<NV>(fv, lane);
    const int kv = v_tsum_k<NV>(lane), vt = kv / S, vs = kv % S;
    const bool wl = (lane & (64 / NV - 1)) == 0;              // one writer lane per value
    if (wl && r0 + vs < N) y[(size_t) vt * rsf + r0 + vs] = val;
    if (yp0) {      // the peer inbox: the unit's 8 S x VR values through LDS, 16-byte stores (N % 32 == 0)
        constexpr int RU = 8 * S;
        float * tile = cyc_lds;                                         // [T][RU]
        const int base = g * 32 + half * RU;
        __syncthreads();
        if (wl) tile[vt * RU + wave * S + vs] = val;
        __syncthreads();
        const int tid = threadIdx.x;
        if (tid < T * RU / 4) {
            const int t = tid / (RU / 4), c = tid % (RU / 4);
            yp0.put4((size_t) t * D + base + 4 * c, *(const float4 *) (tile + t * RU + 4 * c));
        }
    }
}
// c_ctrl_down_all over the nr active rows. Each (job, row pair) loads its weights once and runs
// the decode's per-row sums for every row in turn, row t at cur / msg / g + t rsf.
template <int NJ> VR_NI void v_ctrl_down(CK cjob * cj, int rsf, int tid, int nr) {
    asm volatile ("" : "+v"(tid));
    __shared__ float red[2 * 4 * 2];
    constexpr int NV = D / 256, MR = 2;
    const int bpj = gridDim.x / NJ, j = blockIdx.x / bpj, bj = blockIdx.x % bpj, h = tid >> 8, t = tid & 255;
    if (j >= NJ) return;                                   // the leftover blocks (block-uniform)
    const float * cur0 = qz_gp(cj[j].cur), * msg0 = qz_gp(cj[j].msg), * wd = qz_gp(cj[j].wd);
    float * g0 = qz_gp(cj[j].g);
    for (int r0 = bj; r0 < GRANK; r0 += 2 * MR * bpj) {
        float wa[MR][NV], wb[MR][NV];
        #pragma unroll
        for (int m = 0; m < MR; m++) {
            const int r = r0 + bpj * (h + 2 * m);
            if (r < GRANK) {
                const float * w = wd + (size_t) r * 2 * D;
                #pragma unroll
                for (int k = 0; k < NV; k++) { wa[m][k] = w[t + 256 * k]; wb[m][k] = w[D + t + 256 * k]; }
            } else {
                #pragma unroll
                for (int k = 0; k < NV; k++) { wa[m][k] = 0.f; wb[m][k] = 0.f; }
            }
        }
        #pragma unroll 1
        for (int row = 0; row < nr; row++) {   // the active rows
            const float * cur = cur0 + (size_t) row * rsf, * msg = msg0 + (size_t) row * rsf;
            float x[NV], y[NV];
            #pragma unroll
            for (int k = 0; k < NV; k++) { x[k] = cur[t + 256 * k]; y[k] = msg[t + 256 * k]; }
            float ss[2] = {0.f, 0.f};
            #pragma unroll
            for (int k = 0; k < NV; k++) { ss[0] += x[k] * x[k]; ss[1] += y[k] * y[k]; }
            hsum<2>(ss, red);
            const float rc = rsqrtf(ss[0] / D + 1e-6f), rm = rsqrtf(ss[1] / D + 1e-6f);
            float a[MR];
            #pragma unroll
            for (int m = 0; m < MR; m++) {
                a[m] = 0.f;
                #pragma unroll
                for (int k = 0; k < NV; k++) a[m] += wa[m][k] * (x[k] * rc) + wb[m][k] * (y[k] * rm);
            }
            hsum<MR>(a, red);
            if (t < MR) {
                #pragma unroll
                for (int m = 0; m < MR; m++) {
                    const int r = r0 + bpj * (h + 2 * m);
                    if (t == m && r < GRANK) g0[(size_t) row * rsf + r] = a[m];
                }
            }
        }
    }
    __syncthreads();
}
// Fused v_ctrl_up + rmsnorm + quant32 over NJ jobs and T rows, partitioned on 32-element
// groups so every block keeps its T outputs in registers across one grid barrier (for the
// global sum of squares) and immediately normalizes and quantizes without a separate c_norm.
template <int NJ, int T = VR> VR_NI void v_ctrl_up_norm(CK cjob * cj, CK njob * nj, float eps,
                                                        float * ss_buf, int rs, int rsf,
                                                        unsigned int * vbst, int & kb, int tid, int nr) {
    asm volatile ("" : "+v"(tid));
    float * g = cyc_lds;                                   // [T][NJ][GRANK]
    float * sf = cyc_lds + T * NJ * GRANK;                 // [T][256]
    float * gs = sf + T * 256;                             // [T][8]
    float * rsv = gs + T * 8;                              // [T][NJ]
    float * sbl = rsv + T * NJ;                            // [T * NJ][64]
    __syncthreads();
    for (int i = tid; i < T * NJ * GRANK; i += 512) {
        const int row = i / (NJ * GRANK), jj = (i / GRANK) % NJ, k = i % GRANK;
        g[i] = qz_gp(cj[jj].g)[(size_t) row * rsf + k];
    }
    __syncthreads();
    constexpr int NG = D / 32, NTG = NJ * NG, KC = 32;
    const int nb = gridDim.x;
    const int g0 = blockIdx.x * NTG / nb, g1 = (blockIdx.x + 1) * NTG / nb;
    const int ng = g1 - g0, n = ng * 32;
    const int which = tid >= n, q = tid - (which ? n : 0);
    const int grp = g0 + (min(q, max(n - 1, 0)) >> 5), j = grp / NG, i = (grp % NG) * 32 + (q & 31);
    float a[T];
    #pragma unroll
    for (int row = 0; row < T; row++) a[row] = 0.f;
    if (tid < 2 * n) {
        const float * w = qz_gp(which ? cj[j].wft : cj[j].wgt) + i;
        #pragma unroll 1
        for (int k0 = 0; k0 < GRANK; k0 += KC) {
            float v[KC];
            #pragma unroll
            for (int k = 0; k < KC; k++) v[k] = w[(size_t) (k0 + k) * D];
            #pragma unroll
            for (int row = 0; row < T; row++)
                #pragma unroll
                for (int k = 0; k < KC; k++) a[row] += g[(row * NJ + j) * GRANK + k0 + k] * v[k];
        }
        if (which)
            #pragma unroll
            for (int row = 0; row < T; row++) sf[row * 256 + q] = a[row];
    }
    __syncthreads();
    float val[T];
    #pragma unroll
    for (int row = 0; row < T; row++) {
        val[row] = 0.f;
        if (tid < n && row < nr) {
            const size_t o = (size_t) row * rsf;
            const float x = qz_gp(cj[j].cur)[o + i], fo = sf[row * 256 + q], ga = a[row];
            val[row] = (x - tanhf(fo) * x) + tanhf(ga) * qz_gp(cj[j].msg)[o + i];
        }
        float v2 = val[row] * val[row];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v2 += __shfl_xor(v2, off, 32);
        if (tid < n && (tid & 31) == 0) gs[row * 8 + (tid >> 5)] = v2;
    }
    __syncthreads();
    if (tid < T * NJ) {
        const int row = tid / NJ, jj = tid % NJ;
        if (row < nr) {
            float s = 0.f;
            for (int u = 0; u < ng; u++) if ((g0 + u) / NG == jj) s += gs[row * 8 + u];
            qz_gp(ss_buf)[tid * nb + blockIdx.x] = s;
        }
    }
    cyc_barrier(vbst, ++kb);
    {
        const int wave = tid >> 6, lane = tid & 63;
        for (int p = wave; p < T * NJ; p += 8)
            if (lane < nb) sbl[p * 64 + lane] = qz_gp(ss_buf)[p * nb + lane];
    }
    __syncthreads();
    if (tid < T * NJ) {
        float tot = 0.f;
        const float * sb = sbl + tid * 64;
        for (int b = 0; b < nb; b++) tot += sb[b];
        rsv[tid] = rsqrtf(tot / D + eps);
    }
    __syncthreads();
    if (tid < n) {
        CK njob * njj = nj + j;
        const float wv = qz_gp(njj->w)[i];
        #pragma unroll
        for (int row = 0; row < T; row++) {
            if (row < nr) {
                const float yv = val[row] * rsv[row * NJ + j] * wv;
                if (njj->y) qz_gp(njj->y)[(size_t) row * rsf + i] = yv;
                quant32(yv, qz_gp(njj->q) + (size_t) row * rs, qz_gp(njj->qs) + (size_t) row * rsf, i);
            }
        }
    }
}

// The attention of the VR rows: phase A, one (kv head, split) unit per block with wave w =
// row w, every wave running the flash body's 3 q heads against the block's one K/V staging
// (ink2_flash.h). Phase B, after the grid barrier: unit (row, kv head), thread (head, dim),
// the splits merged in ascending order on top of the sink, then the gate and the quant.
constexpr size_t VWS = (size_t) HKVL * FL_NS * GQ * ATT_PW;   // ws: [VR][HKVL][FL_NS][GQ][ATT_PW]
static_assert(FL_LDS8 <= VER_LDS, "the attention fits the verify's dynamic LDS");
VR_NI void v_attn_a(CK appk * tab, int napp, int ai, int pos, int v, float * ws, int tid, int nr) {
    asm volatile ("" : "+v"(tid));
    const int kvh = v % HKVL, sp = v / HKVL;      // KV head minor, so the units with keys spread over the grid
    const attn_args * at0 = &((CK appk *) tab + ai)->at;
    fl_attn_rows(at0, (size_t) napp * sizeof(appk), qz_gp(ws), VWS, pos, kvh, sp, tid, nr, cyc_lds);
}
VR_NI void v_attn_b(CK appk * tab, int napp, int ai, int u, const float * ws, int tid, int nr) {
    asm volatile ("" : "+v"(tid));
    const int row = u / HKVL, kvh = u % HKVL;
    if (row >= nr) return;                                       // block-uniform: an inactive row
    if (tid >= GQ * HD) return;
    CK attn_args * ea = &((CK appk *) tab + (size_t) row * napp + ai)->at;
    fl_merge(qz_gp(ws) + (size_t) row * VWS + (size_t) kvh * FL_NS * GQ * ATT_PW,
             qz_gp(ea->gate), qz_gp(ea->oq), qz_gp(ea->os), kvh * GQ, tid >> 7, tid & (HD - 1));
}

// The conv history commit: hist[k][ch] = s[m + k], with s[3 + j] row j's u from uh.
static __global__ void k_vcommit(float * hist, const float * uh, int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NAPP * CHL) return;
    const int slot = i / CHL, ch = i % CHL;
    float * h = hist + (size_t) slot * 3 * CHL + ch;
    const float * u = uh + (size_t) slot * VR * CHL + ch;
    float nv[3];
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        const int q = m + k;
        nv[k] = q < 3 ? h[(size_t) q * CHL] : u[(size_t) (q - 3) * CHL];
    }
    #pragma unroll
    for (int k = 0; k < 3; k++) h[(size_t) k * CHL] = nv[k];
}

// Per application the row's chain work runs on chain group r, blocks 4 r .. 4 r + 3, as
// ink2_cycle's blocks 0 .. CG - 1 in the same order. Then the grid's phases for all VR
// rows: the qkv and out GEMVs, the head posts, the attention, the router, the experts. One
// TP2 flag per exchange, rows at the disjoint inbox region 2 D + (parity * VR + r) D.
// f(std::integral_constant<int, T>) with T = 2, 4 or 8 rows, the fewest that cover the
// nr active rows: the dense GEMVs cost per row, so a short verify runs the short body
template <class F> VR_NI void v_by_rows(int nr, F f) {
    if (nr <= 2) f(std::integral_constant<int, 2>{});
    else if (nr <= 4) f(std::integral_constant<int, 4>{});
    else f(std::integral_constant<int, VR>{});
}
__global__ __launch_bounds__(512) void ink2_verify(unsigned int * vbst, const vargs * Vg, vtok tv, int pos, int head, int nr_) {
    __shared__ int sids[VR * NE_USED];
    __shared__ float swts[VR * NE_USED];
    __shared__ vdist dd;
    const int tid_k = threadIdx.x, nb = gridDim.x;
    const bool b0 = blockIdx.x == 0;
    int kb = 0;
    CK vargs * Vp = (CK vargs *) Vg;
    asm volatile ("" : "+s"(Vp));
    const int rs = Vp->rs, rsf = Vp->rsf;
    const int cr = blockIdx.x >> 2, cw = blockIdx.x & 3;   // the chain group's row, this block's columns
    const int nr = min(max(nr_, 1), VR);                   // the active rows: 2 .. VR, the rest idle
    const bool chain = cr < nr;
    const int tr = chain ? cr : 0;
    unsigned int xs = *Vp->A.xseq;
    // pick, distinct and up run in one phase, with no grid barrier between them
    // the embedding: chain group r's leader block, then the group's sync
    if (chain && cw == 0)
        d_embed(Vp->A.ev, Vp->A.es, tv.t[tr], Vp->A.embd_norm, Vp->A.eps,
                (float *) ((char *) Vp->A.X + (size_t) tr * rs), tid_k, tid_k < 256);
    const cgrp g = {vbst + CYC_NBMAX + cr * CG, (unsigned long long *) (vbst + VBS_SL) + (size_t) cr * CP_SZ, cw};
    unsigned int ce = 0;
    if (chain) cg_sync(g, ++ce);
    const int napp = Vp->A.napp;
    for (int ai = 0; ai < napp; ai++) {
        // the args and the table reloaded per application (no pointers held live)
        CK vargs * Vq = (CK vargs *) Vg;
        asm volatile ("" : "+s"(Vq));
        CK cyc_args & Av = Vq->A;
        CK appk * e0 = (CK appk *) Av.tab + ai;                      // the application's flags
        CK appk * er = (CK appk *) Av.tab + (size_t) tr * napp + ai;  // this block's row
        int tid = tid_k;
        asm volatile ("" : "+v"(tid));
        const int wave = tid >> 6, lane = tid & 63;
        const int pre = e0->pre, msg = e0->msg, nna = e0->nna;
        float * const xr = (float *) ((char *) Av.X + (size_t) tr * rs);
        float * const pr = (float *) ((char *) Av.P + (size_t) tr * rs);

        // ---- P0, the row's chain group: the previous ffn sum and conv, the pass copies,
        // the res mix and norms, the msg mix and norms
        if (chain) {
            if (ai > 0) {
                CK appk * ep = er - 1;
                cyc_xwait2(Av, xs, 2 * D + (size_t) (xs & 1) * VR * D, Av.PART1, rsf, nr, blockIdx.x, nr * CG);
                const int slot = (int) ((ep->hsf - Vq->hist) / (3 * CHL));
                v_cxconv(Av.PART1, INK_XSRC(Av, 2 * D + (size_t) (xs & 1) * VR * D), ep->fconv, ep->hsf, xr, nullptr, cw, tid,
                         tr, rsf, Vq->uh + ((size_t) (slot * VR + tr)) * CHL + H_F, er->tap);
            }
            if (pre) c_cpre(pre, pr, xr, er->pre_out, cw, tid);
            if (er->has_res && !msg) {
                if (er->mres_m == 1) c_cmix<1, true>(&er->mres, er->qres, er->res_out, er->na.j, Av.eps, g, ce, tid);
                else c_cmix<3, true>(&er->mres, er->qres, er->res_out, er->na.j, Av.eps, g, ce, tid);
            } else {
                if (er->has_res) {
                    if (er->mres_m == 1) c_cmix<1, false>(&er->mres, er->qres, er->res_out, nullptr, 0.f, g, ce, tid);
                    else c_cmix<3, false>(&er->mres, er->qres, er->res_out, nullptr, 0.f, g, ce, tid);
                }
                // the msg mix, chain group 2 when !pre. It reads OUT, which a pass copy writes when pre.
                if (msg) c_cmix<4, false>(&er->mmsg, er->qmsg, er->msg_out, nullptr, 0.f, g, ce, tid);
                if (!msg) {
                    if (nna == 1) c_cnorm<1>(er->na.j, Av.eps, g, ce, tid);
                    else c_cnorm<3>(er->na.j, Av.eps, g, ce, tid);
                }
            }
        }
        cyc_barrier(vbst, ++kb);
        if (msg) {
            v_ctrl_down<3>(e0->ca.j, rsf, tid, nr);
            cyc_barrier(vbst, ++kb);
            v_by_rows(nr, [&](auto t) { v_ctrl_up_norm<3, decltype(t)::value>(e0->ca.j, e0->na.j, Av.eps, Av.H, rs, rsf, vbst, kb, tid, nr); });
            cyc_barrier(vbst, ++kb);
        }
        // ---- attention: the qkv GEMVs, the head posts with the cache rows, the attention
        v_by_rows(nr, [&](auto t) { v_ga_gemv<VG_S, decltype(t)::value>(&e0->ga, e0->nga, rs, rsf, tid); });
        cyc_barrier(vbst, ++kb);
        {
            // a block's nper units are of one kind (q / k / v): TP4's 6 q / 2 k / 2 v units allow 2
            constexpr int NPMAX = HQL % 4 == 0 && HKVL % 4 == 0 ? 4 : HQL % 2 == 0 && HKVL % 2 == 0 ? 2 : 1;
            int nper = nr <= 2 ? 1 : (nr <= 4 ? 2 : 4);
            if constexpr (NPMAX < 4) nper = min(nper, NPMAX);
            const int nu = (HQL + 2 * HKVL) / nper, tb_st = 512 / nper;
            for (int v = blockIdx.x; v < nr * nu; v += nb) {
                const int r = v / nu, u = (v % nu) * nper + (tid / tb_st);
                v_head_post(&VROW(Av.tab, r, ai)->hp, pos + r, tv.tau[r], u, tid, r, Vq->uh, rsf, Vq->hist, nper);
            }
        }
        cyc_barrier(vbst, ++kb);
        for (int v = blockIdx.x; v < HKVL * FL_NS; v += nb) v_attn_a((CK appk *) Av.tab, napp, ai, pos, v, Vq->ws, tid, nr);
        cyc_barrier(vbst, ++kb);
        for (int u = blockIdx.x; u < nr * HKVL; u += nb) v_attn_b((CK appk *) Av.tab, napp, ai, u, Vq->ws, tid, nr);
        cyc_barrier(vbst, ++kb);
        ++xs;
        v_by_rows(nr, [&](auto t) {
            v_go_gemv<decltype(t)::value>(&e0->go.j[0], rs, rsf, INK_XDST(Av, 2 * D + (size_t) (xs & 1) * VR * D), tid);
        });
        __threadfence_system();
        cyc_barrier(vbst, ++kb);
        // ---- P1, the row's chain group: the attention sum and conv, the ffn mix and norm
        if (chain) {
            cyc_xwait2(Av, xs, 2 * D + (size_t) (xs & 1) * VR * D, Av.PART0, rsf, nr, blockIdx.x, nr * CG);
            const int slot = (int) ((er->hsa - Vq->hist) / (3 * CHL));
            v_cxconv(Av.PART0, INK_XSRC(Av, 2 * D + (size_t) (xs & 1) * VR * D), er->pconv, er->hsa, xr, er->blk, cw, tid,
                     tr, rsf, Vq->uh + ((size_t) (slot * VR + tr)) * CHL + H_A);
            if (msg) c_cmix<1, false>(&er->mff, er->qff, (float *) ((char *) Av.F + (size_t) tr * rs), nullptr, 0.f, g, ce, tid);
            else c_cmix<1, true>(&er->mff, er->qff, (float *) ((char *) Av.F + (size_t) tr * rs), er->nb.j, Av.eps, g, ce, tid);
        }
        cyc_barrier(vbst, ++kb);
        if (msg) {
            v_ctrl_down<1>(e0->cb.j, rsf, tid, nr);
            cyc_barrier(vbst, ++kb);
            v_by_rows(nr, [&](auto t) { v_ctrl_up_norm<1, decltype(t)::value>(e0->cb.j, e0->nb.j, Av.eps, Av.H, rs, rsf, vbst, kb, tid, nr); });
            cyc_barrier(vbst, ++kb);
        }
        // ---- the experts: the router, the pick, gate | up + SITU, the down sum
        v_by_rows(nr, [&](auto t) {
            for (int ex = blockIdx.x * 8 + wave; ex < NE; ex += 8 * nb) v_router<decltype(t)::value>(Av.FN, e0->gin, e0->pb, Av.H, rs, ex, lane);
        });
        cyc_barrier(vbst, ++kb);
        v_pick_w(Av.H, rsf, Av.route_scale, sids, swts, tid, nr);
        if (b0 && tid < VR * NE_USED && tid / NE_USED < nr) {
            const int r = tid / NE_USED, k = tid % NE_USED;
            ((int *) ((char *) Av.IDS + (size_t) r * rs))[k] = sids[tid];
            ((float *) ((char *) Av.WTS + (size_t) r * rs))[k] = swts[tid];
        }
        v_distinct(sids, dd, tid, nr);
        if (nr <= 2) v_up_dd<2>(e0->guv, e0->gus, sids, dd, Av.aq3, Av.as3, Av.beta, Av.lb, Av.hq, Av.hs, tid, rs, rsf, nr);
        else v_up_dd_r4(e0->guv, e0->gus, sids, dd, Av.aq3, Av.as3, Av.beta, Av.lb, Av.hq, Av.hs, tid, rs, rsf, nr);
        cyc_barrier(vbst, ++kb);
        ++xs;
        v_down_dd(e0->dnv, e0->dns, sids, swts, dd, Av.hq, Av.hs, Av.PART1, INK_XDST(Av, 2 * D + (size_t) (xs & 1) * VR * D),
                  tid, rs, rsf, cyc_lds, nr);
        __threadfence_system();
        cyc_barrier(vbst, ++kb);
    }
    const int wave = tid_k >> 6, lane = tid_k & 63;
    // the last ffn sum and conv, with the window, one block per active row
    if (blockIdx.x < (unsigned) nr) {
        const int r = blockIdx.x;
        CK appk * ep = (CK appk *) Vp->A.tab + (size_t) r * napp + (napp - 1);
        cyc_xwait2(Vp->A, xs, 2 * D + (size_t) (xs & 1) * VR * D, Vp->A.PART1, rsf, nr, r, nr);
        const int slot = (int) ((ep->hsf - Vp->hist) / (3 * CHL));
        v_xconv_b0(Vp->A.PART1, INK_XSRC(Vp->A, 2 * D + (size_t) (xs & 1) * VR * D), ep->fconv, ep->hsf,
                   (float *) ((char *) Vp->A.X + (size_t) r * rs), nullptr, tid_k, r, rsf,
                   Vp->uh + ((size_t) (slot * VR + r)) * CHL + H_F);
        if (blockIdx.x == 0 && tid_k == 0) *Vp->A.xseq = xs;
    }
    if (head) {
        if (blockIdx.x < (unsigned) nr) {    // the head mix and norm of row r, its normed input to the drafter
            const int r = blockIdx.x;
            __syncthreads();
            sptrs S;
            S.n = Vp->A.hmix.n;
            #pragma unroll
            for (int s = 0; s < 6; s++) S.p[s] = qz_gp((const float *) ((const char *) Vp->A.hmix.p[s < S.n ? s : 0] + (size_t) r * rs));
            d_mix_lds<1>(S, qz_gp(Vp->A.hq_out), qz_gp((float *) ((char *) Vp->A.F + (size_t) r * rs)), tid_k, tid_k < 256, cyc_lds);
            njob hn;
            hn.x = qz_gp((float *) ((char *) Vp->A.F + (size_t) r * rs));
            hn.w = qz_gp(Vp->A.hn.w);
            hn.y = Vp->vtap ? Vp->vtap + (size_t) r * 4 * D + 3 * D : nullptr;
            hn.q = qz_gp((unsigned char *) ((char *) Vp->A.aq3 + (size_t) r * rs));
            hn.qs = qz_gp((float *) ((char *) Vp->A.as3 + (size_t) r * rs));
            d_norm_quant(hn, Vp->A.eps, tid_k, tid_k < 256);
        }
        cyc_barrier(vbst, ++kb);
        v_by_rows(nr, [&](auto tc) {   // the T-row head GEMV, c_head_gemv's rows and values, softcap on rows < nvalid
            constexpr int T = decltype(tc)::value;
            const int N = Vp->A.hg.N, NV = Vp->A.nvalid;
            unsigned char * la = (unsigned char *) cyc_lds;
            float * ls = (float *) (la + T * D);
            __syncthreads();
            for (int i = tid_k; i < T * D / 16; i += 512) {
                const int r = i / (D / 16), c = i % (D / 16);
                ((uint4 *) la)[i] = *(const uint4 *) (qz_gp(Vp->A.hg.av) + (size_t) r * rs + c * 16);
            }
            for (int i = tid_k; i < T * (D / 32); i += 512) {
                const int r = i / (D / 32), c = i % (D / 32);
                ls[i] = qz_gp(Vp->A.hg.as)[(size_t) r * rsf + c];
            }
            __syncthreads();
            constexpr int S_ = 4;
            for (int gg = blockIdx.x; gg * (8 * S_) < N; gg += nb) {
                const int r0 = gg * (8 * S_) + wave * S_;
                float acc[T][S_];
                v_core_acc<T, S_>(la, ls, D, D / 32, D / 32,
                                  Vp->A.hg.wv, Vp->A.hg.ws, D, D / 32, N, r0, lane, acc);
                constexpr int NVv = T * S_;
                float (&fv)[NVv] = *(float (*)[NVv]) &acc[0][0];
                const float val = v_tsum<NVv>(fv, lane);
                const int kv = v_tsum_k<NVv>(lane), vt = kv / S_, vs = kv % S_;
                if ((lane & (64 / NVv - 1)) == 0 && r0 + vs < NV)
                    Vp->LOGV[(size_t) vt * NV + r0 + vs] = Vp->A.softcap * tanhf(val / Vp->A.softcap);
            }
        });
        cyc_barrier(vbst, ++kb);
    }
}
#undef VROW
#undef VR_NI
#pragma clang fp contract(fast)
