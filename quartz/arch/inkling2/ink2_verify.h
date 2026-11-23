// inkling2 verify kernel. Runs the VR = 8 rows of a speculative round (the anchor
// and 7 drafts, positions pos .. pos + 7) through the 48 layer applications and the
// head, in one launch per card. Each row runs the decode bodies on its own copy of
// the activations (byte stride RS from row 0), so it is bit-identical to a single
// ink2_cycle step at its position. Row r sees KV rows up to pos + r; its own K and V
// are written before the attention phase.
//
// The host builds the row replicas in ink2_engine::vplan: one table entry per (row,
// application), every activation pointer shifted by the row's stride, so the chain and
// grid bodies run unchanged. Included from ink2_engine.h after ink2_cycle.h.
#pragma once

// fp contract(on) so the bodies round like ink2_cycle.h; contract(fast) again at the end
#pragma clang fp contract(on)

#define VR_NI __device__ __forceinline__
constexpr int VR = 8;
#ifndef VG_S
#define VG_S 2
#endif

struct vtok { int t[VR]; float tau[VR]; };        // tau from the host's tau_at

// The verify arguments: the decode args with tab at the VR-row table, cpart at the VR
// groups, aq3 / hq / ... at row 0, plus the row-replicated buffers.
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
VR_NI float v_s3(const float * hbase, const float * own0, const float * peer0, int i, int k, int rsf) {
    if (k < 3) return hbase[(size_t) k * CHL + i];
    const int q = k - 3;
    return own0[(size_t) q * rsf + i] + __builtin_nontemporal_load(peer0 + (size_t) q * D + i);
}

// c_cxconv with the verify window: no history write, row r's u into its uh row
VR_NI void v_cxconv(const float * own0, const float * peer0, const float * w, float * hbase, float * x, float * blk, int cw, int tid,
                    int r, int rsf, float * uhrow, float * tap = nullptr) {
    own0 = qz_gp(own0); peer0 = qz_gp(peer0); w = qz_gp(w); hbase = qz_gp(hbase);
    x = qz_gp(x); blk = blk ? qz_gp(blk) : nullptr; tap = tap ? qz_gp(tap) : nullptr; uhrow = qz_gp(uhrow);
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            const int i = ccol(cw, q);
            const float u = own0[(size_t) r * rsf + i] + __builtin_nontemporal_load(peer0 + (size_t) r * D + i);
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
VR_NI void v_xconv_b0(const float * own0, const float * peer0, const float * w, const float * hbase, float * x, float * blk,
                      int tid, int r, int rsf, float * uhrow) {
    constexpr int NT = 512, NEL = (D + NT - 1) / NT;
    own0 = qz_gp(own0); peer0 = qz_gp(peer0); w = qz_gp(w); hbase = qz_gp(hbase); x = qz_gp(x); blk = blk ? qz_gp(blk) : nullptr; uhrow = qz_gp(uhrow);
    float u[NEL], h0[NEL], h1[NEL], h2[NEL], xv[NEL];
    float4 wi[NEL];
    #pragma unroll
    for (int k = 0; k < NEL; k++) {
        const int i = min(tid + NT * k, D - 1);          // clamped: unconditional loads, guarded stores
        u[k] = own0[(size_t) r * rsf + i] + __builtin_nontemporal_load(peer0 + (size_t) r * D + i);
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
VR_NI float v_gsum2(float x, float * red, int grp) {
    x = qz_wsum_dpp(x);
    __syncthreads();
    if ((threadIdx.x & 63) == 0) red[grp * 2 + ((threadIdx.x >> 6) & 1)] = x;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < 2; i++) s += red[grp * 2 + i];
    return s;
}
VR_NI void d_head_post_v(const hpost & a, int b, int t, bool act, int dmax, int r, float * uh, int rsf, const float * hb, int grp) {
    __shared__ float red[4 * 2];
    __shared__ float rvs[4][DREL];
    float * rv = rvs[grp];
    if (b < HQL) {
        const int h = b, i = h * HD + t;
        float x = act ? a.qr[i] * a.qclip[i] : 0.f;
        float n2[1] = {v_gsum2(x * x, red, grp)};
        const float nr = sqrtf(n2[0]) + 1e-6f;
        x *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        if (act) a.q[i] = x * a.tau;
        float r_ = t < DREL ? a.rr[h * DREL + t] * a.rclip[h * DREL + t] : 0.f;
        float r2[1] = {v_gsum2(r_ * r_, red, grp)};
        const float rn = sqrtf(r2[0]) + 1e-6f;
        if (t < DREL) rv[t] = r_ * fminf(fmaxf(a.rel_clip / rn, 0.f), 1.f) * a.tau;
        __syncthreads();
        const float sink = a.sk[h];
        if (act)
            for (int d = t; d < dmax; d += 128) {
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
    float n2[1] = {v_gsum2(y * y, red, grp)};
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
// c_head_post with the verify window
VR_NI void v_head_post(CK hpost * e, int pos, float tau, int v, int tid, int r, float * uh, int rsf, const float * hb) {
    asm volatile ("" : "+v"(tid));
    hpost h = *e;
    const int row = e->dk ? pos : pos % PF_RING;          // dk: full layers only
    h.qr = qz_gp(h.qr); h.rr = qz_gp(h.rr); h.sk = qz_gp(h.sk); h.kr = qz_gp(h.kr); h.vr = qz_gp(h.vr);
    h.dk = h.dk ? qz_gp(h.dk) : nullptr; h.dv = h.dv ? qz_gp(h.dv) : nullptr;
    h.qclip = qz_gp(h.qclip); h.rclip = qz_gp(h.rclip); h.relb = qz_gp(h.relb); h.kconv = qz_gp(h.kconv); h.vconv = qz_gp(h.vconv);
    h.hist = qz_gp(h.hist); h.q = qz_gp(h.q); h.tb = qz_gp(h.tb);
    h.kc = qz_gp(h.kc + (size_t) row * KVL); h.vc = qz_gp(h.vc + (size_t) row * KVL);
    h.tau = e->dk ? tau : 1.f;
    d_head_post_v(h, v, tid & 127, true, min(TBW, pos + 1), r, uh, rsf, hb, tid >> 7);
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
VR_NI float v_wdot(const v_wbuf & b, const unsigned char * la, const float * ls, int lane, float (&acc)[VR][2]) {
    static_assert(D / 32 - 64 == 8 && VR == 8, "the 8 tail units, one row per 8 lanes");
    constexpr int NU = D / 32;
    #pragma unroll
    for (int t = 0; t < VR; t++) {
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
    const int tt = lane >> 3, ut = 64 + (lane & 7);
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
    for (int t = 0; t < VR; t++) {
        const float st = __shfl(sat, t * 8 + (lane & 7), 64);
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            const float v = __shfl(q[s], t * 8 + (lane & 7), 64);
            if (lane < 8) acc[t][s] += v * st;
        }
    }
    constexpr int NV = VR * 2;
    float (&fv)[NV] = *(float (*)[NV]) &acc[0][0];
    return v_tsum<NV>(fv, lane);                                            // value (row, gate|up) = lane >> 2
}


// d_router for the VR rows at once: a wave per expert, its weights loaded once. Each row's
// sum follows d_router's order, and v_tsum reduces the row sums on the butterfly's pairs.
VR_NI void v_router(const float * F0, const float * wr, const float * bias, float * H0, int rs, int e, int lane) {
    F0 = qz_gp(F0); wr = qz_gp(wr); bias = qz_gp(bias); H0 = qz_gp(H0);
    float s[VR];
    #pragma unroll
    for (int t = 0; t < VR; t++) s[t] = 0.f;
    #pragma unroll 1
    for (int c0 = 0; c0 < D / 64; c0 += 12) {
        float wv[12];
        #pragma unroll
        for (int k = 0; k < 12; k++) wv[k] = wr[(size_t) e * D + lane + 64 * (c0 + k)];
        #pragma unroll
        for (int t = 0; t < VR; t++) {
            const float * f = (const float *) ((const char *) F0 + (size_t) t * rs);
            float fv[12];
            #pragma unroll
            for (int k = 0; k < 12; k++) fv[k] = f[lane + 64 * (c0 + k)];
            #pragma unroll
            for (int k = 0; k < 12; k++) s[t] += fv[k] * wv[k];
        }
    }
    const float v = v_tsum<VR>(s, lane);                  // row lane >> 3
    if ((lane & 7) == 0) {
        float * ps = (float *) ((char *) H0 + (size_t) (lane >> 3) * rs);
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
VR_NI void v_pick_w(const float * H0, int rsf, float scale, int * sids, float * swts, int tid) {
    static_assert(VR == 8 && NE == 128, "a wave per row, 2 candidates per lane");
    asm volatile ("" : "+v"(tid));
    const int r = tid >> 6, lane = tid & 63;
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
    __syncthreads();
}

// The experts the VR rows picked, each once: dex[0 .. nd-1] ascending, dmk[u] the rows
// that picked dex[u] as bit r. Every block builds it identically, after v_pick_all.
struct vdist { int nd; int dex[VR * NE_USED]; unsigned dmk[VR * NE_USED]; };
VR_NI void v_distinct(const int * sids, vdist & dd, int tid) {
    __shared__ int wc[2];
    unsigned m = 0;
    const int lane = tid & 63, wave = tid >> 6;
    if (tid < NE) {
        #pragma unroll
        for (int r = 0; r < VR; r++)
            #pragma unroll
            for (int k = 0; k < NE_USED; k++) m |= (sids[r * NE_USED + k] == tid ? 1u : 0u) << r;
    }
    const unsigned long long b = __ballot(m != 0);
    if (tid < NE && lane == 0) wc[wave] = __popcll(b);
    __syncthreads();
    if (tid < NE && m) {
        const int idx = __popcll(b & ((1ull << lane) - 1)) + (wave ? wc[0] : 0);
        dd.dex[idx] = tid; dd.dmk[idx] = m;
    }
    if (tid == 0) dd.nd = wc[0] + wc[1];
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
// 32 gate and up row pairs for all VR rows at once, weights read once, keeps the SITU
// values in its LDS tile, then quantizes each picking row's group into its slot.
VR_NI void v_up_dd(const char * wv, const __half * ws, const int * sids, const vdist & dd,
                   const unsigned char * av0, const float * as0, float beta, float lb,
                   unsigned char * hq0, float * hs0, int tid, int rs, int rsf) {
    asm volatile ("" : "+v"(tid));
    av0 = qz_gp(av0); as0 = qz_gp(as0); hq0 = qz_gp(hq0); hs0 = qz_gp(hs0);
    const int wave = tid >> 6, lane = tid & 63, nw = gridDim.x * 8;
    // the VR activation rows (int8 and scales) in dynamic LDS, read by every task
    unsigned char * la = (unsigned char *) cyc_lds;                       // [VR][D]
    float * ls = cyc_lds + VR * D / 4;                                     // [VR][D / 32]
    __syncthreads();
    for (int i = tid; i < VR * D / 16; i += 512) {
        const int r = i / (D / 16), c = i % (D / 16);
        ((uint4 *) la)[i] = *(const uint4 *) (av0 + (size_t) r * rs + c * 16);
    }
    for (int i = tid; i < VR * D / 32; i += 512) {
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
        const size_t base = (size_t) e * FFE;
        const size_t r0 = base + (size_t) g * 64;                           // gate rows r0 .. r0 + 31, up + 32
        const int kv = lane >> 2, vr = kv >> 1, vs = kv & 1;
        const bool wl = (lane & 3) == 0 && vs == 0 && (mk >> vr & 1);
        v_wbuf A, B;
        float acc[VR][2];
        v_wload(A, wv, ws, r0, r0 + 32, lane);
        #pragma unroll 1
        for (int a = 0; a < 32; a += 2) {
            v_wload(B, wv, ws, r0 + a + 1, r0 + a + 33, lane);
            float val = v_wdot(A, la, ls, lane, acc), upv = __shfl_down(val, 4, 64);
            if (wl) lq[vr * 32 + a] = situ_v(val, upv, beta, lb);
            // unconditional on purpose: a conditional reload here miscompiles (NaN)
            v_wload(A, wv, ws, r0 + min(a + 2, 31), r0 + min(a + 2, 31) + 32, lane);
            val = v_wdot(B, la, ls, lane, acc); upv = __shfl_down(val, 4, 64);
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
// the down rows over the distinct experts, c_moe_down_q's quads and per-(row, slot) values.
// Each expert's 4 weight rows load once for the rows that picked it, from its own row list,
// the rows' SITU activations staged in LDS. Then per row the same k-order fold of wk *
// acc. LDS: lacc [VR][MD_Q][NE_USED][4], then the activations (VER_LDS).
constexpr int VER_LDS = (VR * MD_Q * NE_USED * 4) * 4 + VR * NE_USED * FFL + VR * NE_USED * (FFL / 32) * 4;
static_assert(VER_LDS >= CYC_LDS, "verify's dynamic LDS covers the decode's");
VR_NI void v_down_dd(const char * wv, const __half * ws, const int * sids, const float * swts, const vdist & dd,
                     const unsigned char * aq0, const float * as0, float * y0, float * yp0, int tid, int rs, int rsf, float * lacc) {
    static_assert(FFL / 32 <= 32, "one unit per lane");
    asm volatile ("" : "+v"(tid));
    constexpr int NU = FFL / 32, NQ = D / 4, LR = MD_Q * NE_USED * 4;
    aq0 = qz_gp(aq0); as0 = qz_gp(as0); y0 = qz_gp(y0); yp0 = qz_gp(yp0);
    const int wave = tid >> 6, lane = tid & 63, gq = lane >> 4, l = lane & 15;
    const int q0 = blockIdx.x * NQ / gridDim.x, q1 = (blockIdx.x + 1) * NQ / gridDim.x;
    const int np = (dd.nd + 3) / 4;
    // the VR rows' SITU activations, all NE_USED slots, staged in LDS after the partials
    unsigned char * la = (unsigned char *) (lacc + VR * LR);
    float * ls = (float *) (la + VR * NE_USED * FFL);
    for (int i = tid; i < VR * NE_USED * FFL / 16; i += 512) {
        const int r = i / (NE_USED * FFL / 16), o = i % (NE_USED * FFL / 16);
        ((uint4 *) la)[i] = ((const uint4 *) (aq0 + (size_t) r * rs))[o];
    }
    for (int i = tid; i < VR * NE_USED * NU; i += 512) {
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
            const int st = (l >> 1) & 3, ut = 16 + (l & 1);
            const char * w = wv + ((size_t) e * D + r0) * FFL;
            const __half * sw = ws + ((size_t) e * D + r0) * NU;
            qz_v4u w0[4], w1[4], t0, t1;
            float wsc[4], tsc;
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                const qz_gu4 * wr = (const qz_gu4 *) (w + (size_t) s * FFL + l * 32);
                w0[s] = wr[0]; w1[s] = wr[1];
                wsc[s] = __half2float(__ushort_as_half(((const qz_gh *) sw)[(size_t) s * NU + l]));
            }
            {
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
                const uint4 a0 = *(const uint4 *) (aq + l * 32), a1 = *(const uint4 *) (aq + l * 32 + 16);
                const uint4 b0 = *(const uint4 *) (aq + ut * 32), b1 = *(const uint4 *) (aq + ut * 32 + 16);
                const float sa = as[l], sb = as[ut];
                float acc[4];
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    int d = 0;
                    DOT4(d, w0[s].x, a0.x); DOT4(d, w0[s].y, a0.y); DOT4(d, w0[s].z, a0.z); DOT4(d, w0[s].w, a0.w);
                    DOT4(d, w1[s].x, a1.x); DOT4(d, w1[s].y, a1.y); DOT4(d, w1[s].z, a1.z); DOT4(d, w1[s].w, a1.w);
                    acc[s] = (float) d * wsc[s] * sa;
                }
                int dt = 0;
                DOT4(dt, t0.x, b0.x); DOT4(dt, t0.y, b0.y); DOT4(dt, t0.z, b0.z); DOT4(dt, t0.w, b0.w);
                DOT4(dt, t1.x, b1.x); DOT4(dt, t1.y, b1.y); DOT4(dt, t1.z, b1.z); DOT4(dt, t1.w, b1.w);
                const float q = (float) dt * tsc * sb;
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    const float tv = __shfl(q, (lane & ~15) + 2 * s + (l & 1), 64);
                    acc[s] += l < 2 ? tv : 0.f;
                }
                const float val = v_tsum<4, 16>(acc, l);                    // value s in lanes 4 s ..
                if ((l & 3) == 0 && on) lacc[r * LR + (qd * NE_USED + k) * 4 + (l >> 2)] = val;
            }
        }
        __syncthreads();
        for (int i = tid; i < VR * nq * 4; i += 512) {
            const int r = i / (nq * 4), qd = (i >> 2) % nq, s = i & 3, row = (c0 + qd) * 4 + s;
            float tot = 0.f;
            #pragma unroll
            for (int k = 0; k < NE_USED; k++) tot += swts[r * NE_USED + k] * lacc[r * LR + (qd * NE_USED + k) * 4 + s];
            y0[(size_t) r * rsf + row] = tot;
            yp0[(size_t) r * D + row] = tot;
        }
        __syncthreads();
    }
}

// c_gemv2 over the VR rows at once: activation row t at av + t rs, output at y + t rsf,
// yp0 the peer inbox rows at stride D. Weights are read once, and each row's value is
// c_gemv2's, since qz_gemv_core's per-row order does not depend on T.
template <int S> VR_NI void v_gemv(CK gjobs * J, int hb, int rs, int rsf, float * yp0) {
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
    float acc[VR][S];
    v_core_acc<VR, S>(qz_gp(j->av), qz_gp(j->as), rs, rsf, K / 32, j->wv, j->ws, K, K / 32, N, r0, lane, acc);
    constexpr int NV = VR * S;
    float (&fv)[NV] = *(float (*)[NV]) &acc[0][0];
    const float val = v_tsum<NV>(fv, lane);
    const int kv = v_tsum_k<NV>(lane), vt = kv / S, vs = kv % S;
    const bool wl = (lane & (64 / NV - 1)) == 0;              // one writer lane per value
    if (wl && r0 + vs < N) y[(size_t) vt * rsf + r0 + vs] = val;
    if (yp0) {      // the peer inbox: the unit's 8 S x VR values through LDS, 16-byte stores (N % 32 == 0)
        constexpr int RU = 8 * S;
        float * tile = cyc_lds;                                         // [VR][RU]
        const int base = g * 32 + half * RU;
        __syncthreads();
        if (wl) tile[vt * RU + wave * S + vs] = val;
        __syncthreads();
        const int tid = threadIdx.x;
        if (tid < VR * RU / 4) {
            const int t = tid / (RU / 4), c = tid % (RU / 4);
            *(float4 *) (qz_gp(yp0) + (size_t) t * D + base + 4 * c) = *(const float4 *) (tile + t * RU + 4 * c);
        }
    }
}
// c_ctrl_down_all over the VR rows. Each (job, row pair) loads its weights once and runs
// the decode's per-row sums for every row in turn, row t at cur / msg / g + t rsf.
template <int NJ> VR_NI void v_ctrl_down(CK cjob * cj, int rsf, int tid) {
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
            const int r = min(r0 + bpj * (h + 2 * m), GRANK - 1);
            const float * w = wd + (size_t) r * 2 * D;
            #pragma unroll
            for (int k = 0; k < NV; k++) { wa[m][k] = w[t + 256 * k]; wb[m][k] = w[D + t + 256 * k]; }
        }
        #pragma unroll 1
        for (int row = 0; row < VR; row++) {
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
// c_ctrl_up_all over the VR rows. Each weight chunk loads once, the VR rows' sums in
// c_ctrl_up_all's k order. g and the forget sums of all rows sit in dynamic LDS.
template <int NJ> VR_NI void v_ctrl_up(CK cjob * cj, int rsf, int tid) {
    asm volatile ("" : "+v"(tid));
    float * g = cyc_lds;                                   // [VR][NJ][GRANK]
    float * sf = cyc_lds + VR * NJ * GRANK;                // [VR][256]
    __syncthreads();
    for (int i = tid; i < VR * NJ * GRANK; i += 512) {
        const int row = i / (NJ * GRANK), jj = (i / GRANK) % NJ, k = i % GRANK;
        g[i] = qz_gp(cj[jj].g)[(size_t) row * rsf + k];
    }
    __syncthreads();
    constexpr int NR = NJ * D, KC = 32;
    const int r0 = blockIdx.x * NR / gridDim.x, r1 = (blockIdx.x + 1) * NR / gridDim.x;
    for (int c0 = r0; c0 < r1; c0 += 256) {
        const int n = min(256, r1 - c0), which = tid >= n, q = tid - (which ? n : 0);
        const int r = c0 + min(q, n - 1), j = r / D, i = r % D;
        float a[VR];
        #pragma unroll
        for (int row = 0; row < VR; row++) a[row] = 0.f;
        if (tid < 2 * n) {
            const float * w = qz_gp(which ? cj[j].wft : cj[j].wgt) + i;
            #pragma unroll 1
            for (int k0 = 0; k0 < GRANK; k0 += KC) {
                float v[KC];
                #pragma unroll
                for (int k = 0; k < KC; k++) v[k] = w[(size_t) (k0 + k) * D];
                #pragma unroll
                for (int row = 0; row < VR; row++)
                    #pragma unroll
                    for (int k = 0; k < KC; k++) a[row] += g[(row * NJ + j) * GRANK + k0 + k] * v[k];
            }
            if (which)
                #pragma unroll
                for (int row = 0; row < VR; row++) sf[row * 256 + q] = a[row];
        }
        __syncthreads();
        if (tid < n) {
            #pragma unroll 1
            for (int row = 0; row < VR; row++) {
                const size_t o = (size_t) row * rsf;
                const float x = qz_gp(cj[j].cur)[o + i], fo = sf[row * 256 + q], ga = a[row];
                qz_gp(cj[j].out)[o + i] = (x - tanhf(fo) * x) + tanhf(ga) * qz_gp(cj[j].msg)[o + i];
            }
        }
        __syncthreads();
    }
}

// The attention of the VR rows in two phases. Phase A, one wave per unit (KV head,
// split, decode wave w2) with wave w = row w, runs exactly decode wave w2's groups
// (sp * 8 + w2) * 4 + lane / 16 of d_attn, the key loop and the xor joins, and stores the
// wave's state to ws. A wave whose 4 groups have no keys (lo + group > pos) skips, its
// state being (-inf, 0, 0), which every att_join leaves unchanged. Phase B, after a grid
// barrier: unit (row, KV head), thread (head, dim), d_attn's folds in their order, per
// split from its first non-empty wave, then the sink and the splits, the gate and the quant.
constexpr size_t VWS = (size_t) HKVL * ATT_SPL * 8 * GQ * ATT_PW; // ws: [VR][HKVL][ATT_SPL][8][GQ][ATT_PW]
VR_NI void v_attn_a(CK appk * tab, int napp, int ai, int pos, int v, float * ws, int tid) {
    asm volatile ("" : "+v"(tid));
    const int kvh = v % HKVL, sp = v / (HKVL * 8), w2 = (v / HKVL) % 8;   // KV head minor, so the units with keys spread over the grid
    const int wave = tid >> 6, lane = tid & 63, li = lane & 15, row = wave;
    CK attn_args * ea = &((CK appk *) tab + (size_t) row * napp + ai)->at;
    const int full = ea->full, rp = pos + row;
    const int lo = full ? 0 : max(0, rp - (WIN - 1));
    if (lo + (sp * 8 + w2) * 4 > rp) return;                     // wave-uniform: no keys
    const float * aq = qz_gp(ea->q), * atb = qz_gp(ea->tb);
    const __half * kc = qz_gp(ea->kc), * vc = qz_gp(ea->vc);
    const int h0 = kvh * GQ, c0 = kvh * HD + li * 8, g = (sp * 8 + w2) * 4 + (lane >> 4);
    float m[GQ], l[GQ], o[GQ][8];
    {
        const float scale = 1.f / sqrtf((float) HD);
        float q[GQ][8];
        #pragma unroll
        for (int hh = 0; hh < GQ; hh++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) { q[hh][k] = aq[(h0 + hh) * HD + li * 8 + k] * scale; o[hh][k] = 0.f; }
            m[hh] = -INFINITY; l[hh] = 0.f;
        }
        for (int j0 = lo + g; j0 <= rp; j0 += 2 * ATT_G) {
            const int j1 = j0 + ATT_G;
            const bool two = j1 <= rp;
            const int r0 = full ? j0 : j0 % PF_RING, r1 = two ? (full ? j1 : j1 % PF_RING) : r0;
            const uint4 k0u = *(const uint4 *) (kc + (size_t) r0 * KVL + c0);
            const uint4 k1u = *(const uint4 *) (kc + (size_t) r1 * KVL + c0);
            const uint4 v0u = *(const uint4 *) (vc + (size_t) r0 * KVL + c0);
            const uint4 v1u = *(const uint4 *) (vc + (size_t) r1 * KVL + c0);
            const int d0 = min(rp - j0, REL_EXT), d1 = min(rp - j1, REL_EXT);
            float kf[8], vf[8];
            float s0[GQ], s1[GQ];
            att_h8(k0u, kf);
            #pragma unroll
            for (int hh = 0; hh < GQ; hh++) {
                float x = 0.f;
                #pragma unroll
                for (int k = 0; k < 8; k++) x += q[hh][k] * kf[k];
                s0[hh] = att_row16(x) + atb[(h0 + hh) * TBW + d0];
            }
            att_h8(k1u, kf);
            #pragma unroll
            for (int hh = 0; hh < GQ; hh++) {
                float x = 0.f;
                #pragma unroll
                for (int k = 0; k < 8; k++) x += q[hh][k] * kf[k];
                x = att_row16(x);
                s1[hh] = two ? x + atb[(h0 + hh) * TBW + max(d1, 0)] : -INFINITY;
            }
            float p0[GQ], p1[GQ];
            #pragma unroll
            for (int hh = 0; hh < GQ; hh++) {
                const float mn = fmaxf(m[hh], fmaxf(s0[hh], s1[hh])), c = expf(m[hh] - mn);
                p0[hh] = expf(s0[hh] - mn); p1[hh] = two ? expf(s1[hh] - mn) : 0.f;
                l[hh] = l[hh] * c + p0[hh] + p1[hh];
                #pragma unroll
                for (int k = 0; k < 8; k++) o[hh][k] *= c;
                m[hh] = mn;
            }
            att_h8(v0u, vf);
            #pragma unroll
            for (int hh = 0; hh < GQ; hh++)
                #pragma unroll
                for (int k = 0; k < 8; k++) o[hh][k] += p0[hh] * vf[k];
            att_h8(v1u, vf);
            #pragma unroll
            for (int hh = 0; hh < GQ; hh++)
                #pragma unroll
                for (int k = 0; k < 8; k++) o[hh][k] += p1[hh] * vf[k];
        }
    }
    #pragma unroll
    for (int off = 16; off <= 32; off <<= 1) {
        #pragma unroll
        for (int hh = 0; hh < GQ; hh++) {
            float o2[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) o2[k] = __shfl_xor(o[hh][k], off, 64);
            const float m2 = __shfl_xor(m[hh], off, 64), l2 = __shfl_xor(l[hh], off, 64);
            att_join(m[hh], l[hh], o[hh], m2, l2, o2, 8);
        }
    }
    if (lane < 16) {
        float * W = qz_gp(ws) + (size_t) row * VWS + ((((size_t) kvh * ATT_SPL + sp) * 8 + w2) * GQ) * ATT_PW;
        #pragma unroll
        for (int hh = 0; hh < GQ; hh++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) W[hh * ATT_PW + 2 + li * 8 + k] = o[hh][k];
            if (lane == 0) { W[hh * ATT_PW] = m[hh]; W[hh * ATT_PW + 1] = l[hh]; }
        }
    }
}
VR_NI void v_attn_b(CK appk * tab, int napp, int ai, int pos, int u, const float * ws, int tid) {
    asm volatile ("" : "+v"(tid));
    const int row = u / HKVL, kvh = u % HKVL;
    if (tid >= GQ * HD) return;
    CK attn_args * ea = &((CK appk *) tab + (size_t) row * napp + ai)->at;
    const int full = ea->full, rp = pos + row;
    const int lo = full ? 0 : max(0, rp - (WIN - 1));
    const int hh = tid >> 7, d = tid & (HD - 1), h0 = kvh * GQ;
    const float * W = qz_gp(ws) + (size_t) row * VWS + (size_t) kvh * ATT_SPL * 8 * GQ * ATT_PW + hh * ATT_PW;
    float M = 0.f, L = 1.f, O = 0.f;                       // the sink: logit 0, value 0
    #pragma unroll 1
    for (int sp = 0; sp < ATT_SPL; sp++) {
        if (lo + sp * 32 > rp) break;                      // no keys here or in the later splits
        const float * B = W + (size_t) sp * 8 * GQ * ATT_PW;
        const int nw = min(8, (rp - lo - sp * 32) / 4 + 1);   // the non-empty waves, wave 0 never empty
        float sm[8], sl[8], so[8];
        #pragma unroll
        for (int w2 = 0; w2 < 8; w2++) {                    // all in flight, the empty waves' slots unused
            const float * S = B + (size_t) w2 * GQ * ATT_PW;
            sm[w2] = S[0]; sl[w2] = S[1]; so[w2] = S[2 + d];
        }
        float pm = sm[0], pl = sl[0], po = so[0];
        #pragma unroll
        for (int w2 = 1; w2 < 8; w2++)
            if (w2 < nw) att_join(pm, pl, &po, sm[w2], sl[w2], &so[w2], 1);
        att_join(M, L, &O, pm, pl, &po, 1);
    }
    const int i = (h0 + hh) * HD + d;
    const float y = O / L * sigm(qz_gp(ea->gate)[i]);
    quant32(y, qz_gp(ea->oq), qz_gp(ea->os), i);
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
__global__ __launch_bounds__(512) void ink2_verify(unsigned int * vbst, const vargs * Vg, vtok tv, int pos, int head) {
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
    const bool chain = cr < VR;
    const int tr = chain ? cr : 0;
    unsigned int xs = *Vp->A.xseq;
    // pick, distinct and up run in one phase, with no grid barrier between them
    // the embedding: chain group r's leader block, then the group's sync
    if (chain && cw == 0)
        d_embed(Vp->A.ev, Vp->A.es, tv.t[tr], Vp->A.embd_norm, Vp->A.eps,
                (float *) ((char *) Vp->A.X + (size_t) tr * rs), tid_k, tid_k < 256);
    const cgrp g = {vbst + CYC_NBMAX + cr * CG, Vp->A.cpart + (size_t) cr * CP_SZ, cw};
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
                cyc_xwait(Av, xs);
                const int slot = (int) ((ep->hsf - Vq->hist) / (3 * CHL));
                v_cxconv(Av.PART1, Av.xin_mine + 2 * D + (size_t) (xs & 1) * VR * D, ep->fconv, ep->hsf, xr, nullptr, cw, tid,
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
            v_ctrl_down<3>(e0->ca.j, rsf, tid);
            cyc_barrier(vbst, ++kb);
            v_ctrl_up<3>(e0->ca.j, rsf, tid);
            cyc_barrier(vbst, ++kb);
            for (int v = blockIdx.x; v < VR * nna; v += nb) {
                const int r = v / nna, j = v % nna;
                c_norm(&VROW(Av.tab, r, ai)->na.j[j], Av.eps, tid);
            }
            cyc_barrier(vbst, ++kb);
        }
        // ---- attention: the qkv GEMVs, the head posts with the cache rows, the attention
        for (int v = blockIdx.x; v < 4 / VG_S * e0->nga; v += nb) v_gemv<VG_S>(&e0->ga, v, rs, rsf, nullptr);
        cyc_barrier(vbst, ++kb);
        static_assert(HQL % 4 == 0 && HKVL % 4 == 0, "a block's 4 units of one kind (q / k / v)");
        for (int v = blockIdx.x; v < VR * (HQL + 2 * HKVL) / 4; v += nb) {
            const int nu = (HQL + 2 * HKVL) / 4, r = v / nu, u = (v % nu) * 4 + (tid >> 7);   // 4 units, 128 threads each
            v_head_post(&VROW(Av.tab, r, ai)->hp, pos + r, tv.tau[r], u, tid, r, Vq->uh, rsf, Vq->hist);
        }
        cyc_barrier(vbst, ++kb);
        for (int v = blockIdx.x; v < HKVL * ATT_SPL * 8; v += nb) v_attn_a((CK appk *) Av.tab, napp, ai, pos, v, Vq->ws, tid);
        cyc_barrier(vbst, ++kb);
        for (int u = blockIdx.x; u < VR * HKVL; u += nb) v_attn_b((CK appk *) Av.tab, napp, ai, pos, u, Vq->ws, tid);
        cyc_barrier(vbst, ++kb);
        ++xs;
        for (int v = blockIdx.x; v < 4 / VG_S * e0->ngo; v += nb) v_gemv<VG_S>(&e0->go, v, rs, rsf, Av.xin_peer + 2 * D + (size_t) (xs & 1) * VR * D);
        __threadfence_system();
        cyc_barrier(vbst, ++kb);
        // ---- P1, the row's chain group: the attention sum and conv, the ffn mix and norm
        if (chain) {
            cyc_xwait(Av, xs);
            const int slot = (int) ((er->hsa - Vq->hist) / (3 * CHL));
            v_cxconv(Av.PART0, Av.xin_mine + 2 * D + (size_t) (xs & 1) * VR * D, er->pconv, er->hsa, xr, er->blk, cw, tid,
                     tr, rsf, Vq->uh + ((size_t) (slot * VR + tr)) * CHL + H_A);
            if (msg) c_cmix<1, false>(&er->mff, er->qff, (float *) ((char *) Av.F + (size_t) tr * rs), nullptr, 0.f, g, ce, tid);
            else c_cmix<1, true>(&er->mff, er->qff, (float *) ((char *) Av.F + (size_t) tr * rs), er->nb.j, Av.eps, g, ce, tid);
        }
        cyc_barrier(vbst, ++kb);
        if (msg) {
            v_ctrl_down<1>(e0->cb.j, rsf, tid);
            cyc_barrier(vbst, ++kb);
            v_ctrl_up<1>(e0->cb.j, rsf, tid);
            cyc_barrier(vbst, ++kb);
            if (blockIdx.x < VR) c_norm(&VROW(Av.tab, blockIdx.x, ai)->nb.j[0], Av.eps, tid);
            cyc_barrier(vbst, ++kb);
        }
        // ---- the experts: the router, the pick, gate | up + SITU, the down sum
        for (int ex = blockIdx.x * 8 + wave; ex < NE; ex += 8 * nb) v_router(Av.FN, e0->gin, e0->pb, Av.H, rs, ex, lane);
        cyc_barrier(vbst, ++kb);
        v_pick_w(Av.H, rsf, Av.route_scale, sids, swts, tid);
        if (b0 && tid < VR * NE_USED) {
            const int r = tid / NE_USED, k = tid % NE_USED;
            ((int *) ((char *) Av.IDS + (size_t) r * rs))[k] = sids[tid];
            ((float *) ((char *) Av.WTS + (size_t) r * rs))[k] = swts[tid];
        }
        v_distinct(sids, dd, tid);
        v_up_dd(e0->guv, e0->gus, sids, dd, Av.aq3, Av.as3, Av.beta, Av.lb, Av.hq, Av.hs, tid, rs, rsf);
        cyc_barrier(vbst, ++kb);
        ++xs;
        v_down_dd(e0->dnv, e0->dns, sids, swts, dd, Av.hq, Av.hs, Av.PART1, Av.xin_peer + 2 * D + (size_t) (xs & 1) * VR * D,
                  tid, rs, rsf, cyc_lds);
        __threadfence_system();
        cyc_barrier(vbst, ++kb);
    }
    const int wave = tid_k >> 6, lane = tid_k & 63;
    // the last ffn sum and conv, with the window, one block per row
    if (blockIdx.x < VR) {
        const int r = blockIdx.x;
        CK appk * ep = (CK appk *) Vp->A.tab + (size_t) r * napp + (napp - 1);
        cyc_xwait(Vp->A, xs);
        const int slot = (int) ((ep->hsf - Vp->hist) / (3 * CHL));
        v_xconv_b0(Vp->A.PART1, Vp->A.xin_mine + 2 * D + (size_t) (xs & 1) * VR * D, ep->fconv, ep->hsf,
                   (float *) ((char *) Vp->A.X + (size_t) r * rs), nullptr, tid_k, r, rsf,
                   Vp->uh + ((size_t) (slot * VR + r)) * CHL + H_F);
        if (blockIdx.x == 0 && tid_k == 0) *Vp->A.xseq = xs;
    }
    if (head) {
        if (blockIdx.x < VR) {          // the head mix and norm of row r, its normed input to the drafter
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
        {   // the VR-row head GEMV, c_head_gemv's rows and values, softcap on rows < nvalid
            const int N = Vp->A.hg.N, NV = Vp->A.nvalid;
            const int rsf_ = rsf;
            constexpr int S_ = 4;
            for (int gg = blockIdx.x; gg * (8 * S_) < N; gg += nb) {
                const int r0 = gg * (8 * S_) + wave * S_;
                float acc[VR][S_];
                v_core_acc<VR, S_>(Vp->A.hg.av, Vp->A.hg.as, (int) rs, rsf_, Vp->A.hg.K / 32,
                                   Vp->A.hg.wv, Vp->A.hg.ws, Vp->A.hg.K, Vp->A.hg.K / 32, N, r0, lane, acc);
                constexpr int NVv = VR * S_;
                float (&fv)[NVv] = *(float (*)[NVv]) &acc[0][0];
                const float val = v_tsum<NVv>(fv, lane);
                const int kv = v_tsum_k<NVv>(lane), vt = kv / S_, vs = kv % S_;
                if ((lane & (64 / NVv - 1)) == 0 && r0 + vs < NV)
                    Vp->LOGV[(size_t) vt * NV + r0 + vs] = Vp->A.softcap * tanhf(val / Vp->A.softcap);
            }
        }
        cyc_barrier(vbst, ++kb);
    }
}
#undef VROW
#undef VR_NI
#pragma clang fp contract(fast)
