// Batched prefill. Runs prompt rows through the trunk as GEMMs on both cards (TP2)
// and writes the K / V, DeltaNet, conv and MTP state that decode resumes from.
#pragma once

#include <vector>

#include "core/devices.h"
#include "core/link.h"

// Per-card geometry. The PF_ prefix avoids the D, S, G macros engine.h defines.
constexpr int PF_D = 5120, PF_CCH = 5120, PF_HK = 8, PF_HV = 24, PF_S = 128, PF_G = 3072, PF_AQ = 6144,
              PF_AKV = 512, PF_HQ = 12, PF_HKV = 2, PF_HD = 256, PF_FF = 8704;
#include "kernels/pf_gemm.h"

// ------------------------------------------------------- elementwise ----
// Token embeddings. With irow, row m takes image row irow[m] when that is >= 0.
__global__ void pf_embed(const int32_t * __restrict__ tok, const int * __restrict__ ev, const __half * __restrict__ es,
                         float * x, int n, int Mp, const int * __restrict__ irow = nullptr, const float * __restrict__ img = nullptr) {
    const int m = blockIdx.x;
    if (irow && m < n && irow[m] >= 0) {
        const float * r = img + (size_t) irow[m] * PF_D;
        for (int k = threadIdx.x; k < PF_D; k += blockDim.x) x[(size_t) m * PF_D + k] = r[k];
        return;
    }
    const int id = m < n ? tok[m] : 0;
    const signed char * q = (const signed char *) (ev + (size_t) id * (PF_D / 4));
    for (int k = threadIdx.x; k < PF_D; k += blockDim.x)
        x[(size_t) m * PF_D + k] = (float) q[k] * __half2float(es[(size_t) id * (PF_D / 32) + k / 32]);
}

// Causal conv1d (4 taps) and silu over all rows at once. Rows below q0 read the
// window cs, which holds qkv rows at stride PF_AQ.
__device__ __forceinline__ float pf_conv_in(const float * qkv, const float * cs, int k, int c, int q0) {
    return k >= q0 ? qkv[(size_t) k * PF_AQ + c] : cs[(3 + k - q0) * PF_CCH + c];
}
__device__ __forceinline__ float pf_conv_at(const float * qkv, const float * cs, const float * w4, int m, int c, int q0) {
    const float * w = w4 + c * 4;
    const float o = pf_conv_in(qkv, cs, m - 3, c, q0) * w[0] + pf_conv_in(qkv, cs, m - 2, c, q0) * w[1] +
                    pf_conv_in(qkv, cs, m - 1, c, q0) * w[2] + qkv[(size_t) m * PF_AQ + c] * w[3];
    return o / (1.f + expf(-o));
}
// The conv window before row snap, the prefix cache's mid-chunk snapshot.
__global__ void pf_conv_snap(const float * __restrict__ qkv, int snap, const float * __restrict__ cs, float * out, int q0 = 0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= PF_CCH) return;
    out[c] = pf_conv_in(qkv, cs, snap - 3, c, q0); out[PF_CCH + c] = pf_conv_in(qkv, cs, snap - 2, c, q0); out[2 * PF_CCH + c] = pf_conv_in(qkv, cs, snap - 1, c, q0);
}
// The chunk's conv window afterwards: its last three inputs. cs may alias ws.
__global__ void pf_conv_commit(const float * __restrict__ qkv, int n, const float * cs, float * ws, int q0 = 0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= PF_CCH || n <= 0) return;
    const float a = pf_conv_in(qkv, cs, n - 3, c, q0), b = pf_conv_in(qkv, cs, n - 2, c, q0), d = pf_conv_in(qkv, cs, n - 1, c, q0);
    ws[c] = a; ws[PF_CCH + c] = b; ws[2 * PF_CCH + c] = d;
}

// Conv and DeltaNet input prep, one pass per row. q and k go through the conv into
// their l2 norms and are never stored. The v channels are written out for the scan.
__global__ void __launch_bounds__(256) pf_conv_prep(
        const float * __restrict__ qkv, const float * __restrict__ cs, const float * __restrict__ w4, float * conv,
        const float * __restrict__ ba, float eps,
        const float * __restrict__ dt_bias, const float * __restrict__ ssm_a,
        float * qn, float * kn, float * beta, float * dec, int m0 = 0, int q0 = 0) {
    __shared__ float red[4];
    const int m = m0 + blockIdx.x, t = threadIdx.x, wave = t >> 6, lane = t & 63;
    const float epsn = eps * (1.f / 128.f), l2i = 1.f / sqrtf(128.f);
    // the v channels the scan reads, taken through the conv
    for (int c = 2 * PF_HK * PF_S + t; c < PF_CCH; c += 256) conv[(size_t) m * PF_CCH + c] = pf_conv_at(qkv, cs, w4, m, c, q0);
    for (int v = 0; v < 2 * PF_HK; v += 2) {          // one head per pass, both its q and its k
        const int kh = v / 2;
        const int vec = wave >> 1;                      // waves 0, 1 do q, waves 2, 3 do k
        const int e = (wave & 1) * 64 + lane;
        const float x = pf_conv_at(qkv, cs, w4, m, vec * PF_S * PF_HK + kh * PF_S + e, q0);   // q or k: conv here, never stored
        float s = x * x;
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) s += __shfl_down(s, off, 64);
        if (lane == 0) red[wave] = s;
        __syncthreads();
        const float inv = 1.f / sqrtf((red[vec * 2] + red[vec * 2 + 1]) * (1.f / 128.f) + epsn) * l2i;
        if (vec == 0) qn[((size_t) m * PF_HK + kh) * PF_S + e] = x * inv * (1.f / sqrtf(128.f));
        else          kn[((size_t) m * PF_HK + kh) * PF_S + e] = x * inv;
        __syncthreads();
    }
    if (t < PF_HV) {
        const float b = ba[(size_t) m * 2 * PF_HV + t], al = ba[(size_t) m * 2 * PF_HV + PF_HV + t];
        beta[(size_t) m * PF_HV + t] = 1.f / (1.f + expf(-b));
        const float a0 = al + dt_bias[t];
        const float sp = a0 > 20.f ? a0 : logf(1.f + expf(a0));
        dec[(size_t) m * PF_HV + t] = expf(sp * ssm_a[t]);
    }
}

// The DeltaNet recurrence over n rows in the decode scan's order: one wave per head
// and 16 value columns. Row m + 1's inputs load while row m computes.
struct pf_dn_in { float k[32], q[32], v, bt, dc; };
__device__ __forceinline__ void pf_dn_load(pf_dn_in & a, const float * __restrict__ qn, const float * __restrict__ kn,
                                           const float * __restrict__ conv, const float * __restrict__ beta,
                                           const float * __restrict__ dec, int m, int h, int kh, int r, int j) {
    const float * kr = kn + ((size_t) m * PF_HK + kh) * PF_S + r * 32;
    const float * qr = qn + ((size_t) m * PF_HK + kh) * PF_S + r * 32;
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const float4 x = *(const float4 *) (kr + i), y = *(const float4 *) (qr + i);
        a.k[i] = x.x; a.k[i + 1] = x.y; a.k[i + 2] = x.z; a.k[i + 3] = x.w;
        a.q[i] = y.x; a.q[i + 1] = y.y; a.q[i + 2] = y.z; a.q[i + 3] = y.w;
    }
    a.v = conv[(size_t) m * PF_CCH + 2 * PF_S * PF_HK + h * PF_S + j];
    a.bt = beta[(size_t) m * PF_HV + h];
    a.dc = dec[(size_t) m * PF_HV + h];
}
__device__ __forceinline__ void pf_dn_row(float (&s)[32], const pf_dn_in & a, float * o, int m, int h, int r, int jl, int j) {
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = s[i] * a.dc;
    // explicit fmas so the result does not depend on where the compiler contracts
    float p = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) p = __builtin_fmaf(s[i], a.k[i], p);
    const float sk = __shfl(p, jl, 64) + __shfl(p, 16 + jl, 64) + __shfl(p, 32 + jl, 64) + __shfl(p, 48 + jl, 64);
    const float d = (a.v - sk) * a.bt;
    float p2 = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        s[i] = __builtin_fmaf(a.k[i], d, s[i]);
        p2 = __builtin_fmaf(s[i], a.q[i], p2);
    }
    const float oj = __shfl(p2, jl, 64) + __shfl(p2, 16 + jl, 64) + __shfl(p2, 32 + jl, 64) + __shfl(p2, 48 + jl, 64);
    if (r == 0) o[((size_t) m * PF_HV + h) * PF_S + j] = oj;
}
__global__ void __launch_bounds__(64) pf_dn_scan(
        const float * __restrict__ qn, const float * __restrict__ kn, const float * __restrict__ conv,
        const float * __restrict__ beta, const float * __restrict__ dec, float * state, float * o, int n, int m0 = 0,
        int snap = -1, float * __restrict__ snap_state = nullptr) {
    // snap_state copies the state before row snap, the prefix cache's mid-chunk snapshot
    const int h = blockIdx.x / 8, cg = blockIdx.x % 8, lane = threadIdx.x;
    const int r = lane >> 4, jl = lane & 15, j = cg * 16 + jl, kh = h % PF_HK;
    float * st = state + (size_t) h * PF_S * PF_S;
    float s[32];
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = st[(r * 32 + i) * PF_S + j];
    pf_dn_in a, b;
    if (m0 < n) pf_dn_load(a, qn, kn, conv, beta, dec, m0, h, kh, r, j);
    int m = m0;
    float * sst = snap_state ? snap_state + (size_t) h * PF_S * PF_S : nullptr;
    auto snap_at = [&](int row) {
        if (row == snap && sst) {
#pragma unroll
            for (int i = 0; i < 32; i++) sst[(r * 32 + i) * PF_S + j] = s[i];
        }
    };
    for (; m + 1 < n; m += 2) {
        pf_dn_load(b, qn, kn, conv, beta, dec, m + 1, h, kh, r, j);
        snap_at(m);
        pf_dn_row(s, a, o, m, h, r, jl, j);
        if (m + 2 < n) pf_dn_load(a, qn, kn, conv, beta, dec, m + 2, h, kh, r, j);
        snap_at(m + 1);
        pf_dn_row(s, b, o, m + 1, h, r, jl, j);
    }
    if (m < n) { snap_at(m); pf_dn_row(s, a, o, m, h, r, jl, j); }
#pragma unroll
    for (int i = 0; i < 32; i++) st[(r * 32 + i) * PF_S + j] = s[i];
}

// Gated rms norm per head, o * rsqrt(mean(o^2) + eps) * w * silu(z), quantized as the
// ssm_out input. One block of 128 threads per row and head.
__global__ void __launch_bounds__(128) pf_dn_out(
        const float * __restrict__ o, const float * __restrict__ z, const float * __restrict__ nw, float eps,
        int8_t * Ag, float * sag, int Mp, int m0 = 0, int n = 1 << 30) {
    __shared__ float red[2];
    const int m = m0 + blockIdx.x / PF_HV, h = blockIdx.x % PF_HV, t = threadIdx.x;
    if (m >= n) {   // padded rows of the ssm_out input get zeros
        Ag[((size_t) ((h * PF_S + t) / 128) * Mp + m) * 128 + (h * PF_S + t) % 128] = 0;
        if ((t & 31) == 0) sag[((size_t) ((h * PF_S + t) / 128) * Mp + m) * 4 + ((h * PF_S + t) % 128) / 32] = 0.f;
        return;
    }
    const float v = o[((size_t) m * PF_HV + h) * PF_S + t];
    float s = v * v;
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) s += __shfl_down(s, off, 64);
    if ((t & 63) == 0) red[t >> 6] = s;
    __syncthreads();
    const float inv = 1.f / sqrtf((red[0] + red[1]) * (1.f / 128.f) + eps);
    const float zj = z[(size_t) m * PF_G + h * PF_S + t];
    pf_quant_unit((v * inv) * nw[t] * (zj / (1.f + expf(-zj))), h * PF_S + t, m, Mp, Ag, sag, t & 31);
}

// Attention q and k: per-head RMS norm then MRoPE. q goes to q16, and k and v go to
// the arena at p0 + m. rp holds per-row (t, h, w) rope positions; null means p0 + m.
__global__ void __launch_bounds__(256) pf_attn_qk(
        const float * __restrict__ yq, const float * __restrict__ yk, const float * __restrict__ yv,
        const float * __restrict__ qw, const float * __restrict__ kw, float eps, int p0,
        __half * q16, __half * kv_k, __half * kv_v, int hh0, int nhh, int m0 = 0, const int * __restrict__ rp = nullptr) {
    __shared__ __half aq[PF_HD];
    __shared__ float red[9];
    const int m = m0 + blockIdx.x / nhh, hh = hh0 + blockIdx.x % nhh, t = threadIdx.x;
    const bool isq = hh < PF_HQ;
    const int h = isq ? hh : hh - PF_HQ;
    const float * src = isq ? yq + (size_t) m * PF_AQ + h * 2 * PF_HD : yk + (size_t) m * PF_AKV + h * PF_HD;
    const float * w = isq ? qw : kw;
    aq[t] = __float2half(src[t]);
    __syncthreads();
    float v = 0.f;
    { const float x = __half2float(aq[t]); v = x * x; }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
    if ((t & 63) == 0) red[t >> 6] = v;
    __syncthreads();
    if (t == 0) red[8] = 1.f / sqrtf((red[0] + red[1] + red[2] + red[3]) / PF_HD + eps);
    __syncthreads();
    aq[t] = __float2half(__half2float(aq[t]) * red[8] * w[t]);
    __syncthreads();
    const int pos = p0 + m;
    __half * dst = isq ? q16 + ((size_t) m * PF_HQ + h) * PF_HD : kv_k + qz_kv_row(pos, PF_HKV, h);
    if (t < 32) {
        const int rpos = rp ? rp[3 * m + t % 3] : pos;
        const float ang = (float) rpos * powf(1e7f, -(float) t / 32.f);
        const float cs = cosf(ang), sn = sinf(ang);
        const float x0 = __half2float(aq[t]), x1 = __half2float(aq[t + 32]);
        dst[t]      = __float2half(x0 * cs - x1 * sn);
        dst[t + 32] = __float2half(x0 * sn + x1 * cs);
    } else if (t >= 64) {
        dst[t] = aq[t];
    }
    if (!isq) kv_v[qz_kv_row(pos, PF_HKV, h) + t] = __float2half(yv[(size_t) m * PF_AKV + h * PF_HD + t]);
}

// MTP entry inputs: enorm(emb(tok[r])) followed by hnorm of the hidden at the previous
// position. One block per row.
__global__ void __launch_bounds__(256) pf_mtp_in(
        const int32_t * __restrict__ tok, const int * __restrict__ ev, const __half * __restrict__ es,
        const float * __restrict__ hn, int hoff, const float * __restrict__ ew, const float * __restrict__ hw,
        float eps, int R, float * cat, const int * __restrict__ irow = nullptr, const float * __restrict__ img = nullptr) {
    __shared__ float red[2][4];
    const int r = blockIdx.x, t = threadIdx.x;
    float * out = cat + (size_t) r * 2 * PF_D;
    if (r >= R) {
        for (int k = t; k < 2 * PF_D; k += 256) out[k] = 0.f;
        return;
    }
    const signed char * q = (const signed char *) (ev + (size_t) tok[r] * (PF_D / 4));
    const float * ir = irow && irow[r] >= 0 ? img + (size_t) irow[r] * PF_D : nullptr;   // image row, as in pf_embed
    const float * hr = r + hoff >= 0 ? hn + (size_t) (r + hoff) * PF_D : nullptr;
    float se = 0.f, sh = 0.f;
    for (int k = t; k < PF_D; k += 256) {
        const float e = ir ? ir[k] : (float) q[k] * __half2float(es[(size_t) tok[r] * (PF_D / 32) + k / 32]);
        out[k] = e;
        se += e * e;
        if (hr) sh += hr[k] * hr[k];
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) { se += __shfl_down(se, off, 64); sh += __shfl_down(sh, off, 64); }
    if ((t & 63) == 0) { red[0][t >> 6] = se; red[1][t >> 6] = sh; }
    __syncthreads();
    const float re = 1.f / sqrtf((red[0][0] + red[0][1] + red[0][2] + red[0][3]) / PF_D + eps);
    const float rh = 1.f / sqrtf((red[1][0] + red[1][1] + red[1][2] + red[1][3]) / PF_D + eps);
    for (int k = t; k < PF_D; k += 256) {
        out[k] = out[k] * re * ew[k];
        out[PF_D + k] = hr ? hr[k] * rh * hw[k] : 0.f;
    }
}

typedef _Float16 pf_h2 __attribute__((ext_vector_type(2)));


// pf_flash2: causal attention as two register-tiled GEMMs. A block is 64 queries of
// one head, wave w holds q dims w * 32 .. + 31, and keys come in tiles of 16.
#define PF2_KT 16
#define PF2_KS 264                                         // K / V row stride in halves, 256 + 8 so rows kg + G j hit distinct banks
#define PF2_QKP 1                                          // QK: how many steps ahead the K reads are issued (2 or more spill)
#define PF2_NW 8                                           // waves: wave w holds dims w * DW .. w * DW + DW - 1 of the QB = 64 queries
__global__ void __launch_bounds__(PF2_NW * 64, 1) pf_flash2(
        const __half * __restrict__ q16, const __half * __restrict__ kv_k, const __half * __restrict__ kv_v,
        const float * __restrict__ yq, int p0, int n, float * __restrict__ out, int mb = 0) {
    constexpr int NW = PF2_NW, NT = NW * 64, DW = 256 / NW, QB = 8 * NW, QL = 2, KG = 2;
    constexpr int KL = PF2_KT / KG;                        // keys per lane in QK, kg + KG j
    static_assert((QB / QL) * KG == 64, "QK lane tile");
    static_assert((QB / 4) * (DW / 8) == 64, "PV lane tile");
    __shared__ __attribute__((aligned(16))) __half Ks[PF2_KT * PF2_KS], Vs[PF2_KT * PF2_KS];
    // Per wave, partial scores [query][key], rows unpadded. Query q's chunk c sits at
    // c ^ ((q >> 1) & 3), which keeps QK b128 writes and softmax b64 reads conflict-free.
    __shared__ __attribute__((aligned(16))) float Sp[NW][QB][PF2_KT];
    auto spc = [](int q, int c) { return c ^ ((q >> 1) & 3); };
    // QK lane kg's keys are kg * KL + j. kcol swizzles K rows onto distinct banks.
    auto qkey = [](int kg, int j) { return kg * KL + j; };
    auto kcol = [](int row, int col) { return col ^ ((row / KL) * 8); };
    __shared__ __attribute__((aligned(16))) float Pt[PF2_KT][QB + 4];   // probabilities [key][query]
    __shared__ __attribute__((aligned(16))) float cr[QB], ls[QB];       // per query: this tile's rescale and the final l
    const int h = blockIdx.y, kh = h / 6, t = threadIdx.x, w = t >> 6, lane = t & 63;
    const int m0 = mb + (gridDim.x - 1 - blockIdx.x) * QB;   // longest causal blocks dispatched first
    const int kmax = p0 + min(m0 + QB - 1, n - 1);
    // QK lane: queries qg + r * QB / 2, keys qkey(kg, j)
    const int qg = lane / KG, kg = lane % KG;
    pf_h2 qv[QL][DW / 2];
#pragma unroll
    for (int r = 0; r < QL; r++) {
        const int m = m0 + qg + (QB / QL) * r;
        const pf_h2 * src = (const pf_h2 *) (q16 + ((size_t) (m < n ? m : 0) * PF_HQ + h) * PF_HD + w * DW);
#pragma unroll
        for (int i = 0; i < DW / 2; i++) qv[r][i] = m < n ? src[i] : pf_h2{ 0, 0 };
    }
    // softmax lane: query sq (8 lanes per query)
    const int sq = t >> 3, sqpos = p0 + m0 + sq;
    float mx = -INFINITY, l = 0.f;
    // PV lane: queries pq .. pq + 3, dims pd .. pd + 7
    const int pq = (lane / (DW / 8)) * 4, pd = w * DW + (lane % (DW / 8)) * 8;
    float o[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int d = 0; d < 8; d++) o[i][d] = 0.f;
    // tile loads: each thread takes LPT items of (key, 8 halves), for K and for V
    constexpr int LPT = PF2_KT * 32 / NT;
    uint4 nk[LPT];
    // V is loaded as key pairs, so the pair dwords are built in-lane with no shuffle.
    static_assert(NT == 8 * 64, "V load: 8 key pairs x 64 lanes");
    uint2 ve = make_uint2(0, 0), vo = make_uint2(0, 0);
    auto load = [&](int kt) {
#pragma unroll
        for (int u = 0; u < LPT; u++) {
            const int idx = t + u * NT, kp = kt + (idx >> 5), lc = (idx & 31) * 8;
            nk[u] = make_uint4(0, 0, 0, 0);
            if (kp <= kmax) nk[u] = *(const uint4 *) (kv_k + qz_kv_row(kp, PF_HKV, kh) + lc);
        }
        {   // keys kt .. kt + 15 sit in one 64-position page of qz_kv_row
            const int ke = kt + 2 * (t >> 6), dd = (t & 63) * 4;
            ve = vo = make_uint2(0, 0);
            if (ke <= kmax)     ve = *(const uint2 *) (kv_v + qz_kv_row(ke, PF_HKV, kh) + dd);
            if (ke + 1 <= kmax) vo = *(const uint2 *) (kv_v + qz_kv_row(ke + 1, PF_HKV, kh) + dd);
        }
    };
    auto storeK = [&]() {
#pragma unroll
        for (int u = 0; u < LPT; u++) {
            const int idx = t + u * NT, lk = idx >> 5, lc = (idx & 31) * 8;
            *(uint4 *) &Ks[lk * PF2_KS + kcol(lk, lc)] = nk[u];
        }
    };
    load(0);
    for (int kt = 0; kt <= kmax; kt += PF2_KT) {
        __syncthreads();                                   // the last tile's QK and PV are done with Ks, Vs and Pt
        {   // V as key pairs: dword (key 2j, key 2j + 1) of one dim, at pair row j
            auto lo = [](uint32_t a, uint32_t b) { return __builtin_amdgcn_perm(b, a, 0x05040100u); };
            auto hi = [](uint32_t a, uint32_t b) { return __builtin_amdgcn_perm(b, a, 0x07060302u); };
            *(uint4 *) ((uint32_t *) Vs + (t >> 6) * PF2_KS + (t & 63) * 4) =
                make_uint4(lo(ve.x, vo.x), hi(ve.x, vo.x), lo(ve.y, vo.y), hi(ve.y, vo.y));
        }
        storeK();
        __syncthreads();
        if (kt + PF2_KT <= kmax) load(kt + PF2_KT);        // the next tile's global loads, in flight behind QK
        {   // QK partials over dims w * DW .. + DW - 1
            float s[QL][KL];
#pragma unroll
            for (int r = 0; r < QL; r++)
#pragma unroll
                for (int j = 0; j < KL; j++) s[r][j] = 0.f;
            {   // software-pipelined, so the read of step it + PF2_QKP is in flight
                constexpr int NS = DW / 8 * KL;
                auto kaddr = [&](int it) { const int row = qkey(kg, it % KL); return (const uint4 *) &Ks[row * PF2_KS + kcol(row, w * DW + (it / KL) * 8)]; };
                uint4 kb[PF2_QKP + 1];
#pragma unroll
                for (int it = 0; it < PF2_QKP; it++) kb[it] = *kaddr(it);
#pragma unroll
                for (int it = 0; it < NS; it++) {
                    if (it + PF2_QKP < NS) kb[(it + PF2_QKP) % (PF2_QKP + 1)] = *kaddr(it + PF2_QKP);
                    const uint4 k4 = kb[it % (PF2_QKP + 1)];
                    const int c = it / KL, j = it % KL;
                    const uint32_t kk[4] = { k4.x, k4.y, k4.z, k4.w };
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const pf_h2 k2 = __builtin_bit_cast(pf_h2, kk[u]);
#pragma unroll
                        for (int r = 0; r < QL; r++) s[r][j] = __builtin_amdgcn_fdot2(qv[r][c * 4 + u], k2, s[r][j], false);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < QL; r++)
#pragma unroll
                for (int j = 0; j < KL; j += 4) *(float4 *) &Sp[w][qg + (QB / QL) * r][4 * spc(qg + (QB / QL) * r, (kg * KL + j) >> 2)] = make_float4(s[r][j], s[r][j + 1], s[r][j + 2], s[r][j + 3]);
        }
        __syncthreads();
        {   // softmax: query sq, keys sk and sk + 1
            // indices rebuilt from the lane id: kept live they spill
            int lid = __lane_id();
            asm volatile("" : "+v"(lid));
            const int sk = (lid & 7) * 2, sq = __builtin_amdgcn_readfirstlane(w) * 8 + (lid >> 3);
            float x[2], a2[2] = { 0.f, 0.f };
#pragma unroll
            for (int v = 0; v < NW; v++) {                // the waves' partials, in wave order
                const float2 f = *(const float2 *) &Sp[v][sq][4 * spc(sq, sk >> 2) + (sk & 3)];
                a2[0] += f.x; a2[1] += f.y;
            }
#pragma unroll
            for (int u = 0; u < 2; u++)                   // softmax in base 2: scores in log2 units, then v_exp_f32
                x[u] = kt + sk + u > sqpos ? -INFINITY : a2[u] * (1.4426950408889634f / 16.f);
            float tm = fmaxf(x[0], x[1]);
            // max over the query's 8 lanes on DPP; the three steps can run in any order
            tm = fmaxf(tm, qz_dpp<0xb1, false>(tm)); tm = fmaxf(tm, qz_dpp<0x4e, false>(tm)); tm = fmaxf(tm, qz_dpp<0x141, false>(tm));
            const float nm = fmaxf(mx, tm);               // finite: key 0 is in tile 0 and visible to every query
            const float corr = mx == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(mx - nm);
            float ps = 0.f, pr[2];
#pragma unroll
            for (int u = 0; u < 2; u++) {
                float p = x[u] == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(x[u] - nm);
                p = (float) (_Float16) p;                 // P x V takes p rounded to f16, so l sums the rounded values
                pr[u] = p;
                ps += p;
            }
            // the lane's key pair (sk, sk + 1) as one f16 pair at [pair][query] in Pt
            ((pf_h2 *) &Pt[0][0])[(sk / 2) * (QB + 4) + sq] = pf_h2{ (_Float16) pr[0], (_Float16) pr[1] };
            // the xor 1 / 2 / 4 butterfly on DPP gives the same sums. After xor 2 the
            // quad agrees, so row_half_mirror (lane 7 - i) supplies lane i ^ 4.
            ps += qz_dpp<0xb1, false>(ps); ps += qz_dpp<0x4e, false>(ps); ps += qz_dpp<0x141, false>(ps);
            l = l * corr + ps;
            mx = nm;
            if ((t & 7) == 0) cr[sq] = corr;
        }
        __syncthreads();
        {   // PV
            const float4 c4 = *(const float4 *) &cr[pq];
            const float cc[4] = { c4.x, c4.y, c4.z, c4.w };
            // lazy rescale: once the running max settles, corr is exactly 1, so a wave in
            // which no lane's queries moved skips the multiplies and keeps the same bits.
            if (__any(cc[0] != 1.f || cc[1] != 1.f || cc[2] != 1.f || cc[3] != 1.f))
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int d = 0; d < 8; d++) o[i][d] *= cc[i];
#pragma unroll 2
            for (int j = 0; j < PF2_KT / 2; j++) {
                const uint4 pp = *(const uint4 *) ((const pf_h2 *) &Pt[0][0] + j * (QB + 4) + pq);   // queries pq .. pq + 3, keys 2j, 2j + 1
                const pf_h2 p2[4] = { __builtin_bit_cast(pf_h2, pp.x), __builtin_bit_cast(pf_h2, pp.y),
                                      __builtin_bit_cast(pf_h2, pp.z), __builtin_bit_cast(pf_h2, pp.w) };
                const uint4 va = *(const uint4 *) ((const uint32_t *) Vs + j * PF2_KS + pd);
                const uint4 vb = *(const uint4 *) ((const uint32_t *) Vs + j * PF2_KS + pd + 4);
                const uint32_t vv[8] = { va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w };
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int d = 0; d < 8; d++) o[i][d] = __builtin_amdgcn_fdot2(p2[i], __builtin_bit_cast(pf_h2, vv[d]), o[i][d], false);
            }
        }
    }
    if ((t & 7) == 0) ls[sq] = l;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + pq + i;
        if (m >= n) continue;
        const float il = 1.f / ls[pq + i];
        const float * g = yq + (size_t) m * PF_AQ + h * 2 * PF_HD + PF_HD + pd;
        float * dst = out + (size_t) m * PF_HQ * PF_HD + h * PF_HD + pd;
#pragma unroll
        for (int d = 0; d < 8; d++) dst[d] = o[i][d] * il * (1.f / (1.f + expf(-g[d])));
    }
}

// DSpine target features: the residual after these trunk layers, 0-based, captured on
// card 0 right after the next layer's attn norm.
constexpr int PF_FEAT_LAYERS[5] = { 1, 16, 31, 46, 61 };

// ------------------------------------------------------------ driver ----
// A prompt's images: per position the image row (-1 for a token) and rope (t, h, w)
struct qz_mm_view {
    const int * irow = nullptr;
    const int * rpos = nullptr;
    int n_pos = 0, shift = 0;
    const float * rows[2] = {};
    int  img(int p) const { return p < n_pos ? irow[p] : -1; }
    void rope(int p, int * o) const {
        if (p < n_pos) { o[0] = rpos[3 * p]; o[1] = rpos[3 * p + 1]; o[2] = rpos[3 * p + 2]; }
        else o[0] = o[1] = o[2] = p + shift;
    }
};

struct qz_prefill {
    int Mmax = 0, Mp = 0;
    std::vector<int> mm_irow, mm_rpos;   // host staging of a chunk's qz_mm_view rows
    // DSpine feature capture, [5][Mmax][PF_D] on card 0; null means off
    float * feat = nullptr;
    struct card {
        float * x = nullptr, * part = nullptr, * y = nullptr, * conv = nullptr,
              * z = nullptr, * ba = nullptr, * qn = nullptr, * kn = nullptr, * beta = nullptr, * dec = nullptr,
              * o = nullptr, * fg = nullptr, * fu = nullptr, * yk = nullptr, * yv = nullptr,
              * hn = nullptr, * cat = nullptr,
              * edge = nullptr;   // [recurrent layer][3][PF_CCH]: half 0's last three qkv rows, half 1's conv window
        int8_t * aq = nullptr, * aq2 = nullptr; float * as = nullptr, * as2 = nullptr;   // grouped int8 GEMM input and scales, two sets for ffn in and out
        void * pl[2] = {}, * pr[2] = {};                 // partials, bf16 or f32: this card's own, the peer's written by its GEMM
        __half * q16 = nullptr;
        int32_t * tok = nullptr;
        int * irow = nullptr, * rpos = nullptr;          // a chunk's image rows and rope positions (qz_mm_view)
        size_t scr = 0;                                  // bytes carved from the engine's scratch; 0 means own buffers
        hipEvent_t ev = nullptr, ev_aux = nullptr, evp[4] = {};
        hipStream_t aux = nullptr;
    } c[2];

    // the per-card buffers of init(max_rows): (pointer, bytes) for M rows.
    static std::vector<std::pair<void **, size_t>> bufs(card & k, size_t M) {
        const size_t xb = 2, pb = M * PF_D * xb;         // bytes per bf16 cross-card partial
        std::vector<std::pair<void **, size_t>> v = { { (void **) &k.x, M * PF_D * 4 }, { (void **) &k.part, M * PF_D * 4 }, { (void **) &k.y, M * PF_AQ * 4 },
                 { (void **) &k.conv, M * PF_CCH * 4 }, { (void **) &k.z, M * PF_G * 4 }, { (void **) &k.ba, M * 2 * PF_HV * 4 },
                 { (void **) &k.qn, M * PF_HK * PF_S * 4 }, { (void **) &k.kn, M * PF_HK * PF_S * 4 },
                 { (void **) &k.beta, M * PF_HV * 4 }, { (void **) &k.dec, M * PF_HV * 4 }, { (void **) &k.o, M * PF_HV * PF_S * 4 },
                 { (void **) &k.yk, M * PF_AKV * 4 }, { (void **) &k.yv, M * PF_AKV * 4 },
                 { (void **) &k.aq, M * 2 * PF_D }, { (void **) &k.as, M * 2 * PF_D / 32 * 4 },
                 { (void **) &k.aq2, M * 2 * PF_D }, { (void **) &k.as2, M * 2 * PF_D / 32 * 4 },
                 { (void **) &k.hn, M * PF_D * 4 }, { (void **) &k.cat, M * 2 * PF_D * 4 },
                 { (void **) &k.q16, M * PF_HQ * PF_HD * 2 }, { (void **) &k.tok, M * 4 }, { (void **) &k.edge, (size_t) 64 * 3 * PF_CCH * 4 },
                 { &k.pl[0], pb }, { &k.pr[0], pb }, { &k.pl[1], pb }, { &k.pr[1], pb },
                 { (void **) &k.irow, M * 4 }, { (void **) &k.rpos, M * 3 * 4 } };
        return v;
    }
    static size_t rows(int max_rows) { return (size_t) ((max_rows + PF_MT - 1) / PF_MT * PF_MT + PF_MT); }
    // device bytes per card that init(max_rows) needs, buffers 256-byte aligned
    static size_t bytes(int max_rows) {
        card k;
        std::vector<size_t> v;
        for (const auto & b : bufs(k, rows(max_rows))) v.push_back(b.second);
        return qz_scr_size(v);
    }
    int feat_init() {
        if (feat) return 0;
        qz_mem_scope mem("prefill feat");
        CHECK(hipSetDevice(qz_dev_of(0)));
        CHECK(qz_dmalloc(&feat, (size_t) 5 * Mmax * PF_D * 4));
        CHECK(hipMemset(feat, 0, (size_t) 5 * Mmax * PF_D * 4));
        CHECK(hipSetDevice(qz_dev_of(0)));
        return 0;
    }
    int init(qz_engine & e, int max_rows) {
        Mmax = (int) rows(max_rows);   // one extra tile: the MTP pass has n + 1 rows
        qz_mem_scope mem("prefill");
        for (int i = 0; i < 2; i++) {
            card & k = c[i];
            CHECK(hipSetDevice(qz_dev_of(i)));
            const auto b = bufs(k, Mmax);
            // carve from the engine's scratch when it has room, else own buffers
            if (e.C[i].scr_carve(b)) k.scr = bytes(max_rows);
            else for (const auto & x : b) CHECK(qz_dmalloc(x.first, x.second));
            CHECK(hipEventCreateWithFlags(&k.ev, hipEventDisableTiming));
            CHECK(hipEventCreateWithFlags(&k.ev_aux, hipEventDisableTiming));
            for (int j = 0; j < 4; j++) CHECK(hipEventCreateWithFlags(&k.evp[j], hipEventDisableTiming));
            CHECK(hipStreamCreateWithFlags(&k.aux, hipStreamNonBlocking));
            CHECK(hipEventRecord(k.ev_aux, k.aux));
        }
        CHECK(hipSetDevice(qz_dev_of(0)));
        return 0;
    }

    // Two token halves on two streams per card, so one half's small kernels run beside
    // the other half's GEMMs.
    struct half { int r0 = 0, r1 = 0, ty0 = 0, nty = 0, rp = 0; };
    half hv[2];
    int nh = 1;
    int hset[2] = {0, 0}, hpn[2] = {0, 0};          // per half: the partial set in use (hset) and a pending sum flag (hpn)
    hipEvent_t evx[2][2] = {}, evh[2][4] = {}, evc[2][2] = {};
                                                    // [card][half]: a partial is written. [card][kind]: half 0's conv, scan or kv is done, or the join.
                                                    // evc[card][half]: the copy of the peer's partial into that card has landed (host mode)

    hipStream_t hs(qz_engine & e, int i, int h) { return h == 0 ? e.C[i].st : c[i].aux; }

    // C = A x W over the half's token tiles. A is the grouped int8 input, aq by default.
    void gemm(qz_engine & e, int i, int h, const char * W, const __half * sw, int N, int K, float * C, int ldc,
              const int8_t * A = nullptr, const float * As = nullptr) {
        const half & H = hv[h];
        hipLaunchKernelGGL(pf_gemm<0>, dim3((N + 63) / 64, H.nty), dim3(256), 0, hs(e, i, h), A ? A : c[i].aq, As ? As : c[i].as, Mp,
                           W, sw, K, K, C, ldc, N, H.ty0, 0);
    }
    // Row-parallel projection. Each card writes its bf16 partial locally and to the peer.
    // The next norm() applies x += p0 + p1. Two partial sets alternate.
    // Reaching the peer: under p2p the GEMM's C2 epilogue stores straight into its VRAM.
    // In host mode it cannot, so the GEMM writes only the local pl and one staged copy per
    // card moves the tile-aligned rows into the peer's pr, which its owner then reads
    // without leaving the card. Putting pr in pinned host memory instead (the epilogue stores
    // into it, the peer's addnorm reads it over PCIe) needs no copy but was slower. Neither
    // changes the arithmetic, only where the partial is written.
    void rowpar(qz_engine & e, int h, const char * const * W, const __half * const * sw, int K, bool ffn) {
        const half & H = hv[h];
        hset[h] ^= 1;
        const int s_ = hset[h];
        const bool xc = qz_link == QZ_LINK_HOST;   // partial by copy, no peer store
        for (int i = 0; i < 2; i++) {
            CHECK(hipSetDevice(qz_dev_of(i)));
            hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_D / 64, H.nty), dim3(256), 0, hs(e, i, h), ffn ? c[i].aq2 : c[i].aq, ffn ? c[i].as2 : c[i].as, Mp, W[i], sw[i],
                               K, K, (float *) c[i].pl[s_], PF_D, PF_D, H.ty0, 1,
                               xc ? (unsigned short *) nullptr : (unsigned short *) c[1 - i].pr[s_]);
            CHECK(hipEventRecord(evx[i][h], hs(e, i, h)));
        }
        if (xc) {
            // the rows this half's GEMM wrote, straight into the peer's pr: the source is
            // this card's stream, the destination the peer's, and evc says it has landed
            const size_t off = (size_t) H.r0 * PF_D * 2, nb = (size_t) (H.rp - H.r0) * PF_D * 2;
            for (int i = 0; i < 2; i++) {
                CHECK(qz_xcopy((char *) c[1 - i].pr[s_] + off, 1 - i, hs(e, 1 - i, h),
                               (const char *) c[i].pl[s_] + off, i, hs(e, i, h), nb));
                CHECK(hipSetDevice(qz_dev_of(1 - i)));
                CHECK(hipEventRecord(evc[1 - i][h], hs(e, 1 - i, h)));
            }
        }
        hpn[h] = 1;
    }
    // RMS norm of the half's rows of x, quantized into Ag and optionally as f32 into
    // outf. A pending cross-card sum is added first.
    void norm(qz_engine & e, int i, int h, const float * w, int8_t * Ag, float * outf, int rows_end = -1, hipStream_t st_ = nullptr) {
        const half & H = hv[h];
        card & k = c[i];
        hipStream_t st = st_ ? st_ : hs(e, i, h);
        const int r1 = rows_end >= 0 ? rows_end : H.rp;
        if (hpn[h]) {
            CHECK(hipStreamWaitEvent(st, evx[1 - i][h], 0));
            if (qz_link == QZ_LINK_HOST)   // the peer's partial arrived by copy
                CHECK(hipStreamWaitEvent(st, evc[i][h], 0));
            const int s_ = hset[h];
            const void * p0 = i == 0 ? k.pl[s_] : k.pr[s_], * p1 = i == 0 ? k.pr[s_] : k.pl[s_];
            hipLaunchKernelGGL(pf_addnorm_quant<PF_D>, dim3(r1 - H.r0), dim3(256), 0, st, k.x, (const unsigned short *) p0,
                               (const unsigned short *) p1, H.r1, w, e.eps, Ag, k.as, Mp, outf, H.r0);
        } else {
            hipLaunchKernelGGL(pf_norm_quant, dim3(r1 - H.r0), dim3(256), 0, st, k.x, PF_D, w, e.eps, PF_D, Ag, k.as, Mp, outf, H.r0);
        }
    }
    void ev(qz_engine & e, int i, int h, int kind) { CHECK(hipEventRecord(evh[i][kind], hs(e, i, h))); }
    void wait(qz_engine & e, int i, int h, int kind) { CHECK(hipStreamWaitEvent(hs(e, i, h), evh[i][kind], 0)); }

    // Stage the image rows and rope positions of positions first .. first + R - 1 into
    // the given card's irow and rpos, on its main stream; card -1 means both cards.
    void mm_stage(qz_engine & e, const qz_mm_view * mm, int first, int R, int card = -1) {
        mm_irow.resize(R); mm_rpos.resize((size_t) R * 3);
        for (int r = 0; r < R; r++) { mm_irow[r] = mm->img(first + r); mm->rope(first + r, &mm_rpos[(size_t) r * 3]); }
        for (int i = card < 0 ? 0 : card; i < (card < 0 ? 2 : card + 1); i++) {
            CHECK(hipSetDevice(qz_dev_of(i)));
            CHECK(hipMemcpyAsync(c[i].irow, mm_irow.data(), (size_t) R * 4, hipMemcpyHostToDevice, e.C[i].st));
            CHECK(hipMemcpyAsync(c[i].rpos, mm_rpos.data(), (size_t) R * 12, hipMemcpyHostToDevice, e.C[i].st));
        }
    }

    // MTP K / V entries for p0 + 1 .. p0 + n, plus entry 0 when p0 == 0. Runs after both
    // halves have joined.
    void mtp_pass(qz_engine & e, const int32_t * toks, int p0, int n, int32_t next, const qz_mm_view * mm = nullptr) {
        const int first = p0 == 0 ? 0 : p0 + 1;               // first entry position
        const int R = p0 + n + 1 - first;                     // entries first .. p0 + n
        const int Mr = (R + PF_MT - 1) / PF_MT * PF_MT;
        std::vector<int32_t> mt(R);
        for (int r = 0; r < R; r++) {
            const int pos = first + r;
            mt[r] = pos - p0 < n ? toks[pos - p0] : next;
        }
        const int hoff = first - 1 - p0;                      // entry r's hidden lives in hn row r + hoff
        const int save = Mp;
        if (mm) mm_stage(e, mm, first, R);
        for (int i = 0; i < 2; i++) {
            qz_card & K = e.C[i];
            card & k = c[i];
            CHECK(hipSetDevice(qz_dev_of(i)));
            // the MTP pairing input: the output-normed trunk hidden of every row
            Mp = save;
            for (int h = 0; h < nh; h++) norm(e, i, h, K.A.out_norm, (int8_t *) nullptr, k.hn, hv[h].r1, K.st);
            CHECK(hipMemcpyAsync(k.tok, mt.data(), (size_t) R * 4, hipMemcpyHostToDevice, K.st));
            hipLaunchKernelGGL(pf_mtp_in, dim3(Mr), dim3(256), 0, K.st, k.tok, K.A.emb_vals, K.A.emb_scal, k.hn, hoff,
                               K.mw.enorm, K.mw.hnorm, e.eps, R, k.cat, mm ? (const int *) k.irow : nullptr, mm ? mm->rows[i] : nullptr);
            Mp = Mr;
            hipLaunchKernelGGL(pf_quant, dim3(Mr), dim3(256), 0, K.st, k.cat, 2 * PF_D, 2 * PF_D, k.aq, k.as, Mr, 0);
            hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_D / 64, Mr / 128), dim3(256), 0, K.st, k.aq, k.as, Mr, K.mw.eh_vals, K.mw.eh_scal,
                               (2 * PF_D), 2 * PF_D, k.part, PF_D, PF_D, 0, 0);
            hipLaunchKernelGGL(pf_norm_quant, dim3(Mr), dim3(256), 0, K.st, k.part, PF_D, K.mw.L.attn_norm, e.eps, PF_D,
                               k.aq, k.as, Mr, (float *) nullptr, 0);
            hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_AKV / 64, Mr / 128), dim3(256), 0, K.st, k.aq, k.as, Mr, K.mw.L.k_vals, K.mw.L.k_scal,
                               PF_D, PF_D, k.yk, PF_AKV, PF_AKV, 0, 0);
            hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_AKV / 64, Mr / 128), dim3(256), 0, K.st, k.aq, k.as, Mr, K.mw.L.v_vals, K.mw.L.v_scal,
                               PF_D, PF_D, k.yv, PF_AKV, PF_AKV, 0, 0);
            hipLaunchKernelGGL(pf_attn_qk, dim3(R * PF_HKV), dim3(256), 0, K.st, (const float *) nullptr, k.yk, k.yv,
                               (const float *) nullptr, K.mw.L.k_norm_w, e.eps, first, (__half *) nullptr,
                               K.mw.L.kv_k, K.mw.L.kv_v, PF_HQ, PF_HKV, 0, mm ? (const int *) k.rpos : nullptr);
        }
        Mp = save;
        hpn[0] = hpn[1] = 0;
    }

    // MTP K / V entries first .. first + R - 1 from staged rows. Entry r pairs toks[r]
    // with hn[card][r], the output-normed hidden of the position before.
    void mtp_rows(qz_engine & e, const int32_t * toks, float * const * hn, int first, int R) {
        for (int r0 = 0; r0 < R; r0 += Mmax - PF_MT) {
            const int n = std::min(Mmax - PF_MT, R - r0), Mr = (n + PF_MT - 1) / PF_MT * PF_MT;
            for (int i = 0; i < 2; i++) {
                qz_card & K = e.C[i];
                card & k = c[i];
                CHECK(hipSetDevice(qz_dev_of(i)));
                CHECK(hipMemcpyAsync(k.tok, toks + r0, (size_t) n * 4, hipMemcpyHostToDevice, K.st));
                hipLaunchKernelGGL(pf_mtp_in, dim3(Mr), dim3(256), 0, K.st, k.tok, K.A.emb_vals, K.A.emb_scal,
                                   hn[i] + (size_t) r0 * PF_D, 0, K.mw.enorm, K.mw.hnorm, e.eps, n, k.cat,
                                   (const int *) nullptr, (const float *) nullptr);
                hipLaunchKernelGGL(pf_quant, dim3(Mr), dim3(256), 0, K.st, k.cat, 2 * PF_D, 2 * PF_D, k.aq, k.as, Mr, 0);
                hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_D / 64, Mr / 128), dim3(256), 0, K.st, k.aq, k.as, Mr, K.mw.eh_vals, K.mw.eh_scal,
                                   (2 * PF_D), 2 * PF_D, k.part, PF_D, PF_D, 0, 0);
                hipLaunchKernelGGL(pf_norm_quant, dim3(Mr), dim3(256), 0, K.st, k.part, PF_D, K.mw.L.attn_norm, e.eps, PF_D,
                                   k.aq, k.as, Mr, (float *) nullptr, 0);
                hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_AKV / 64, Mr / 128), dim3(256), 0, K.st, k.aq, k.as, Mr, K.mw.L.k_vals, K.mw.L.k_scal,
                                   PF_D, PF_D, k.yk, PF_AKV, PF_AKV, 0, 0);
                hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_AKV / 64, Mr / 128), dim3(256), 0, K.st, k.aq, k.as, Mr, K.mw.L.v_vals, K.mw.L.v_scal,
                                   PF_D, PF_D, k.yv, PF_AKV, PF_AKV, 0, 0);
                hipLaunchKernelGGL(pf_attn_qk, dim3(n * PF_HKV), dim3(256), 0, K.st, (const float *) nullptr, k.yk, k.yv,
                                   (const float *) nullptr, K.mw.L.k_norm_w, e.eps, first + r0, (__half *) nullptr,
                                   K.mw.L.kv_k, K.mw.L.kv_v, PF_HQ, PF_HKV, 0, (const int *) nullptr);
                CHECK(hipGetLastError());
            }
        }
        for (int i = 0; i < 2; i++) { CHECK(hipSetDevice(qz_dev_of(i))); CHECK(hipStreamSynchronize(e.C[i].st)); }
        CHECK(hipSetDevice(qz_dev_of(0)));
    }

    // Runs positions p0 .. p0 + n - 1 of toks, next being the token at p0 + n. snap > 0
    // also copies the recurrent state before row snap for the prefix cache. Returns 0.
    int run(qz_engine & e, const int32_t * toks, int p0, int n, int32_t next,
            int snap = -1, float * const * snap_dn = nullptr, float * const * snap_cs = nullptr,
            const qz_mm_view * mm = nullptr) {
        if (snap <= 0 || snap >= n || !snap_dn) snap = -1;
        int nrl = 0;   // recurrent layers so far
        if (n <= 0 || n + 1 > Mmax) return 1;
        Mp = (n + PF_MT - 1) / PF_MT * PF_MT;
        const float eps = e.eps;
        // two halves only when each gets at least 2 tiles
        const int split = (n / 2 + PF_MT - 1) / PF_MT * PF_MT;
        nh = split >= 2 * PF_MT && split < n ? 2 : 1;
        if (nh == 2) { hv[0] = { 0, split, 0, split / PF_MT, split }; hv[1] = { split, n, split / PF_MT, (Mp - split) / PF_MT, Mp }; }
        else hv[0] = { 0, n, 0, Mp / PF_MT, Mp };
        hpn[0] = hpn[1] = 0;
        for (int i = 0; i < 2; i++) {
            qz_card & K = e.C[i];
            CHECK(hipSetDevice(qz_dev_of(i)));
            if (!evx[i][0]) {
                for (int h = 0; h < 2; h++) CHECK(hipEventCreateWithFlags(&evx[i][h], hipEventDisableTiming));
                for (int j = 0; j < 4; j++) CHECK(hipEventCreateWithFlags(&evh[i][j], hipEventDisableTiming));
                for (int h = 0; h < 2; h++) CHECK(hipEventCreateWithFlags(&evc[i][h], hipEventDisableTiming));
            }
            // in the shared scratch the buffers still hold the decode's partials, so zero
            // them like fresh buffers. Rows past n are padding the kernels skip.
            if (c[i].scr) CHECK(hipMemsetAsync(e.C[i].scr, 0, c[i].scr, K.st));
            CHECK(hipEventRecord(c[i].ev, K.st));
        }
        for (int i = 0; i < 2; i++) {
            qz_card & K = e.C[i];
            CHECK(hipSetDevice(qz_dev_of(i)));
            // the first row-parallel partial lands in the peer's scratch, so wait until
            // the peer has zeroed it: the peer may still be busy with the last chunk's
            // snapshot copies, and its memset would then wipe this card's partial. In host
            // mode the partial arrives by copy instead, ordered by evc, so this wait is no
            // longer load-bearing there; it stays so both modes queue work the same way.
            CHECK(hipStreamWaitEvent(K.st, c[1 - i].ev, 0));
            CHECK(hipMemcpyAsync(c[i].tok, toks, (size_t) n * 4, hipMemcpyHostToDevice, K.st));
            if (mm) { mm_stage(e, mm, p0, n, i); CHECK(hipSetDevice(qz_dev_of(i))); }   // after the scratch memset
            hipLaunchKernelGGL(pf_embed, dim3(Mp), dim3(256), 0, K.st, c[i].tok, K.A.emb_vals, K.A.emb_scal, c[i].x, n, Mp,
                               mm ? (const int *) c[i].irow : nullptr, mm ? mm->rows[i] : nullptr);
            if (nh == 2) { ev(e, i, 0, 3); wait(e, i, 1, 3); }   // half 1 starts only after the embedding
        }
        enum { EV_CONV = 0, EV_SCAN = 1, EV_KV = 2 };
        for (int il = 0; il < e.n_layer; il++) {
            const bool recr = e.C[0].wh[il].recr;
            for (int h = 0; h < nh; h++) {
                const half & H = hv[h];
                const int nr = H.r1 - H.r0, np = H.rp - H.r0;
                for (int i = 0; i < 2; i++) {
                    qz_card & K = e.C[i];
                    const qz_trunk_w & W = K.wh[il];
                    card & k = c[i];
                    CHECK(hipSetDevice(qz_dev_of(i)));
                    hipStream_t st = hs(e, i, h);
                    norm(e, i, h, W.attn_norm, k.aq, (float *) nullptr);
                    // DSpine features: the residual is complete once layer il's attn norm
                    // has applied layer il - 1's pending sum.
                    for (int ft = 0; feat && i == 0 && ft < 5; ft++)
                        if (il == PF_FEAT_LAYERS[ft] + 1)
                            CHECK(hipMemcpyAsync(feat + ((size_t) ft * Mmax + H.r0) * PF_D, k.x + (size_t) H.r0 * PF_D,
                                                 (size_t) (H.r1 - H.r0) * PF_D * 4, hipMemcpyDeviceToDevice, st));
                    if (recr) {
                        gemm(e, i, h, W.bet_vals, W.bet_scal, PF_HV, PF_D, k.ba, 2 * PF_HV);
                        gemm(e, i, h, W.alp_vals, W.alp_scal, PF_HV, PF_D, k.ba + PF_HV, 2 * PF_HV);
                        // qkv at row stride PF_AQ: k.y is shared with attention, and the
                        // halves may be a layer apart.
                        gemm(e, i, h, W.qkv_vals, W.qkv_scal, PF_CCH, PF_D, k.y, PF_AQ);
                        gemm(e, i, h, W.gat_vals, W.gat_scal, PF_G, PF_D, k.z, PF_G);
                        // conv: half 1's window is half 0's last three qkv rows,
                        // copied into k.edge once per recurrent layer.
                        float * edge = k.edge + (size_t) nrl * 3 * PF_CCH;
                        const float * cs = h == 1 ? edge : W.conv_state;
                        const int q0 = h == 1 ? H.r0 : 0;
                        if (h == 0 && nh == 2)
                            CHECK(hipMemcpy2DAsync(edge, PF_CCH * 4, k.y + (size_t) (hv[1].r0 - 3) * PF_AQ, PF_AQ * 4, PF_CCH * 4, 3,
                                                   hipMemcpyDeviceToDevice, st));
                        if (h == 1) wait(e, i, h, EV_CONV);
                        hipLaunchKernelGGL(pf_conv_prep, dim3(nr), dim3(256), 0, st, k.y, cs, W.conv1d, k.conv,
                                           k.ba, eps, W.dt_bias, W.ssm_a, k.qn, k.kn, k.beta, k.dec, H.r0, q0);
                        // snapshot window: the half whose rows hold snap - 3 .. snap - 1.
                        // Half 0 still sees the old window; half 1 commits after EV_CONV.
                        if (snap >= 0 && (nh == 1 || (h == 0) == (snap <= hv[1].r0)))
                            hipLaunchKernelGGL(pf_conv_snap, dim3((PF_CCH + 255) / 256), dim3(256), 0, st, k.y, snap, cs,
                                               snap_cs[i] + (size_t) nrl * 3 * PF_CCH, q0);
                        if (h == 0 && nh == 2) ev(e, i, h, EV_CONV);
                        if (h == nh - 1) hipLaunchKernelGGL(pf_conv_commit, dim3((PF_CCH + 255) / 256), dim3(256), 0, st, k.y, n, cs, W.conv_state, q0);
                        if (h == 1) wait(e, i, h, EV_SCAN);   // the state after half 0
                        hipLaunchKernelGGL(pf_dn_scan, dim3(PF_HV * 8), dim3(64), 0, st, k.qn, k.kn, k.conv, k.beta, k.dec, W.dn_state, k.o, H.r1, H.r0,
                                           snap, snap >= 0 ? snap_dn[i] + (size_t) nrl * PF_HV * PF_S * PF_S : (float *) nullptr);
                        if (h == 0 && nh == 2) ev(e, i, h, EV_SCAN);
                        hipLaunchKernelGGL(pf_dn_out, dim3(np * PF_HV), dim3(128), 0, st, k.o, k.z, W.ssm_norm_w, eps, k.aq, k.as, Mp, H.r0, n);
                    } else {
                        gemm(e, i, h, W.q_vals, W.q_scal, PF_AQ, PF_D, k.y, PF_AQ);
                        gemm(e, i, h, W.k_vals, W.k_scal, PF_AKV, PF_D, k.yk, PF_AKV);
                        gemm(e, i, h, W.v_vals, W.v_scal, PF_AKV, PF_D, k.yv, PF_AKV);
                        hipLaunchKernelGGL(pf_attn_qk, dim3(nr * (PF_HQ + PF_HKV)), dim3(256), 0, st, k.y, k.yk, k.yv,
                                           W.q_norm_w, W.k_norm_w, eps, p0, k.q16, W.kv_k, W.kv_v, 0, PF_HQ + PF_HKV, H.r0,
                                           mm ? (const int *) k.rpos : nullptr);
                        if (h == 0 && nh == 2) ev(e, i, h, EV_KV);
                        if (h == 1) wait(e, i, h, EV_KV);    // keys of half 0
                        if (H.rp > n) CHECK(hipMemsetAsync(k.o + (size_t) n * PF_HQ * PF_HD, 0, (size_t) (H.rp - n) * PF_HQ * PF_HD * 4, st));
                        hipLaunchKernelGGL(pf_flash2, dim3((nr + 63) / 64, PF_HQ), dim3(512), 0, st, k.q16, W.kv_k, W.kv_v, k.y, p0, H.r1, k.o, H.r0);
                        hipLaunchKernelGGL(pf_quant, dim3(np), dim3(256), 0, st, k.o, PF_HQ * PF_HD, PF_HQ * PF_HD, k.aq, k.as, Mp, H.r0);
                    }
                }
                {   // ssm_out or attn_output, both K = 3072 per card, plus the cross-card sum
                    const char * Wo[2] = { e.C[0].wh[il].out_vals, e.C[1].wh[il].out_vals };
                    const __half * So[2] = { e.C[0].wh[il].out_scal, e.C[1].wh[il].out_scal };
                    rowpar(e, h, Wo, So, PF_G, false);
                }
                for (int i = 0; i < 2; i++) {
                    qz_card & K = e.C[i];
                    const qz_trunk_w & W = K.wh[il];
                    card & k = c[i];
                    CHECK(hipSetDevice(qz_dev_of(i)));
                    norm(e, i, h, W.post_norm, k.aq, (float *) nullptr);
                    // gate, up and SwiGLU plus quantize in one GEMM, aq in and aq2 out
                    hipLaunchKernelGGL(pf_gemm<1>, dim3(PF_FF / 32, H.nty), dim3(256), 0, hs(e, i, h), k.aq, k.as, Mp, W.fg_vals, W.fg_scal,
                                       PF_D, PF_D, (float *) nullptr, 0, PF_FF, H.ty0, 0, (unsigned short *) nullptr,
                                       W.fu_vals, W.fu_scal, k.aq2, k.as2);
                }
                {
                    const char * Wd[2] = { e.C[0].wh[il].fd_vals, e.C[1].wh[il].fd_vals };
                    const __half * Sd[2] = { e.C[0].wh[il].fd_scal, e.C[1].wh[il].fd_scal };
                    rowpar(e, h, Wd, Sd, PF_FF, true);
                }
            }
            if (recr) nrl++;
        }
        for (int i = 0; i < 2; i++)
            if (nh == 2) { CHECK(hipSetDevice(qz_dev_of(i))); ev(e, i, 1, 3); wait(e, i, 0, 3); }   // join: main waits for half 1
        mtp_pass(e, toks, p0, n, next, mm);
        for (int i = 0; i < 2; i++) { CHECK(hipSetDevice(qz_dev_of(i))); CHECK(hipMemsetAsync(e.C[i].ctl_d, 0, 8 * sizeof(int), e.C[i].st)); }
        e.sync2();
        hpn[0] = hpn[1] = 0;
        return 0;
    }
};
