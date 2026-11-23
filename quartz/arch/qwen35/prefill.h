// Batched prefill: prompt rows through the trunk as GEMMs on both cards (TP2),
// writing the K / V, DeltaNet, conv and MTP state the decode continues from.
#pragma once

#include <vector>

#include "devices.h"

// per-card geometry (PF_ prefix: engine.h defines D, S, G, ... as macros)
constexpr int PF_D = 5120, PF_CCH = 5120, PF_HK = 8, PF_HV = 24, PF_S = 128, PF_G = 3072, PF_AQ = 6144,
              PF_AKV = 512, PF_HQ = 12, PF_HKV = 2, PF_HD = 256, PF_FF = 8704;
constexpr int PF_MT = 128;
constexpr int PF_GG = 4;    // pf_gemm: token tiles per dispatch group

// ---------------------------------------------------------------- GEMM ----
// C[m][n] = A[m] . W[n] with per-32 scales; block 64 rows x 128 tokens
typedef unsigned int pf_u32x4 __attribute__((ext_vector_type(4)));
struct pf_xrow { pf_u32x4 q[4][2]; pf_u32x4 pad; };

// SWI = 1: gate (W) and up (W2) in one GEMM, the epilogue writes the quantized
// silu(g) * u for the down projection
template <int SWI>
__global__ void __launch_bounds__(256, 2) pf_gemm(
        const int8_t * __restrict__ Ag, const float * __restrict__ sag, int Mp,
        const char * __restrict__ W, const __half * __restrict__ sw, int ldw, int K,
        float * __restrict__ C, int ldc, int N, int ty0, int obf16, unsigned short * __restrict__ C2 = nullptr,
        const char * __restrict__ W2 = nullptr, const __half * __restrict__ sw2 = nullptr,
        int8_t * __restrict__ Qo = nullptr, float * __restrict__ Qs = nullptr) {
    __shared__ pf_u32x4 sWlo[64][4], sWhi[64][4];
    __shared__ unsigned short sWd[4][64];
    __shared__ pf_xrow sX[128];
    __shared__ float sXd[128][5];
    const int t = threadIdx.x, tx = t & 63, ty = t >> 6;
    // grouped dispatch order: consecutive blocks cover PF_GG token tiles x successive row blocks, so the
    // blocks in flight share both weight rows and activation tiles in L2 (same work per block: exact)
    int bx, by;
    {
        const int gx = gridDim.x, lin = blockIdx.y * gx + blockIdx.x, per = PF_GG * gx, fy = lin / per * PF_GG;
        const int gs = min((int) gridDim.y - fy, PF_GG), r = lin - (fy / PF_GG) * per;
        bx = r / gs; by = fy + r % gs;
    }
    const int row0 = bx * 64, tok0 = (by + ty0) * 128;
    const int nkb = K / 32;
    float acc[16][2] = {};
    pf_u32x4 pwlo, pwhi, pxlo[2], pxhi[2];
    unsigned short pwd;
    float pxd[2];
    unsigned wrow;
    if (SWI) {
        wrow = (unsigned) (bx * 32 + ((t / 4) & 31));
        if (t / 4 >= 32) { W = W2; sw = sw2; }
    } else {
        wrow = (unsigned) min(row0 + t / 4, N - 1);   // rows past N: clamped, outputs dropped
    }
    const unsigned w_off = wrow * (unsigned) ldw + (t % 4) * 32u;
    const unsigned wd_off = wrow * (unsigned) nkb + t % 4;
    unsigned x_off[2], xd_off[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int e = t + 256 * i;
        x_off[i] = (unsigned) (tok0 + e / 4) * 128u + (e % 4) * 32u;
        xd_off[i] = (unsigned) (tok0 + e / 4) * 4u + e % 4;
    }
    auto prefetch = [&](int sb0) {
        const pf_u32x4 * ws = (const pf_u32x4 *) (W + w_off + (unsigned) sb0 * 32u);
        pwlo = ws[0]; pwhi = ws[1];
        pwd = ((const unsigned short *) sw)[wd_off + (unsigned) sb0];
        const unsigned grp = (unsigned) (sb0 / 4) * (unsigned) Mp;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const pf_u32x4 * xs = (const pf_u32x4 *) (Ag + grp * 128u + x_off[i]);
            pxlo[i] = xs[0]; pxhi[i] = xs[1];
            pxd[i] = sag[grp * 4u + xd_off[i]];
        }
    };
    auto store = [&]() {
        sWlo[t / 4][t % 4] = pwlo; sWhi[t / 4][t % 4] = pwhi; sWd[t % 4][t / 4] = pwd;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int e = t + 256 * i;
            sX[e / 4].q[e % 4][0] = pxlo[i]; sX[e / 4].q[e % 4][1] = pxhi[i]; sXd[e / 4][e % 4] = pxd[i];
        }
    };
    prefetch(0);
    for (int sb0 = 0; sb0 < nkb; sb0 += 4) {
        store();
        __syncthreads();
        if (sb0 + 4 < nkb) prefetch(sb0 + 4);
#pragma unroll 1
        for (int kk = 0; kk < 4; kk++) {
            int xq[2][8];
            float dx[2];
#pragma unroll
            for (int n = 0; n < 2; n++) {
                const pf_u32x4 q0 = sX[tx + 64 * n].q[kk][0], q1 = sX[tx + 64 * n].q[kk][1];
                xq[n][0] = q0.x; xq[n][1] = q0.y; xq[n][2] = q0.z; xq[n][3] = q0.w;
                xq[n][4] = q1.x; xq[n][5] = q1.y; xq[n][6] = q1.z; xq[n][7] = q1.w;
                dx[n] = sXd[tx + 64 * n][kk];
            }
            const int rb = ty * 16;
            pf_u32x4 wlo = sWlo[rb][kk], whi = sWhi[rb][kk];
#pragma unroll
            for (int g = 0; g < 16; g += 8) {
                const pf_u32x4 dh4 = *(const pf_u32x4 *) &sWd[kk][rb + g];
#pragma unroll
                for (int rr = 0; rr < 8; rr++) {
                    const int r = g + rr;
                    const pf_u32x4 lo = wlo, hi = whi;
                    if (r + 1 < 16) { wlo = sWlo[rb + r + 1][kk]; whi = sWhi[rb + r + 1][kk]; }
                    const unsigned dw32 = rr < 2 ? dh4.x : rr < 4 ? dh4.y : rr < 6 ? dh4.z : dh4.w;
                    const float d = __half2float(__ushort_as_half((unsigned short) ((rr & 1) ? dw32 >> 16 : dw32 & 0xffffu)));
#pragma unroll
                    for (int n = 0; n < 2; n++) {
                        int isum = 0;
                        isum = __builtin_amdgcn_sdot4(xq[n][0], (int) lo.x, isum, false);
                        isum = __builtin_amdgcn_sdot4(xq[n][1], (int) lo.y, isum, false);
                        isum = __builtin_amdgcn_sdot4(xq[n][2], (int) lo.z, isum, false);
                        isum = __builtin_amdgcn_sdot4(xq[n][3], (int) lo.w, isum, false);
                        isum = __builtin_amdgcn_sdot4(xq[n][4], (int) hi.x, isum, false);
                        isum = __builtin_amdgcn_sdot4(xq[n][5], (int) hi.y, isum, false);
                        isum = __builtin_amdgcn_sdot4(xq[n][6], (int) hi.z, isum, false);
                        isum = __builtin_amdgcn_sdot4(xq[n][7], (int) hi.w, isum, false);
                        acc[r][n] = __builtin_fmaf((float) isum, dx[n] * d, acc[r][n]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (SWI) {
        // two passes of 64 tokens through LDS (the X tile's space): sAcc[row][token]
        float (*sAcc)[65] = (float (*)[65]) &sX[0];
        static_assert(sizeof(sX) >= 64 * 65 * 4, "epilogue tile");
#pragma unroll
        for (int n = 0; n < 2; n++) {
            __syncthreads();
#pragma unroll
            for (int g = 0; g < 16; g++) sAcc[ty * 16 + g][tx] = acc[g][n];
            __syncthreads();
            // thread: token tt, dims part * 8 .. + 7 of the block's 32; amax over the 32 by a 4-lane shuffle
            const int tt = t >> 2, part = t & 3, m = tok0 + 64 * n + tt;
            float h[8], ax = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float a = sAcc[part * 8 + j][tt], u = sAcc[32 + part * 8 + j][tt];
                h[j] = a / (1.f + expf(-a)) * u;
                ax = fmaxf(ax, fabsf(h[j]));
            }
            ax = fmaxf(ax, __shfl_xor(ax, 1, 64));
            ax = fmaxf(ax, __shfl_xor(ax, 2, 64));
            const float d = ax / 127.f;
            const float id = d != 0.f ? 1.f / d : 0.f;
            unsigned w2[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int q = (int) roundf(h[j] * id);
                q = q > 127 ? 127 : q < -127 ? -127 : q;
                w2[j / 4] |= ((unsigned) q & 0xffu) << (8 * (j % 4));
            }
            const int ff = bx * 32 + part * 8;
            const unsigned g = (unsigned) (ff / 128), sub = (unsigned) (ff % 128);
            *(uint2 *) (Qo + ((size_t) g * Mp + m) * 128 + sub) = make_uint2(w2[0], w2[1]);
            if (part == 0) Qs[((size_t) g * Mp + m) * 4 + sub / 32] = __half2float(__float2half(d));
        }
        return;
    }
    if (obf16) {   // row-parallel partials for the cross-card sum: bf16 (round to nearest even), as llama's all-reduce;
                   // C2: the same 32 B also stored straight into the peer card's memory (P2P write, overlapped with the GEMM)
#pragma unroll
        for (int n = 0; n < 2; n++) {
            unsigned b[8];
#pragma unroll
            for (int g = 0; g < 16; g += 2) {
                const unsigned u0 = __float_as_uint(acc[g][n]), u1 = __float_as_uint(acc[g + 1][n]);
                b[g / 2] = ((u0 + 0x7fffu + ((u0 >> 16) & 1u)) >> 16) | (((u1 + 0x7fffu + ((u1 >> 16) & 1u)) >> 16) << 16);
            }
            const size_t off = (size_t) (tok0 + tx + 64 * n) * ldc + row0 + ty * 16;
            pf_u32x4 * d1 = (pf_u32x4 *) ((unsigned short *) C + off);
            d1[0] = pf_u32x4{b[0], b[1], b[2], b[3]}; d1[1] = pf_u32x4{b[4], b[5], b[6], b[7]};
            if (C2) {
                pf_u32x4 * d2 = (pf_u32x4 *) (C2 + off);
                d2[0] = pf_u32x4{b[0], b[1], b[2], b[3]}; d2[1] = pf_u32x4{b[4], b[5], b[6], b[7]};
            }
        }
        return;
    }
#pragma unroll
    for (int n = 0; n < 2; n++) {
        float * dst = C + (size_t) (tok0 + tx + 64 * n) * ldc + row0 + ty * 16;
        if (row0 + ty * 16 + 16 <= N) {
#pragma unroll
            for (int g = 0; g < 16; g += 4)
                *(float4 *) (dst + g) = make_float4(acc[g][n], acc[g + 1][n], acc[g + 2][n], acc[g + 3][n]);
            if (C2) {   // f32 partial also straight into the peer card's memory
                float * d2 = (float *) C2 + (size_t) (tok0 + tx + 64 * n) * ldc + row0 + ty * 16;
#pragma unroll
                for (int g = 0; g < 16; g += 4)
                    *(float4 *) (d2 + g) = make_float4(acc[g][n], acc[g + 1][n], acc[g + 2][n], acc[g + 3][n]);
            }
        } else {
#pragma unroll
            for (int g = 0; g < 16; g++)
                if (row0 + ty * 16 + g < N) dst[g] = acc[g][n];
        }
    }
}

// ------------------------------------------------------- quantization ----
// int8 per 32 values (f16-rounded scale), grouped [K/128][Mp][128]
__device__ __forceinline__ void pf_quant_unit(float v, int k, int m, int Mp, int8_t * Ag, float * sag, int lane32) {
    float ax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ax = fmaxf(ax, __shfl_xor(ax, off, 32));
    const float d = ax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    int q = (int) roundf(v * id);
    q = q > 127 ? 127 : q < -127 ? -127 : q;
    const unsigned g = (unsigned) (k / 128), sub = (unsigned) (k % 128);
    Ag[((size_t) g * Mp + m) * 128 + sub] = (int8_t) q;
    if (lane32 == 0) sag[((size_t) g * Mp + m) * 4 + sub / 32] = __half2float(__float2half(d));
}

// one block per row: y = x * rsqrt(mean(x^2) + eps) * w, quantized (and
// optionally kept in f32 for the MTP hidden input)
__global__ void __launch_bounds__(256) pf_norm_quant(
        const float * __restrict__ x, int ldx, const float * __restrict__ w, float eps, int K,
        int8_t * Ag, float * sag, int Mp, float * outf, int m0 = 0) {
    __shared__ float red[8];
    const int m = m0 + blockIdx.x, t = threadIdx.x;
    const float * xr = x + (size_t) m * ldx;
    float ss = 0.f;
    for (int k = t; k < K; k += 256) ss += xr[k] * xr[k];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off, 64);
    if ((t & 63) == 0) red[t >> 6] = ss;
    __syncthreads();
    const float r = 1.f / sqrtf((red[0] + red[1] + red[2] + red[3]) / K + eps);
    for (int k = t; k < K; k += 256) {
        const float y = xr[k] * r * w[k];
        if (outf) outf[(size_t) m * K + k] = y;
        if (Ag) pf_quant_unit(y, k, m, Mp, Ag, sag, t & 31);
    }
}

// the cross-card sum x += p0 + p1 (bf16 partials) fused into the next RMS norm
__device__ __forceinline__ float pf_ld_part(const unsigned short * p, size_t i) { return __uint_as_float((unsigned) p[i] << 16); }
__global__ void __launch_bounds__(256) pf_addnorm_quant(
        float * __restrict__ x, const unsigned short * __restrict__ p0, const unsigned short * __restrict__ p1, int n,
        const float * __restrict__ w, float eps, int8_t * Ag, float * sag, int Mp, float * outf, int m0 = 0) {
    __shared__ float red[8];
    const int m = m0 + blockIdx.x, t = threadIdx.x;
    float * xr = x + (size_t) m * PF_D;
    constexpr int NE = PF_D / 256;
    float xv[NE];                     // the row stays in registers between the add and the norm
    float ss = 0.f;
    if (m < n) {
        const unsigned short * a = p0 + (size_t) m * PF_D, * b = p1 + (size_t) m * PF_D;
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const int k = t + 256 * j;
            const float v = xr[k] + (pf_ld_part(a, k) + pf_ld_part(b, k));
            xr[k] = v;
            xv[j] = v;
            ss = __builtin_fmaf(v, v, ss);
        }
    } else {
#pragma unroll
        for (int j = 0; j < NE; j++) { xv[j] = xr[t + 256 * j]; ss = __builtin_fmaf(xv[j], xv[j], ss); }
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off, 64);
    if ((t & 63) == 0) red[t >> 6] = ss;
    __syncthreads();
    const float r = 1.f / sqrtf((red[0] + red[1] + red[2] + red[3]) / PF_D + eps);
#pragma unroll
    for (int j = 0; j < NE; j++) {
        const int k = t + 256 * j;
        const float y = xv[j] * r * w[k];
        if (outf) outf[(size_t) m * PF_D + k] = y;
        if (Ag) pf_quant_unit(y, k, m, Mp, Ag, sag, t & 31);
    }
}

// quantize rows of y (row stride ldy) as they are
__global__ void __launch_bounds__(256) pf_quant(const float * __restrict__ y, int ldy, int K, int8_t * Ag, float * sag, int Mp, int m0 = 0) {
    const int m = m0 + blockIdx.x;
    for (int k = threadIdx.x; k < K; k += 256) pf_quant_unit(y[(size_t) m * ldy + k], k, m, Mp, Ag, sag, threadIdx.x & 31);
}

// ------------------------------------------------------- elementwise ----
// token embeddings; with irow, row m takes image row irow[m] (>= 0) instead
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

// causal conv1d (4 taps) + silu over all rows at once; rows below q0 come from
// the window cs (qkv rows at stride PF_AQ)
__device__ __forceinline__ float pf_conv_in(const float * qkv, const float * cs, int k, int c, int q0) {
    return k >= q0 ? qkv[(size_t) k * PF_AQ + c] : cs[(3 + k - q0) * PF_CCH + c];
}
__device__ __forceinline__ float pf_conv_at(const float * qkv, const float * cs, const float * w4, int m, int c, int q0) {
    const float * w = w4 + c * 4;
    const float o = pf_conv_in(qkv, cs, m - 3, c, q0) * w[0] + pf_conv_in(qkv, cs, m - 2, c, q0) * w[1] +
                    pf_conv_in(qkv, cs, m - 1, c, q0) * w[2] + qkv[(size_t) m * PF_AQ + c] * w[3];
    return o / (1.f + expf(-o));
}
// the window as it is before row snap of the chunk (the prefix cache's mid-chunk snapshot)
__global__ void pf_conv_snap(const float * __restrict__ qkv, int snap, const float * __restrict__ cs, float * out, int q0 = 0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= PF_CCH) return;
    out[c] = pf_conv_in(qkv, cs, snap - 3, c, q0); out[PF_CCH + c] = pf_conv_in(qkv, cs, snap - 2, c, q0); out[2 * PF_CCH + c] = pf_conv_in(qkv, cs, snap - 1, c, q0);
}
// the window after the chunk: its last three inputs (cs == ws is fine)
__global__ void pf_conv_commit(const float * __restrict__ qkv, int n, const float * cs, float * ws, int q0 = 0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= PF_CCH || n <= 0) return;
    const float a = pf_conv_in(qkv, cs, n - 3, c, q0), b = pf_conv_in(qkv, cs, n - 2, c, q0), d = pf_conv_in(qkv, cs, n - 1, c, q0);
    ws[c] = a; ws[PF_CCH + c] = b; ws[2 * PF_CCH + c] = d;
}

// conv + DeltaNet input prep in one pass per row: q / k go from the conv straight
// into their l2 norms (not stored), the v channels are written for the scan
__global__ void __launch_bounds__(256) pf_conv_prep(
        const float * __restrict__ qkv, const float * __restrict__ cs, const float * __restrict__ w4, float * conv,
        const float * __restrict__ ba, float eps,
        const float * __restrict__ dt_bias, const float * __restrict__ ssm_a,
        float * qn, float * kn, float * beta, float * dec, int m0 = 0, int q0 = 0) {
    __shared__ float red[4];
    const int m = m0 + blockIdx.x, t = threadIdx.x, wave = t >> 6, lane = t & 63;
    const float epsn = eps * (1.f / 128.f), l2i = 1.f / sqrtf(128.f);
    // the v channels the scan reads, through the conv
    for (int c = 2 * PF_HK * PF_S + t; c < PF_CCH; c += 256) conv[(size_t) m * PF_CCH + c] = pf_conv_at(qkv, cs, w4, m, c, q0);
    for (int v = 0; v < 2 * PF_HK; v += 2) {          // two vectors per pass: q / k of one head
        const int kh = v / 2;
        const int vec = wave >> 1;                      // waves 0,1: q; 2,3: k
        const int e = (wave & 1) * 64 + lane;
        const float x = pf_conv_at(qkv, cs, w4, m, vec * PF_S * PF_HK + kh * PF_S + e, q0);   // q / k: conv in place
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

// the DeltaNet recurrence over n rows: one wave per (head, 16 value columns),
// the decode scan's order; row m + 1's inputs load while row m computes
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
    // explicit fmas: the result does not depend on where the compiler contracts
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
    // snap_state: a copy of the state as it is before row snap (the prefix cache's mid-chunk snapshot)
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

// gated rms norm per head (o * rsqrt(mean(o^2) + eps) * w * silu(z)),
// quantized as the ssm_out input; one block (128 threads) per (row, head)
__global__ void __launch_bounds__(128) pf_dn_out(
        const float * __restrict__ o, const float * __restrict__ z, const float * __restrict__ nw, float eps,
        int8_t * Ag, float * sag, int Mp, int m0 = 0, int n = 1 << 30) {
    __shared__ float red[2];
    const int m = m0 + blockIdx.x / PF_HV, h = blockIdx.x % PF_HV, t = threadIdx.x;
    if (m >= n) {   // padded rows of the ssm_out input: zeros
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

// attention q / k: per-head RMS norm + MRoPE; q -> q16, k / v -> the arena at
// p0 + m; rp: per-row (t, h, w) rope positions (null: p0 + m)
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

// MTP entry inputs: [enorm(emb(tok[r])) | hnorm(hidden of the previous
// position)], one block per row
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
    const float * ir = irow && irow[r] >= 0 ? img + (size_t) irow[r] * PF_D : nullptr;   // image row (as pf_embed)
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

// v from another lane by DPP control C (0xb1 xor 1, 0x4e xor 2, 0x141 half
// mirror): no LDS round trip, unlike __shfl_xor
template <int C> __device__ __forceinline__ float pf_dppf(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), C, 0xf, 0xf, false));
}

// pf_flash2: causal attention as two register-tiled GEMMs; block = 64 queries
// of one head, wave w holds q dims w * 32 .. + 31, key tiles of 16
#define PF2_KT 16
#define PF2_KS 264                                         // K / V row stride in halves (256 + 8: rows kg + G j on distinct banks)
#define PF2_QKP 1                                          // QK: K reads issued this many steps ahead (2+ spill)
#define PF2_NW 8                                           // waves: wave w holds dims w * DW .. + DW - 1 of QB = 64 queries
__global__ void __launch_bounds__(PF2_NW * 64, 1) pf_flash2(
        const __half * __restrict__ q16, const __half * __restrict__ kv_k, const __half * __restrict__ kv_v,
        const float * __restrict__ yq, int p0, int n, float * __restrict__ out, int mb = 0) {
    constexpr int NW = PF2_NW, NT = NW * 64, DW = 256 / NW, QB = 8 * NW, QL = 2, KG = 2;
    constexpr int KL = PF2_KT / KG;                        // keys per lane in QK: kg + KG j
    static_assert((QB / QL) * KG == 64, "QK lane tile");
    static_assert((QB / 4) * (DW / 8) == 64, "PV lane tile");
    __shared__ __attribute__((aligned(16))) __half Ks[PF2_KT * PF2_KS], Vs[PF2_KT * PF2_KS];
    // per wave: partial scores [query][key], unpadded rows of KT; 16-byte chunk c of query q's row stored at
    // c ^ ((q >> 1) & 3), which keeps the QK b128 writes and the softmax b64 reads conflict-free
    __shared__ __attribute__((aligned(16))) float Sp[NW][QB][PF2_KT];
    auto spc = [](int q, int c) { return c ^ ((q >> 1) & 3); };
    // QK lane kg's keys: kg * KL + j (contiguous, K rows swizzled: 16-byte chunk ^ the row's key group
    // so the KG rows read together sit on distinct banks)
    auto qkey = [](int kg, int j) { return kg * KL + j; };
    auto kcol = [](int row, int col) { return col ^ ((row / KL) * 8); };
    __shared__ __attribute__((aligned(16))) float Pt[PF2_KT][QB + 4];   // probabilities [key][query]
    __shared__ __attribute__((aligned(16))) float cr[QB], ls[QB];       // per query: rescale of this tile, final l
    const int h = blockIdx.y, kh = h / 6, t = threadIdx.x, w = t >> 6, lane = t & 63;
    const int m0 = mb + (gridDim.x - 1 - blockIdx.x) * QB;   // longest causal blocks dispatched first
    const int kmax = p0 + min(m0 + QB - 1, n - 1);
    // QK lane: queries qg + r * QB / 2; keys qkey(kg, j)
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
    // tile loads: LPT x (key, 8 halves) of K and of V per thread
    constexpr int LPT = PF2_KT * 32 / NT;
    uint4 nk[LPT];
    // V is loaded straight as key pairs: thread = (pair j = t >> 6, dims (t & 63) * 4 .. + 3),
    // 8 B of key 2j and 8 B of key 2j + 1, so the pair dwords are built in-lane (no cross-lane shuffle)
    static_assert(NT == 8 * 64, "V load: 8 key pairs x 64 lanes");
    uint2 ve = make_uint2(0, 0), vo = make_uint2(0, 0);
    auto load = [&](int kt) {
#pragma unroll
        for (int u = 0; u < LPT; u++) {
            const int idx = t + u * NT, kp = kt + (idx >> 5), lc = (idx & 31) * 8;
            nk[u] = make_uint4(0, 0, 0, 0);
            if (kp <= kmax) nk[u] = *(const uint4 *) (kv_k + qz_kv_row(kp, PF_HKV, kh) + lc);
        }
        {   // keys kt .. kt + 15 lie in one 64-position page of qz_kv_row
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
        __syncthreads();                                   // the last tile's QK / PV are done with Ks, Vs, Pt
        {   // V as key pairs: dword (key 2j, key 2j + 1) of one dim; pair row j, stride PF2_KS dwords
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
            {   // software-pipelined: the read of step it + PF2_QKP is in flight while step it computes
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
        {   // softmax: query sq, keys sk, sk + 1
            // indices rebuilt from the lane id here (kept live they spill)
            int lid = __lane_id();
            asm volatile("" : "+v"(lid));
            const int sk = (lid & 7) * 2, sq = __builtin_amdgcn_readfirstlane(w) * 8 + (lid >> 3);
            float x[2], a2[2] = { 0.f, 0.f };
#pragma unroll
            for (int v = 0; v < NW; v++) {                // the waves' partials in wave order
                const float2 f = *(const float2 *) &Sp[v][sq][4 * spc(sq, sk >> 2) + (sk & 3)];
                a2[0] += f.x; a2[1] += f.y;
            }
#pragma unroll
            for (int u = 0; u < 2; u++)                   // softmax in base 2: scores in log2 units, v_exp_f32
                x[u] = kt + sk + u > sqpos ? -INFINITY : a2[u] * (1.4426950408889634f / 16.f);
            float tm = fmaxf(x[0], x[1]);
            // max over the query's 8 lanes on DPP (xor 1, xor 2, then the other quad: any order gives the max)
            tm = fmaxf(tm, pf_dppf<0xb1>(tm)); tm = fmaxf(tm, pf_dppf<0x4e>(tm)); tm = fmaxf(tm, pf_dppf<0x141>(tm));
            const float nm = fmaxf(mx, tm);               // finite: key 0 is in tile 0 and visible to every query
            const float corr = mx == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(mx - nm);
            float ps = 0.f, pr[2];
#pragma unroll
            for (int u = 0; u < 2; u++) {
                float p = x[u] == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(x[u] - nm);
                p = (float) (_Float16) p;                 // P x V uses p rounded to f16, so l sums the rounded values
                pr[u] = p;
                ps += p;
            }
            // the lane's key pair (sk, sk + 1) as one f16 pair, [pair][query] in Pt's space
            ((pf_h2 *) &Pt[0][0])[(sk / 2) * (QB + 4) + sq] = pf_h2{ (_Float16) pr[0], (_Float16) pr[1] };
            // the xor 1 / 2 / 4 butterfly on DPP, the same sums: after xor 2 all four lanes of a quad hold the
            // same value, so row_half_mirror (lane 7 - i, the other quad) supplies what lane i ^ 4 held
            ps += pf_dppf<0xb1>(ps); ps += pf_dppf<0x4e>(ps); ps += pf_dppf<0x141>(ps);
            l = l * corr + ps;
            mx = nm;
            if ((t & 7) == 0) cr[sq] = corr;
        }
        __syncthreads();
        {   // PV
            const float4 c4 = *(const float4 *) &cr[pq];
            const float cc[4] = { c4.x, c4.y, c4.z, c4.w };
            // lazy rescale: once the running max settles corr is exactly 1 (expf(0)), so a wave
            // where no lane's queries moved skips the multiplies (same bits)
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

// DSpine target features: the residual after these trunk layers (0-based),
// captured on card 0 right after the next layer's attn norm
constexpr int PF_FEAT_LAYERS[5] = { 1, 16, 31, 46, 61 };

// ------------------------------------------------------------ driver ----
// a prompt's images: per position, its image row (-1: token) and rope (t, h, w)
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
    int pset = 0, pn = 0;   // the pending cross-card sum: partial set, rows (pn 0: none)
    // DSpine feature capture [5][Mmax][PF_D] on card 0 (null: off)
    float * feat = nullptr;
    struct card {
        float * x = nullptr, * part = nullptr, * y = nullptr, * conv = nullptr,
              * z = nullptr, * ba = nullptr, * qn = nullptr, * kn = nullptr, * beta = nullptr, * dec = nullptr,
              * o = nullptr, * fg = nullptr, * fu = nullptr, * yk = nullptr, * yv = nullptr,
              * hn = nullptr, * cat = nullptr,
              * edge = nullptr;   // [recurrent layer][3][CCH]: half 0's last three qkv rows, half 1's conv window
        int8_t * aq = nullptr, * aq2 = nullptr; float * as = nullptr, * as2 = nullptr;   // grouped int8 GEMM input + scales (two: ffn in / out)
        void * pl[2] = {}, * pr[2] = {};                 // partials (bf16 or f32), two sets: this card's own, the peer's (written by its GEMM)
        __half * q16 = nullptr;
        int32_t * tok = nullptr;
        int * irow = nullptr, * rpos = nullptr;          // a chunk's image rows / rope positions (qz_mm_view)
        size_t scr = 0;                                  // bytes carved from the engine's scratch (0: own buffers)
        hipEvent_t ev = nullptr, ev_aux = nullptr, evp[4] = {};
        hipStream_t aux = nullptr;
    } c[2];
    double ms_total = 0;

    // the per-card buffers of init(max_rows): (pointer, bytes) for M rows
    static std::vector<std::pair<void **, size_t>> bufs(card & k, size_t M) {
        const size_t xb = 2;                             // bf16 cross-card partials
        return { { (void **) &k.x, M * PF_D * 4 }, { (void **) &k.part, M * PF_D * 4 }, { (void **) &k.y, M * PF_AQ * 4 },
                 { (void **) &k.conv, M * PF_CCH * 4 }, { (void **) &k.z, M * PF_G * 4 }, { (void **) &k.ba, M * 2 * PF_HV * 4 },
                 { (void **) &k.qn, M * PF_HK * PF_S * 4 }, { (void **) &k.kn, M * PF_HK * PF_S * 4 },
                 { (void **) &k.beta, M * PF_HV * 4 }, { (void **) &k.dec, M * PF_HV * 4 }, { (void **) &k.o, M * PF_HV * PF_S * 4 },
                 { (void **) &k.yk, M * PF_AKV * 4 }, { (void **) &k.yv, M * PF_AKV * 4 },
                 { (void **) &k.aq, M * 2 * PF_D }, { (void **) &k.as, M * 2 * PF_D / 32 * 4 },
                 { (void **) &k.aq2, M * 2 * PF_D }, { (void **) &k.as2, M * 2 * PF_D / 32 * 4 },
                 { (void **) &k.hn, M * PF_D * 4 }, { (void **) &k.cat, M * 2 * PF_D * 4 },
                 { (void **) &k.q16, M * PF_HQ * PF_HD * 2 }, { (void **) &k.tok, M * 4 }, { (void **) &k.edge, (size_t) 64 * 3 * PF_CCH * 4 },
                 { &k.pl[0], M * PF_D * xb }, { &k.pr[0], M * PF_D * xb }, { &k.pl[1], M * PF_D * xb }, { &k.pr[1], M * PF_D * xb },
                 { (void **) &k.irow, M * 4 }, { (void **) &k.rpos, M * 3 * 4 } };
    }
    static size_t rows(int max_rows) { return (size_t) ((max_rows + PF_MT - 1) / PF_MT * PF_MT + PF_MT); }
    // device bytes per card init(max_rows) needs (256-byte aligned buffers)
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
        Mmax = (int) rows(max_rows);   // + one tile: the MTP pass has n + 1 rows
        qz_mem_scope mem("prefill");
        for (int i = 0; i < 2; i++) {
            card & k = c[i];
            CHECK(hipSetDevice(qz_dev_of(i)));
            const auto b = bufs(k, Mmax);
            // the engine's scratch when it holds them (qz_engine::scratch_min), else own buffers
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

    // two token halves on two streams per card, so one half's small kernels
    // run beside the other's GEMMs
    struct half { int r0 = 0, r1 = 0, ty0 = 0, nty = 0, rp = 0; };
    half hv[2];
    int nh = 1;
    int hset[2] = {0, 0}, hpn[2] = {0, 0};          // per half: the pending cross-card sum (set, rows; 0 none)
    hipEvent_t evx[2][2] = {}, evh[2][4] = {};      // [card][half]: partial written; [card][kind]: half 0 conv / scan / kv done, join

    hipStream_t hs(qz_engine & e, int i, int h) { return h == 0 ? e.C[i].st : c[i].aux; }

    // C = A x W over the half's token tiles (A: the grouped int8 input, default aq)
    void gemm(qz_engine & e, int i, int h, const char * W, const __half * sw, int N, int K, float * C, int ldc,
              const int8_t * A = nullptr, const float * As = nullptr) {
        const half & H = hv[h];
        hipLaunchKernelGGL(pf_gemm<0>, dim3((N + 63) / 64, H.nty), dim3(256), 0, hs(e, i, h), A ? A : c[i].aq, As ? As : c[i].as, Mp,
                           W, sw, K, K, C, ldc, N, H.ty0, 0);
    }
    // row-parallel projection: each card writes its bf16 partial locally and to
    // the peer; the next norm() applies x += p0 + p1 (two partial sets alternate)
    void rowpar(qz_engine & e, int h, const char * const * W, const __half * const * sw, int K, bool ffn) {
        const half & H = hv[h];
        hset[h] ^= 1;
        for (int i = 0; i < 2; i++) {
            CHECK(hipSetDevice(qz_dev_of(i)));
            hipLaunchKernelGGL(pf_gemm<0>, dim3(PF_D / 64, H.nty), dim3(256), 0, hs(e, i, h), ffn ? c[i].aq2 : c[i].aq, ffn ? c[i].as2 : c[i].as, Mp, W[i], sw[i],
                               K, K, (float *) c[i].pl[hset[h]], PF_D, PF_D, H.ty0, 1, (unsigned short *) c[1 - i].pr[hset[h]]);
            CHECK(hipEventRecord(evx[i][h], hs(e, i, h)));
        }
        hpn[h] = 1;
    }
    // RMS norm of the half's rows of x (quantized into aq, and / or f32 to
    // outf), the half's pending cross-card sum first when there is one
    void norm(qz_engine & e, int i, int h, const float * w, int8_t * Ag, float * outf, int rows_end = -1, hipStream_t st_ = nullptr) {
        const half & H = hv[h];
        card & k = c[i];
        hipStream_t st = st_ ? st_ : hs(e, i, h);
        const int r1 = rows_end >= 0 ? rows_end : H.rp;
        if (hpn[h]) {
            CHECK(hipStreamWaitEvent(st, evx[1 - i][h], 0));
            const int s_ = hset[h];
            const void * p0 = i == 0 ? k.pl[s_] : k.pr[s_], * p1 = i == 0 ? k.pr[s_] : k.pl[s_];
            hipLaunchKernelGGL(pf_addnorm_quant, dim3(r1 - H.r0), dim3(256), 0, st, k.x, (const unsigned short *) p0,
                               (const unsigned short *) p1, H.r1, w, e.eps, Ag, k.as, Mp, outf, H.r0);
        } else {
            hipLaunchKernelGGL(pf_norm_quant, dim3(r1 - H.r0), dim3(256), 0, st, k.x, PF_D, w, e.eps, PF_D, Ag, k.as, Mp, outf, H.r0);
        }
    }
    void ev(qz_engine & e, int i, int h, int kind) { CHECK(hipEventRecord(evh[i][kind], hs(e, i, h))); }
    void wait(qz_engine & e, int i, int h, int kind) { CHECK(hipStreamWaitEvent(hs(e, i, h), evh[i][kind], 0)); }

    // the image rows / rope positions of positions first .. first + R - 1 to
    // card `card`'s irow / rpos (-1: both cards), on the card's main stream
    void mm_stage(qz_engine & e, const qz_mm_view * mm, int first, int R, int card = -1) {
        mm_irow.resize(R); mm_rpos.resize((size_t) R * 3);
        for (int r = 0; r < R; r++) { mm_irow[r] = mm->img(first + r); mm->rope(first + r, &mm_rpos[(size_t) r * 3]); }
        for (int i = card < 0 ? 0 : card; i < (card < 0 ? 2 : card + 1); i++) {
            CHECK(hipSetDevice(qz_dev_of(i)));
            CHECK(hipMemcpyAsync(c[i].irow, mm_irow.data(), (size_t) R * 4, hipMemcpyHostToDevice, e.C[i].st));
            CHECK(hipMemcpyAsync(c[i].rpos, mm_rpos.data(), (size_t) R * 12, hipMemcpyHostToDevice, e.C[i].st));
        }
    }

    // MTP K / V entries for p0 + 1 .. p0 + n (and entry 0 when p0 == 0), after
    // both halves joined
    void mtp_pass(qz_engine & e, const int32_t * toks, int p0, int n, int32_t next, const qz_mm_view * mm = nullptr) {
        const int first = p0 == 0 ? 0 : p0 + 1;               // first entry position
        const int R = p0 + n + 1 - first;                     // entries first .. p0 + n
        const int Mr = (R + PF_MT - 1) / PF_MT * PF_MT;
        std::vector<int32_t> mt(R);
        for (int r = 0; r < R; r++) {
            const int pos = first + r;
            mt[r] = pos - p0 < n ? toks[pos - p0] : next;
        }
        const int hoff = first - 1 - p0;                      // hn row of entry r's hidden: r + hoff
        const int save = Mp;
        if (mm) mm_stage(e, mm, first, R);
        for (int i = 0; i < 2; i++) {
            qz_card & K = e.C[i];
            card & k = c[i];
            CHECK(hipSetDevice(qz_dev_of(i)));
            // trunk hidden of every row, output-normed (the MTP pairing input); per half (its pending sum), main stream
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

    // MTP K / V entries first .. first + R - 1 from staged rows: entry r pairs
    // toks[r] with hn[card][r], the output-normed hidden of the position before
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

    // positions p0 .. p0 + n - 1 of toks, next = the token at p0 + n; snap > 0 also
    // copies the recurrent state before row snap (prefix cache). Returns 0
    int run(qz_engine & e, const int32_t * toks, int p0, int n, int32_t next,
            int snap = -1, float * const * snap_dn = nullptr, float * const * snap_cs = nullptr,
            const qz_mm_view * mm = nullptr) {
        if (snap <= 0 || snap >= n || !snap_dn) snap = -1;
        int nrl = 0;   // recurrent layers so far
        if (n <= 0 || n + 1 > Mmax) return 1;
        Mp = (n + PF_MT - 1) / PF_MT * PF_MT;
        const float eps = e.eps;
        // halves: at least 2 tiles each, else one
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
            }
            // in the shared scratch the buffers hold the decode's partials: zero them, as
            // fresh buffers were (rows past n are padding the kernels never mix in)
            if (c[i].scr) CHECK(hipMemsetAsync(e.C[i].scr, 0, c[i].scr, K.st));
            CHECK(hipMemcpyAsync(c[i].tok, toks, (size_t) n * 4, hipMemcpyHostToDevice, K.st));
            if (mm) { mm_stage(e, mm, p0, n, i); CHECK(hipSetDevice(qz_dev_of(i))); }   // after the scratch memset
            hipLaunchKernelGGL(pf_embed, dim3(Mp), dim3(256), 0, K.st, c[i].tok, K.A.emb_vals, K.A.emb_scal, c[i].x, n, Mp,
                               mm ? (const int *) c[i].irow : nullptr, mm ? mm->rows[i] : nullptr);
            if (nh == 2) { ev(e, i, 0, 3); wait(e, i, 1, 3); }   // half 1 starts after the embedding
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
                    // DSpine features: complete once layer il's attn norm applied
                    // layer il - 1's pending sum
                    for (int ft = 0; feat && i == 0 && ft < 5; ft++)
                        if (il == PF_FEAT_LAYERS[ft] + 1)
                            CHECK(hipMemcpyAsync(feat + ((size_t) ft * Mmax + H.r0) * PF_D, k.x + (size_t) H.r0 * PF_D,
                                                 (size_t) (H.r1 - H.r0) * PF_D * 4, hipMemcpyDeviceToDevice, st));
                    if (recr) {
                        gemm(e, i, h, W.bet_vals, W.bet_scal, PF_HV, PF_D, k.ba, 2 * PF_HV);
                        gemm(e, i, h, W.alp_vals, W.alp_scal, PF_HV, PF_D, k.ba + PF_HV, 2 * PF_HV);
                        // qkv at row stride PF_AQ: k.y is shared with attention, and the
                        // halves may be a layer apart
                        gemm(e, i, h, W.qkv_vals, W.qkv_scal, PF_CCH, PF_D, k.y, PF_AQ);
                        gemm(e, i, h, W.gat_vals, W.gat_scal, PF_G, PF_D, k.z, PF_G);
                        // conv: half 1's window is half 0's last three qkv rows, copied to
                        // k.edge per layer
                        float * edge = k.edge + (size_t) nrl * 3 * PF_CCH;
                        const float * cs = h == 1 ? edge : W.conv_state;
                        const int q0 = h == 1 ? H.r0 : 0;
                        if (h == 0 && nh == 2)
                            CHECK(hipMemcpy2DAsync(edge, PF_CCH * 4, k.y + (size_t) (hv[1].r0 - 3) * PF_AQ, PF_AQ * 4, PF_CCH * 4, 3,
                                                   hipMemcpyDeviceToDevice, st));
                        if (h == 1) wait(e, i, h, EV_CONV);
                        hipLaunchKernelGGL(pf_conv_prep, dim3(nr), dim3(256), 0, st, k.y, cs, W.conv1d, k.conv,
                                           k.ba, eps, W.dt_bias, W.ssm_a, k.qn, k.kn, k.beta, k.dec, H.r0, q0);
                        // the snapshot window: by the half whose rows hold snap - 3 .. snap - 1 (half 0
                        // still sees the old window: half 1 commits only after EV_CONV)
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
                {   // ssm_out / attn_output (both K = 3072 per card) + the cross-card sum
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
                    // gate + up + SwiGLU + quantize in one GEMM: input aq, output aq2 (the down projection's input)
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
