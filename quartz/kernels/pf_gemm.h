// Prefill GEMM over int8 row tiles (W . A with per-32 scales) and the row norm /
// quant kernels that feed it. Kernels here are static, so every arch TU gets its own
// copy. Rows are padded to PF_MT tokens and pf_addnorm_quant<DM> wants DM % 256 == 0.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

constexpr int PF_MT = 128;
constexpr int PF_GG = 4;    // token tiles per pf_gemm dispatch group

// ---------------------------------------------------------------- GEMM ----
// C[m][n] = A[m] . W[n] with per-32 scales, in blocks of 64 rows x 128 tokens
typedef unsigned int pf_u32x4 __attribute__((ext_vector_type(4)));
struct pf_xrow { pf_u32x4 q[4][2]; pf_u32x4 pad; };

// SWI = 1 runs the gate (W) and the up (W2) in one GEMM, and the epilogue writes
// the quantized silu(g) * u for the down projection.
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
    // Grouped dispatch order: consecutive blocks cover PF_GG token tiles across
    // successive row blocks, so blocks in flight share weights and X tiles in L2.
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
        wrow = (unsigned) min(row0 + t / 4, N - 1);   // rows past N clamp, their outputs drop
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
        // two passes of 64 tokens through the X tile's LDS, as sAcc[row][token]
        float (*sAcc)[65] = (float (*)[65]) &sX[0];
        static_assert(sizeof(sX) >= 64 * 65 * 4, "epilogue tile");
#pragma unroll
        for (int n = 0; n < 2; n++) {
            __syncthreads();
#pragma unroll
            for (int g = 0; g < 16; g++) sAcc[ty * 16 + g][tx] = acc[g][n];
            __syncthreads();
            // token tt and dims part * 8 .. + 7, amax over the 32 by a 4-lane shuffle
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
            const float id = qz_q8_inv(d);
            unsigned w2[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int q = qz_q8_code(h[j], id);
                w2[j / 4] |= ((unsigned) q & 0xffu) << (8 * (j % 4));
            }
            const int ff = bx * 32 + part * 8;
            const unsigned g = (unsigned) (ff / 128), sub = (unsigned) (ff % 128);
            *(uint2 *) (Qo + ((size_t) g * Mp + m) * 128 + sub) = make_uint2(w2[0], w2[1]);
            if (part == 0) Qs[((size_t) g * Mp + m) * 4 + sub / 32] = qz_q8_scale(d);
        }
        return;
    }
    if (obf16) {   // row-parallel partials for the cross-card sum, bf16 rounded to
                   // nearest even like llama's all-reduce. C2 takes the same 32 B straight
                   // into the peer card's memory, overlapped with the GEMM
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
            if (C2) {   // the f32 partial also goes straight into the peer card's memory
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
// int8 with one f16-rounded scale per 32 values, laid out [K/128][Mp][128]
__device__ __forceinline__ void pf_quant_unit(float v, int k, int m, int Mp, int8_t * Ag, float * sag, int lane32) {
    float ax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ax = fmaxf(ax, __shfl_xor(ax, off, 32));
    const float d = ax / 127.f;
    const float id = qz_q8_inv(d);
    int q = qz_q8_code(v, id);
    const unsigned g = (unsigned) (k / 128), sub = (unsigned) (k % 128);
    Ag[((size_t) g * Mp + m) * 128 + sub] = (int8_t) q;
    if (lane32 == 0) sag[((size_t) g * Mp + m) * 4 + sub / 32] = qz_q8_scale(d);
}

// one block per row: y = x * rsqrt(mean(x^2) + eps) * w, then quantized, and
// optionally also kept in f32 for the MTP hidden input
static __global__ void __launch_bounds__(256) pf_norm_quant(
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

// the cross-card sum x += p0 + p1 over bf16 partials, fused into the next RMS norm
__device__ __forceinline__ float pf_ld_part(const unsigned short * p, size_t i) { return __uint_as_float((unsigned) p[i] << 16); }
template <int DM>
__global__ void __launch_bounds__(256) pf_addnorm_quant(
        float * __restrict__ x, const unsigned short * __restrict__ p0, const unsigned short * __restrict__ p1, int n,
        const float * __restrict__ w, float eps, int8_t * Ag, float * sag, int Mp, float * outf, int m0 = 0) {
    __shared__ float red[8];
    const int m = m0 + blockIdx.x, t = threadIdx.x;
    float * xr = x + (size_t) m * DM;
    constexpr int NE = DM / 256;
    float xv[NE];                     // the row stays in registers across add and norm
    float ss = 0.f;
    if (m < n) {
        const unsigned short * a = p0 + (size_t) m * DM, * b = p1 + (size_t) m * DM;
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
    const float r = 1.f / sqrtf((red[0] + red[1] + red[2] + red[3]) / DM + eps);
#pragma unroll
    for (int j = 0; j < NE; j++) {
        const int k = t + 256 * j;
        const float y = xv[j] * r * w[k];
        if (outf) outf[(size_t) m * DM + k] = y;
        if (Ag) pf_quant_unit(y, k, m, Mp, Ag, sag, t & 31);
    }
}

// quantizes rows of y, which has row stride ldy
static __global__ void __launch_bounds__(256) pf_quant(const float * __restrict__ y, int ldy, int K, int8_t * Ag, float * sag, int Mp, int m0 = 0) {
    const int m = m0 + blockIdx.x;
    for (int k = threadIdx.x; k < K; k += 256) pf_quant_unit(y[(size_t) m * ldy + k], k, m, Mp, Ag, sag, threadIdx.x & 31);
}
