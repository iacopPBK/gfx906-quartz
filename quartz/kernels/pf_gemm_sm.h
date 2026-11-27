// Small-M prefill GEMM: pf_gemm's int8 x int8 tile with per-32 scales, narrowed in the
// token direction, so a short prompt stops paying for 128 padded rows.
//
// Same arithmetic as pf_gemm and therefore bit-identical: one int32 dot per 32 K values
// and one fma into fp32, over K in increasing order. What changes is the tile and the
// thread mapping. A wave (64 lanes) splits into RG = 64 / JT row groups of JT token
// lanes, so every lane still covers one token and RR = RW / (4 * RG) weight rows, and a
// block covers RW rows x JT tokens instead of 64 x 64 (or 64 x 128).
//
// JT = tokens per block (8, 16, 32, 64), RW = weight rows per block, CH = 32-value
// K blocks staged per LDS round. RW * JT / 256 = rows per thread, so the small tiles
// that keep the CUs fed at M < 64 also keep a handful of loads in flight per thread.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "kernels/pf_gemm.h"

template <int CH> struct pf_xtile { pf_u32x4 q[CH][2]; pf_u32x4 pad; };   // CH k blocks of one token, padded

template <int SWI, int JT, int RW, int CH>
__global__ void __launch_bounds__(256, 2) pf_gemm_sm(
        const int8_t * __restrict__ Ag, const float * __restrict__ sag, int Mp,
        const char * __restrict__ W, const __half * __restrict__ sw, int ldw, int K,
        float * __restrict__ C, int ldc, int N, int ty0, int obf16, unsigned short * __restrict__ C2 = nullptr,
        const char * __restrict__ W2 = nullptr, const __half * __restrict__ sw2 = nullptr,
        int8_t * __restrict__ Qo = nullptr, float * __restrict__ Qs = nullptr) {
    constexpr int RG = 64 / JT;                // row groups per wave
    constexpr int RR = RW / (4 * RG);          // weight rows per thread
    constexpr int SW = RW * CH > 256 ? RW * CH / 256 : 1;   // weight slots per thread
    constexpr int SX = JT * CH > 256 ? JT * CH / 256 : 1;   // activation slots per thread
    static_assert(JT == 8 || JT == 16 || JT == 32 || JT == 64, "token tile: 8, 16, 32 or 64");
    static_assert(RW == 8 || RW == 16 || RW == 32 || RW == 64, "row tile: 8, 16, 32 or 64");
    static_assert((CH == 4 || CH == 8) && RR >= 1 && 4 * RG * RR == RW, "tile shape");
    static_assert(!SWI || (RW == 64 && 32 % RR == 0), "the fused gate/up epilogue wants a 64-row tile");
    __shared__ union {
        struct {
            pf_u32x4 wlo[RW][CH], whi[RW][CH];
            unsigned short wd[CH][RW];
            pf_xtile<CH> x[JT];
            float xd[JT][CH];
        } g;
        float a[64][JT + 1];                   // fused gate/up epilogue tile
    } sm;
    const int t = threadIdx.x, tx = t % JT, g = (t / JT) % RG, ty = t / 64;
    const int row0 = blockIdx.x * RW, tok0 = ty0 * 128 + blockIdx.y * JT;
    const int nkb = K / 32, rb = (ty * RG + g) * RR;
    // weight slots: RW * CH of them, thread t taking t, t + 256, ... in row-major order.
    // With SWI rows 0..31 hold the gate (W) and rows 32..63 the up (W2).
    const char * Wp[SW];
    const __half * swp[SW];
    unsigned w_off[SW], wd_off[SW];
    bool wok[SW];
#pragma unroll
    for (int i = 0; i < SW; i++) {
        const int e = t + 256 * i, r = e / CH;
        wok[i] = e < RW * CH;
        const int rr_ = wok[i] ? r : 0;
        Wp[i] = SWI && rr_ >= 32 ? W2 : W;
        swp[i] = SWI && rr_ >= 32 ? sw2 : sw;
        const unsigned row = SWI ? (unsigned) (blockIdx.x * 32 + (rr_ & 31)) : (unsigned) min(row0 + rr_, N - 1);
        w_off[i] = row * (unsigned) ldw;
        wd_off[i] = row * (unsigned) nkb;
    }
    // activation slots: JT * CH of them, thread t taking t, t + 256, ...
    unsigned x_off[SX], xd_off[SX];
    bool xok[SX];
#pragma unroll
    for (int i = 0; i < SX; i++) {
        const int e = t + 256 * i, tok = e / CH, s = e % CH;
        xok[i] = e < JT * CH;
        x_off[i] = (unsigned) (tok0 + tok) * 128u + (unsigned) (s % 4) * 32u;
        xd_off[i] = (unsigned) (tok0 + tok) * 4u + (unsigned) (s % 4);
    }
    pf_u32x4 plo[SW], phi[SW];
    unsigned short pdl[SW];
    pf_u32x4 pxa[SX], pxb[SX];
    float pxd[SX];
    // kb0 is a multiple of CH and even of 4, so a slot's 128-value group is fixed
    auto prefetch = [&](int kb0) {
#pragma unroll
        for (int i = 0; i < SW; i++) {
            if (wok[i]) {
                const int off = (t + 256 * i) % CH;
                const pf_u32x4 * ws = (const pf_u32x4 *) (Wp[i] + w_off[i] + (unsigned) (kb0 + off) * 32u);
                plo[i] = ws[0]; phi[i] = ws[1];
                pdl[i] = ((const unsigned short *) swp[i])[wd_off[i] + (unsigned) (kb0 + off)];
            }
        }
#pragma unroll
        for (int i = 0; i < SX; i++) {
            if (xok[i]) {
                const int off = (t + 256 * i) % CH;
                const unsigned grp = (unsigned) ((kb0 + off) / 4) * (unsigned) Mp;
                const pf_u32x4 * xs = (const pf_u32x4 *) (Ag + grp * 128u + x_off[i]);
                pxa[i] = xs[0]; pxb[i] = xs[1];
                pxd[i] = sag[grp * 4u + xd_off[i]];
            }
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int i = 0; i < SW; i++) {
            if (!wok[i]) continue;
            const int e = t + 256 * i, r = e / CH, s = e % CH;
            sm.g.wlo[r][s] = plo[i]; sm.g.whi[r][s] = phi[i]; sm.g.wd[s][r] = pdl[i];
        }
#pragma unroll
        for (int i = 0; i < SX; i++) {
            if (!xok[i]) continue;
            const int e = t + 256 * i, tok = e / CH, s = e % CH;
            sm.g.x[tok].q[s][0] = pxa[i]; sm.g.x[tok].q[s][1] = pxb[i]; sm.g.xd[tok][s] = pxd[i];
        }
    };
    float acc[RR];
#pragma unroll
    for (int r = 0; r < RR; r++) acc[r] = 0.f;
    prefetch(0);
    for (int kb0 = 0; kb0 < nkb; kb0 += CH) {
        store();
        __syncthreads();
        if (kb0 + CH < nkb) prefetch(kb0 + CH);
#pragma unroll 1
        for (int kk = 0; kk < CH; kk++) {
            int xq[8];
            float dx;
            {
                const pf_u32x4 q0 = sm.g.x[tx].q[kk][0], q1 = sm.g.x[tx].q[kk][1];
                xq[0] = q0.x; xq[1] = q0.y; xq[2] = q0.z; xq[3] = q0.w;
                xq[4] = q1.x; xq[5] = q1.y; xq[6] = q1.z; xq[7] = q1.w;
                dx = sm.g.xd[tx][kk];
            }
            pf_u32x4 wlo = sm.g.wlo[rb][kk], whi = sm.g.whi[rb][kk];
#pragma unroll
            for (int r = 0; r < RR; r++) {
                const pf_u32x4 lo = wlo, hi = whi;
                if (r + 1 < RR) { wlo = sm.g.wlo[rb + r + 1][kk]; whi = sm.g.whi[rb + r + 1][kk]; }
                const float d = __half2float(__ushort_as_half(sm.g.wd[kk][rb + r]));
                int isum = 0;
                isum = __builtin_amdgcn_sdot4(xq[0], (int) lo.x, isum, false);
                isum = __builtin_amdgcn_sdot4(xq[1], (int) lo.y, isum, false);
                isum = __builtin_amdgcn_sdot4(xq[2], (int) lo.z, isum, false);
                isum = __builtin_amdgcn_sdot4(xq[3], (int) lo.w, isum, false);
                isum = __builtin_amdgcn_sdot4(xq[4], (int) hi.x, isum, false);
                isum = __builtin_amdgcn_sdot4(xq[5], (int) hi.y, isum, false);
                isum = __builtin_amdgcn_sdot4(xq[6], (int) hi.z, isum, false);
                isum = __builtin_amdgcn_sdot4(xq[7], (int) hi.w, isum, false);
                acc[r] = __builtin_fmaf((float) isum, dx * d, acc[r]);
            }
        }
        __syncthreads();
    }
    if (SWI) {
        // gate rows 0..31 and up rows 32..63, quantized per 32 values over tokens: one
        // thread per (token, 8 dims), amax over the 32 by a 4-lane shuffle
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RR; r++) sm.a[rb + r][tx] = acc[r];
        __syncthreads();
        if (t < JT * 4) {
            const int tt = t >> 2, part = t & 3, m = tok0 + tt;
            float h[8], ax = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float a = sm.a[part * 8 + j][tt], u = sm.a[32 + part * 8 + j][tt];
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
                const int q = qz_q8_code(h[j], id);
                w2[j / 4] |= ((unsigned) q & 0xffu) << (8 * (j % 4));
            }
            const int ff = blockIdx.x * 32 + part * 8;
            const unsigned gg = (unsigned) (ff / 128), sub = (unsigned) (ff % 128);
            *(uint2 *) (Qo + ((size_t) gg * Mp + m) * 128 + sub) = make_uint2(w2[0], w2[1]);
            if (part == 0) Qs[((size_t) gg * Mp + m) * 4 + sub / 32] = qz_q8_scale(d);
        }
        return;
    }
    if (obf16) {
#pragma unroll
        for (int r = 0; r < RR; r++) {
            const int row = row0 + rb + r;
            if (row >= N) continue;
            const unsigned u = __float_as_uint(acc[r]);
            const unsigned short b = (unsigned short) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
            const size_t off = (size_t) (tok0 + tx) * ldc + row;
            ((unsigned short *) C)[off] = b;
            if (C2) ((unsigned short *) C2)[off] = b;
        }
        return;
    }
#pragma unroll
    for (int r = 0; r < RR; r++) {
        const int row = row0 + rb + r;
        if (row >= N) continue;
        const size_t off = (size_t) (tok0 + tx) * ldc + row;
        C[off] = acc[r];
        if (C2) C2[off] = acc[r];
    }
}

// Launch the small-M kernel with a fixed tile. rows = valid token rows of this call.
template <int SWI, int JT, int RW, int CH> inline void pf_gemm_sm_do(int rows, hipStream_t st,
        const int8_t * Ag, const float * sag, int Mp,
        const char * W, const __half * sw, int ldw, int K,
        float * C, int ldc, int N, int ty0, int obf16, unsigned short * C2 = nullptr,
        const char * W2 = nullptr, const __half * sw2 = nullptr,
        int8_t * Qo = nullptr, float * Qs = nullptr) {
    constexpr int GX = SWI ? 32 : RW;   // weight rows per block of the caller's grid
    hipLaunchKernelGGL((pf_gemm_sm<SWI, JT, RW, CH>), dim3((N + GX - 1) / GX, (rows + JT - 1) / JT), dim3(256), 0, st,
                       Ag, sag, Mp, W, sw, ldw, K, C, ldc, N, ty0, obf16, C2, W2, sw2, Qo, Qs);
}

// Pick the tile from the call's rows and fall back to pf_gemm when they do not fit.
// The tile table is the one measured in REPORT.md: JT tracks the call's rows, RW gives
// at least ~2 blocks per CU, CH keeps the LDS tile small enough for two blocks.
template <int SWI> inline void pf_gemm_sm_go(int ntiles, int rows, dim3 grid, hipStream_t st,
        const int8_t * Ag, const float * sag, int Mp,
        const char * W, const __half * sw, int ldw, int K,
        float * C, int ldc, int N, int ty0, int obf16, unsigned short * C2 = nullptr,
        const char * W2 = nullptr, const __half * sw2 = nullptr,
        int8_t * Qo = nullptr, float * Qs = nullptr) {
    // Tile table measured in REPORT.md (M <= 32 on every shape, M <= 64 only for the
    // narrow matrices, where the card has few blocks per launch).
    const bool ch8 = (K & 255) == 0, wide = N >= 1024, narrow = N <= 1024;
    const bool small = rows > 0 && (K & 127) == 0 &&
                       (rows <= 32 || (rows <= 64 && narrow));
    if (small) {
#define PF_SM(JTv, RWv, CHv) pf_gemm_sm_do<SWI, JTv, RWv, CHv>(rows, st, Ag, sag, Mp, W, sw, ldw, K, C, ldc, N, ty0, obf16, C2, W2, sw2, Qo, Qs)
        if constexpr (SWI) {
            if (rows <= 8) { if (ch8) PF_SM(8, 64, 8); else PF_SM(8, 64, 4); }
            else if (rows <= 16) { if (ch8) PF_SM(16, 64, 8); else PF_SM(16, 64, 4); }
            else if (rows <= 32) { if (ch8) PF_SM(32, 64, 8); else PF_SM(32, 64, 4); }
            else { if (ch8) PF_SM(32, 64, 8); else PF_SM(32, 64, 4); }
        } else {
            if (rows <= 8) { if (wide) PF_SM(8, 64, 8); else PF_SM(8, 32, 8); }
            else if (rows <= 16) PF_SM(16, 16, 4);
            else if (rows <= 32) { if (wide) { if (ch8) PF_SM(32, 64, 8); else PF_SM(32, 64, 4); }
                                   else { if (ch8) PF_SM(32, 16, 8); else PF_SM(32, 16, 4); } }
            else { if (ch8) PF_SM(32, 16, 8); else PF_SM(32, 16, 4); }
        }
#undef PF_SM
        return;
    }
    pf_gemm_go<SWI>(ntiles, rows, grid, st, Ag, sag, Mp, W, sw, ldw, K, C, ldc, N, ty0, obf16, C2, W2, sw2, Qo, Qs);
}
