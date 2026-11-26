// T-row int8 GEMV over the Q8_0 planes of quartz/repack.h, with the core and the
// row / slice walkers every decode phase is built from.
#pragma once

#include "kernels/dev.h"

// T-row GEMV core. acc[t][s] = row r0+s (clamped to N-1) . activation row t, with the
// result in lane 0. BUF selects buffer loads, the default for T <= QZ_GEMV_BUF_TMAX;
// plain global loads give the same bits.
#ifndef QZ_GEMV_BUF_TMAX
#define QZ_GEMV_BUF_TMAX 3
#endif
#define QZ_GEMV_INL __forceinline__
// RS > 0 takes rows r0 .. r0 + S/2 - 1 and then r0 + RS .. r0 + RS + S/2 - 1
template <int T, int S, int RS = 0, bool BUF = (T <= QZ_GEMV_BUF_TMAX)> __device__ QZ_GEMV_INL void qz_gemv_core(
        const unsigned char * a_vals, const float * a_scal, int lda, int lsa, int n_u,
        const char * w_vals, const __half * w_scal, int ldw, int lsw, int N, int r0, int lane, float (&acc)[T][S]) {
    #pragma unroll
    for (int t = 0; t < T; t++)
        #pragma unroll
        for (int s = 0; s < S; s++) acc[t][s] = 0.f;
    auto compute = [&](int u, const qz_v4u (&w0)[S], const qz_v4u (&w1)[S], const float (&sw)[S]) {
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
    };
    if constexpr (!BUF) {
        for (int u = lane; u < n_u; u += 64) {
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
            compute(u, w0, w1, sw);
        }
    } else {
        const int rb = __builtin_amdgcn_readfirstlane(r0);      // wave-uniform: descriptors in SGPRs
        __amdgpu_buffer_rsrc_t rv[S], rs[S];
        #pragma unroll
        for (int s = 0; s < S; s++) {
            const int r = min(rb + s + (RS && s >= S / 2 ? RS - S / 2 : 0), N - 1);
            rv[s] = __builtin_amdgcn_make_buffer_rsrc((void *)(w_vals + (size_t) r * ldw), (short) 0, n_u * 32, 0x00027000);
            rs[s] = __builtin_amdgcn_make_buffer_rsrc((void *)(w_scal + (size_t) r * lsw), (short) 0, n_u * 2, 0x00027000);
        }
        for (int u = lane; u < n_u; u += 64) {
            qz_v4u w0[S], w1[S];
            float sw[S];
            #pragma unroll
            for (int s = 0; s < S; s++) {
                w0[s] = __builtin_bit_cast(qz_v4u, __builtin_amdgcn_raw_buffer_load_b128(rv[s], u * 32, 0, 0));
                w1[s] = __builtin_bit_cast(qz_v4u, __builtin_amdgcn_raw_buffer_load_b128(rv[s], u * 32 + 16, 0, 0));
                sw[s] = __half2float(__ushort_as_half(__builtin_amdgcn_raw_buffer_load_b16(rs[s], u * 2, 0, 0)));
            }
            compute(u, w0, w1, sw);
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++)
        #pragma unroll
        for (int s = 0; s < S; s++)
#ifdef QZ_GEMV_WSUM
            QZ_GEMV_WSUM(acc[t][s]);          // an arch's equivalent wave sum, same lane 0 value
#else
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) acc[t][s] += __shfl_down(acc[t][s], off, 64);
#endif
}

template <int T, int S, int RS = 0> __device__ __forceinline__ void qz_gemv_rows_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, int N, int r0, int lane, float (&acc)[T][S]) {
    qz_gemv_core<T, S, RS>(a_vals, a_scal, K, K / 32, K / 32, w_vals, w_scal, K, K / 32, N, r0, lane, acc);
}

// T-row phase GEMV into y, which is [T][N]. off rotates the slab walk
template <int T> __device__ __forceinline__ void qz_gemv_slice_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, int N, float * y, int off = 0) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int S = 4;
    int g0 = (int) blockIdx.x - off;
    while (g0 < 0) g0 += gridDim.x;      // off may pass the grid when it is under 60
    for (int g = g0; g * (8 * S) < N; g += gridDim.x) {
        const int r0 = g * (8 * S) + wave * S;
        float acc[T][S];
        qz_gemv_rows_t<T, S>(a_vals, a_scal, K, w_vals, w_scal, N, r0, lane, acc);
        if (lane == 0)
            #pragma unroll
            for (int t = 0; t < T; t++)
                #pragma unroll
                for (int s = 0; s < S; s++)
                    if (r0 + s < N) y[(size_t) t * N + r0 + s] = acc[t][s];
    }
}

// qz_gemv_rows_t over n_uh units of a K slice, with pointers at the slice start
template <int T, int S> __device__ __forceinline__ void qz_gemv_rows_tk(
        const unsigned char * a_vals, const float * a_scal, int lda, int n_uh,
        const char * w_vals, const __half * w_scal, int ldw, int N, int r0, int lane, float (&acc)[T][S]) {
    qz_gemv_core<T, S>(a_vals, a_scal, lda, lda / 32, n_uh, w_vals, w_scal, ldw, ldw / 32, N, r0, lane, acc);
}
