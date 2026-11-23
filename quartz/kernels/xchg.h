// TP2 exchange between the two cards: per-block flags, the row-parallel
// GEMV push and the combine. XA is the arch's args struct; it carries the
// exchange pointers and its width as XA::DM (a multiple of 32).
#pragma once

#include "kernels/gemv.h"

// ---- TP2 exchange ----------------------------------------------
// push partials to the peer inbox (slot tok & 1), wait for its flag, y = p0 + p1
#define QZ_XSPIN 2000000000ll     // clock64 ticks before a wait gives up

template <class XA> __device__ __forceinline__ void qz_xflag(const XA & A, unsigned int tok) {
    __threadfence_system();
    __hip_atomic_store(A.xflag_peer, tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// per-block exchange flags: block b only needs the peer's block b push (slot 0
// doubles as the head-merge flag)
template <class XA> __device__ __forceinline__ void qz_xflag_b(const XA & A, unsigned int tok, int b) {
    __threadfence_system();
    __hip_atomic_store(A.xflag_peer + b, tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
template <class XA> __device__ __forceinline__ void qz_xwait_b(const XA & A, unsigned int tok, int b) {
    const long long t0 = clock64();
    for (;;) {
        const unsigned int v = __hip_atomic_load(A.xflag_mine + b, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if ((int) (v - tok) >= 0) return;
        if (__hip_atomic_load(&A.xerr[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) return;
        if (clock64() - t0 > QZ_XSPIN) {
            if (atomicCAS(&A.xerr[1], 0, 1) == 0) { A.xerr[2] = 1; A.xerr[3] = (int) tok; A.xerr[4] = (int) v; }
            return;
        }
    }
}

// one thread: wait for the peer's flag >= tok. Past the spin limit, or once
// any wait gave up, set the abort word and return so the launch drains.
template <class XA> __device__ __forceinline__ void qz_xwait(const XA & A, unsigned int tok) {
    const long long t0 = clock64();
    for (;;) {
        const unsigned int v = __hip_atomic_load(A.xflag_mine, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if ((int) (v - tok) >= 0) return;
        if (__hip_atomic_load(&A.xerr[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) return;
        if (clock64() - t0 > QZ_XSPIN) {
            if (atomicCAS(&A.xerr[1], 0, 1) == 0) { A.xerr[2] = 1; A.xerr[3] = (int) tok; A.xerr[4] = (int) v; }
            return;
        }
    }
}

// row-parallel GEMV over the card's K slice: the block's slabs stage in LDS and
// push after the loop (one fence); activations optionally staged in LDS region 0
// CX: the exchange buffers' row capacity (the cycle's rows, or the verify's);
// LX: LDS activation staging (off for the verify: it does not fit)
template <int T, int CX, bool LX = T <= 4, class XA> __device__ __forceinline__ void qz_gemv_xpush_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, const XA & A, unsigned int tok, float * stage,
        unsigned char * lds0) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tid = threadIdx.x;
    const int S = 4;
    const int par = tok & 1;
    if constexpr (LX) {
        unsigned char * lv = lds0;
        float * ls = (float *)(lds0 + (size_t) T * K);
        __syncthreads();
        for (int i = tid; i < T * K / 16; i += blockDim.x) ((uint4 *) lv)[i] = ((const uint4 *) a_vals)[i];
        for (int i = tid; i < T * (K / 32); i += blockDim.x) ls[i] = a_scal[i];
        __syncthreads();
        a_vals = lv;
        a_scal = ls;
    } else {
        (void) lds0;
    }
    float * dst = A.xin_peer + (size_t) par * CX * XA::DM;
    int ns = 0;
    for (int g = blockIdx.x; g * 32 < XA::DM; g += gridDim.x, ns++) {
        const int r0 = g * 32 + wave * S;
        float acc[T][S];
        qz_gemv_rows_tk<T, S>(a_vals, a_scal, K, K / 32, w_vals, w_scal, K, XA::DM, r0, lane, acc);
        if (lane == 0)
            #pragma unroll
            for (int t = 0; t < T; t++)
                #pragma unroll
                for (int s = 0; s < S; s++) stage[(ns * T + t) * 32 + wave * S + s] = acc[t][s];
    }
    __syncthreads();
    for (int i = tid; i < ns * 8 * T; i += blockDim.x) {
        const int k = i / (8 * T), t = (i >> 3) % T, q = i & 7;
        const int g = blockIdx.x + k * gridDim.x;
        const float4 v = *(const float4 *)(stage + (k * T + t) * 32 + q * 4);
        const size_t o = (size_t) t * XA::DM + g * 32 + q * 4;
        *(float4 *)(A.y_part + o) = v;
        *(float4 *)(dst + o) = v;
    }
    __threadfence_system();
}

// after qz_gemv_xpush_t: flag the peer, wait for its flag, y = p0 + p1 over this
// block's slabs; BX = false drops the trailing grid barrier (standalone launch)
template <int T, int CX, bool BX = true, class XA> __device__ __forceinline__ void qz_xfinish_t(
        qz_barrier_state * bst, int & kb, const XA & A, unsigned int tok, float * y) {
    const int tid = threadIdx.x;
    const int par = tok & 1;
    __syncthreads();                 // every thread's remote stores are fenced (push)
    if (tid == 0) { qz_xflag_b(A, tok, blockIdx.x); qz_xwait_b(A, tok, blockIdx.x); }
    __syncthreads();
    const float * src = A.xin_mine + (size_t) par * CX * XA::DM;
    for (int g = blockIdx.x; g * 32 < XA::DM; g += gridDim.x) {
        if (tid < 32 * T) {
            const int t = tid >> 5, i = tid & 31;
            const size_t o = (size_t) t * XA::DM + g * 32 + i;
            const float own = __builtin_nontemporal_load(A.y_part + o);
            const float peer = __builtin_nontemporal_load(src + o);
            y[o] = A.tp_card == 0 ? __fadd_rn(own, peer) : __fadd_rn(peer, own);
        }
    }
    if constexpr (BX) qz_grid_barrier(bst, ++kb);
}

// one row-parallel phase with its exchange: gemv + push, then flag, wait
// and combine
template <int T, int CX, bool LX = T <= 4, bool BX = true, class XA> __device__ __forceinline__ void qz_xrow_t(
        qz_barrier_state * bst, int & kb, const XA & A, unsigned int tok,
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, float * stage, float * y, unsigned char * lds0) {
    qz_gemv_xpush_t<T, CX, LX>(a_vals, a_scal, K, w_vals, w_scal, A, tok, stage, lds0);
    // BX forwarded: a standalone launch has no barrier state
    qz_xfinish_t<T, CX, BX>(bst, kb, A, tok, y);
}

// head merge: larger value wins, ties to the lower token id
__device__ __forceinline__ int qz_xpick(float ov, int oi, float pv, int pi) {
    return (pv > ov || (pv == ov && pi < oi)) ? pi : oi;
}
