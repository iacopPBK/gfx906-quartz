// T-row cycle kernel: one persistent launch runs T rows through the trunk and
// lm_head and verifies on device; recurrent state commits lazily via a replay ring.
#pragma once
#include "flash9c.h"

// per-card geometry (tp2.h); TR: the replay ring's row capacity
template <int TR = QZ_TMAX> struct qz_geo {
    static constexpr int HV = QZ_HV / 2, HK = QZ_HK / 2, CCH = QZ_CCH / 2, G = QZ_G / 2, AQ = QZ_AQ / 2,
                         AKV = QZ_AKV / 2, HQ = QZ_HQ / 2, HKV = QZ_HKV / 2, FF = QZ_FF / 2;
    static constexpr int RING_CONV = 0, RING_QKV = TR * CCH, RING_BETA = 2 * TR * CCH,
                         RING_ALPH = 2 * TR * CCH + TR * HV, RING_PAR = 2 * TR * CCH + 2 * TR * HV;
};

// qz_dn_rows: one head's DN phase, n_rep replay rows from rp then T new rows from
// rc; only the state recurrence is serial (n_rep <= CAP, the ring's capacity)
template <int T, int CAP = QZ_TMAX> struct qz_dn_rows_lds {
    float q[CAP + T][QZ_S];
    float k[CAP + T][QZ_S];
    float v[CAP + T][QZ_S];
    float qr[CAP + T][4];         // l2 partials: q half 0/1, k half 0/1
    float inv[CAP + T][2];        // q / k inverse norms
    float beta[CAP + T], dec[CAP + T];
    float part[4][QZ_S];
    float d[QZ_S];
    float opb[4][QZ_S];         // per-band o partials of the last row (summed during the next row)
    float op[T][QZ_S];          // o of the new rows (band sum, same order)
    float orr[T][2];            // rms partials of the new rows
};

// TR: the ring capacity rc / rp index into (QZ_TMAX, or 12 for the verify)
template <int T, int CAP = QZ_TMAX, int TR = QZ_TMAX> __device__ __forceinline__ void qz_dn_rows(
        float * st, int n_rep, const float * rp, const float * rc, const float * y_gate,
        const float * dt_bias, const float * ssm_a, const float * ssm_norm_w,
        unsigned char * pl_vals, float * pl_scal, int h, float eps,
        qz_dn_rows_lds<T, CAP> * L, int tid) {
    using Gm = qz_geo<TR>;
    const int kh = h % Gm::HK;
    const int nr = n_rep + T;                       // rows, replay first
    const int wave = tid >> 6, lane = tid & 63;
    const float epsn = eps * (1.f / 128.f);
    // prep: unit = (row, q/k, half), one wave each; v rows and beta/decay
    for (int u = wave; u < nr * 4; u += 8) {
        const int b = u >> 2, w = u & 3;
        const float * src = (b < n_rep ? rp + Gm::RING_CONV + (size_t) b * Gm::CCH : rc + Gm::RING_CONV + (size_t)(b - n_rep) * Gm::CCH);
        const int e = (w & 1) * 64 + lane;
        float x;
        if (w < 2) { x = src[kh * QZ_S + e]; L->q[b][e] = x; }
        else       { x = src[QZ_S * Gm::HK + kh * QZ_S + e]; L->k[b][e] = x; }
        float sq;
        {   // unfused square + shuffle add, as the step's branch-guarded tree
            #pragma clang fp contract(off)
            sq = x * x;
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) sq += __shfl_down(sq, off, 64);
        if (lane == 0) L->qr[b][w] = sq;
    }
    for (int i = tid; i < nr * QZ_S; i += 512) {
        const int b = i >> 7, j = i & (QZ_S - 1);
        const float * src = (b < n_rep ? rp + Gm::RING_CONV + (size_t) b * Gm::CCH : rc + Gm::RING_CONV + (size_t)(b - n_rep) * Gm::CCH);
        L->v[b][j] = src[2 * QZ_S * Gm::HK + h * QZ_S + j];
    }
    if (tid < nr) {
        const int b = tid;
        const float * pb = b < n_rep ? rp + Gm::RING_BETA + (size_t) b * Gm::HV
                                     : rc + Gm::RING_BETA + (size_t)(b - n_rep) * Gm::HV;
        const float * pa = b < n_rep ? rp + Gm::RING_ALPH + (size_t) b * Gm::HV
                                     : rc + Gm::RING_ALPH + (size_t)(b - n_rep) * Gm::HV;
        L->beta[b] = 1.f / (1.f + expf(-pb[h]));
        const float a0 = pa[h] + dt_bias[h];
        const float sp = a0 > 20.f ? a0 : logf(1.f + expf(a0));            // softplus
        L->dec[b] = expf(sp * ssm_a[h]);
    }
    __syncthreads();
    if (tid < nr) {
        const int b = tid;
        const float l2i = 1.f / sqrtf(128.f);
        L->inv[b][0] = 1.f / sqrtf((L->qr[b][0] + L->qr[b][1]) * (1.f / 128.f) + epsn) * l2i;
        L->inv[b][1] = 1.f / sqrtf((L->qr[b][2] + L->qr[b][3]) * (1.f / 128.f) + epsn) * l2i;
    }
    __syncthreads();
    for (int i = tid; i < nr * QZ_S; i += 512) {
        const int b = i >> 7, j = i & (QZ_S - 1);
        L->q[b][j] = L->q[b][j] * L->inv[b][0] * (1.f / sqrtf(128.f));  // + q scale
        L->k[b][j] = L->k[b][j] * L->inv[b][1];
    }
    __syncthreads();

    // serial recurrence, state strip in registers
    const int j = tid & (QZ_S - 1), r = tid >> 7;
    float s[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) s[i] = st[i * QZ_S];
    for (int b = 0; b < nr; b++) {
        const float dec = L->dec[b], beta_h = L->beta[b];
        #pragma unroll
        for (int i = 0; i < 32; i++) s[i] = s[i] * dec;
        float p = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; i++) p += s[i] * L->k[b][r * 32 + i];
        L->part[r][j] = p;
        __syncthreads();
        if (tid < QZ_S) {
            const float sk_j = L->part[0][j] + L->part[1][j] + L->part[2][j] + L->part[3][j];
            L->d[j] = (L->v[b][j] - sk_j) * beta_h;
            if (b > n_rep) L->op[b - 1 - n_rep][j] = L->opb[0][j] + L->opb[1][j] + L->opb[2][j] + L->opb[3][j];
        }
        __syncthreads();
        float p2 = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            s[i] += L->k[b][r * 32 + i] * L->d[j];
            p2 += s[i] * L->q[b][r * 32 + i];
        }
        if (b >= n_rep) L->opb[r][j] = p2;
        if (b == n_rep - 1)
            #pragma unroll
            for (int i = 0; i < 32; i++) st[i * QZ_S] = s[i];
    }
    __syncthreads();
    if (tid < QZ_S) L->op[T - 1][j] = L->opb[0][j] + L->opb[1][j] + L->opb[2][j] + L->opb[3][j];
    __syncthreads();

    // tails: wave pair per new row, element e = half * 64 + lane
    for (int w = wave; w < 2 * T; w += 8) {
        const int t = w >> 1, e = (w & 1) * 64 + lane;
        const float o = L->op[t][e];
        float so;
        {
            #pragma clang fp contract(off)
            so = o * o;
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) so += __shfl_down(so, off, 64);
        if (lane == 0) L->orr[t][w & 1] = so;
    }
    __syncthreads();
    for (int w = wave; w < 2 * T; w += 8) {
        const int t = w >> 1, e = (w & 1) * 64 + lane;
        const float inv2 = 1.f / sqrtf((L->orr[t][0] + L->orr[t][1]) * (1.f / 128.f) + eps);
        const float z_j = y_gate[(size_t) t * Gm::G + h * QZ_S + e];
        const float o = (L->op[t][e] * inv2) * ssm_norm_w[e] * (z_j / (1.f + expf(-z_j)));
        // head requant, 32-lane units, one element per lane
        const int u = e >> 5, l = e & 31;
        float ax = fabsf(o);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ax = fmaxf(ax, __shfl_down(ax, off, 32));
        ax = __shfl(ax, 0, 32);                    // per-unit amax to all lanes
        const float dq = ax / 127.f;
        const float id = dq != 0.f ? 1.f / dq : 0.f;
        const float d16 = __half2float(__float2half(dq));
        int q = (int) roundf(o * id);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        pl_vals[(size_t) t * Gm::G + (h * 4 + u) * 32 + l] = (unsigned char) q;
        if (l == 0) pl_scal[(size_t) t * (Gm::G / 32) + h * 4 + u] = d16;
    }
    __syncthreads();
}

// T-row GEMV core: acc[t][s] = row r0+s (clamped to N-1) . activation row t,
// in lane 0; buffer loads for T <= QZ_GEMV_BUF_TMAX (bit-identical either way)
#ifndef QZ_GEMV_BUF_TMAX
#define QZ_GEMV_BUF_TMAX 3
#endif
#define QZ_GEMV_INL __forceinline__
template <int T, int S> __device__ QZ_GEMV_INL void qz_gemv_core(
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
    if constexpr (T > QZ_GEMV_BUF_TMAX) {
        for (int u = lane; u < n_u; u += 64) {
            qz_v4u w0[S], w1[S];
            float sw[S];
            #pragma unroll
            for (int s = 0; s < S; s++) {
                const int r = min(r0 + s, N - 1);
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
            const int r = min(rb + s, N - 1);
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
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) acc[t][s] += __shfl_down(acc[t][s], off, 64);
}

template <int T, int S> __device__ __forceinline__ void qz_gemv_rows_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, int N, int r0, int lane, float (&acc)[T][S]) {
    qz_gemv_core<T, S>(a_vals, a_scal, K, K / 32, K / 32, w_vals, w_scal, K, K / 32, N, r0, lane, acc);
}

// T-row phase GEMV, y is [T][N]; off rotates the slab walk
template <int T> __device__ __forceinline__ void qz_gemv_slice_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, int N, float * y, int off = 0) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int S = 4;
    int g0 = (int) blockIdx.x - off;
    if (g0 < 0) g0 += gridDim.x;
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

// qz_gemv_rows_t over n_uh units of a K slice (pointers at the slice start)
template <int T, int S> __device__ __forceinline__ void qz_gemv_rows_tk(
        const unsigned char * a_vals, const float * a_scal, int lda, int n_uh,
        const char * w_vals, const __half * w_scal, int ldw, int N, int r0, int lane, float (&acc)[T][S]) {
    qz_gemv_core<T, S>(a_vals, a_scal, lda, lda / 32, n_uh, w_vals, w_scal, ldw, ldw / 32, N, r0, lane, acc);
}

// ---- TP2 exchange ----------------------------------------------
// push partials to the peer inbox (slot tok & 1), wait for its flag, y = p0 + p1
#define QZ_XSPIN 2000000000ll     // clock64 ticks before a wait gives up

__device__ __forceinline__ void qz_xflag(const qz_trunk_args & A, unsigned int tok) {
    __threadfence_system();
    __hip_atomic_store(A.xflag_peer, tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// per-block exchange flags: block b only needs the peer's block b push (slot 0
// doubles as the head-merge flag)
__device__ __forceinline__ void qz_xflag_b(const qz_trunk_args & A, unsigned int tok, int b) {
    __threadfence_system();
    __hip_atomic_store(A.xflag_peer + b, tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void qz_xwait_b(const qz_trunk_args & A, unsigned int tok, int b) {
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
__device__ __forceinline__ void qz_xwait(const qz_trunk_args & A, unsigned int tok) {
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
#define QZ_XSLAB 4                                   // slabs per block, grid >= 40
#define QZ_XSTAGE_BYTES (QZ_XSLAB * QZ_TMAX * 32 * 4)
// CX: the exchange buffers' row capacity (QZ_TMAX, or MV_T for the verify);
// LX: LDS activation staging (off for the verify: it does not fit)
template <int T, int CX = QZ_TMAX, bool LX = T <= 4> __device__ __forceinline__ void qz_gemv_xpush_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, const qz_trunk_args & A, unsigned int tok, float * stage,
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
    float * dst = A.xin_peer + (size_t) par * CX * QZ_D;
    int ns = 0;
    for (int g = blockIdx.x; g * 32 < QZ_D; g += gridDim.x, ns++) {
        const int r0 = g * 32 + wave * S;
        float acc[T][S];
        qz_gemv_rows_tk<T, S>(a_vals, a_scal, K, K / 32, w_vals, w_scal, K, QZ_D, r0, lane, acc);
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
        const size_t o = (size_t) t * QZ_D + g * 32 + q * 4;
        *(float4 *)(A.y_part + o) = v;
        *(float4 *)(dst + o) = v;
    }
    __threadfence_system();
}

// after qz_gemv_xpush_t: flag the peer, wait for its flag, y = p0 + p1 over this
// block's slabs; BX = false drops the trailing grid barrier (standalone launch)
template <int T, int CX = QZ_TMAX, bool BX = true> __device__ __forceinline__ void qz_xfinish_t(
        qz_barrier_state * bst, int & kb, const qz_trunk_args & A, unsigned int tok, float * y) {
    const int tid = threadIdx.x;
    const int par = tok & 1;
    __syncthreads();                 // every thread's remote stores are fenced (push)
    if (tid == 0) { qz_xflag_b(A, tok, blockIdx.x); qz_xwait_b(A, tok, blockIdx.x); }
    __syncthreads();
    const float * src = A.xin_mine + (size_t) par * CX * QZ_D;
    for (int g = blockIdx.x; g * 32 < QZ_D; g += gridDim.x) {
        if (tid < 32 * T) {
            const int t = tid >> 5, i = tid & 31;
            const size_t o = (size_t) t * QZ_D + g * 32 + i;
            const float own = __builtin_nontemporal_load(A.y_part + o);
            const float peer = __builtin_nontemporal_load(src + o);
            y[o] = A.tp_card == 0 ? __fadd_rn(own, peer) : __fadd_rn(peer, own);
        }
    }
    if constexpr (BX) qz_grid_barrier(bst, ++kb);
}

// one row-parallel phase with its exchange: gemv + push, then flag, wait
// and combine
template <int T, int CX = QZ_TMAX, bool LX = T <= 4, bool BX = true> __device__ __forceinline__ void qz_xrow_t(
        qz_barrier_state * bst, int & kb, const qz_trunk_args & A, unsigned int tok,
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

// T-row qkv GEMV + conv1d + silu: the window commits the previous cycle's
// n_rep rows, then the T new rows run without committing
template <int T> __device__ __forceinline__ void qz_gemv_qkv_conv_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, int N, float * y_qkv,
        float * conv_state, const float * conv1d, float * conv_out,
        const float * prev_qkv, int n_rep) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int S = 4;
    for (int g = blockIdx.x; g * (8 * S) < N; g += gridDim.x) {
        const int r0 = g * (8 * S) + wave * S;
        float acc[T][S];
        qz_gemv_rows_t<T, S>(a_vals, a_scal, K, w_vals, w_scal, N, r0, lane, acc);
        float cur[T];
        #pragma unroll
        for (int t = 0; t < T; t++) {
            cur[t] = 0.f;
            #pragma unroll
            for (int s = 0; s < S; s++) {
                const float v = __shfl(acc[t][s], 0, 64);
                if (lane == s) cur[t] = v;
            }
        }
        const int c = r0 + lane;
        if (lane < S && c < N) {
            const float * w = conv1d + c * 4;
            float c0_ = conv_state[c], c1_ = conv_state[N + c], c2_ = conv_state[2 * N + c];
            // replay: the previous cycle's accepted raw rows enter the
            // committed window
            for (int t = 0; t < n_rep; t++) {
                c0_ = c1_; c1_ = c2_; c2_ = prev_qkv[(size_t) t * N + c];
            }
            if (n_rep > 0) {
                conv_state[c] = c0_;
                conv_state[N + c] = c1_;
                conv_state[2 * N + c] = c2_;
            }
            // new rows: window advances in registers only (lazy commit)
            #pragma unroll
            for (int t = 0; t < T; t++) {
                y_qkv[(size_t) t * N + c] = cur[t];
                const float o = c0_ * w[0] + c1_ * w[1] + c2_ * w[2] + cur[t] * w[3];
                conv_out[(size_t) t * N + c] = o / (1.f + expf(-o));
                c0_ = c1_; c1_ = c2_; c2_ = cur[t];
            }
        }
    }
}

// T-row fused ffn gate/up + silu*up + requant; slot is [T][64] f32 in LDS,
// wave t requantizes row t
template <int T> __device__ __forceinline__ void qz_gemv_ffn_fused_t(
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * gv, const __half * gs,
        const char * uv, const __half * us, int N,
        unsigned char * ffn_q, float * ffn_sc, float * slot) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int S = 4;
    for (int g = blockIdx.x; g * (8 * S) < N; g += gridDim.x) {
        const int r0 = g * (8 * S) + wave * S;
        for (int m = 0; m < 2; m++) {
            const char * w_vals = m ? uv : gv;
            const __half * w_scal = m ? us : gs;
            float acc[T][S];
            qz_gemv_rows_t<T, S>(a_vals, a_scal, K, w_vals, w_scal, N, r0, lane, acc);
            if (lane == 0)
                #pragma unroll
                for (int t = 0; t < T; t++)
                    #pragma unroll
                    for (int s = 0; s < S; s++)
                        slot[t * 64 + m * 32 + wave * S + s] = acc[t][s];
        }
        __syncthreads();
        if (wave < T) {
            const float * sl = slot + wave * 64;
            float xu = 0.f, ax = 0.f;
            if (lane < 32) {
                const float gg = sl[lane];
                xu = (gg / (1.f + expf(-gg))) * sl[32 + lane];
                ax = fabsf(xu);
            }
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) ax = fmaxf(ax, __shfl_xor(ax, off, 64));
            const float d = ax / 127.f;
            const float id = d != 0.f ? 1.f / d : 0.f;
            if (lane < 32) {
                int q = (int) roundf(xu * id);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                ffn_q[(size_t) wave * N + g * 32 + lane] = (unsigned char) q;
            }
            if (lane == 0) ffn_sc[(size_t) wave * (N / 32) + g] = __half2float(__float2half(d));
        }
        __syncthreads();
    }
}

// residual row for a norm: xn = a + b (b may be null), ss = this thread's sum
// of squares, optional copy to st
__device__ __forceinline__ float qz_res_row(const float * a, const float * b,
                                            float * xn, float * st, int tid) {
    float xr[QZ_D / 512];
    #pragma unroll
    for (int k = 0; k < QZ_D / 512; k++) {
        const int i = tid + 512 * k;
        xr[k] = b ? a[i] + b[i] : a[i];
    }
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < QZ_D / 512; k++) {
        const int i = tid + 512 * k;
        xn[i] = xr[k];
        ss = fmaf(xr[k], xr[k], ss);
        if (st) st[i] = xr[k];
    }
    return ss;
}

// rms norm + requant of the row in xn: ss is this thread's partial over
// i = tid + 512k (same order as the T=1 phases), nwr the norm weights
__device__ __forceinline__ void qz_norm_requant_row(
        float * xn, const float (&nwr)[QZ_D / 512], float ss, float eps,
        unsigned char * qv, float * qs, float * red, int tid) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off, 64);
    const int wave = tid >> 6, lane = tid & 63;
    if (lane == 0) red[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float ms = 0.f;
        for (int w = 0; w < 8; w++) ms += red[w];
        red[8] = 1.f / sqrtf(ms / QZ_D + eps);
    }
    __syncthreads();
    // scale + requant in one pass: each 32-lane half-wave covers one unit;
    // xn keeps the scaled row
    const float rs = red[8];
    #pragma unroll
    for (int k = 0; k < QZ_D / 512; k++) {
        const int i = tid + 512 * k;
        const float x = xn[i] * rs * nwr[k];
        xn[i] = x;
        const float amax = qz_amax32(fabsf(x));
        const float d = amax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        int q = (int) roundf(x * id);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        qv[i] = (unsigned char) q;
        if ((tid & 31) == 0) qs[i >> 5] = __half2float(__float2half(d));
    }
    __syncthreads();
}

// qz_norm_requant_row split across the grid: every block reduces the full row,
// then requants only its own units into gq / gs (bit-equal)
__device__ __forceinline__ void qz_norm_requant_slice(
        const float * xn, const float * nw, float ss, float eps,
        unsigned char * gq, float * gs, float * red, int tid) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off, 64);
    const int wave = tid >> 6, lane = tid & 63;
    if (lane == 0) red[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float ms = 0.f;
        for (int w = 0; w < 8; w++) ms += red[w];
        red[8] = 1.f / sqrtf(ms / QZ_D + eps);
    }
    __syncthreads();
    const float rs = red[8];
    const int u = blockIdx.x + gridDim.x * (tid >> 5), l = tid & 31;
    if (u < QZ_D / 32) {             // whole half-waves: the 32-lane amax stays in one unit
        const int i = u * 32 + l;
        const float x = xn[i] * rs * nw[i];
        const float amax = qz_amax32(fabsf(x));
        const float d = amax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        int q = (int) roundf(x * id);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        gq[i] = (unsigned char) q;
        if (l == 0) gs[u] = __half2float(__float2half(d));
    }
    __syncthreads();                 // xn / red are reused by the next row
}

// after the grid barrier: the T published plane rows into this block's LDS
template <int T> __device__ __forceinline__ void qz_planes_to_lds(
        const unsigned char * gq, const float * gs, unsigned char * act_vals, float * act_scales, int tid) {
    for (int i = tid; i < T * QZ_D / 16; i += 512) ((uint4 *) act_vals)[i] = ((const uint4 *) gq)[i];
    for (int i = tid; i < T * (QZ_D / 32); i += 512) act_scales[i] = gs[i];
    __syncthreads();
}

// q (or k) head row: f16 staging in aq, per-head rms norm, rope at position
// pos on dims (j, j+32), j < 32; rope'd row stays in aq
__device__ __forceinline__ void qz_head_norm_rope(
        const float * src, const float * nw, int pos, float eps, __half * aq, float * red, int tid) {
    if (tid < QZ_HD) aq[tid] = __float2half(src[tid]);
    __syncthreads();
    float v = 0.f;
    if (tid < QZ_HD) { const float x = __half2float(aq[tid]); v = x * x; }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
    const int wave = tid >> 6, lane = tid & 63;
    if (lane == 0) red[wave] = v;
    __syncthreads();
    if (tid == 0) {
        float sq = 0.f;
        for (int w = 0; w < 8; w++) sq += red[w];
        red[8] = 1.f / sqrtf(sq / QZ_HD + eps);
    }
    __syncthreads();
    if (tid < QZ_HD) aq[tid] = __float2half(__half2float(aq[tid]) * red[8] * nw[tid]);
    __syncthreads();
    if (tid < 32) {
        const float ang = (float) pos * powf(1e7f, -(float) tid / 32.f);
        const float cs = cosf(ang), sn = sinf(ang);
        const float x0 = __half2float(aq[tid]), x1 = __half2float(aq[tid + 32]);
        aq[tid]      = __float2half(x0 * cs - x1 * sn);
        aq[tid + 32] = __float2half(x0 * sn + x1 * cs);
    }
    __syncthreads();
}

// cycle LDS: region 0 = T plane rows + the f32 norm row, overlaid by the K/V
// tiles and the DN scratch; then red, aq, scores + sml, part, ffn slot [T][64]
template <int T> __host__ __device__ constexpr size_t qz_cycle_r0() {
    const size_t planes = (size_t) T * (QZ_D + (QZ_D / 32) * 4) + (size_t) QZ_D * 4;
    const size_t tiles = (size_t) 2 * 32 * QZ_HD * 2;
    const size_t r = planes > tiles ? planes : tiles;
    const size_t dn = sizeof(qz_dn_lds) > sizeof(qz_dn_rows_lds<T>) ? sizeof(qz_dn_lds) : sizeof(qz_dn_rows_lds<T>);
    return ((r > dn ? r : dn) + 15) & ~(size_t) 15;
}
// fslot (and the TP2 xrow stage at fslot + T*64) must be 16 B aligned
template <int T> constexpr bool qz_cycle_fslot_aligned() {
    return (qz_cycle_r0<T>() + 12 * 4 + (size_t) QZ_HD * 2 + 36 * 4 + (size_t) 8 * QZ_HD * 4) % 16 == 0 &&
           ((size_t) T * 64 * 4) % 16 == 0;
}
static_assert(qz_cycle_fslot_aligned<1>() && qz_cycle_fslot_aligned<2>() &&
              qz_cycle_fslot_aligned<3>() && qz_cycle_fslot_aligned<4>(), "cycle LDS: fslot not 16 B aligned");
// the exchange GEMVs stage their activation slice in region 0
template <int T> constexpr bool qz_cycle_r0_holds_xrow() {
    return qz_cycle_r0<T>() >= (size_t) T * (QZ_FF / 2) * (1 + 4 / 32.0) &&
           qz_cycle_r0<T>() >= (size_t) T * (QZ_G / 2) * (1 + 4 / 32.0);
}
static_assert(qz_cycle_r0_holds_xrow<1>() && qz_cycle_r0_holds_xrow<2>() &&
              qz_cycle_r0_holds_xrow<3>() && qz_cycle_r0_holds_xrow<4>(), "cycle LDS: region 0 < xrow slice");
template <int T> static size_t qz_cycle_smem() {
    // + the fl9c reduce state (rw2, 64 floats); the TP2 push stage starts at
    // the same place and runs QZ_XSTAGE_BYTES past the end (the launch adds them)
    return qz_cycle_r0<T>() + 12 * 4 + (size_t) QZ_HD * 2 + 36 * 4 + (size_t) 8 * QZ_HD * 4 + (size_t) T * 64 * 4 + 64 * 4;
}

// one MTP block pass: T rows (token ptok[t], hidden ph + t*ph_str) at p0 + t;
// HEAD runs the lm_head and folds row am_row's argmax into *am_out
template <int T, bool HEAD>
__device__ __forceinline__ void qz_mtp_pass(
        qz_barrier_state * bst, int & kb, qz_trunk_args & A, const qz_mtp_w * M,
        int p0, const int (&ptok)[T], const float * ph, int ph_str,
        int am_row, int * am_out, unsigned int & xs) {
    using Gm = qz_geo<>;
    extern __shared__ char lds_raw[];
    unsigned char * act_vals = (unsigned char *) lds_raw;                   // [T][D]
    float * act_scales = (float *)(lds_raw + (size_t) T * QZ_D);           // [T][D/32]
    float * xn = (float *)(lds_raw + (size_t) T * (QZ_D + (QZ_D / 32) * 4));
    float * red = (float *)(lds_raw + qz_cycle_r0<T>());
    __half * aq = (__half *)(red + 12);    // red [9] padded to 16 B
    float * scores = (float *)(aq + QZ_HD);
    float * sml = scores + 32;
    float * part = sml + 4;                // sml [2] padded to 16 B
    float * fslot = part + 8 * QZ_HD;      // 16 B aligned: the TP2 stage is read as float4
    float * rw2 = fslot + T * 64;          // fl9c reduce state, later the TP2 push stage
    __half * kt = (__half *) lds_raw;      // fl9c K / V tile (region 0)
    const qz_trunk_w * W = &M->L;
    int tid = threadIdx.x;
    asm volatile ("" : "+v"(tid));           // per pass: see qz_cycle
    const int n_u = QZ_D / 32;
    float nwr[QZ_D / 512];

    // eh_proj input: block 0 requants [enorm(embed) | hnorm(h)] per row into
    // a gmem concat plane (too big for LDS)
    if (blockIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < QZ_D / 512; k++) nwr[k] = M->enorm[tid + 512 * k];
        for (int t = 0; t < T; t++) {
            const int tok = ptok[t];
            unsigned char * ev = A.cat_vals + (size_t) t * 2 * QZ_D;
            float * es = A.cat_scal + (size_t) t * 2 * n_u;
            __syncthreads();
            for (int i = tid; i < QZ_D / 4; i += 512)
                ((int *) ev)[i] = A.emb_vals[(size_t) tok * (QZ_D / 4) + i];
            for (int i = tid; i < n_u; i += 512)
                es[i] = __half2float(A.emb_scal[(size_t) tok * n_u + i]);
            __syncthreads();
            float ss = 0.f;
            for (int i = tid; i < QZ_D; i += 512) {
                const float x = (float) ((signed char) ev[i]) * es[i >> 5];
                xn[i] = x;
                ss += x * x;
            }
            qz_norm_requant_row(xn, nwr, ss, A.eps, ev, es, red, tid);
        }
        #pragma unroll
        for (int k = 0; k < QZ_D / 512; k++) nwr[k] = M->hnorm[tid + 512 * k];
        for (int t = 0; t < T; t++) {
            const float * hr = ph ? ph + (size_t) t * ph_str : nullptr;
            unsigned char * hv = A.cat_vals + (size_t) t * 2 * QZ_D + QZ_D;
            float * hs = A.cat_scal + (size_t) t * 2 * n_u + n_u;
            __syncthreads();
            float ss = 0.f;
            for (int i = tid; i < QZ_D; i += 512) {
                const float x = hr ? hr[i] : 0.f;
                xn[i] = x;
                ss += x * x;
            }
            qz_norm_requant_row(xn, nwr, ss, A.eps, hv, hs, red, tid);
        }
    }
    qz_grid_barrier(bst, ++kb);

    // inpSA = eh_proj @ concat, then attn_norm planes per row
    qz_gemv_slice_t<T>(A.cat_vals, A.cat_scal, 2 * QZ_D, M->eh_vals, M->eh_scal, QZ_D, A.y_g);
    qz_grid_barrier(bst, ++kb);
    #pragma unroll
    for (int k = 0; k < QZ_D / 512; k++) nwr[k] = W->attn_norm[tid + 512 * k];
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float ss = 0.f;
        __syncthreads();
        for (int i = tid; i < QZ_D; i += 512) {
            const float x = A.y_g[(size_t) t * QZ_D + i];
            xn[i] = x;
            ss += x * x;
        }
        qz_norm_requant_row(xn, nwr, ss, A.eps, act_vals + (size_t) t * QZ_D,
                            act_scales + (size_t) t * n_u, red, tid);
    }

    // projections, then q/k norm + rope at p0+t and arena rows
    qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->q_vals, W->q_scal, Gm::AQ, A.y_q);
    qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->k_vals, W->k_scal, Gm::AKV, A.y_k, (Gm::AQ / 32) % 60);
    qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->v_vals, W->v_scal, Gm::AKV, A.y_v, ((Gm::AQ + Gm::AKV) / 32) % 60);
    qz_grid_barrier(bst, ++kb);

    for (int it = blockIdx.x; it < T * Gm::HQ; it += gridDim.x) {
        const int t = it / Gm::HQ, h = it % Gm::HQ;
        qz_head_norm_rope(A.y_q + (size_t) t * Gm::AQ + (size_t) h * 2 * QZ_HD, W->q_norm_w,
                          p0 + t + A.rope_shift, A.eps, aq, red, tid);
        if (tid < QZ_HD)
            A.q_f16[(size_t) t * Gm::HQ * QZ_HD + (size_t) h * QZ_HD + tid] = aq[tid];
        __syncthreads();
    }
    for (int it = blockIdx.x; it < T * Gm::HKV; it += gridDim.x) {
        const int t = it / Gm::HKV, kh = it % Gm::HKV;
        const int p = p0 + t;
        const size_t row = qz_kv_row(p, Gm::HKV, kh);
        qz_head_norm_rope(A.y_k + (size_t) t * Gm::AKV + (size_t) kh * QZ_HD, W->k_norm_w,
                          p + A.rope_shift, A.eps, aq, red, tid);
        if (tid < QZ_HD) W->kv_k[row + tid] = aq[tid];
        if (tid < QZ_HD) W->kv_v[row + tid] = __float2half(A.y_v[(size_t) t * Gm::AKV + (size_t) kh * QZ_HD + tid]);
        __syncthreads();
    }
    qz_grid_barrier(bst, ++kb);

    // attention (fl9c, flash9c.h): split-K, FL9_NS partials per (row, head),
    // rows causal to p0+t
    {
        qz_fl9c_flash<T, Gm::HQ, Gm::HKV>(W->kv_k, W->kv_v, A.q_f16, A.pm, A.pl, A.pacc, p0,
                                          kt, (__half *) part, tid);
        qz_grid_barrier(bst, ++kb);
        // per (row, head) merge, gate, requant
        for (int it = blockIdx.x; it < T * Gm::HQ; it += gridDim.x) {
            const int t = it / Gm::HQ;
            qz_fl9c_reduce<T, Gm::HQ>(A.pm, A.pl, A.pacc, A.y_q + (size_t) t * Gm::AQ + QZ_HD, 2 * QZ_HD,
                                      A.o_preg + (size_t) t * Gm::HQ * QZ_HD,
                                      A.pl_vals + (size_t) t * Gm::G, A.pl_scal + (size_t) t * (Gm::G / 32),
                                      rw2, it, tid);
        }
        qz_grid_barrier(bst, ++kb);
    }

    // attn_output GEMV, then h = inpSA + core out (block 0 keeps the row),
    // post norm planes
    qz_xrow_t<T>(bst, kb, A, ++xs, A.pl_vals, A.pl_scal, Gm::G, W->out_vals, W->out_scal, fslot + T * 64, A.y_core, (unsigned char *) lds_raw);
    #pragma unroll
    for (int k = 0; k < QZ_D / 512; k++) nwr[k] = W->post_norm[tid + 512 * k];
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float ss = 0.f;
        __syncthreads();
        for (int i = tid; i < QZ_D; i += 512) {
            const float x = A.y_g[(size_t) t * QZ_D + i] + A.y_core[(size_t) t * QZ_D + i];
            xn[i] = x;
            ss += x * x;
            if (blockIdx.x == 0) A.y_u[(size_t) t * QZ_D + i] = x;
        }
        qz_norm_requant_row(xn, nwr, ss, A.eps, act_vals + (size_t) t * QZ_D,
                            act_scales + (size_t) t * n_u, red, tid);
    }

    // ffn fused gate/up + down GEMV, same as the trunk phases
    qz_gemv_ffn_fused_t<T>(act_vals, act_scales, QZ_D, W->fg_vals, W->fg_scal, W->fu_vals, W->fu_scal,
                           Gm::FF, A.ffn_q, A.ffn_sc, fslot);
    qz_grid_barrier(bst, ++kb);
    qz_xrow_t<T>(bst, kb, A, ++xs, A.ffn_q, A.ffn_sc, Gm::FF, W->fd_vals, W->fd_scal, fslot + T * 64, A.y_ffn, (unsigned char *) lds_raw);

    // head input: out = h + ffn out, shared_head_norm; the normed row is the
    // next draft step's hidden (block 0 keeps it in mtp_h)
    #pragma unroll
    for (int k = 0; k < QZ_D / 512; k++) nwr[k] = M->shnorm[tid + 512 * k];
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float ss = 0.f;
        __syncthreads();
        for (int i = tid; i < QZ_D; i += 512) {
            const float x = A.y_u[(size_t) t * QZ_D + i] + A.y_ffn[(size_t) t * QZ_D + i];
            xn[i] = x;
            ss += x * x;
        }
        qz_norm_requant_row(xn, nwr, ss, A.eps, act_vals + (size_t) t * QZ_D,
                            act_scales + (size_t) t * n_u, red, tid);
        if (blockIdx.x == 0)
            for (int i = tid; i < QZ_D; i += 512) A.mtp_h[(size_t) t * QZ_D + i] = xn[i];
    }

    if (HEAD) {
        // only row am_row produces a draft: the head GEMV streams the same
        // weights for one row as for T, so score just that one
        const int hn = A.hrows;
        qz_gemv_slice_t<1>(act_vals + (size_t) am_row * QZ_D, act_scales + (size_t) am_row * n_u,
                           QZ_D, A.hd_vals, A.hd_scal, hn, A.mtp_yhead);
        qz_grid_barrier(bst, ++kb);
        if (blockIdx.x == 0) {
            const float * yh = A.mtp_yhead;
            float bv = -3.4028235e38f;
            int bi = hn - 1;
            for (int r = tid; r < hn; r += 512) {
                const float v = yh[r];
                if (v > bv) { bv = v; bi = r; }
            }
            part[tid] = bv;
            ((int *)(part + 512))[tid] = bi;
            __syncthreads();
            if (tid == 0) {
                float b0 = part[0];
                int i0 = ((int *)(part + 512))[0];
                for (int t2 = 1; t2 < 512; t2++) {
                    const float v = part[t2];
                    const int ix = ((int *)(part + 512))[t2];
                    if (v > b0 || (v == b0 && ix < i0)) { b0 = v; i0 = ix; }
                }
                // merge with the peer's half: one pair, block 0 alone
                const unsigned int tok = xs + 1;
                const int gid = i0 + A.tp_card * A.hrows;
                float * ph = A.xhm_peer + (size_t)(tok & 1) * QZ_TMAX * 2;
                *(float2 *) ph = make_float2(b0, __int_as_float(gid));
                qz_xflag(A, tok);
                qz_xwait(A, tok);
                const float * mh = A.xhm_mine + (size_t)(tok & 1) * QZ_TMAX * 2;
                *am_out = qz_xpick(b0, gid, __builtin_nontemporal_load(mh), __float_as_int(__builtin_nontemporal_load(mh + 1)));
            }
        }
        xs++;
    }
}

// the cycle: tokens A.toks[0..T-1] at A.pos..; outputs y_head [T][V] and
// out_tok[t] = argmax of row t
template <int T>
__global__ void __launch_bounds__(512)
qz_cycle(qz_barrier_state * bst, qz_trunk_args A) {
    using Gm = qz_geo<>;
    extern __shared__ char lds_raw[];
    unsigned char * act_vals = (unsigned char *) lds_raw;                   // [T][D]
    float * act_scales = (float *)(lds_raw + (size_t) T * QZ_D);           // [T][D/32]
    float * xn = (float *)(lds_raw + (size_t) T * (QZ_D + (QZ_D / 32) * 4));
    float nwr[QZ_D / 512];
    float * red = (float *)(lds_raw + qz_cycle_r0<T>());
    __half * aq = (__half *)(red + 12);    // red [9] padded to 16 B
    float * scores = (float *)(aq + QZ_HD);
    float * sml = scores + 32;
    float * part = sml + 4;                // sml [2] padded to 16 B
    float * fslot = part + 8 * QZ_HD;      // 16 B aligned: the TP2 stage is read as float4
    float * rw2 = fslot + T * 64;          // fl9c reduce state, later the TP2 push stage
    __half * kt = (__half *) lds_raw;      // fl9c K / V tile (region 0)

    // tid is made opaque every layer so per-thread addresses are not hoisted
    // out of the loop (they spilled)
    const int tid_k = threadIdx.x;
    int tid = tid_k;
    const int n_u = QZ_D / 32;
    const int n_rep = A.ctl[0];               // previous cycle's accepted rows
    const int par = A.cycle & 1;
    int kb = 0;
    unsigned int xs = 0;
    xs = __hip_atomic_load(A.xseq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    for (int il = 0; il < A.n_layer; il++) {
        tid = tid_k;
        asm volatile ("" : "+v"(tid));
        const qz_trunk_w * W = &A.w[il];
        float * xprev = A.x_rows + (size_t)(il & 1) * T * QZ_D;
        float * xin = A.x_rows + (size_t)((il + 1) & 1) * T * QZ_D;
        // recr rows live in this cycle's ring slot; the other slot holds the
        // previous cycle's rows for replay
        float * rcur = W->recr ? W->ring + (size_t) par * Gm::RING_PAR : nullptr;
        const float * rprev = W->recr ? W->ring + (size_t)(par ^ 1) * Gm::RING_PAR : nullptr;
        // DSpine features: the residual after layers 1, 16, 31, 46, 61 = this layer's input
        const int fq = !A.fcap ? -1 : il == 2 ? 0 : il == 17 ? 1 : il == 32 ? 2 : il == 47 ? 3 : il == 62 ? 4 : -1;

        // phase A: per row, h + previous ffn out (or the embedding) -> attn_norm
        // -> requant split over the grid into A.nq_*, then copied to LDS
        for (int t = 0; t < T; t++) {
            unsigned char * qv = act_vals + (size_t) t * QZ_D;
            float * qs = act_scales + (size_t) t * n_u;
            float ss = 0.f;
            __syncthreads();
            if (il > 0) {
                // all loads first (the xin store may alias them, which would
                // serialize load -> store -> load), same sum order
                float xr[QZ_D / 512];
                #pragma unroll
                for (int k = 0; k < QZ_D / 512; k++) {
                    const int i = tid + 512 * k;
                    xr[k] = xprev[(size_t) t * QZ_D + i] + A.y_ffn[(size_t) t * QZ_D + i];
                }
                #pragma unroll
                for (int k = 0; k < QZ_D / 512; k++) {
                    const int i = tid + 512 * k;
                    xn[i] = xr[k];
                    // explicit fma keeps the rolled loop's v_fmac order
                    ss = fmaf(xr[k], xr[k], ss);
                    if (blockIdx.x == 0) xin[(size_t) t * QZ_D + i] = xr[k];
                    if (fq >= 0 && blockIdx.x == 1)
                        A.fcap[((size_t) fq * A.fcap_cap + (A.pos + t) % A.fcap_cap) * QZ_D + i] = xr[k];
                }
            } else {
                const int tok = A.toks[t];
                for (int i = tid; i < QZ_D / 4; i += 512)
                    ((int *) qv)[i] = A.emb_vals[(size_t) tok * (QZ_D / 4) + i];
                for (int i = tid; i < n_u; i += 512)
                    qs[i] = __half2float(A.emb_scal[(size_t) tok * n_u + i]);
                __syncthreads();
                for (int i = tid; i < QZ_D; i += 512) {
                    const float x = (float) ((signed char) qv[i]) * qs[i >> 5];
                    xn[i] = x;
                    ss += x * x;
                    if (blockIdx.x == 0) xin[(size_t) t * QZ_D + i] = x;
                }
            }
            qz_norm_requant_slice(xn, W->attn_norm, ss, A.eps, A.nq_vals + (size_t) t * QZ_D,
                                  A.nq_scal + (size_t) t * n_u, red, tid);
        }
        qz_grid_barrier(bst, ++kb);
        qz_planes_to_lds<T>(A.nq_vals, A.nq_scal, act_vals, act_scales, tid);

        // phase B: core projections over T rows
        if (W->recr) {
            qz_gemv_qkv_conv_t<T>(act_vals, act_scales, QZ_D, W->qkv_vals, W->qkv_scal, Gm::CCH,
                                  rcur + Gm::RING_QKV, W->conv_state, W->conv1d, rcur + Gm::RING_CONV,
                                  rprev + Gm::RING_QKV, n_rep);
            qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->gat_vals, W->gat_scal, Gm::G, A.y_gate, (Gm::CCH / 32) % 60);
            qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->bet_vals, W->bet_scal, Gm::HV, rcur + Gm::RING_BETA, ((Gm::CCH + Gm::G) / 32) % 60);
            qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->alp_vals, W->alp_scal, Gm::HV, rcur + Gm::RING_ALPH, ((Gm::CCH + Gm::G) / 32 + 2) % 60);
        } else {
            qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->q_vals, W->q_scal, Gm::AQ, A.y_q);
            qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->k_vals, W->k_scal, Gm::AKV, A.y_k, (Gm::AQ / 32) % 60);
            qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, W->v_vals, W->v_scal, Gm::AKV, A.y_v, ((Gm::AQ + Gm::AKV) / 32) % 60);
        }
        qz_grid_barrier(bst, ++kb);

        if (W->recr) {
            // phase D: replay the previous cycle's accepted rows and commit,
            // then run the T new rows without storing the state
            if (blockIdx.x < Gm::HV) {
                const int h = blockIdx.x;
                const int j = tid & (QZ_S - 1), r = tid >> 7;
                qz_dn_rows<T>(W->dn_state + ((size_t) h << 14) + (size_t) r * 32 * QZ_S + j, n_rep,
                                  rprev, rcur, A.y_gate, W->dt_bias, W->ssm_a, W->ssm_norm_w,
                                  A.pl_vals, A.pl_scal, h, A.eps, (qz_dn_rows_lds<T> *) lds_raw, tid);
            }
            qz_grid_barrier(bst, ++kb);
        } else {
            // phase C: q rows (norm + rope at pos+t) and arena k/v rows
            for (int it = blockIdx.x; it < T * Gm::HQ; it += gridDim.x) {
                const int t = it / Gm::HQ, h = it % Gm::HQ;
                qz_head_norm_rope(A.y_q + (size_t) t * Gm::AQ + (size_t) h * 2 * QZ_HD, W->q_norm_w,
                                  A.pos + t + A.rope_shift, A.eps, aq, red, tid);
                if (tid < QZ_HD)   // dims 0..63 roped, 64..255 pass through
                    A.q_f16[(size_t) t * Gm::HQ * QZ_HD + (size_t) h * QZ_HD + tid] = aq[tid];
                __syncthreads();
            }
            for (int it = blockIdx.x; it < T * Gm::HKV; it += gridDim.x) {
                const int t = it / Gm::HKV, kh = it % Gm::HKV;
                const int p = A.pos + t;
                const size_t row = qz_kv_row(p, Gm::HKV, kh);
                qz_head_norm_rope(A.y_k + (size_t) t * Gm::AKV + (size_t) kh * QZ_HD, W->k_norm_w,
                                  p + A.rope_shift, A.eps, aq, red, tid);
                if (tid < QZ_HD) W->kv_k[row + tid] = aq[tid];
                if (tid < QZ_HD) W->kv_v[row + tid] = __float2half(A.y_v[(size_t) t * Gm::AKV + (size_t) kh * QZ_HD + tid]);
                __syncthreads();
            }
            qz_grid_barrier(bst, ++kb);

            // phase D: attention (fl9c, flash9c.h): split-K, FL9_NS partials
            // per (row, head), rows causal to pos+t
            qz_fl9c_flash<T, Gm::HQ, Gm::HKV>(W->kv_k, W->kv_v, A.q_f16, A.pm, A.pl, A.pacc, A.pos,
                                              kt, (__half *) part, tid);
            qz_grid_barrier(bst, ++kb);

            // phase E: per (row, head) merge, gate, requant
            for (int it = blockIdx.x; it < T * Gm::HQ; it += gridDim.x) {
                const int t = it / Gm::HQ;
                qz_fl9c_reduce<T, Gm::HQ>(A.pm, A.pl, A.pacc, A.y_q + (size_t) t * Gm::AQ + QZ_HD, 2 * QZ_HD,
                                          A.o_preg + (size_t) t * Gm::HQ * QZ_HD,
                                          A.pl_vals + (size_t) t * Gm::G, A.pl_scal + (size_t) t * (Gm::G / 32),
                                          rw2, it, tid);
            }
            qz_grid_barrier(bst, ++kb);

        }

        // phase F: core-out GEMV off the gmem planes
        qz_xrow_t<T>(bst, kb, A, ++xs, A.pl_vals, A.pl_scal, Gm::G, W->out_vals, W->out_scal, fslot + T * 64, A.y_core, (unsigned char *) lds_raw);

        // phase G: per row, h = x + core out -> post norm -> requant as phase A;
        // T <= 3 prefetches the next row's y_core (slower at T = 4)
        constexpr bool PF = T <= 3;
        float gb[QZ_D / 512];
        if constexpr (PF)
            #pragma unroll
            for (int k = 0; k < QZ_D / 512; k++) gb[k] = A.y_core[tid + 512 * k];
        for (int t = 0; t < T; t++) {
            float ss = 0.f;
            __syncthreads();
            float xr[QZ_D / 512];
            if constexpr (PF) {
                #pragma unroll
                for (int k = 0; k < QZ_D / 512; k++) xr[k] = xin[(size_t) t * QZ_D + tid + 512 * k] + gb[k];
                if (t + 1 < T)
                    #pragma unroll
                    for (int k = 0; k < QZ_D / 512; k++) gb[k] = A.y_core[(size_t)(t + 1) * QZ_D + tid + 512 * k];
            } else {
                #pragma unroll
                for (int k = 0; k < QZ_D / 512; k++)
                    xr[k] = xin[(size_t) t * QZ_D + tid + 512 * k] + A.y_core[(size_t) t * QZ_D + tid + 512 * k];
            }
            #pragma unroll
            for (int k = 0; k < QZ_D / 512; k++) {
                xn[tid + 512 * k] = xr[k];
                ss = fmaf(xr[k], xr[k], ss);   // as the rolled loop's v_fmac
            }
            qz_norm_requant_slice(xn, W->post_norm, ss, A.eps, A.nq_vals + (size_t) t * QZ_D,
                                  A.nq_scal + (size_t) t * n_u, red, tid);
        }
        qz_grid_barrier(bst, ++kb);
        qz_planes_to_lds<T>(A.nq_vals, A.nq_scal, act_vals, act_scales, tid);

        // phase H/I: fused gate/up + silu*up + requant, T rows
        qz_gemv_ffn_fused_t<T>(act_vals, act_scales, QZ_D, W->fg_vals, W->fg_scal, W->fu_vals, W->fu_scal,
                               Gm::FF, A.ffn_q, A.ffn_sc, fslot);
        qz_grid_barrier(bst, ++kb);

        // FFN residual is h: promote the x rows, spread over the grid
        for (int i = blockIdx.x * 512 + tid; i < T * QZ_D; i += gridDim.x * 512)
            xin[i] = xin[i] + A.y_core[i];

        // phase J: ffn down GEMV
        qz_xrow_t<T>(bst, kb, A, ++xs, A.ffn_q, A.ffn_sc, Gm::FF, W->fd_vals, W->fd_scal, fslot + T * 64, A.y_ffn, (unsigned char *) lds_raw);
    }

    // head: per row, x_out = h + ffn out -> out_norm -> planes; lm_head GEMV
    {
        float * xlast = A.x_rows + (size_t)(A.n_layer & 1) * T * QZ_D;
        #pragma unroll
        for (int k = 0; k < QZ_D / 512; k++) nwr[k] = A.out_norm[tid + 512 * k];
        for (int t = 0; t < T; t++) {
            float ss = 0.f;
            __syncthreads();
            ss = qz_res_row(xlast + (size_t) t * QZ_D, A.y_ffn + (size_t) t * QZ_D, xn, nullptr, tid);
            qz_norm_requant_row(xn, nwr, ss, A.eps, act_vals + (size_t) t * QZ_D,
                                act_scales + (size_t) t * n_u, red, tid);
            // the normed row is the trunk h_nextn: the MTP catch-up input.
            // Pure copy after the requant, trunk arithmetic untouched.
            if (A.mtp && blockIdx.x == 0)
                for (int i = tid; i < QZ_D; i += 512)
                    A.h_nextn[(size_t) t * QZ_D + i] = xn[i];
        }
    }
    const int hn = A.hrows;
    qz_gemv_slice_t<T>(act_vals, act_scales, QZ_D, A.hd_vals, A.hd_scal, hn, A.y_head);
    qz_grid_barrier(bst, ++kb);
    const unsigned int htok = ++xs;

    // argmax per row: block t folds row t
    if (blockIdx.x < T) {
        const float * yh = A.y_head + (size_t) blockIdx.x * hn;
        float bv = -3.4028235e38f;
        int bi = hn - 1;
        for (int r = tid; r < hn; r += 512) {
            const float v = yh[r];
            if (v > bv) { bv = v; bi = r; }
        }
        part[tid] = bv;
        ((int *)(part + 512))[tid] = bi;
        __syncthreads();
        if (tid == 0) {
            float b0 = part[0];
            int i0 = ((int *)(part + 512))[0];
            for (int t2 = 1; t2 < 512; t2++) {
                const float v = part[t2];
                const int ix = ((int *)(part + 512))[t2];
                if (v > b0 || (v == b0 && ix < i0)) { b0 = v; i0 = ix; }
            }
            // own pair to y_part (block 0 merges), the same pair to the peer
            const int gid = i0 + A.tp_card * A.hrows;
            A.y_part[2 * blockIdx.x] = b0;
            A.y_part[2 * blockIdx.x + 1] = __int_as_float(gid);
            float * ph = A.xhm_peer + (size_t)(htok & 1) * QZ_TMAX * 2;
            *(float2 *)(ph + 2 * blockIdx.x) = make_float2(b0, __int_as_float(gid));
            __threadfence_system();
        }
    }
    // greedy verify: accept rows while row t's argmax equals row t+1's draft;
    // the next cycle starts from the bonus token
    qz_grid_barrier(bst, ++kb);
    if (blockIdx.x == 0 && tid == 0) {
        qz_xflag(A, htok);
        qz_xwait(A, htok);
        const float * mh = A.xhm_mine + (size_t)(htok & 1) * QZ_TMAX * 2;
        for (int t = 0; t < T; t++)
            A.out_tok[t] = qz_xpick(__builtin_nontemporal_load(A.y_part + 2 * t),
                                    __float_as_int(__builtin_nontemporal_load(A.y_part + 2 * t + 1)),
                                    __builtin_nontemporal_load(mh + 2 * t),
                                    __float_as_int(__builtin_nontemporal_load(mh + 2 * t + 1)));
    }
    if (blockIdx.x == 0 && tid == 0) {
        int m = 0;
        if (A.verify) {
            while (m < T - 1 && A.out_tok[m] == A.toks[m + 1]) m++;
        } else {
            m = T - 1;
        }
        A.ctl[0] = m + 1;
        A.ctl[1] = A.out_tok[m];
        A.ctl[2] = A.pos + m + 1;
        *A.xseq = xs;
    }
}

// MTP draft kernel (right after qz_cycle, no host sync): catch-up over the
// accepted rows, then drafts the next cycle's tokens into ctl[3..]
template <int N> __device__ __forceinline__ void qz_mtp_catchup(
        qz_barrier_state * bst, int & kb, qz_trunk_args & A, int bonus, unsigned int & xs) {
    int ptk[N];
    #pragma unroll
    for (int t = 0; t < N; t++) ptk[t] = t < N - 1 ? A.toks[t + 1] : bonus;
    qz_mtp_pass<N, true>(bst, kb, A, A.mtp, A.pos + 1, ptk, A.h_nextn, QZ_D, N - 1, &A.ctl[3], xs);
}

template <int T>
__global__ void __launch_bounds__(512)
qz_mtp(qz_barrier_state * bst, qz_trunk_args A) {
    int kb = 0;
    unsigned int xs = 0;
    xs = __hip_atomic_load(A.xseq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const int n_acc = A.ctl[0];
    const int bonus = A.ctl[1];
    if (A.pos == 0) {
        int pzt[T];
        #pragma unroll
        for (int t = 0; t < T; t++) pzt[t] = A.toks[0];
        qz_mtp_pass<T, false>(bst, kb, A, A.mtp, 0, pzt, nullptr, 0, 0, nullptr, xs);
    }
    // catch-up over the accepted prefix only (n_acc is uniform: ctl[0])
    if (n_acc == 1)      qz_mtp_catchup<1>(bst, kb, A, bonus, xs);
    else if (n_acc == 2) qz_mtp_catchup<2>(bst, kb, A, bonus, xs);
    else if (n_acc == 3) { if constexpr (T >= 3) qz_mtp_catchup<3>(bst, kb, A, bonus, xs); }
    else                 { if constexpr (T >= 4) qz_mtp_catchup<4>(bst, kb, A, bonus, xs); }
    int hrow = n_acc - 1;            // hidden row feeding the draft chain
    int tn = T;                      // next cycle's rows
    if (A.mtp_t > 0 && A.mtp_t < T) tn = A.mtp_t;
    else if (A.mtp_t < 0) {          // -(Tc + 8 * slack)
        const int tc = (-A.mtp_t) & 7, slack = (-A.mtp_t) >> 3;
        tn = n_acc >= tc - slack ? T : T - 1;
    }
    if (A.mtp_t != 0 && blockIdx.x == 0 && threadIdx.x == 0) A.ctl[7] = tn;   // ctl holds 8 ints
    for (int i = 1; i <= tn - 2; i++) {
        qz_grid_barrier(bst, ++kb);   // d_i's store (last pass argmax) is
                                      // behind that pass's final barrier
        const int pdt[1] = { A.ctl[2 + i] };
        qz_mtp_pass<1, true>(bst, kb, A, A.mtp, A.pos + n_acc + i, pdt,
                             A.mtp_h + (size_t) hrow * QZ_D, 0, 0, &A.ctl[2 + i + 1], xs);
        hrow = 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *A.xseq = xs;
}
