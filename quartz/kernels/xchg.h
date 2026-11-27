// TP exchange between the NC cards: per-block flags, the row-parallel GEMV push and
// the combine. XA is the arch's args struct: it holds the exchange pointers, its
// width as XA::DM, a multiple of 32, and the card count as XA::NC.
//
// Slots. Each card's inbox, head-merge pairs and flags hold NC - 1 slots. For card
// me, peer k (1..NC-1) is card (me + k) % NC; me writes into that card's slot
// NC - 1 - k (the *_px[k - 1] pointers, set by the host) and reads peer k from its
// own slot k - 1. At NC = 2 both are slot 0, the TP2 layout.
#pragma once

#include "kernels/gemv.h"
#include "core/xcheck.h"

// ---- TP exchange -----------------------------------------------
// push partials to every peer's inbox in parity tok & 1, wait for their flags, then
// y = sum of the NC partials in card order
#define QZ_XSPIN 2000000000ll     // clock64 ticks before a wait gives up
#define QZ_XFS 128                // flags per slot (512 B)

// inbox slot stride in floats for CX rows: data [2][CX][DM] and checksums [2][CX][DM/32]
template <int CX, int DM> constexpr size_t qz_xin_slot() { return (size_t) 2 * CX * DM + (size_t) 2 * CX * (DM / 32); }
// card me's peer k and the slot me writes into it, for the host wiring
__host__ __device__ constexpr int qz_xpeer(int nc, int me, int k) { return (me + k) % nc; }
__host__ __device__ constexpr int qz_xslot_in_peer(int nc, int k) { return nc - 1 - k; }
// the card index of an args struct: tp_card (Qwen) or card (Inkling), for the fan-out check
template <class XA> __device__ __forceinline__ auto qz_xcard(const XA & A, int) -> decltype(A.tp_card) { return A.tp_card; }
template <class XA> __device__ __forceinline__ auto qz_xcard(const XA & A, long) -> decltype(A.card) { return A.card; }
template <class XA> __device__ __forceinline__ int qz_xcard(const XA & A) { return qz_xcard(A, 0); }

// wait in one thread for inbox flag b (slot * QZ_XFS + block) to reach tok
template <class XA> __device__ __forceinline__ void qz_xwait_slot(const XA & A, unsigned int tok, int b) {
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
// per-block exchange flags, set in every peer and waited on from every peer. Block b
// only needs the peers' pushes of block b, and flag 0 doubles as the head-merge flag.
template <class XA> __device__ __forceinline__ void qz_xflag_b(const XA & A, unsigned int tok, int b) {
    __threadfence_system();
    QZ_XCK(qz_xck_end(qz_xcard(A)));
    #pragma unroll
    for (int k = 0; k < XA::NC - 1; k++) {
        QZ_XCK(qz_xck_store(qz_xcard(A), k + 1, XS_FLAG));
        __hip_atomic_store(A.xflag_px[k] + b, tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    QZ_XCK(qz_xck_end(qz_xcard(A)));
}
template <class XA> __device__ __forceinline__ void qz_xwait_b(const XA & A, unsigned int tok, int b) {
    #pragma unroll
    for (int k = 0; k < XA::NC - 1; k++) qz_xwait_slot(A, tok, k * QZ_XFS + b);
}

// Two hops at NC = 4. On some hosts one GPU writing to several peers at once collapses
// P2P (0.3 GB/s), while one peer per GPU stays at full speed with all four busy. So the
// row exchange never fans out: hop 1 swaps the partials with the pair partner me ^ 1
// (peer k = 1 for an even card, 3 for an odd one), hop 2 swaps the pair sums with the
// cross partner me ^ 2 (k = 2). Every card ends with (p0 + p1) + (p2 + p3), the same
// bits as the one-hop card-order sum. At NC = 2 hop 1 is the whole exchange.
template <class XA> __device__ __forceinline__ int qz_xkpair(const XA & A) {
    if constexpr (XA::NC == 2) return 1;
    else return (A.tp_card & 1) ? 3 : 1;
}
// the inbox slot me writes in hop 1; a select, not an index, keeps the pointers in SGPRs
template <class XA> __device__ __forceinline__ float * qz_xhop1_px(const XA & A) {
    if constexpr (XA::NC == 2) return A.xin_px[0];
    else return (A.tp_card & 1) ? A.xin_px[2] : A.xin_px[0];
}
// flag b in peer k only
template <class XA> __device__ __forceinline__ void qz_xflag_k(const XA & A, unsigned int tok, int k, int b) {
    static_assert(XA::NC == 4, "one-peer flags are for the two-hop exchange");
    __threadfence_system();
    unsigned int * f = k == 1 ? A.xflag_px[0] : k == 2 ? A.xflag_px[1] : A.xflag_px[2];
    QZ_XCK(qz_xck_end(qz_xcard(A)));
    QZ_XCK(qz_xck_store(qz_xcard(A), k, XS_FLAGK));
    __hip_atomic_store(f + b, tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    QZ_XCK(qz_xck_end(qz_xcard(A)));
}

// block 0's flag, the whole-card exchange: set it, or wait in one thread for the
// peers' flags to reach tok. Past the spin limit, or once any wait gave up, a wait
// sets the abort word and returns so the launch drains.
template <class XA> __device__ __forceinline__ void qz_xflag(const XA & A, unsigned int tok) { qz_xflag_b(A, tok, 0); }
template <class XA> __device__ __forceinline__ void qz_xwait(const XA & A, unsigned int tok) { qz_xwait_b(A, tok, 0); }

// row-parallel GEMV over the card's K slice. The block's slabs stage in LDS and are
// pushed after the loop, under one fence. CX is the row capacity of the exchange
// buffers, the cycle's rows or the verify's. LX stages the activations in LDS
// region 0, off for the verify, where they do not fit.
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
        QZ_XCK(qz_xck_store(qz_xcard(A), qz_xkpair(A), XS_PUSH));
        *(float4 *)(qz_xhop1_px(A) + (size_t) par * CX * XA::DM + o) = v;
    }
    __threadfence_system();
    QZ_XCK(qz_xck_end(qz_xcard(A)));
}

// the NC partials of one element summed in card order, (p0 + p1) + (p2 + p3) at
// NC = 4. own is this card's, peer k's sits in inbox slot k - 1 at offset o.
template <int CX, class XA> __device__ __forceinline__ float qz_xsum(const XA & A, float own, const float * src, size_t o) {
    if constexpr (XA::NC == 2) {
        const float peer = __builtin_nontemporal_load(src + o);
        return A.tp_card == 0 ? __fadd_rn(own, peer) : __fadd_rn(peer, own);
    } else {
        static_assert(XA::NC == 4, "card-order sum for 2 or 4 cards");
        constexpr size_t XS = qz_xin_slot<CX, XA::DM>();
        auto v = [&](int c) {
            const int k = (c - A.tp_card + XA::NC) % XA::NC;
            return k == 0 ? own : __builtin_nontemporal_load(src + (size_t) (k - 1) * XS + o);
        };
        return __fadd_rn(__fadd_rn(v(0), v(1)), __fadd_rn(v(2), v(3)));
    }
}

// after qz_gemv_xpush_t: flag the peers, wait for their flags and sum the partials over
// this block's slabs. BX = false drops the trailing grid barrier.
template <int T, int CX, bool BX = true, class XA> __device__ __forceinline__ void qz_xfinish_t(
        qz_barrier_state * bst, int & kb, const XA & A, unsigned int tok, float * y) {
    const int tid = threadIdx.x;
    const int par = tok & 1;
    __syncthreads();                 // every thread's remote stores are fenced by the push
    if constexpr (XA::NC == 2) {
        if (tid == 0) { qz_xflag_b(A, tok, blockIdx.x); qz_xwait_b(A, tok, blockIdx.x); }
        __syncthreads();
        const float * src = A.xin_mine + (size_t) par * CX * XA::DM;
        for (int g = blockIdx.x; g * 32 < XA::DM; g += gridDim.x) {
            if (tid < 32 * T) {
                const int t = tid >> 5, i = tid & 31;
                const size_t o = (size_t) t * XA::DM + g * 32 + i;
                const float own = __builtin_nontemporal_load(A.y_part + o);
                y[o] = qz_xsum<CX>(A, own, src, o);
            }
        }
    } else {
        constexpr size_t XS = qz_xin_slot<CX, XA::DM>();
        const int b = blockIdx.x, kp = qz_xkpair(A);
        const bool odd = A.tp_card & 1, hi = A.tp_card >= 2;
        const float * src = A.xin_mine + (size_t) par * CX * XA::DM;
        float * dst = A.xin_px[1] + (size_t) par * CX * XA::DM;     // the cross partner's slot
        // hop 1: the pair sum, lower card first, kept in y and pushed to the cross partner
        if (tid == 0) { qz_xflag_k(A, tok, kp, b); qz_xwait_slot(A, tok, (kp - 1) * QZ_XFS + b); }
        __syncthreads();
        for (int g = b; g * 32 < XA::DM; g += gridDim.x) {
            if (tid < 32 * T) {
                const int t = tid >> 5, i = tid & 31;
                const size_t o = (size_t) t * XA::DM + g * 32 + i;
                const float own = __builtin_nontemporal_load(A.y_part + o);
                const float pv = __builtin_nontemporal_load(src + (size_t) (kp - 1) * XS + o);
                const float sp = odd ? __fadd_rn(pv, own) : __fadd_rn(own, pv);
                y[o] = sp;
                QZ_XCK(qz_xck_store(qz_xcard(A), 2, XS_HOP2));
                dst[o] = sp;
            }
        }
        __threadfence_system();
        QZ_XCK(qz_xck_end(qz_xcard(A)));
        __syncthreads();
        // hop 2: lower pair's sum first
        if (tid == 0) { qz_xflag_k(A, tok, 2, b); qz_xwait_slot(A, tok, QZ_XFS + b); }
        __syncthreads();
        for (int g = b; g * 32 < XA::DM; g += gridDim.x) {
            if (tid < 32 * T) {
                const int t = tid >> 5, i = tid & 31;
                const size_t o = (size_t) t * XA::DM + g * 32 + i;
                const float sp = y[o];
                const float pc = __builtin_nontemporal_load(src + XS + o);
                y[o] = hi ? __fadd_rn(pc, sp) : __fadd_rn(sp, pc);
            }
        }
    }
    if constexpr (BX) qz_grid_barrier(bst, ++kb);
}

// one row-parallel phase with its exchange: the GEMV and the push, then the flag,
// the wait and the combine
template <int T, int CX, bool LX = T <= 4, bool BX = true, class XA> __device__ __forceinline__ void qz_xrow_t(
        qz_barrier_state * bst, int & kb, const XA & A, unsigned int tok,
        const unsigned char * a_vals, const float * a_scal, int K,
        const char * w_vals, const __half * w_scal, float * stage, float * y, unsigned char * lds0) {
    qz_gemv_xpush_t<T, CX, LX>(a_vals, a_scal, K, w_vals, w_scal, A, tok, stage, lds0);
    // BX is forwarded, since a standalone launch has no barrier state
    qz_xfinish_t<T, CX, BX>(bst, kb, A, tok, y);
}

// head merge. The larger value wins, and a tie goes to the lower token id: a total
// order, so the NC-way merge is the same in any order.
__device__ __forceinline__ int qz_xpick(float ov, int oi, float pv, int pi) {
    return (pv > ov || (pv == ov && pi < oi)) ? pi : oi;
}
// fold one {value, id} pair into the running best, the same rule
__device__ __forceinline__ void qz_xfold(float & bv, int & bi, float pv, int pi) {
    if (pv > bv || (pv == bv && pi < bi)) { bv = pv; bi = pi; }
}

// ---- exchange handles for kernels that push and sum inline (Inkling) ---------------
// qz_xpush: this card's slot in the one peer it pushes to, at one offset: the peer at two
// cards, the pair partner (hop 1) at four; the arch runs hop 2 (Inkling: cyc_xwait2).
// A null p[0] means no push; at four cards p[1], p[2] are unused.
template <int NC> struct qz_xpush {
    float * p[NC - 1];
#ifdef QZ_XCHECK
    int card = 0, k = 1;
#endif
    __device__ __forceinline__ explicit operator bool() const { return p[0] != nullptr; }
    __device__ __forceinline__ void put(size_t i, float v) const {
        QZ_XCK(qz_xck_store(card, k, XS_INK_PUT));
        qz_gp(p[0])[i] = v;
    }
    __device__ __forceinline__ void put4(size_t i, float4 v) const {
        QZ_XCK(qz_xck_store(card, k, XS_INK_PUT4));
        *(float4 *) (qz_gp(p[0]) + i) = v;
    }
};
// qz_xinbox: this card's inbox at one offset, slot stride `slot` floats, card me. add()
// sums own with the peers' values at i in card order: own + peer at two cards (the
// same bits either way round). At four, after the two hops, the pair partner's partial
// sits in slot 0 (even card) or 2 (odd) and the cross partner's pair sum in slot 1, and
// add() gives (p0 + p1) + (p2 + p3), the bits of the one-hop card-order sum.
template <int NC> struct qz_xinbox {
    const float * base;
    size_t slot;
    int me;
    __device__ __forceinline__ float add(float own, size_t i) const {
        if constexpr (NC == 2) {
            return own + __builtin_nontemporal_load(qz_gp(base) + i);
        } else {
            static_assert(NC == 4, "card-order sum for 2 or 4 cards");
            const bool odd = me & 1, hi = me >= 2;
            const float pv = __builtin_nontemporal_load(qz_gp(base) + (odd ? 2 * slot : 0) + i);
            const float sp = odd ? __fadd_rn(pv, own) : __fadd_rn(own, pv);
            const float pc = __builtin_nontemporal_load(qz_gp(base) + slot + i);
            return hi ? __fadd_rn(pc, sp) : __fadd_rn(sp, pc);
        }
    }
};
