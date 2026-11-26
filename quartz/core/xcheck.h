// Fan-out check (check builds only: make tp4 XCHECK=1, which defines QZ_XCHECK). On some
// hosts one GPU writing to several peers at once collapses P2P (0.3 GB/s on a 4x MI50 EPYC
// box), so the exchanges write to one peer per step. Emulated cards cannot show the
// collapse, so this build records the pattern instead:
//  - device: each thread remembers the peer it last stored to; a store to a different peer
//    before a system fence, or after its own flag, is a fan-out at that site;
//  - host: a copy from a card to one peer while its copy to another peer, queued on a
//    different stream, has not finished.
// qz_xck_poll() prints the new counts per site (from the end of each generate). Normal
// builds compile none of it.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>

#include "core/devices.h"

// device sites, and their names for the report
enum qz_xsite { XS_PUSH, XS_HOP2, XS_FLAG, XS_FLAGK, XS_HM_MTP, XS_HM_CYC, XS_HM_VER, XS_VPUSH, XS_INK_PUT, XS_INK_PUT4, XS_N };
inline const char * qz_xsite_name(int s) {
    static const char * n[XS_N] = { "row push (kernels/xchg.h)", "hop-2 push (kernels/xchg.h)", "flag to every peer (qz_xflag_b)",
                                     "one-peer flag (qz_xflag_k)", "head merge, MTP (qwen35_cycle.hip)", "head merge, cycle (qwen35_cycle.hip)",
                                     "head merge, verify (dspine_verify.h)", "verify row push (dspine_verify.h)",
                                     "Inkling push put (qz_xpush)", "Inkling push put4 (qz_xpush)" };
    return s >= 0 && s < XS_N ? n[s] : "?";
}

#ifdef QZ_XCHECK
#define QZ_XCK_B 256
static __device__ signed char qz_xck_last[QZ_NC_MAX][QZ_XCK_B][512];   // [card][block & 255][thread]: last peer k, 0 none
static __device__ unsigned long long qz_xck_cnt[XS_N][QZ_NC_MAX];       // fan-out events per site and card

// a store from card `card` into its peer k (1 .. NC-1) at site s
__device__ __forceinline__ void qz_xck_store(int card, int k, int s) {
    signed char & l = qz_xck_last[card][blockIdx.x & (QZ_XCK_B - 1)][threadIdx.x & 511];
    if (l > 0 && l != k) atomicAdd(&qz_xck_cnt[s][card], 1ull);
    l = (signed char) k;
}
// the step ends: a system fence, or the thread's own flag store
__device__ __forceinline__ void qz_xck_end(int card) { qz_xck_last[card][blockIdx.x & (QZ_XCK_B - 1)][threadIdx.x & 511] = 0; }
// one store that goes to n peers at once (the Inkling push handle)
__device__ __forceinline__ void qz_xck_fan(int card, int n, int s) { if (n > 1) atomicAdd(&qz_xck_cnt[s][card], 1ull); }
#define QZ_XCK(x) x
#else
#define QZ_XCK(x)
#endif

// host copies: per source card, the last copies (stream, peer, event)
inline void qz_xck_copy(int src, int dst, hipStream_t st, const char * file, int line) {
#ifdef QZ_XCHECK
    struct rec { hipStream_t st = nullptr; int dst = -1; hipEvent_t ev = nullptr; };
    static rec R[QZ_NC_MAX][8];
    static int nxt[QZ_NC_MAX];
    struct site { const char * f; int l; unsigned long long n; };
    static site S[64];
    if (!file) {   // the report
        for (auto & x : S)
            if (x.f && x.n) { fprintf(stderr, "XCHECK host: %s:%d: a card copied to two peers at once, %llu times\n", x.f, x.l, x.n); x.n = 0; }
        return;
    }
    int dev0 = 0;
    (void) hipGetDevice(&dev0);
    if (src < 0 || src >= QZ_NC_MAX) return;
    for (auto & r : R[src])
        if (r.ev && r.st != st && r.dst != dst && hipEventQuery(r.ev) == hipErrorNotReady) {
            for (auto & x : S)
                if (!x.f || (x.f == file && x.l == line)) { x.f = file; x.l = line; x.n++; break; }
            break;
        }
    rec & r = R[src][nxt[src]++ & 7];
    (void) hipSetDevice(qz_dev_of(src));
    if (!r.ev) (void) hipEventCreateWithFlags(&r.ev, hipEventDisableTiming);
    r.st = st; r.dst = dst;
    (void) hipEventRecord(r.ev, st);
    (void) hipSetDevice(dev0);
#else
    (void) src; (void) dst; (void) st; (void) file; (void) line;
#endif
}

// print the new fan-out counts of this translation unit's kernels, per site and card
static inline void qz_xck_poll(int ncards) {
#ifdef QZ_XCHECK
    static unsigned long long seen[XS_N][QZ_NC_MAX];
    int dev0 = 0;
    (void) hipGetDevice(&dev0);
    bool any = false;
    for (int c = 0; c < ncards && c < QZ_NC_MAX; c++) {
        unsigned long long cnt[XS_N][QZ_NC_MAX];
        (void) hipSetDevice(qz_dev_of(c));
        (void) hipDeviceSynchronize();
        if (hipMemcpyFromSymbol(cnt, HIP_SYMBOL(qz_xck_cnt), sizeof(cnt)) != hipSuccess) continue;
        for (int s = 0; s < XS_N; s++)
            if (cnt[s][c] > seen[s][c]) {
                fprintf(stderr, "XCHECK: card %d, %s: %llu stores to a second peer in one step\n", c, qz_xsite_name(s), cnt[s][c] - seen[s][c]);
                seen[s][c] = cnt[s][c];
                any = true;
            }
    }
    (void) hipSetDevice(dev0);
    qz_xck_copy(0, -1, nullptr, nullptr, 0);   // the host report
    if (!any) fprintf(stderr, "XCHECK: no device fan-out\n");
#else
    (void) ncards;
#endif
}
