// Tensor-parallel card plumbing shared by the archs: peer access, each card's exchange
// buffers, the exchange error words, and the residency check that keeps a persistent
// grid-barrier kernel from hanging.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "core/devices.h"
#include "core/link.h"
#include "mem.h"

// Peer access both ways between every two cards on different GPUs (devices.h); false,
// with a message. Two virtual cards on one GPU need none. Host mode has none to enable:
// the mailboxes are pinned host memory (core/link.h).
inline bool qz_peer_enable() {
    if (qz_link == QZ_LINK_HOST) return true;
    for (int a = 0; a < qz_ncard; a++)
        for (int b = 0; b < qz_ncard; b++) {
            if (qz_card_same_gpu(a, b)) continue;
            if (hipSetDevice(qz_dev_of(a)) != hipSuccess) return false;
            hipError_t e = hipDeviceEnablePeerAccess(qz_dev_of(b), 0);
            if (e == hipErrorPeerAccessAlreadyEnabled) { (void) hipGetLastError(); e = hipSuccess; }
            if (e != hipSuccess) {
                fprintf(stderr, "no peer access %d -> %d: %s\n", qz_dev_of(a), qz_dev_of(b), hipGetErrorString(e));
                return false;
            }
        }
    return true;
}

// One card's exchange buffers (kernels/xchg.h), nslot = cards - 1 slots of each mailbox:
// the inbox [2][rows][dm] with its checksums [2][rows][dm/32], the head-merge pairs
// [2][rows][2] of {value, id bits}, the flags [QZ_XFS] in their own allocation; then the
// exchange counter and the error words [8]. The mailboxes are cross-card (qz_xalloc);
// the counter and the error words are this card's own.
struct qz_xbufs {
    float * xin = nullptr, * xhm = nullptr;
    unsigned int * xflag = nullptr, * xseq = nullptr;
    int * xerr = nullptr;
    size_t xin_slot = 0, xhm_slot = 0;   // slot strides in floats
    static constexpr size_t xflag_slot = 128;   // QZ_XFS

    // on the current device, zeroed; card is the owner, for the link's ledger
    hipError_t alloc(int rows, int dm, int card, int nslot) {
        hipError_t e;
        xin_slot = (size_t) 2 * rows * dm + (size_t) 2 * rows * (dm / 32);
        xhm_slot = (size_t) 2 * rows * 2;
        if ((e = qz_xalloc((void **) &xin, nslot * xin_slot * 4, card, "xin")) ||
            (e = qz_xalloc((void **) &xhm, nslot * xhm_slot * 4, card, "xhm")) ||
            (e = qz_xalloc((void **) &xflag, nslot * xflag_slot * 4, card, "xflag"))) return e;
        if ((e = qz_dmalloc((void **) &xseq, 64)) || (e = hipMemset(xseq, 0, 64))) return e;
        if ((e = qz_dmalloc((void **) &xerr, 8 * sizeof(int))) || (e = hipMemset(xerr, 0, 8 * sizeof(int)))) return e;
        return hipSuccess;
    }
};

// The host-side error words of a card must be 0 after a sync, else report and exit.
inline void qz_xerr_check_h(int card, const int * e, long long n_launch) {
    if (e[0] || e[1]) {
        fprintf(stderr, "exchange error card %d after launch %lld: chk mismatches %d (first tok %d slab %d row %d), "
                        "abort %d code %d tok %d seen %d\n", card, n_launch, e[0], e[5], e[6], e[7], e[1], e[2], e[3], e[4]);
        exit(1);
    }
}
inline void qz_xerr_check(int card, const int * xerr, long long n_launch) {
    int e[8];
    if (hipMemcpy(e, xerr, sizeof(e), hipMemcpyDeviceToHost) != hipSuccess) { fprintf(stderr, "card %d: xerr read failed\n", card); exit(1); }
    qz_xerr_check_h(card, e, n_launch);
}

// A persistent kernel's grid must be resident at once: its grid barrier spins. False,
// with a message, if `grid` blocks of `threads` and `smem` bytes do not fit.
struct qz_resident_kernel { const void * f; int smem; const char * name; };
inline bool qz_resident(int dev, const qz_resident_kernel * ks, int n, int grid, int threads) {
    int cu = 0;
    if (hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess) return false;
    for (int i = 0; i < n; i++) {
        int per_cu = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, ks[i].f, threads, ks[i].smem) != hipSuccess) return false;
        if (per_cu * cu < grid) {
            fprintf(stderr, "card %d: %s fits %d x %d CU blocks < %d (LDS %d B): grid barrier would hang\n",
                    dev, ks[i].name, per_cu, cu, grid, ks[i].smem);
            return false;
        }
    }
    return true;
}
