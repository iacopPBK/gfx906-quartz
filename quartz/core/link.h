// The cross-card link. With peer access (QZ_LINK=p2p) a mailbox is uncached VRAM on the
// owner's card and the peer reaches it directly. Without it (QZ_LINK=host) the mailbox is
// pinned host memory instead: fine-grained and coherent, so the exchange's release stores
// and acquire loads work on it unchanged, and both cards address it with one pointer. The
// mode is resolved with the devices (core/devices.h); kernels never see the difference.
#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "core/devices.h"    // qz_link, qz_dev_of, qz_dev_max
#include "mem.h"             // qz_dmalloc_uncached, qz_dfree, the ledger

// A mailbox both cards read and write, zeroed, valid on `card` and on its peer: uncached
// VRAM on the current device (a peer's stores must land in HBM, not behind this card's L2)
// under p2p, else pinned host memory the cards map and reach over PCIe. `card` is the
// owner's index, for the ledger and the messages.
inline hipError_t qz_xalloc(void ** p, size_t bytes, int card, const char * sub) {
    *p = nullptr;
    if (qz_link == QZ_LINK_P2P) {
        const hipError_t e = qz_dmalloc_uncached(p, bytes, sub);
        if (e != hipSuccess) return e;
        return hipMemset(*p, 0, bytes);
    }
    const unsigned int fl = hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped;
    void * hp = nullptr;
    hipError_t e = hipHostMalloc(&hp, bytes, fl);
    if (e != hipSuccess) return e;
    memset(hp, 0, bytes);                       // the host's own zeroing: no launch, no stream
    // one pointer for both cards: a kernel writes the peer's mailbox through the same
    // pointer its owner reads it with, so the two mappings must be the same address
    void * dp[2] = { nullptr, nullptr };
    int dev0 = 0;
    (void) hipGetDevice(&dev0);
    for (int c = 0; c < 2 && e == hipSuccess; c++)
        if ((e = hipSetDevice(qz_dev_of(c))) == hipSuccess) e = hipHostGetDevicePointer(&dp[c], hp, 0);
    (void) hipSetDevice(dev0);
    if (e == hipSuccess && dp[0] != dp[1]) {
        fprintf(stderr, "quartz: host mailbox (card %d, %zu B): HIP %d maps it at %p, HIP %d at %p; "
                        "the two cards must address it with one pointer\n",
                card, bytes, qz_dev_of(0), dp[0], qz_dev_of(1), dp[1]);
        e = hipErrorInvalidValue;
    }
    if (e != hipSuccess) { (void) hipHostFree(hp); return e; }
    qz_mem_host_add(dp[0], hp, bytes, sub);
    *p = dp[0];
    return hipSuccess;
}

// Release a mailbox from qz_xalloc. The cards see the device pointer, the runtime wants
// the host one back.
inline hipError_t qz_xfree(void * p) {
    if (!p) return hipSuccess;
    if (qz_link == QZ_LINK_P2P) return qz_dfree(p);
    void * hp = qz_mem_host_del(p);
    if (!hp) {
        fprintf(stderr, "quartz: qz_xfree: %p is not a host mailbox\n", p);
        return hipErrorInvalidValue;
    }
    return hipHostFree(hp);
}

// ---- bulk copies (prefill, DSpine, DSpark) ----
// p2p: hipMemcpyPeerAsync on the source's stream. host: D2H into a pinned staging slot on
// s_src, an event, then H2D on s_dst. Either way the copy is complete in both streams'
// order, so a caller keeps its existing event handoff.
#define QZ_XRING 8           // staging slots per card pair: ~6 copies are in flight between two syncs

struct qz_xslot {
    char * buf = nullptr;        // pinned staging, grown on demand
    size_t bytes = 0;
    hipEvent_t ev_src = nullptr; // the D2H through it, on the source card
    hipEvent_t ev_dst = nullptr; // its H2D, on the destination card
};
inline qz_xslot qz_xring[qz_dev_max][qz_dev_max][QZ_XRING];
inline int qz_xring_next[qz_dev_max][qz_dev_max];

// A slot of `bytes` for a copy from `src` to `dst`. A slot comes back only once its last
// H2D has finished, so the device is never still reading a buffer the host overwrites.
inline hipError_t qz_xslot_get(int src, int dst, size_t bytes, qz_xslot ** slot) {
    qz_xslot & R = qz_xring[src][dst][qz_xring_next[src][dst]];
    qz_xring_next[src][dst] = (qz_xring_next[src][dst] + 1) % QZ_XRING;
    int dev0 = 0;
    (void) hipGetDevice(&dev0);
    hipError_t e = hipSuccess;
    if (R.ev_dst && (e = hipSetDevice(dst)) == hipSuccess) e = hipEventSynchronize(R.ev_dst);
    if (e == hipSuccess && R.bytes < bytes) {
        if (R.buf) { (void) hipHostFree(R.buf); R.buf = nullptr; R.bytes = 0; }
        if ((e = hipHostMalloc((void **) &R.buf, bytes, hipHostMallocPortable)) == hipSuccess) R.bytes = bytes;
        else R.buf = nullptr;
    }
    if (e == hipSuccess && (!R.ev_src || !R.ev_dst)) {   // one event per side; the D2H's, then the H2D's
        if ((e = hipSetDevice(src)) == hipSuccess) e = hipEventCreateWithFlags(&R.ev_src, hipEventDisableTiming);
        if (e == hipSuccess && (e = hipSetDevice(dst)) == hipSuccess) e = hipEventCreateWithFlags(&R.ev_dst, hipEventDisableTiming);
    }
    (void) hipSetDevice(dev0);
    if (e == hipSuccess) *slot = &R;
    return e;
}

inline hipError_t qz_xcopy(void * dst, int dst_card, hipStream_t s_dst,
                           const void * src, int src_card, hipStream_t s_src, size_t bytes) {
    if (!bytes) return hipSuccess;
    if (src_card < 0 || src_card >= qz_dev_max || dst_card < 0 || dst_card >= qz_dev_max) return hipErrorInvalidDevice;
    int dev0 = 0;
    (void) hipGetDevice(&dev0);
    hipError_t e = hipSuccess;
    if (qz_link == QZ_LINK_P2P) {
        e = hipMemcpyPeerAsync(dst, dst_card, src, src_card, bytes, s_src);
        if (e != hipSuccess || !s_dst || s_dst == s_src) return e;
        // the destination stream is ordered behind the copy too, as it is under the host
        // staging: one event per source card, re-recorded per copy (single launch thread)
        static hipEvent_t ev[qz_dev_max] = {};
        if ((e = hipSetDevice(src_card)) == hipSuccess && !ev[src_card])
            e = hipEventCreateWithFlags(&ev[src_card], hipEventDisableTiming);
        if (e == hipSuccess) e = hipEventRecord(ev[src_card], s_src);
        if (e == hipSuccess) e = hipStreamWaitEvent(s_dst, ev[src_card], 0);
    } else {
        qz_xslot * R = nullptr;
        if ((e = qz_xslot_get(src_card, dst_card, bytes, &R)) == hipSuccess)
            e = hipSetDevice(src_card);
        if (e == hipSuccess) e = hipMemcpyAsync(R->buf, src, bytes, hipMemcpyDeviceToHost, s_src);
        if (e == hipSuccess) e = hipEventRecord(R->ev_src, s_src);
        if (e == hipSuccess) e = hipSetDevice(dst_card);
        if (e == hipSuccess) e = hipStreamWaitEvent(s_dst, R->ev_src, 0);
        if (e == hipSuccess) e = hipMemcpyAsync(dst, R->buf, bytes, hipMemcpyHostToDevice, s_dst);
        if (e == hipSuccess) e = hipEventRecord(R->ev_dst, s_dst);
    }
    (void) hipSetDevice(dev0);
    return e;
}

// ---- the start-up check's deadline ----
// A stream must finish within `timeout_s`: a stuck link reports instead of hanging. 1 when
// the stream failed, 2 on a timeout: the kernel may still be stuck, so the caller returns at
// once without freeing or destroying anything (a free would block on the stuck kernel).
inline int qz_link_wait(hipStream_t st, const char * tag, const char * what, const char * hang, double timeout_s) {
    const auto t0 = std::chrono::steady_clock::now();
    hipError_t q;
    while ((q = hipStreamQuery(st)) == hipErrorNotReady) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            fprintf(stderr, "%s: %s did not finish in %.0f s: %s\n", tag, what, timeout_s, hang);
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (q != hipSuccess) {
        fprintf(stderr, "%s: %s failed: %s\n", tag, what, hipGetErrorString(q));
        return 1;
    }
    return 0;
}
