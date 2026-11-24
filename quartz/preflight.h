// Start-up link check, so a broken link fails in seconds instead of minutes into the
// weight load. p2p: writes cross the link in both directions, then the words are read back.
// host: a kernel on one card fills the pinned mailbox and a kernel on the other checks it,
// both ways, then a flag ping-pong says what a round trip costs (core/link.cpp, its own
// translation unit so the archs' device code stays as it was). The per-architecture memory
// check lives in arch/<arch>/*_preflight.h.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/devices.h"
#include "core/link.h"     // qz_xalloc / qz_xfree, qz_link

static __global__ void qz_p2p_fill(unsigned * __restrict__ dst, size_t n, unsigned seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = seed ^ (unsigned) (i * 2654435761u);
}

// One card's kernel writes into the other card's memory, on the uncached pages
// the engine's exchange uses, and the words are read back from the owner. Returns
// 0 when both directions are intact, 1 on refusal with the reason printed.
inline int qz_link_check_p2p(double timeout_s, double * gbs) {
    const size_t n = (size_t) 16 << 20;              // 64 MB of words
    for (int c = 0; c < 2; c++) {
        if (hipSetDevice(qz_dev_of(c)) != hipSuccess) return 1;
        hipError_t e = hipDeviceEnablePeerAccess(qz_dev_of(1 - c), 0);
        if (e == hipErrorPeerAccessAlreadyEnabled) { (void) hipGetLastError(); e = hipSuccess; }
        if (e != hipSuccess) {
            fprintf(stderr, "quartz: peer access HIP %d -> HIP %d cannot be enabled: %s\n",
                    qz_dev_of(c), qz_dev_of(1 - c), hipGetErrorString(e));
            return 1;
        }
    }
    std::vector<unsigned> host(n);
    for (int c = 0; c < 2; c++) {
        const int src = qz_dev_of(c), dst = qz_dev_of(1 - c);
        unsigned * buf = nullptr;
        hipStream_t st = nullptr;
        hipEvent_t e0 = nullptr, e1 = nullptr;
        int rc = 0;
        if (hipSetDevice(dst) != hipSuccess || hipExtMallocWithFlags((void **) &buf, n * 4, hipDeviceMallocUncached) != hipSuccess ||
            hipMemset(buf, 0, n * 4) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            fprintf(stderr, "quartz: peer check: cannot allocate 64 MB on HIP %d\n", dst);
            return 1;
        }
        const unsigned seed = 0x9e3779b9u + (unsigned) c;
        char what[64];
        snprintf(what, sizeof what, "HIP %d writing into HIP %d", src, dst);
        (void) hipSetDevice(src);
        (void) hipStreamCreateWithFlags(&st, hipStreamNonBlocking);
        (void) hipEventCreate(&e0); (void) hipEventCreate(&e1);
        for (int rep = 0; rep < 2 && !rc; rep++) {       // first pass warms up, second is timed
            (void) hipEventRecord(e0, st);
            hipLaunchKernelGGL(qz_p2p_fill, dim3(1024), dim3(256), 0, st, buf, n, seed);
            (void) hipEventRecord(e1, st);
            rc = qz_link_wait(st, "quartz: peer check", what, "the peer link hangs", timeout_s);
        }
        if (rc == 2) return 2;                           // stuck: nothing can be freed safely
        float ms = 0.f;
        if (!rc && hipEventElapsedTime(&ms, e0, e1) == hipSuccess && ms > 0) gbs[c] = n * 4 / (ms * 1e6);
        if (!rc) {
            (void) hipSetDevice(dst);
            if (hipMemcpy(host.data(), buf, n * 4, hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "quartz: peer check: cannot read back HIP %d\n", dst);
                rc = 1;
            } else {
                size_t bad = 0;
                for (size_t i = 0; i < n; i++) bad += host[i] != (seed ^ (unsigned) (i * 2654435761u));
                if (bad) {
                    fprintf(stderr, "quartz: peer check: HIP %d writing into HIP %d: %zu of %zu words arrived wrong; "
                                    "cross-card writes are not reliable on this machine\n", src, dst, bad, n);
                    rc = 1;
                }
            }
        }
        (void) hipSetDevice(src);
        (void) hipEventDestroy(e0); (void) hipEventDestroy(e1);
        (void) hipStreamDestroy(st);
        (void) hipSetDevice(dst);
        (void) hipFree(buf);
        if (rc) return 1;
    }
    return 0;
}

// host mode, in core/link.cpp: no peer access at all, both cards reach one pinned host
// mailbox through the same pointer, so the test's kernels live there
int qz_link_check_host(double timeout_s, double * gbs, double * rt_us);

// 0 when the link carries the exchange, 1 on refusal with the reason printed. Takes the
// mode from devices.h: with peer access (p2p) as it always did, else through host memory.
inline int qz_link_check() {
    const double timeout_s = 10.0;
    double gbs[2] = {};
    if (qz_link == QZ_LINK_P2P) {
        const int rc = qz_link_check_p2p(timeout_s, gbs);
        // auto: a peer link that fails its check (not a hang: that kernel may still be
        // running) is not used; the exchange goes through host memory instead. Nothing
        // cross-card is allocated yet, so the mode can still change here.
        if (rc == 1 && qz_link_auto) {
            fprintf(stderr, "quartz: the peer link failed its check, so the exchange goes through host memory (QZ_LINK=p2p refuses instead)\n");
            qz_link = QZ_LINK_HOST;
        } else if (rc) {
            return 1;
        }
    }
    if (qz_link == QZ_LINK_P2P) {
        (void) hipSetDevice(qz_dev_of(0));
        fprintf(stderr, "quartz: peer link ok: HIP %d -> %d %.1f GB/s, HIP %d -> %d %.1f GB/s\n",
                qz_dev_of(0), qz_dev_of(1), gbs[0], qz_dev_of(1), qz_dev_of(0), gbs[1]);
        return 0;
    }
    double rt_us = 0;
    if (qz_link_check_host(timeout_s, gbs, &rt_us)) return 1;
    (void) hipSetDevice(qz_dev_of(0));
    fprintf(stderr, "quartz: link host (no P2P): HIP %d -> %d %.1f GB/s, HIP %d -> %d %.1f GB/s, flag round trip %.2f us\n",
            qz_dev_of(0), qz_dev_of(1), gbs[0], qz_dev_of(1), qz_dev_of(0), gbs[1], rt_us);
    return 0;
}
