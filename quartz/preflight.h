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
#include <algorithm>
#include <cstring>
#include <vector>

#include "core/devices.h"
#include "core/link.h"     // qz_xalloc / qz_xfree, qz_link

static __global__ void qz_p2p_fill(unsigned * __restrict__ dst, size_t n, unsigned seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = seed ^ (unsigned) (i * 2654435761u);
}

// One GPU's kernel writes into the other GPU's memory, on the uncached pages the engine's
// exchange uses, and the words are read back from the owner. Returns 0 when the direction
// is intact (its bandwidth in *gbs), 1 on refusal with the reason printed, 2 on a hang.
inline int qz_link_check_dir(int src, int dst, double timeout_s, double * gbs) {
    const size_t n = (size_t) 16 << 20;              // 64 MB of words
    static std::vector<unsigned> host;
    host.resize(n);
    unsigned * buf = nullptr;
    hipStream_t st = nullptr;
    hipEvent_t e0 = nullptr, e1 = nullptr;
    int rc = 0;
    if (hipSetDevice(dst) != hipSuccess || hipExtMallocWithFlags((void **) &buf, n * 4, hipDeviceMallocUncached) != hipSuccess ||
        hipMemset(buf, 0, n * 4) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, "quartz: peer check: cannot allocate 64 MB on HIP %d\n", dst);
        return 1;
    }
    const unsigned seed = 0x9e3779b9u + (unsigned) src * 7u + (unsigned) dst;
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
    if (!rc && hipEventElapsedTime(&ms, e0, e1) == hipSuccess && ms > 0) *gbs = n * 4 / (ms * 1e6);
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
    return rc;
}

// the GPUs the cards sit on: cards 0 .. qz_nphys - 1 are on distinct GPUs
inline int qz_link_gpu(int g) { return qz_dev_of(g); }

// Every directed pair of the GPUs: peer access, then qz_link_check_dir. gbs[a * 4 + b] is
// GPU a writing into GPU b (GPU g: the one card g sits on).
inline int qz_link_check_p2p(double timeout_s, double * gbs) {
    const int ng = qz_nphys;
    for (int a = 0; a < ng; a++)
        for (int b = 0; b < ng; b++) {
            if (a == b) continue;
            if (hipSetDevice(qz_link_gpu(a)) != hipSuccess) return 1;
            hipError_t e = hipDeviceEnablePeerAccess(qz_link_gpu(b), 0);
            if (e == hipErrorPeerAccessAlreadyEnabled) { (void) hipGetLastError(); e = hipSuccess; }
            if (e != hipSuccess) {
                fprintf(stderr, "quartz: peer access HIP %d -> HIP %d cannot be enabled: %s\n",
                        qz_link_gpu(a), qz_link_gpu(b), hipGetErrorString(e));
                return 1;
            }
        }
    for (int a = 0; a < ng; a++)
        for (int b = 0; b < ng; b++) {
            if (a == b) continue;
            const int rc = qz_link_check_dir(qz_link_gpu(a), qz_link_gpu(b), timeout_s, &gbs[a * 4 + b]);
            if (rc) return rc;
        }
    return 0;
}

// the flag round trip between two GPUs over peer writes, in core/link.cpp
int qz_link_rt_p2p(int a, int b, double timeout_s, double * rt_us);

// host mode, in core/link.cpp: no peer access at all, both cards reach one pinned host
// mailbox through the same pointer, so the test's kernels live there
int qz_link_check_host(double timeout_s, double * gbs, double * rt_us);

// 0 when the link carries the exchange, 1 on refusal with the reason printed. Takes the
// mode from devices.h: with peer access (p2p) as it always did, else through host memory.
inline int qz_link_check() {
    const double timeout_s = 10.0;
    double gbs[16] = {};
    if (qz_nphys == 1) {   // virtual cards on one GPU: nothing crosses a link
        fprintf(stderr, "quartz: one GPU (HIP %d), no peer link\n", qz_dev_of(0));
        return 0;
    }
    if (qz_link == QZ_LINK_P2P) {
        const int rc = qz_link_check_p2p(timeout_s, gbs);
        // auto: a peer link that fails its check (not a hang: that kernel may still be
        // running) is not used; the exchange goes through host memory instead. Nothing
        // cross-card is allocated yet, so the mode can still change here.
        if (rc == 1 && qz_link_auto && qz_nphys == 2 && qz_ncard == 2) {
            fprintf(stderr, "quartz: the peer link failed its check, so the exchange goes through host memory (QZ_LINK=p2p refuses instead)\n");
            qz_link = QZ_LINK_HOST;
        } else if (rc) {
            return 1;
        }
    }
    if (qz_link == QZ_LINK_P2P && qz_nphys == 2) {
        (void) hipSetDevice(qz_dev_of(0));
        fprintf(stderr, "quartz: peer link ok: HIP %d -> %d %.1f GB/s, HIP %d -> %d %.1f GB/s\n",
                qz_dev_of(0), qz_dev_of(1), gbs[1], qz_dev_of(1), qz_dev_of(0), gbs[4]);
        return 0;
    }
    if (qz_link == QZ_LINK_P2P) {   // more GPUs: the matrix, then a flag round trip per pair
        const int ng = qz_nphys;
        double rt[16] = {}, lo = 1e30, hi_rt = 0;
        for (int a = 0; a < ng; a++)
            for (int b = a + 1; b < ng; b++) {
                const int rc = qz_link_rt_p2p(qz_link_gpu(a), qz_link_gpu(b), timeout_s, &rt[a * 4 + b]);
                if (rc) return 1;
                rt[b * 4 + a] = rt[a * 4 + b];
                hi_rt = std::max(hi_rt, rt[a * 4 + b]);
            }
        for (int a = 0; a < ng; a++)
            for (int b = 0; b < ng; b++) if (a != b) lo = std::min(lo, gbs[a * 4 + b]);
        (void) hipSetDevice(qz_dev_of(0));
        fprintf(stderr, "quartz: peer link ok: %d GPUs, every pair, slowest %.1f GB/s, flag round trip up to %.2f us\n", ng, lo, hi_rt);
        fprintf(stderr, "quartz: peer writes GB/s (row writes into column) | flag round trip us\n        ");
        for (int b = 0; b < ng; b++) fprintf(stderr, " HIP %-3d", qz_link_gpu(b));
        fprintf(stderr, " |");
        for (int b = 0; b < ng; b++) fprintf(stderr, " HIP %-3d", qz_link_gpu(b));
        fprintf(stderr, "\n");
        for (int a = 0; a < ng; a++) {
            fprintf(stderr, "  HIP %-2d", qz_link_gpu(a));
            for (int b = 0; b < ng; b++) a == b ? fprintf(stderr, "     -  ") : fprintf(stderr, " %7.1f", gbs[a * 4 + b]);
            fprintf(stderr, " |");
            for (int b = 0; b < ng; b++) a == b ? fprintf(stderr, "     -  ") : fprintf(stderr, " %7.2f", rt[a * 4 + b]);
            fprintf(stderr, "\n");
        }
        return 0;
    }
    double rt_us = 0;
    if (qz_link_check_host(timeout_s, gbs, &rt_us)) return 1;
    (void) hipSetDevice(qz_dev_of(0));
    fprintf(stderr, "quartz: link host (no P2P): HIP %d -> %d %.1f GB/s, HIP %d -> %d %.1f GB/s, flag round trip %.2f us\n",
            qz_dev_of(0), qz_dev_of(1), gbs[0], qz_dev_of(1), qz_dev_of(0), gbs[1], rt_us);
    return 0;
}
