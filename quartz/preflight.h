// Start-up check that fails in seconds instead of minutes into the weight load:
// a write-and-read-back across the peer link in both directions. The memory
// check is per architecture (arch/<arch>/*_preflight.h).
#pragma once
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "devices.h"

static __global__ void qz_p2p_fill(unsigned * __restrict__ dst, size_t n, unsigned seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = seed ^ (unsigned) (i * 2654435761u);
}

// one card's kernel writes into the other card's memory (the uncached pages
// the engine's exchange uses), then the words are read back from the owner.
// 0: both directions intact; 1: refused (the reason printed)
inline int qz_p2p_check() {
    const size_t n = (size_t) 16 << 20;              // 64 MB of words
    const double timeout_s = 10.0;
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
    double gbs[2] = {};
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
        (void) hipSetDevice(src);
        (void) hipStreamCreateWithFlags(&st, hipStreamNonBlocking);
        (void) hipEventCreate(&e0); (void) hipEventCreate(&e1);
        for (int rep = 0; rep < 2 && !rc; rep++) {       // the first pass warms up, the second is timed
            (void) hipEventRecord(e0, st);
            hipLaunchKernelGGL(qz_p2p_fill, dim3(1024), dim3(256), 0, st, buf, n, seed);
            (void) hipEventRecord(e1, st);
            const auto t0 = std::chrono::steady_clock::now();
            hipError_t q;
            while ((q = hipStreamQuery(st)) == hipErrorNotReady) {
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
                    fprintf(stderr, "quartz: peer check: HIP %d writing into HIP %d did not finish in %.0f s: the peer link hangs\n",
                            src, dst, timeout_s);
                    return 1;      // the kernel may still be stuck: nothing more can be freed safely
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (q != hipSuccess) {
                fprintf(stderr, "quartz: peer check: HIP %d writing into HIP %d failed: %s\n", src, dst, hipGetErrorString(q));
                rc = 1;
            }
        }
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
    (void) hipSetDevice(qz_dev_of(0));
    fprintf(stderr, "quartz: peer link ok: HIP %d -> %d %.1f GB/s, HIP %d -> %d %.1f GB/s\n",
            qz_dev_of(0), qz_dev_of(1), gbs[0], qz_dev_of(1), qz_dev_of(0), gbs[1]);
    return 0;
}
