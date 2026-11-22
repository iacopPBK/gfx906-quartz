// Start-up checks that fail in seconds, with numbers, instead of minutes into
// the weight load: the memory each card needs for the requested context, and a
// write-and-read-back across the peer link in both directions.
#pragma once
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "devices.h"
#include "engine.h"
#include "loader.h"

// from the per-card ledger (2026-11-22): DSpine takes 3.8 GB at any context;
// the cycle buffers, the snapshot staging, the drafter switch and the HIP
// runtime's own share about 1 GB; the batched prefill wants 256 MB left free
constexpr double QZ_PF_DSPINE = 3.85e9, QZ_PF_FIXED = 1.0e9, QZ_PF_HEADROOM = 0.25e9;
// with --mmproj both cards grow by 1.1-1.2 GB at load and one by ~0.3 GB more
// once an image is encoded (0.93 GB projector): its file size + 1 GB on each
constexpr double QZ_PF_MMPROJ_EXTRA = 1.0e9;

inline double qz_file_bytes(const char * p) {
    struct stat st;
    return p && stat(p, &st) == 0 ? (double) st.st_size : 0.0;
}

// 0: fits (or cannot be sized here: init reports the model problem); 1: refused
inline int qz_vram_check(const char * model, int max_ctx, int steps, size_t scratch, const char * dspine, const char * mmproj) {
    qz_gguf g;
    if (!g.open(model)) return 0;
    double weights[2] = {}, free_b[2] = {}, total_b[2] = {};
    for (int c = 0; c < 2; c++) {
        qz_loader L;
        if (!L.plan(g, c)) return 0;
        weights[c] = (double) L.need_bytes;
        size_t fr = 0, tot = 0;
        if (hipSetDevice(qz_dev_of(c)) != hipSuccess || hipMemGetInfo(&fr, &tot) != hipSuccess) return 0;
        free_b[c] = (double) fr; total_b[c] = (double) tot;
    }
    (void) hipSetDevice(qz_dev_of(0));
    const double kv = (double) qz_kv_bytes(steps), ds = dspine ? QZ_PF_DSPINE : 0.0;
    const double mm = mmproj ? qz_file_bytes(mmproj) + QZ_PF_MMPROJ_EXTRA : 0.0;
    const double buf = (double) scratch + QZ_PF_FIXED + QZ_PF_HEADROOM;
    double need[2];
    for (int c = 0; c < 2; c++) need[c] = weights[c] + kv + ds + mm + buf;
    char extra[64] = "";
    if (ds > 0) snprintf(extra, sizeof extra, " + DSpine %.1f", ds / 1e9);
    if (mm > 0) snprintf(extra + strlen(extra), sizeof extra - strlen(extra), " + vision %.1f", mm / 1e9);
    fprintf(stderr, "quartz: memory per card: weights %.1f + KV %.1f (ctx %d)%s + buffers %.1f = %.1f GB; free %.1f and %.1f GB\n",
            weights[0] / 1e9, kv / 1e9, max_ctx, extra, buf / 1e9, need[0] / 1e9, free_b[0] / 1e9, free_b[1] / 1e9);
    int bad = -1;
    for (int c = 0; c < 2; c++)
        if (need[c] > free_b[c] && (bad < 0 || need[c] - free_b[c] > need[bad] - free_b[bad])) bad = c;
    if (bad < 0) return 0;

    // the largest context whose K / V fits next to everything else on both cards
    int fit = 0;
    for (int lo = 1024, hi = max_ctx; lo <= hi; ) {
        const int mid = lo + (hi - lo) / 2;
        const double kv_m = (double) qz_kv_bytes(mid + 2 * QZ_TMAX);
        bool ok = true;
        for (int c = 0; c < 2; c++) ok = ok && need[c] - kv + kv_m <= free_b[c];
        if (ok) { fit = mid; lo = mid + 1; } else hi = mid - 1;
    }
    fit = fit / 1024 * 1024;
    fprintf(stderr, "quartz: card %d (HIP %d) has %.1f of %.1f GB free and this start needs about %.1f GB\n",
            bad, qz_dev_of(bad), free_b[bad] / 1e9, total_b[bad] / 1e9, need[bad] / 1e9);
    if (total_b[bad] - free_b[bad] > 1.0e9)
        fprintf(stderr, "        %.1f GB of it is already in use: another process (an old server?) may hold it\n",
                (total_b[bad] - free_b[bad]) / 1e9);
    if (fit >= 1024) fprintf(stderr, "        --max-ctx %d fits%s\n", fit, ds > 0 ? ", or drop --dspine for more context" : "");
    else fprintf(stderr, "        even --max-ctx 1024 does not fit%s\n", ds > 0 ? ": free the card or drop --dspine" : ": free the card");
    if (getenv("QZ_NO_PREFLIGHT")) {
        fprintf(stderr, "quartz: QZ_NO_PREFLIGHT set: loading anyway\n");
        return 0;
    }
    return 1;
}

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
