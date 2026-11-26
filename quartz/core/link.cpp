// The host link's start-up check, qz_link_check_host (preflight.h): with no peer access both
// cards reach one pinned host mailbox through the same pointer, so a kernel on one card fills
// it and a kernel on the other checks the words, both ways, then 1000 flag round trips.
//
// Its own translation unit on purpose: the kernels below only exist for this check, and adding
// them to the archs' modules moved an unrelated branch and an unrelated shuffle lowering there
// (same source, same compiler, two more kernels in the module). Here the archs' device code stays
// byte-identical to a build without host mode.
#include "core/link.h"

#include <chrono>
#include <cstdio>

// the pattern the p2p check writes, into host memory this time
static __global__ void qz_link_fill(unsigned * __restrict__ dst, size_t n, unsigned seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = seed ^ (unsigned) (i * 2654435761u);
}

// the other card counts the words the fill did not reach
static __global__ void qz_link_scan(const unsigned * __restrict__ src, size_t n, unsigned seed, int * bad) {
    int b = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        b += src[i] != (seed ^ (unsigned) (i * 2654435761u));
    for (int o = 16; o; o >>= 1) b += __shfl_down(b, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(bad, b);
}

#define QZ_LINK_SPIN 2000000000ll     // clock64 ticks a flag wait spins before giving up

// the flag ping-pong: card 0 raises f[0], card 1 answers on f[1], n times, with system-scope
// release / acquire as kernels/xchg.h uses. Past the spin limit a wait leaves its token in
// err and returns, so the launch drains instead of hanging.
static __global__ void qz_link_flag(unsigned int * f, int mine, int n, unsigned long long * err) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned int * mine_f = f + mine, * peer_f = f + (1 - mine);
    for (int tok = 1; tok <= n; tok++) {
        if (mine == 0) {
            __threadfence_system();
            __hip_atomic_store(mine_f, (unsigned) tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        const long long t0 = clock64();
        for (;;) {
            const unsigned int v = __hip_atomic_load(peer_f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
            if ((int) (v - (unsigned) tok) >= 0) break;
            if (clock64() - t0 > QZ_LINK_SPIN) { *err = (unsigned long long) tok; return; }
        }
        if (mine == 1) {
            __threadfence_system();
            __hip_atomic_store(mine_f, (unsigned) tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

int qz_link_check_host(double timeout_s, double * gbs, double * rt_us) {
    const size_t n = (size_t) 16 << 20;              // 64 MB of words
    const int nrt = 1000;
    void * mp = nullptr;
    hipError_t e = qz_xalloc(&mp, n * 4, 0, "check");
    if (e != hipSuccess) {
        fprintf(stderr, "quartz: link check: cannot pin 64 MB: %s\n", hipGetErrorString(e));
        return 1;
    }
    unsigned * buf = (unsigned *) mp;
    int rc = 0;
    for (int c = 0; c < 2 && !rc; c++) {
        const int src = qz_dev_of(c), dst = qz_dev_of(1 - c);
        const unsigned seed = 0x9e3779b9u + (unsigned) c;
        hipStream_t s0 = nullptr, s1 = nullptr;
        hipEvent_t e0 = nullptr, e1 = nullptr;
        int * bad = nullptr;
        char what[64];
        snprintf(what, sizeof what, "HIP %d writing into host memory for HIP %d", src, dst);
        (void) hipSetDevice(src);
        (void) hipStreamCreateWithFlags(&s0, hipStreamNonBlocking);
        (void) hipEventCreate(&e0); (void) hipEventCreate(&e1);
        (void) hipSetDevice(dst);
        (void) hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
        if (hipMalloc((void **) &bad, sizeof(int)) != hipSuccess || hipMemset(bad, 0, sizeof(int)) != hipSuccess ||
            hipDeviceSynchronize() != hipSuccess) {
            fprintf(stderr, "quartz: link check: cannot allocate on HIP %d\n", dst);
            return 1;
        }
        for (int rep = 0; rep < 2 && !rc; rep++) {   // first pass warms up, second is timed
            (void) hipSetDevice(src);
            (void) hipEventRecord(e0, s0);
            hipLaunchKernelGGL(qz_link_fill, dim3(1024), dim3(256), 0, s0, buf, n, seed);
            (void) hipEventRecord(e1, s0);
            rc = qz_link_wait(s0, "quartz: link check", what, "the host link hangs", timeout_s);
        }
        if (rc == 2) return 2;                        // stuck: leave everything in place
        float ms = 0.f;
        if (!rc && hipEventElapsedTime(&ms, e0, e1) == hipSuccess && ms > 0) gbs[c] = n * 4 / (ms * 1e6);
        if (!rc) {                                    // the other card reads the same words back
            (void) hipSetDevice(dst);
            hipLaunchKernelGGL(qz_link_scan, dim3(1024), dim3(256), 0, s1, (const unsigned *) buf, n, seed, bad);
            snprintf(what, sizeof what, "HIP %d reading host memory written by HIP %d", dst, src);
            rc = qz_link_wait(s1, "quartz: link check", what, "the host link hangs", timeout_s);
            if (rc == 2) return 2;
            int nb = 0;
            if (!rc && hipMemcpy(&nb, bad, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "quartz: link check: cannot read the counter on HIP %d\n", dst);
                rc = 1;
            }
            if (!rc && nb) {
                fprintf(stderr, "quartz: link check: HIP %d writing into host memory: %d of %zu words arrived wrong\n",
                        src, nb, n);
                rc = 1;
            }
        }
        (void) hipSetDevice(src);
        (void) hipEventDestroy(e0); (void) hipEventDestroy(e1);
        (void) hipStreamDestroy(s0);
        (void) hipSetDevice(dst);
        (void) hipStreamDestroy(s1);
        (void) hipFree(bad);
    }
    (void) qz_xfree(mp);
    if (rc) return 1;

    // the flag round trip: 1000 crossings each way, timed from the two launches
    void * fp = nullptr;
    e = qz_xalloc(&fp, 256, 0, "check flag");
    if (e != hipSuccess) {
        fprintf(stderr, "quartz: link check: cannot pin the flag mailbox: %s\n", hipGetErrorString(e));
        return 1;
    }
    unsigned int * f = (unsigned int *) fp;
    hipStream_t sf[2] = { nullptr, nullptr };
    void * errd[2] = { nullptr, nullptr };
    for (int c = 0; c < 2; c++) {
        (void) hipSetDevice(qz_dev_of(c));
        (void) hipStreamCreateWithFlags(&sf[c], hipStreamNonBlocking);
        if (hipMalloc(&errd[c], 8) != hipSuccess || hipMemset(errd[c], 0, 8) != hipSuccess) {
            fprintf(stderr, "quartz: link check: cannot allocate on HIP %d\n", qz_dev_of(c));
            (void) qz_xfree(fp);
            return 1;
        }
    }
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    for (int c = 0; c < 2; c++) {
        (void) hipSetDevice(qz_dev_of(c));
        hipLaunchKernelGGL(qz_link_flag, dim3(1), dim3(32), 0, sf[c], f, c, nrt, (unsigned long long *) errd[c]);
    }
    for (int c = 0; c < 2 && !rc; c++) {
        char what[64];
        snprintf(what, sizeof what, "the flag ping-pong (HIP %d side)", qz_dev_of(c));
        rc = qz_link_wait(sf[c], "quartz: link check", what, "the host link hangs", timeout_s);
    }
    if (rc == 2) return 2;
    if (!rc) *rt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / nrt;
    for (int c = 0; c < 2; c++) {
        unsigned long long ne = 0;
        (void) hipSetDevice(qz_dev_of(c));
        if (!rc && hipMemcpy(&ne, errd[c], 8, hipMemcpyDeviceToHost) == hipSuccess && ne) {
            fprintf(stderr, "quartz: link check: HIP %d saw no answer to its flag %llu within the spin limit\n",
                    qz_dev_of(c), ne);
            rc = 1;
        }
        (void) hipStreamDestroy(sf[c]);
        (void) hipFree(errd[c]);
    }
    (void) qz_xfree(fp);
    return rc;
}

// The peer flag round trip (preflight.h, more than two GPUs): each side spins on a flag in
// its own memory and answers by writing the other side's, so the link carries writes only.
static __global__ void qz_p2p_flag(const unsigned int * rd, unsigned int * wr, int first, int n, unsigned long long * err) {
    if (blockIdx.x || threadIdx.x) return;
    for (int tok = 1; tok <= n; tok++) {
        if (first) {
            __threadfence_system();
            __hip_atomic_store(wr, (unsigned) tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        const long long t0 = clock64();
        for (;;) {
            const unsigned int v = __hip_atomic_load(rd, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
            if ((int) (v - (unsigned) tok) >= 0) break;
            if (clock64() - t0 > QZ_LINK_SPIN) { *err = (unsigned long long) tok; return; }
        }
        if (!first) {
            __threadfence_system();
            __hip_atomic_store(wr, (unsigned) tok, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

int qz_link_rt_p2p(int a, int b, double timeout_s, double * rt_us) {
    const int nrt = 1000, dv[2] = { a, b };
    unsigned int * f[2] = { nullptr, nullptr };
    void * errd[2] = { nullptr, nullptr };
    hipStream_t sf[2] = { nullptr, nullptr };
    int rc = 0;
    for (int c = 0; c < 2 && !rc; c++) {
        (void) hipSetDevice(dv[c]);
        if (hipExtMallocWithFlags((void **) &f[c], 256, hipDeviceMallocUncached) != hipSuccess || hipMemset(f[c], 0, 256) != hipSuccess ||
            hipMalloc(&errd[c], 8) != hipSuccess || hipMemset(errd[c], 0, 8) != hipSuccess ||
            hipStreamCreateWithFlags(&sf[c], hipStreamNonBlocking) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            fprintf(stderr, "quartz: peer check: cannot allocate the flags on HIP %d\n", dv[c]);
            rc = 1;
        }
    }
    if (!rc) {
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        for (int c = 0; c < 2; c++) {
            (void) hipSetDevice(dv[c]);
            hipLaunchKernelGGL(qz_p2p_flag, dim3(1), dim3(32), 0, sf[c], f[c], f[1 - c], c == 0, nrt, (unsigned long long *) errd[c]);
        }
        for (int c = 0; c < 2 && !rc; c++) {
            char what[64];
            snprintf(what, sizeof what, "the flag ping-pong HIP %d <-> HIP %d", a, b);
            rc = qz_link_wait(sf[c], "quartz: peer check", what, "the peer link hangs", timeout_s);
        }
        if (rc == 2) return 2;                        // stuck: leave everything in place
        if (!rc) *rt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / nrt;
    }
    for (int c = 0; c < 2; c++) {
        unsigned long long ne = 0;
        (void) hipSetDevice(dv[c]);
        if (!rc && errd[c] && hipMemcpy(&ne, errd[c], 8, hipMemcpyDeviceToHost) == hipSuccess && ne) {
            fprintf(stderr, "quartz: peer check: HIP %d saw no answer to its flag %llu from HIP %d within the spin limit\n",
                    dv[c], ne, dv[1 - c]);
            rc = 1;
        }
        if (sf[c]) (void) hipStreamDestroy(sf[c]);
        if (errd[c]) (void) hipFree(errd[c]);
        if (f[c]) (void) hipFree(f[c]);
    }
    return rc;
}
