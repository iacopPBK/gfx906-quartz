#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "loader.h"          // qz_dev_tensor / qz_loader

#ifndef DS_CHECK
#define DS_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(1); } } while (0)
#endif

#define DS_H 5120             // target d_model = draft hidden
#define DS_FL 5               // draft layers (all sliding window)
#define DS_W 8192             // attention window
#ifndef DS_RING
#define DS_RING 12288         // ring slots (position mod DS_RING), past the window so a resumed
                              // request does not read rows an earlier request wrote
#endif
#define DS_NKV 8              // draft kv heads
#define DS_NQ 32              // draft q heads
#define DS_HD 128             // head dim (rope pairs 64 + 64)
#define DS_EPS 1e-6f          // rms_norm_eps and the bridge eps_b
#define DS_TR 12                // batched-row cap of the context fill (the served block size)

// --------------------------------------------------------------- device ----
// RMSNorm of `nrows` rows, one warp per row (in place is fine)
__global__ void __launch_bounds__(256) ds_rms(const float * x, const float * w, float * y, int nrows, int n, float eps) {
    const int r = (blockIdx.x << 2) + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    if (r >= nrows) return;
    const float * xr = x + (size_t) r * n;
    float * yr = y + (size_t) r * n;
    float ss = 0.f;
    for (int i = lane; i < n; i += 64) ss += xr[i] * xr[i];
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off, 64);
    ss = __shfl(ss, 0, 64);        // lane 0 holds the total; every lane normalizes with it
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = lane; i < n; i += 64) yr[i] = xr[i] * inv * w[i];
}

// quantize `nrows` f32 rows (row stride n) into Q8 planes (trunk requant
// semantics; one warp per (row, unit))
__global__ void __launch_bounds__(256) ds_qrows(const float * x, unsigned char * q, float * s,
                                                int nrows, int n) {
    const int n32 = n / 32;
    const int u = (blockIdx.x << 3) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if ((size_t) u >= (size_t) nrows * n32) return;
    const int r = u / n32, uu = u % n32;
    const float v = x[(size_t) r * n + (size_t) uu * 32 + l];
    float ax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ax = fmaxf(ax, __shfl_down(ax, off, 32));
    ax = __shfl(ax, 0, 32);
    const float dq = ax / 127.f;
    const float id = qz_q8_inv(dq);
    const float d16 = qz_q8_scale(dq);
    int qq = qz_q8_code(v, id);
    q[(size_t) r * n + (size_t) uu * 32 + l] = (unsigned char) qq;
    if (l == 0) s[(size_t) r * n32 + uu] = d16;
}

// T-row GEMV over a [N, w_k] Q8 plane: a wave per slab of S rows, W slabs per
// block; a row's result does not depend on T
template <int T, int S, int W>
__global__ void __launch_bounds__(64 * W) ds_gemvT(const unsigned char * a_vals, const float * a_scal, int a_k,
                                                   const char * w_vals, const __half * w_scal, int w_k, int N,
                                                   float * y) {
    const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
    const int nsl = (N + S - 1) / S;
    for (int sl = blockIdx.x * W + w; sl < nsl; sl += gridDim.x * W) {
        const int r0 = sl * S;
        float acc[T][S];
        qz_gemv_core<T, S, 0, false>(a_vals, a_scal, a_k, a_k / 32, a_k / 32,
                                     w_vals, w_scal, w_k, w_k / 32, N, r0, lane, acc);
        if (lane == 0)
            #pragma unroll
            for (int t = 0; t < T; t++)
                #pragma unroll
                for (int s = 0; s < S; s++)
                    if (r0 + s < N) y[(size_t) t * N + r0 + s] = acc[t][s];
    }
}

// k head norm + rope + ring store (row (p0 + r) mod W): one block of 128 per
// (local head, row)
__global__ void __launch_bounds__(128) ds_k_head_rows(const float * k, const float * knw, const float * inv,
                                                       float * kring, int h0, int hn, int p0) {
    __shared__ float red[4], qs[DS_HD], rinv;
    const int h = blockIdx.x, r = blockIdx.y, e = threadIdx.x;
    const float * krow = k + (size_t) r * (hn * DS_HD) + (size_t) h * DS_HD;
    float * kring_row = kring + (size_t) ((p0 + r) % DS_RING) * (DS_NKV * DS_HD);
    const float v = krow[e];
    float ss = v * v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down(ss, off, 32);
    if ((e & 31) == 0) red[e >> 5] = ss;
    __syncthreads();
    if (e == 0) rinv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float) DS_HD + DS_EPS);
    __syncthreads();
    qs[e] = v * rinv * knw[e];
    __syncthreads();
    if (e < DS_HD / 2) {
        const float a = (float) (p0 + r) * inv[e];
        const float c = cosf(a), sn = sinf(a);
        const float x0 = qs[e], x1 = qs[e + DS_HD / 2];
        kring_row[(size_t) (h0 + h) * DS_HD + e]              = x0 * c - x1 * sn;
        kring_row[(size_t) (h0 + h) * DS_HD + e + DS_HD / 2]  = x0 * sn + x1 * c;
    }
}

// scatter f32 rows (a v gemv batch, [rows][len]) into a position ring
// [W][NKV*HD] at offset `off`, one block per row (handles the wrap)
__global__ void __launch_bounds__(256) ds_ring_scatter(const float * src, float * ring,
                                                       int len, int off, int p0, int rows) {
    const int r = blockIdx.x, t = threadIdx.x;
    const float * x = src + (size_t) r * len;
    float * y = ring + (size_t) ((p0 + r) % DS_RING) * (DS_NKV * DS_HD) + off;
    for (int i = t; i < len; i += 256) y[i] = x[i];
}

// ----------------------------------------------------------------- host ----
// the drafter's context-cache state on one device
struct qz_ds_ctx {
    int dev = 0;
    hipStream_t st = nullptr;
    const unsigned char * fc_v = nullptr;               // fc.weight planes [H x 5H]
    const unsigned short * fc_s = nullptr;
    const float * hnorm = nullptr;                      // enc.output_norm
    const unsigned char * k_v[DS_FL] = {}, * v_v[DS_FL] = {};   // [NKV*HD x H]
    const unsigned short * k_s[DS_FL] = {}, * v_s[DS_FL] = {};
    const float * knw[DS_FL] = {};                      // attn_k_norm [HD]
    const float * inv = nullptr;                        // rope inv_freq [HD/2]
    float * ring_k[DS_FL] = {}, * ring_v[DS_FL] = {};   // [W][NKV*HD] f32
    int h0 = 0, hn = DS_NKV;                            // this device's kv heads
    // work buffers
    float * f32 = nullptr;                              // [5 * H]
    unsigned char * f_v = nullptr;                     // [5 * H]
    float * f_s = nullptr;                             // [5 * H / 32]
    float * c = nullptr;                              // [H]
    unsigned char * c_v = nullptr;                    // [H]
    float * c_s = nullptr;                           // [H / 32]
    float * y = nullptr;                             // [>= H]
    // batched-fill buffers (qz_ds_ctx_rows; DS_TR rows)
    unsigned char * b_fv = nullptr;                  // [DS_TR][5H]
    float * b_fs = nullptr;                          // [DS_TR][5H/32]
    float * bc = nullptr;                            // [DS_TR][H]
    unsigned char * b_cv = nullptr;                 // [DS_TR][H]
    float * b_cs = nullptr;                          // [DS_TR][H/32]
    float * by = nullptr;                            // [DS_TR][NKV*HD]
    qz_loader ld;                                    // owns the device tensors
};

// load just the context tensors (fc, hidden_norm, per-layer k / v, k_norm);
// everything else is skipped in place on the loader.
inline int qz_ds_ctx_load(qz_ds_ctx & d, const char * draft, int dev, int h0 = 0, int hn = DS_NKV) {
    d.dev = dev;
    d.h0 = h0;
    d.hn = hn;
    qz_mem_scope mem("dspine");
    DS_CHECK(hipSetDevice(dev));
    static auto need_ctx = [](const char * n) {
        if (!strcmp(n, "fc.weight") || !strcmp(n, "enc.output_norm.weight")) return true;
        int l = -1;
        if (sscanf(n, "blk.%d.", &l) == 1 && l >= 0 && l < DS_FL) {
            const char * suf = strchr(n + 4, '.');
            return suf && (!strcmp(suf + 1, "attn_k.weight") || !strcmp(suf + 1, "attn_v.weight") ||
                           !strcmp(suf + 1, "attn_k_norm.weight"));
        }
        return false;
    };
    d.ld.skip = [](const char * n) { return !need_ctx(n); };
    qz_gguf g;
    if (!g.open(draft)) { fprintf(stderr, "cannot open draft gguf %s\n", draft); return 1; }
    if (!d.ld.load(draft, g, dev)) return 1;
    auto fd = [&](const char * n) -> const qz_dev_tensor & {
        const int64_t i = d.ld.find(n);
        if (i < 0) { fprintf(stderr, "draft tensor missing: %s\n", n); exit(1); }
        return d.ld.ts[i];
    };
    const qz_dev_tensor & fc = fd("fc.weight");
    d.fc_v = (const unsigned char *) fc.vals;
    d.fc_s = (const unsigned short *) fc.scal;
    d.hnorm = (const float *) fd("enc.output_norm.weight").vals;
    char n[64];
    for (int l = 0; l < DS_FL; l++) {
        snprintf(n, sizeof(n), "blk.%d.attn_k.weight", l);
        d.k_v[l] = (const unsigned char *) fd(n).vals;
        d.k_s[l] = (const unsigned short *) fd(n).scal;
        snprintf(n, sizeof(n), "blk.%d.attn_v.weight", l);
        d.v_v[l] = (const unsigned char *) fd(n).vals;
        d.v_s[l] = (const unsigned short *) fd(n).scal;
        snprintf(n, sizeof(n), "blk.%d.attn_k_norm.weight", l);
        d.knw[l] = (const float *) fd(n).vals;
    }
    auto fz = [&](void ** p, size_t b) { DS_CHECK(qz_dmalloc(p, b)); DS_CHECK(hipMemset(*p, 0, b)); };
    qz_mem_scope mem_ctx("dspine ctx");
    fz((void **) &d.f32, (size_t) 5 * DS_H * 4);
    fz((void **) &d.f_v, (size_t) 5 * DS_H);
    fz((void **) &d.f_s, (size_t) 5 * DS_H / 32 * 4);
    fz((void **) &d.c, (size_t) DS_H * 4);
    fz((void **) &d.c_v, (size_t) DS_H);
    fz((void **) &d.c_s, (size_t) DS_H / 32 * 4);
    fz((void **) &d.y, (size_t) DS_H * 4);
    fz((void **) &d.b_fv, (size_t) DS_TR * 5 * DS_H);
    fz((void **) &d.b_fs, (size_t) DS_TR * 5 * DS_H / 32 * 4);
    fz((void **) &d.bc, (size_t) DS_TR * DS_H * 4);
    fz((void **) &d.b_cv, (size_t) DS_TR * DS_H);
    fz((void **) &d.b_cs, (size_t) DS_TR * DS_H / 32 * 4);
    fz((void **) &d.by, (size_t) DS_TR * DS_NKV * DS_HD * 4);
    std::vector<float> hinv(DS_HD / 2);
    for (int i = 0; i < DS_HD / 2; i++) hinv[i] = powf(1e7f, -(float) i / (DS_HD / 2));
    void * pi = nullptr;
    fz(&pi, hinv.size() * 4);
    d.inv = (const float *) pi;
    DS_CHECK(hipMemcpy(pi, hinv.data(), hinv.size() * 4, hipMemcpyHostToDevice));
    for (int l = 0; l < DS_FL; l++) {
        fz((void **) &d.ring_k[l], (size_t) DS_RING * DS_NKV * DS_HD * 4);
        fz((void **) &d.ring_v[l], (size_t) DS_RING * DS_NKV * DS_HD * 4);
    }
    DS_CHECK(hipStreamCreateWithFlags(&d.st, hipStreamNonBlocking));
    printf("qz_ds_ctx: dev %d, draft context tensors %.2f GB, kv heads %d..%d\n",
           dev, d.ld.need_bytes / 1e9, d.h0, d.h0 + d.hn - 1);
    return 0;
}

// context fill for a verify's committed rows: feat_rows [nrows][5H] (device)
// at positions p0.., nrows <= DS_TR
inline void qz_ds_ctx_rows(qz_ds_ctx & d, const float * feat_rows, int p0, int nrows) {
    hipStream_t s = d.st;
    const int nr = d.hn * DS_HD;
    ds_qrows<<<(DS_TR * 5 * DS_H / 32 + 7) / 8, 256, 0, s>>>(feat_rows, d.b_fv, d.b_fs, nrows, 5 * DS_H);
    ds_gemvT<DS_TR, 3, 4><<<(DS_H / 3 + 1 + 3) / 4, 256, 0, s>>>(
        d.b_fv, d.b_fs, 5 * DS_H, (const char *) d.fc_v, (const __half *) d.fc_s, 5 * DS_H, DS_H, d.bc);
    ds_rms<<<(DS_TR + 3) / 4, 256, 0, s>>>(d.bc, d.hnorm, d.bc, nrows, DS_H, DS_EPS);
    ds_qrows<<<(DS_TR * DS_H / 32 + 7) / 8, 256, 0, s>>>(d.bc, d.b_cv, d.b_cs, nrows, DS_H);
    for (int l = 0; l < DS_FL; l++) {
        ds_gemvT<DS_TR, 3, 4><<<(nr / 3 + 1 + 3) / 4, 256, 0, s>>>(
            d.b_cv, d.b_cs, DS_H, (const char *) d.k_v[l] + (size_t) d.h0 * DS_HD * DS_H,
            (const __half *) (d.k_s[l] + (size_t) d.h0 * DS_HD * (DS_H / 32)), DS_H, nr, d.by);
        ds_k_head_rows<<<dim3(d.hn, nrows), 128, 0, s>>>(d.by, d.knw[l], d.inv, d.ring_k[l], d.h0, d.hn, p0);
        ds_gemvT<DS_TR, 3, 4><<<(nr / 3 + 1 + 3) / 4, 256, 0, s>>>(
            d.b_cv, d.b_cs, DS_H, (const char *) d.v_v[l] + (size_t) d.h0 * DS_HD * DS_H,
            (const __half *) (d.v_s[l] + (size_t) d.h0 * DS_HD * (DS_H / 32)), DS_H, nr, d.by);
        ds_ring_scatter<<<nrows, 256, 0, s>>>(d.by, d.ring_v[l], nr, d.h0 * DS_HD, p0, nrows);
    }
}
