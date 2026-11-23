// Exact per-row top-K of the lm_head logits by radix select, for the sampled verify.
#pragma once

#include <hip/hip_runtime.h>
#include <cmath>

// ---- sampled verify, temperature > 0: the exact per-row top-K of the lm_head
// logits that the cycle leaves in y_head, sampled on the host by qz_engine::sample_row
#define QZ_TOPK 64        // candidates per row, covering top_k 1 .. 64 exactly
#define QZ_TOPK_NB 16     // blocks per row; the host merges NB x TOPK per card
__device__ __forceinline__ unsigned qz_fkey(float v) {   // float order as unsigned order
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// block (b, t) writes the exact top QZ_TOPK of chunk b of row t as logit / id
// pairs, by radix select. The id -1 pads a short chunk.
static __global__ void qz_topk_rows(const float * y, int n, int id0, float2 * out) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_need;
    __shared__ int s_gt, s_eq;
    const int t = blockIdx.y, b = blockIdx.x;
    const int chunk = (n + gridDim.x - 1) / gridDim.x;
    const int i0 = b * chunk, i1 = min(n, i0 + chunk);
    const int total = max(0, min(QZ_TOPK, i1 - i0));
    const float * row = y + (size_t) t * n;
    float2 * o = out + ((size_t) t * gridDim.x + b) * QZ_TOPK;
    if (threadIdx.x == 0) { s_prefix = 0; s_need = total; s_gt = 0; s_eq = 0; }
    __syncthreads();
    for (int d = 3; d >= 0 && total > 0; d--) {
        const int shift = d * 8;
        const unsigned hmask = d == 3 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned pre = s_prefix;
        for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
            const unsigned k = qz_fkey(row[i]);
            if ((k & hmask) == (pre & hmask)) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned need = s_need;
            unsigned cum = 0;
            int bin = 255;
            for (; bin > 0; bin--) {
                if (cum + hist[bin] >= need) break;
                cum += hist[bin];
            }
            s_need = need - cum;
            s_prefix = pre | ((unsigned) bin << shift);
        }
        __syncthreads();
    }
    const unsigned thr = s_prefix;
    const int n_eq = (int) s_need, n_gt = total - n_eq;
    for (int i = i0 + threadIdx.x; i < i1 && total > 0; i += blockDim.x) {
        const float v = row[i];
        const unsigned k = qz_fkey(v);
        if (k > thr) {
            o[atomicAdd(&s_gt, 1)] = make_float2(v, __int_as_float(id0 + i));
        } else if (k == thr) {
            const int e = atomicAdd(&s_eq, 1);
            if (e < n_eq) o[n_gt + e] = make_float2(v, __int_as_float(id0 + i));
        }
    }
    for (int i = total + threadIdx.x; i < QZ_TOPK; i += blockDim.x) o[i] = make_float2(-INFINITY, __int_as_float(-1));
}
