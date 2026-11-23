// Row ops of the qwen35 persistent cycle: the RMS norm with the int8 requant split
// over the grid, and the copy of the published plane rows into LDS. DM is the row
// width, a multiple of 32. Blocks run 512 threads.
#pragma once

#include "kernels/dev.h"

// qz_norm_requant_row split across the grid. Every block reduces the full row, then
// requants only its own units into gq and gs, so the result is bit-equal.
template <int DM> __device__ __forceinline__ void qz_norm_requant_slice(
        const float * xn, const float * nw, float ss, float eps,
        unsigned char * gq, float * gs, float * red, int tid) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off, 64);
    const int wave = tid >> 6, lane = tid & 63;
    if (lane == 0) red[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float ms = 0.f;
        for (int w = 0; w < 8; w++) ms += red[w];
        red[8] = 1.f / sqrtf(ms / DM + eps);
    }
    __syncthreads();
    const float rs = red[8];
    const int u = blockIdx.x + gridDim.x * (tid >> 5), l = tid & 31;
    if (u < DM / 32) {             // whole half-waves: the 32-lane amax stays in one unit
        const int i = u * 32 + l;
        const float x = xn[i] * rs * nw[i];
        const float amax = qz_amax32(fabsf(x));
        const float d = amax / 127.f;
        const float id = qz_q8_inv(d);
        int q = qz_q8_code(x, id);
        gq[i] = (unsigned char) q;
        if (l == 0) gs[u] = qz_q8_scale(d);
    }
    __syncthreads();                 // xn / red are reused by the next row
}

// after the grid barrier, copy the T published plane rows into this block's LDS
template <int T, int DM> __device__ __forceinline__ void qz_planes_to_lds(
        const unsigned char * gq, const float * gs, unsigned char * act_vals, float * act_scales, int tid) {
    for (int i = tid; i < T * DM / 16; i += 512) ((uint4 *) act_vals)[i] = ((const uint4 *) gq)[i];
    for (int i = tid; i < T * (DM / 32); i += 512) act_scales[i] = gs[i];
    __syncthreads();
}
