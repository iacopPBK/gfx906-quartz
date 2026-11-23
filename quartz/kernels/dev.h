// Device primitives shared by every arch: global-address views, the int8 dot,
// the software grid barrier, wave max reductions.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// global-address-space views: weight streams compile to global_load, not
// flat_load (flat also bumps lgkmcnt, so LDS waits would wait on HBM)
typedef unsigned int qz_v4u __attribute__((ext_vector_type(4)));
typedef const qz_v4u __attribute__((address_space(1))) qz_gu4;
typedef const unsigned short __attribute__((address_space(1))) qz_gh;

#define DOT4(d, wv, av) asm ("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(d) : "v"(wv), "v"(av))

struct qz_barrier_state {
    unsigned int cnt[2];
    unsigned int gen;
};

// Grid barrier: parity alternation, monotonic generation; the wait polls gen
// with a plain load (an atomic spin would fight the arrivals' RMWs).
__device__ __forceinline__ void qz_grid_barrier(qz_barrier_state * st, int k) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int p = (unsigned int)k & 1u;
        __threadfence();
        const unsigned int arrived = atomicAdd(&st->cnt[p], 1u);
        if (arrived == gridDim.x - 1) {
            st->cnt[p] = 0;
            __threadfence();
            atomicAdd(&st->gen, 1u);
        } else {
            while (*(volatile unsigned int *)&st->gen < (unsigned int)k) { }
            __threadfence();
        }
    }
    __syncthreads();
}

// max |x| over each 32-lane half of the wave, in every lane: 4 DPP v_max +
// one ds_swizzle, bit-equal to the __shfl_xor tree.
template <int CTRL> __device__ __forceinline__ float qz_dpp_fmax(float v) {
    const int o = __builtin_amdgcn_update_dpp(__float_as_int(v), __float_as_int(v), CTRL, 0xf, 0xf, false);
    return fmaxf(v, __int_as_float(o));
}
__device__ __forceinline__ float qz_amax32(float v) {
    v = qz_dpp_fmax<0xB1>(v);    // quad_perm [1,0,3,2]
    v = qz_dpp_fmax<0x4E>(v);    // quad_perm [2,3,0,1]
    v = qz_dpp_fmax<0x141>(v);   // row_half_mirror (8 lanes)
    v = qz_dpp_fmax<0x140>(v);   // row_mirror (16 lanes)
    const int o = __builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F);   // bitmode: and 0x1f, xor 0x10
    return fmaxf(v, __int_as_float(o));
}
