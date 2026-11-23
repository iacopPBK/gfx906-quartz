// Device primitives shared by every arch: global-address views, the int8 dot,
// the software grid barrier, cross-lane reads, wave reductions, Q8 rounding.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// global-address-space views, so weight streams compile to global_load and not
// flat_load. flat bumps lgkmcnt, so LDS waits would then wait on HBM.
typedef unsigned int qz_v4u __attribute__((ext_vector_type(4)));
typedef const qz_v4u __attribute__((address_space(1))) qz_gu4;
typedef const unsigned short __attribute__((address_space(1))) qz_gh;

#define DOT4(d, wv, av) asm ("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(d) : "v"(wv), "v"(av))

// a pointer the compiler cannot prove global, because it came from a table or a
// struct: assert it, so loads and stores compile to global_*. flat_* counts in
// lgkmcnt too, so every LDS wait would wait on the outstanding HBM traffic
template <class T> __device__ __forceinline__ T * qz_gp(T * p) {
    __builtin_assume(!__builtin_amdgcn_is_shared((const void *) p));
    __builtin_assume(!__builtin_amdgcn_is_private((const void *) p));
    return p;
}

struct qz_barrier_state {
    unsigned int cnt[2];
    unsigned int gen;
};

// Grid barrier with parity alternation and a monotonic generation. The wait polls
// gen with a plain load, since an atomic spin would fight the arrivals' RMWs.
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

// x from another lane through DPP control C, for example 0xB1 for xor 1, 0x141 for
// row_half_mirror or 0x128 for row_ror:8. No LDS round trip, unlike __shfl_xor.
template <int C, bool BC = true> __device__ __forceinline__ float qz_dpp(float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), C, 0xF, 0xF, BC));
}
// the same read as a plain DPP move, with no old value to keep
template <int C> __device__ __forceinline__ float qz_dpp_mov(float x) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), C, 0xF, 0xF, false));
}
// wave sum in every lane, by the xor butterfly 32, 16, ..., 1
__device__ __forceinline__ float qz_wsum(float x) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) x += __shfl_xor(x, off, 64);
    return x;
}
// qz_wsum with the in-row levels as DPP. After the xor-16 level a lane's value
// depends only on lane & 15, so row_ror:8 is xor 8, and row_ror:4 then pairs lane
// i with the lane holding the value of i ^ 4. The additions are the same bit for
// bit, and only the two shuffles cost an LDS round trip.
__device__ __forceinline__ float qz_wsum_dpp(float x) {
    x += __shfl_xor(x, 32, 64);
    x += __shfl_xor(x, 16, 64);
    x += qz_dpp<0x128>(x);   // row_ror:8
    x += qz_dpp<0x124>(x);   // row_ror:4
    x += qz_dpp<0x4E>(x);    // quad_perm [2,3,0,1]
    x += qz_dpp<0xB1>(x);    // quad_perm [1,0,3,2]
    return x;
}

// max |x| over each 32-lane half of the wave, in every lane, as 4 DPP v_max ops
// plus one ds_swizzle. Bit-equal to the __shfl_xor tree.
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

// Q8_0 rounding of a 32-value unit with scale d = amax / 127. qz_q8_inv gives the
// inverse scale, qz_q8_code a value's int8 code from x * (1 / d) as llama.cpp's
// quantize_row_q8_0 does, and qz_q8_scale the scale as stored, f16-rounded.
__device__ __forceinline__ float qz_q8_inv(float d) { return d != 0.f ? 1.f / d : 0.f; }
__device__ __forceinline__ int qz_q8_code(float x, float id) {
    const int q = (int) roundf(x * id);
    return q > 127 ? 127 : (q < -127 ? -127 : q);
}
__device__ __forceinline__ float qz_q8_scale(float d) { return __half2float(__float2half(d)); }
// the code as x / d, the rounding inkling2's kernels were validated with
__device__ __forceinline__ int qz_q8_code_div(float x, float amax, float d) {
    const int q = amax == 0.f ? 0 : (int) roundf(x / d);
    return q > 127 ? 127 : (q < -127 ? -127 : q);
}
