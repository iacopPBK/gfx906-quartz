// Everything the inkling2 headers include from outside the arch, at global
// scope; the arch's TUs include this first, then their own headers inside
// namespace ink2 (see docs/adding-a-model.md).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "arch.h"
#include "core/cards.h"
#include "core/sampler.h"
#include <hip/hip_runtime.h>
// the xor butterfly (32, 16, 8, 4, 2, 1) with the in-row levels as DPP: after
// the xor-16 level a lane's value depends only on lane & 15, so row_ror:8 is
// xor 8, and row_ror:4 then pairs lane i with one holding the value of i ^ 4;
// the same additions as the shuffle loop, bit for bit, in two LDS round trips
template <int C> __device__ __forceinline__ float dppf(float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), C, 0xF, 0xF, true));
}
__device__ __forceinline__ float wsum_xor(float x) {
    x += __shfl_xor(x, 32, 64);
    x += __shfl_xor(x, 16, 64);
    x += dppf<0x128>(x);   // row_ror:8
    x += dppf<0x124>(x);   // row_ror:4
    x += dppf<0x4E>(x);    // quad_perm [2,3,0,1]
    x += dppf<0xB1>(x);    // quad_perm [1,0,3,2]
    return x;
}
// the GEMV core's per-row wave sum: lane 0 of the shfl_down tree adds the same
// pairs as the xor butterfly (its cone: lanes i < off pair with i + off = i ^ off)
#define QZ_GEMV_WSUM(x) (x) = wsum_xor(x)
#include "kernels/gemv.h"
#include "kernels/dev.h"
#include "kernels/topk.h"
#include "loader.h"
#include "gguf.h"
#include "preflight.h"
