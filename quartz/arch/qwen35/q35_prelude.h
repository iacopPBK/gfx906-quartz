// Everything the qwen35 headers include from outside the arch: system, HIP
// and shared quartz headers, at global scope. The arch's TUs include this first,
// then their own headers inside namespace q35 (the shared headers' #pragma
// once makes the nested includes no-ops), so no qwen35 symbol can collide
// with another arch's in libquartz.so.
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
#include <deque>
#include <functional>
#include <initializer_list>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "arch.h"
#include "core/cards.h"
#include "core/prefix_cache.h"
#include "core/sampler.h"
#include "kernels/gemv.h"
#include "kernels/dev.h"
#include "kernels/pf_gemm.h"
#include "kernels/rowops.h"
#include "kernels/topk.h"
#include "kernels/xchg.h"
#include "loader.h"
#include "gguf.h"
#include "preflight.h"
#include "vision/qwen3vl.h"
