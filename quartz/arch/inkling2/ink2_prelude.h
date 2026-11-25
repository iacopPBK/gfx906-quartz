// Everything the inkling2 headers include from outside the arch, at global scope. The
// arch's TUs include this first, then their own headers inside namespace ink2 (see
// docs/adding-a-model.md).
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
#include "core/nc.h"
#include "core/sampler.h"
#include "vision/inkling2v.h"
#include "kernels/dev.h"
// The GEMV core's per-row wave sum. Lane 0 of the shfl_down tree adds the same pairs as
// the xor butterfly: its cone pairs lanes i < off with i + off = i ^ off. In-row levels
// are DPP here.
#define QZ_GEMV_WSUM(x) (x) = qz_wsum_dpp(x)
#include "kernels/gemv.h"
#include "kernels/topk.h"
#include "loader.h"
#include "gguf.h"
#include "preflight.h"
