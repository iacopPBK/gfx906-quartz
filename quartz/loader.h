// Loads GGUF tensors onto one HIP device, Q8_0 repacked into the planes of
// quartz/repack.h. raw_off keeps the file offsets, so callers can re-read bytes.
#pragma once

#include <hip/hip_runtime.h>

#include "mem.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gguf.h"

// rows lists the source rows in local order, cols the source q8 blocks (Q8_0)
// or elements (F32). An empty list means all of them.
struct qz_slice {
    std::vector<int32_t> rows, cols;
};


struct qz_dev_tensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    int64_t ne0 = 0, ne1 = 1;      // [K, N]
    void * vals = nullptr;         // Q8_0 values plane, otherwise the raw bytes
    int64_t vstride = 0;           // Q8_0 values row stride in bytes, 0 means ne0
    bool shared = false;           // not loaded; vals/scal point at another owner's copy
    void * scal = nullptr;         // Q8_0 scale plane
    int64_t src_ne0 = 0, src_ne1 = 1;   // file dims, they differ from ne0/ne1 when sliced
    int64_t ne[GGML_MAX_DIMS] = {0, 1, 1, 1};   // file dims in ggml order, ne[0] fastest
    qz_slice slice;                // this card's TP2 slice, empty means whole tensor
    int64_t bytes() const {
        if (type == GGML_TYPE_Q8_0) return (vstride ? vstride : ne0) * ne1 + ne0 / 16 * ne1;
        const double bs = (double) ggml_type_size(type) / ggml_blck_size(type);
        return (int64_t)((double) ne0 * ne1 * bs);
    }
};

class qz_loader {
public:
    std::vector<qz_dev_tensor> ts;       // GGUF order
    std::vector<uint64_t> raw_off;       // absolute file offset of raw bytes
    int64_t need_bytes = 0;
    size_t free_before = 0;
    size_t free_after = 0;
    double load_seconds = 0;

    // skip(name) leaves a tensor unallocated and marks it shared
    std::function<bool(const char *)> skip;
    // the arch's TP split of one tensor for one card; false means replicated
    std::function<bool(const char *, int, int64_t, qz_slice &)> slice;

    // card >= 0 loads only that card's slices, otherwise the whole model
    bool load(const char * path, const qz_gguf & g, int dev, int card = -1);
    // fills the tensor table and need_bytes only, no allocation. load() runs it first.
    bool plan(const qz_gguf & g, int card = -1);
    int64_t find(const char * name) const;   // index into ts, -1 when absent
    const qz_dev_tensor * get(const char * name) const { const int64_t i = find(name); return i < 0 ? nullptr : &ts[i]; }
};
