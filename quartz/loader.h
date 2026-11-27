// Loads GGUF tensors onto one HIP device, Q8_0 repacked into the planes of
// quartz/repack.h. raw_off keeps the file offsets, so callers can re-read bytes.
#pragma once

#include <hip/hip_runtime.h>

#include "mem.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "gguf.h"

// rows lists the source rows in local order, cols the source q8 blocks (Q8_0)
// or elements (F32). An empty list means all of them. A row of QZ_SLICE_ZERO is a
// zero row: padding, so that a split's rows come in whole blocks on every card.
constexpr int32_t QZ_SLICE_ZERO = -1;
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
    bool host = false;             // in pinned host memory, read by every GPU over PCIe
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
    int64_t need_bytes = 0;              // VRAM; host tensors are counted in host_bytes
    int64_t host_bytes = 0;
    size_t free_before = 0;
    size_t free_after = 0;
    double load_seconds = 0;

    // skip(name) leaves a tensor unallocated and marks it shared
    std::function<bool(const char *)> skip;
    // the arch's TP split of one tensor for one card; false means replicated
    std::function<bool(const char *, int, int64_t, qz_slice &)> slice;
    // host(name) places a tensor in pinned host memory instead of VRAM: for tensors read
    // a few rows at a time (the token embedding), so one copy serves every card
    std::function<bool(const char *)> host;

    // card >= 0 loads only that card's slices, otherwise the whole model
    bool load(const char * path, const qz_gguf & g, int dev, int card = -1);
    // fills the tensor table and need_bytes only, no allocation. load() runs it first.
    bool plan(const qz_gguf & g, int card = -1);
    int64_t find(const char * name) const;   // index into ts, -1 when absent
    const qz_dev_tensor * get(const char * name) const { const int64_t i = find(name); return i < 0 ? nullptr : &ts[i]; }
};

// One copy of tensor `name` in pinned host memory for all cards, for a tensor read a few
// rows at a time (the token embedding): set before each card's load (or plan), card 0's
// loader places it in host memory and the later cards' skip it. QZ_EMB_VRAM=1 turns it
// off (every card loads its own copy). Returns whether it is on.
inline bool qz_host_once(qz_loader & L, int card, const char * name) {
    if (getenv("QZ_EMB_VRAM")) return false;
    const std::string n = name;
    auto is = [n](const char * x) { return n == x; };
    if (card == 0) L.host = is;
    else L.skip = [is, prev = L.skip](const char * x) { return is(x) || (prev && prev(x)); };
    return true;
}
// after a later card's load: point its skipped copy of `name` at card 0's host one
inline bool qz_host_share(qz_loader & L, const qz_loader & L0, const char * name) {
    const qz_dev_tensor * e0 = L0.get(name);
    const int64_t i = L.find(name);
    if (!e0 || i < 0 || !e0->host) return false;
    qz_dev_tensor & e = L.ts[i];
    e.vals = e0->vals; e.scal = e0->scal; e.vstride = e0->vstride; e.host = true;
    return true;
}
