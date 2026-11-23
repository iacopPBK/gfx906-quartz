// Thin wrapper over the gguf C API. No new reader code is written;
// this only wraps key lookup with defaults for the engine's loader.
#pragma once

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "gguf_lite.h"   // quartz's own reader (was ggml include)

static inline const char * qz_type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return "F32"; case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_BF16: return "BF16"; case GGML_TYPE_Q8_0: return "Q8_0";
        default: return "other";
    }
}

struct qz_gguf {
    gguf_context * ctx = nullptr;

    ~qz_gguf() { if (ctx) gguf_free(ctx); }

    // metadata + tensor map only, no tensor data is read
    bool open(const char * path) {
        gguf_init_params p = {};
        p.no_alloc = true;
        p.ctx = nullptr;
        ctx = gguf_init_from_file(path, p);
        return ctx != nullptr;
    }

    int64_t key(const char * k) const { return gguf_find_key(ctx, k); }

    std::string str(const char * k) const {
        int64_t id = key(k);
        return id < 0 ? std::string() : std::string(gguf_get_val_str(ctx, id));
    }

    uint32_t u32(const char * k, uint32_t def = 0) const {
        int64_t id = key(k);
        return id < 0 ? def : gguf_get_val_u32(ctx, id);
    }

    float f32(const char * k, float def = 0.0f) const {
        int64_t id = key(k);
        return id < 0 ? def : gguf_get_val_f32(ctx, id);
    }

    bool flag(const char * k, bool def = false) const {
        int64_t id = key(k);
        return id < 0 ? def : gguf_get_val_bool(ctx, id);
    }

    int64_t tensor(const char * name) const {   // tensor index, -1 if absent
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx); i++)
            if (strcmp(gguf_get_tensor_name(ctx, i), name) == 0) return i;
        return -1;
    }

    // tensor `name` has exactly the file dims `dims` (ggml order, trailing 1s implied)
    // and, unless type is GGML_TYPE_COUNT, that type; false with a message otherwise
    bool shape(const char * name, std::initializer_list<int64_t> dims, ggml_type type = GGML_TYPE_COUNT) const {
        const int64_t t = tensor(name);
        if (t < 0) { fprintf(stderr, "quartz: missing tensor %s\n", name); return false; }
        const int64_t * ne = gguf_get_tensor_ne(ctx, t);
        bool ok = dims.size() <= GGML_MAX_DIMS && (type == GGML_TYPE_COUNT || gguf_get_tensor_type(ctx, t) == type);
        int d = 0;
        for (int64_t n : dims) if (d < GGML_MAX_DIMS && ne[d++] != n) ok = false;
        for (; d < GGML_MAX_DIMS; d++) if (ne[d] != 1) ok = false;
        if (!ok) {
            fprintf(stderr, "quartz: tensor %s is %s [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], expected [", name,
                    qz_type_name(gguf_get_tensor_type(ctx, t)), ne[0], ne[1], ne[2], ne[3]);
            d = 0;
            for (int64_t n : dims) fprintf(stderr, "%s%" PRId64, d++ ? ", " : "", n);
            fprintf(stderr, "]%s%s\n", type == GGML_TYPE_COUNT ? "" : " ", type == GGML_TYPE_COUNT ? "" : qz_type_name(type));
        }
        return ok;
    }

    // an integer or bool array as int64 (empty when absent or of another type)
    std::vector<int64_t> ints(const char * k) const {
        std::vector<int64_t> v;
        int64_t id = key(k);
        if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return v;
        const size_t n = gguf_get_arr_n(ctx, id);
        const void * d = gguf_get_arr_data(ctx, id);
        if (!d) return v;
        for (size_t i = 0; i < n; i++) {
            switch (gguf_get_arr_type(ctx, id)) {
                case GGUF_TYPE_BOOL: case GGUF_TYPE_UINT8: v.push_back(((const uint8_t *) d)[i]); break;
                case GGUF_TYPE_INT8:   v.push_back(((const int8_t *) d)[i]); break;
                case GGUF_TYPE_UINT16: v.push_back(((const uint16_t *) d)[i]); break;
                case GGUF_TYPE_INT16:  v.push_back(((const int16_t *) d)[i]); break;
                case GGUF_TYPE_UINT32: v.push_back(((const uint32_t *) d)[i]); break;
                case GGUF_TYPE_INT32:  v.push_back(((const int32_t *) d)[i]); break;
                case GGUF_TYPE_UINT64: v.push_back((int64_t) ((const uint64_t *) d)[i]); break;
                case GGUF_TYPE_INT64:  v.push_back(((const int64_t *) d)[i]); break;
                default: return std::vector<int64_t>();
            }
        }
        return v;
    }
};
