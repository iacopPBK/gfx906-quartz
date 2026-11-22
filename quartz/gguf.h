// Thin wrapper over the gguf C API. No new reader code is written;
// this only wraps key lookup with defaults for the engine's loader.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf_lite.h"   // quartz's own reader (was ggml include)

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
};
