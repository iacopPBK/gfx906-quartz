// The model behind the C API. qz_ctx is one loaded model of one architecture, and
// each arch derives from it and registers a create function in capi.cpp.
#pragma once

#include "capi.h"

#include <cstddef>
#include <cstdint>

struct qz_create_opts {
    int max_ctx = 0;       // prompt + generated tokens
    int slot_t = 0;        // decode rows per cycle, arch-specific, 0 means adaptive
    int adapt_slack = 0;
};

struct qz_ctx {
    virtual ~qz_ctx() = default;
    virtual void info(qz_model_info * out) const = 0;
    virtual int  max_ctx() const = 0;
    // images holds ids from image_add, and the k-th placeholder run takes images[k].
    // A null smp means greedy sampling.
    virtual int  generate(const int32_t * prompt, int n_prompt, const int * images, int n_images,
                          int max_new, const int32_t * stop, int n_stop, qz_emit_cb cb, void * user,
                          qz_stats * out, const qz_sampling_c * smp) = 0;
    virtual void abort() = 0;   // any thread, stops the running generate at the next cycle
    virtual int    vision_attach(const char *) { return -1; }
    virtual int    image_add(const void *, size_t, int *, int *) { return -1; }
    virtual double image_last_ms() const { return 0.0; }
};

// arch entry points, nullptr on failure with a message printed
qz_ctx * qz_qwen35_create(const char * path, const qz_create_opts & o);
qz_ctx * qz_inkling2_create(const char * path, const qz_create_opts & o);
