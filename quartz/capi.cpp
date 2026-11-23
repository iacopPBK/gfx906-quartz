// C API: picks the architecture from the GGUF and forwards to its qz_ctx.
#include "arch.h"
#include "devices.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>

struct qz_arch_entry {
    const char * arch;   // GGUF general.architecture
    qz_ctx * (*create)(const char * path, const qz_create_opts & o);
};
static const qz_arch_entry qz_archs[] = {
    { "qwen35", qz_qwen35_create },
};

extern "C" {

qz_ctx * qz_create(const char * model, int max_ctx, int slot_t, int adapt_slack) {
    if (!model) return nullptr;
    if (qz_dev_select()) return nullptr;
    std::string arch;
    {
        qz_gguf g;
        if (!g.open(model)) { fprintf(stderr, "quartz: cannot open %s\n", model); return nullptr; }
        arch = g.str("general.architecture");
    }
    qz_create_opts o;
    o.max_ctx = max_ctx; o.slot_t = slot_t; o.adapt_slack = adapt_slack;
    for (const auto & a : qz_archs)
        if (arch == a.arch) return a.create(model, o);
    fprintf(stderr, "quartz: unsupported architecture '%s'; supported:", arch.c_str());
    for (const auto & a : qz_archs) fprintf(stderr, " %s", a.arch);
    fprintf(stderr, "\n");
    return nullptr;
}

int qz_max_ctx(const qz_ctx * c) { return c ? c->max_ctx() : 0; }

int qz_model_info_get(const qz_ctx * c, qz_model_info * out) {
    if (!c || !out) return -1;
    memset(out, 0, sizeof(*out));
    out->img_pad = -1;
    c->info(out);
    return 0;
}

int qz_generate(qz_ctx * c, const int32_t * prompt, int n_prompt, int max_new,
                const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out) {
    return c ? c->generate(prompt, n_prompt, nullptr, 0, max_new, stop, n_stop, cb, user, out, nullptr) : -1;
}

int qz_generate_s(qz_ctx * c, const int32_t * prompt, int n_prompt, int max_new,
                  const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out,
                  const qz_sampling_c * smp) {
    return c ? c->generate(prompt, n_prompt, nullptr, 0, max_new, stop, n_stop, cb, user, out, smp) : -1;
}

int qz_generate_mm(qz_ctx * c, const int32_t * prompt, int n_prompt, const int * images, int n_images,
                   int max_new, const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out,
                   const qz_sampling_c * smp) {
    return c ? c->generate(prompt, n_prompt, images, n_images, max_new, stop, n_stop, cb, user, out, smp) : -1;
}

int qz_vision_attach(qz_ctx * c, const char * mmproj) { return c && mmproj ? c->vision_attach(mmproj) : -1; }

int qz_image_add(qz_ctx * c, const void * bytes, size_t n, int * nx, int * ny) {
    return c && bytes && n ? c->image_add(bytes, n, nx, ny) : -1;
}

double qz_image_last_ms(const qz_ctx * c) { return c ? c->image_last_ms() : 0.0; }

void qz_abort(qz_ctx * c) { if (c) c->abort(); }

void qz_destroy(qz_ctx * c) { delete c; }   // device memory is released at process exit

}
