#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qz_ctx qz_ctx;

typedef struct qz_stats {
    double prefill_ms, decode_ms;   // prefill = forced prompt cycles incl. the boundary one
    int n_prompt, n_gen;            // prompt tokens, emitted tokens
    int prefill_cycles, decode_cycles, decode_tok;   // decode_tok: accepted (overshoot incl.)
    int stop;                       // 0 max_new / arena, 1 stop id, 2 abort, 3 callback, -1 error
    int n_cached;                   // prompt tokens taken from the prefix cache (not prefilled)
    int dspine;                     // 1: decoded on DSpine (decode_cycles = its rounds)
    int wide_cycles;                // DSpine rounds with the 12-row verify
} qz_stats;

typedef bool (*qz_emit_cb)(void * user, const int32_t * toks, int n);

qz_ctx * qz_create(const char * model, int max_ctx, int slot_t, int adapt_slack);
int      qz_max_ctx(const qz_ctx * c);
int      qz_generate(qz_ctx * c, const int32_t * prompt, int n_prompt, int max_new,
                     const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out);
void     qz_abort(qz_ctx * c);   // any thread: ends the running qz_generate at the next cycle

typedef struct qz_sampling_c {
    float temp;
    int top_k;
    float top_p, min_p;
    int min_keep;
    uint32_t seed;
    int n_order;
    int order[8];
} qz_sampling_c;
int      qz_generate_s(qz_ctx * c, const int32_t * prompt, int n_prompt, int max_new,
                       const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out,
                       const qz_sampling_c * smp);

// images (qwen3vl_merger projector, mmproj GGUF). The projector runs on card 0.
int      qz_vision_attach(qz_ctx * c, const char * mmproj);   // 0 ok
int      qz_image_pad_token(void);                            // <|image_pad|>
// encodes an image (jpg / png / bmp / gif) into the 8-entry LRU cache; returns
// its id or -1; nx * ny = the <|image_pad|> tokens the prompt holds for it
int      qz_image_add(qz_ctx * c, const void * bytes, size_t n, int * nx, int * ny);
double   qz_image_last_ms(const qz_ctx * c);                  // last encode (0: cache hit before)
// qz_generate_s with images: the k-th <|image_pad|> run takes images[k]
int      qz_generate_mm(qz_ctx * c, const int32_t * prompt, int n_prompt, const int * images, int n_images,
                        int max_new, const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out,
                        const qz_sampling_c * smp);
void     qz_destroy(qz_ctx * c);

#ifdef __cplusplus
}
#endif
