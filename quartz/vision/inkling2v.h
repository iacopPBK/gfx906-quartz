// Image encoder for the inkling2v projector (Inkling-2-Mini): a ViT with a CLS token,
// learned position embeddings resized bilinearly to the patch grid, 2D RoPE, and 2 x 2
// patches merged into one row-major row per token. Card 0 only, API as in qwen3vl.h.
#pragma once
#include <stddef.h>

namespace inkling2v {

struct qz_vision;

// Loads the projector; rows come out on device 0 and nullptr means failure. The token
// limit is QZ_IMG_MAX_TOKENS (4096), larger images are scaled down.
qz_vision * qz_vision_load(const char * mmproj);
// Decodes and encodes an image held in memory; returns nx * ny, negative on error. The
// rows stay on the device until the next encode.
int         qz_vision_encode(qz_vision * v, const void * bytes, size_t n, int * nx, int * ny);
// The same from 8-bit RGB pixels, row-major with 3 bytes per pixel.
int         qz_vision_encode_rgb(qz_vision * v, const unsigned char * rgb, int w, int h, int * nx, int * ny);
int         qz_vision_n_embd(const qz_vision * v);
int         qz_vision_copy_rows(const qz_vision * v, float * host);
double      qz_vision_last_ms(const qz_vision * v);   // last encode, upload to rows ready
void        qz_vision_free(qz_vision * v);
// True if the mmproj GGUF is an inkling2v projector.
bool        qz_vision_is(const char * mmproj);

}   // namespace inkling2v
