// Image encoder for the qwen3vl_merger projector, split over the two cards. An image
// becomes nx * ny rows of the projector width, row-major over the token grid.
#pragma once
#include <stddef.h>

namespace qwen3vl {

struct qz_vision;

// Loads the projector; rows come out on device 0 and nullptr means failure. Token limits
// are QZ_IMG_MIN_TOKENS (8) and QZ_IMG_MAX_TOKENS (4096).
qz_vision * qz_vision_load(const char * mmproj);
// Decodes, resizes and encodes an image held in memory; returns nx * ny, negative on
// error. The rows stay on the device until the next encode.
int         qz_vision_encode(qz_vision * v, const void * bytes, size_t n, int * nx, int * ny);
// The same from 8-bit RGB pixels, row-major with 3 bytes per pixel.
int         qz_vision_encode_rgb(qz_vision * v, const unsigned char * rgb, int w, int h, int * nx, int * ny);
int         qz_vision_n_embd(const qz_vision * v);
int         qz_vision_copy_rows(const qz_vision * v, float * host);
double      qz_vision_last_ms(const qz_vision * v);   // last encode, upload to rows ready
void        qz_vision_free(qz_vision * v);

}   // namespace qwen3vl
