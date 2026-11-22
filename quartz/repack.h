// Q8_0 repack into GEMV planes: values [N][K] int8 (K contiguous) and scales
// [N][K/32] fp16. Host code.
#pragma once

#include <cstdint>
#include <cstring>

// one row of K values from raw Q8_0 blocks ([f16 scale][32 x int8] per block,
// K % 32 == 0) -> K/4 little-endian value dwords + K/16 scale bytes
static inline void qz_repack_row_q8(const uint8_t * raw_row, int K,
                                    uint32_t * vals, uint16_t * scales) {
    const int nblk = K / 32;
    for (int b = 0; b < nblk; b++) {
        const uint8_t * blk = raw_row + b * 34;
        memcpy(&scales[b], blk, 2);
        memcpy(&vals[b * 8], blk + 2, 32);
    }
}
