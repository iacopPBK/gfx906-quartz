// Q8_0 repack into the GEMV planes, values as [N][K] int8 with K contiguous and
// scales as [N][K/32] fp16. Host code only.
#pragma once

#include <cstdint>
#include <cstring>

// unpacks one row of K values from raw Q8_0 blocks, each a f16 scale followed by
// 32 int8, into K/4 little-endian value dwords and K/16 scale bytes. K % 32 == 0.
static inline void qz_repack_row_q8(const uint8_t * raw_row, int K,
                                    uint32_t * vals, uint16_t * scales) {
    const int nblk = K / 32;
    for (int b = 0; b < nblk; b++) {
        const uint8_t * blk = raw_row + b * 34;
        memcpy(&scales[b], blk, 2);
        memcpy(&vals[b * 8], blk + 2, 32);
    }
}
