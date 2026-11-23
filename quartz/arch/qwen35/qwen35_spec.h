// Engine spec for Qwen3.8-27B (Q8_0): one sequence, TP2 over two cards,
// persistent-kernel phases. Model-hardcoded by design.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"

// attention cache layout: f16 element offset of position p, kv head kh of
// hkv per card, head dim 256, in pages of 64 positions per head ([page][hkv][64][256])
#if defined(__HIPCC__)
__host__ __device__
#endif
static inline constexpr size_t qz_kv_row(int p, int hkv, int kh) {
    return ((size_t) (p >> 6) * hkv + kh) * (64 * 256) + (size_t) (p & 63) * 256;
}

// Layer classes in the trunk. Attention layers are (i+1) % attn_interval == 0,
// else gated delta net. Mirrors src/models/qwen35.cpp.
enum qz_layer_kind {
    QZ_LAYER_ATTN = 0,
    QZ_LAYER_RECR = 1,
    QZ_LAYER_MTP  = 2,   // nextn block, dense attention
};

// TP2 split of a phase: COL halves the output rows (no exchange), ROW halves
// the contraction dim (one exchange after the phase).
enum qz_split {
    QZ_SPLIT_NONE = 0,
    QZ_SPLIT_COL  = 1,
    QZ_SPLIT_ROW  = 2,
};

struct qz_hparams {
    int n_vocab    = 0;   // 248320
    int d_model    = 0;   // 5120
    int n_ff       = 0;   // 17408
    int n_layer    = 0;   // trunk layers, 64
    int n_mtp      = 0;   // nextn blocks, 1
    int n_attn     = 0;   // full attention layers, 16
    int n_recr     = 0;   // gated delta net layers, 48
    int n_head     = 0;   // 24 q heads
    int n_kv       = 0;   // 4 kv heads
    int head_dim   = 0;   // 256
    int q_dim      = 0;   // n_head * head_dim
    int kv_dim     = 0;   // n_kv * head_dim
    int ssm_state  = 0;   // 128, per-head k/v dim
    int ssm_group  = 0;   // 16 k heads
    int ssm_rank   = 0;   // 48 v heads
    int ssm_conv   = 0;   // conv kernel, 4

    std::vector<uint8_t> layer_kind;   // per trunk+nextn layer, qz_layer_kind

    bool is_recr(int il) const { return layer_kind.empty() ? false : layer_kind[il] == QZ_LAYER_RECR; }

    bool load(const qz_gguf & g);

};
