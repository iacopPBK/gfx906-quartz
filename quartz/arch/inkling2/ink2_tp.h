// Per-card weight slices for nc cards (2 or 4). Card c owns the q heads from 24c/nc, the
// kv heads from 8c/nc, as whole GQA groups, 1/nc of every expert's FF (gate rows
// 1152c/nc.., up rows 1152 + 1152c/nc.., down contraction blocks 36c/nc..) and head rows
// [Vp c/nc, +Vp/nc), Vp the vocabulary padded to whole 32-row blocks per card: padding
// rows load as zeros (QZ_SLICE_ZERO) and their logits are never valid (VOCAB_REAL).
// The rest is replicated.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "loader.h"   // qz_slice, QZ_SLICE_ZERO

static inline void ink2_range(std::vector<int32_t> & v, int a, int n) {
    for (int i = 0; i < n; i++) v.push_back(a + i);
}

// head rows per card: the vocabulary over nc cards, rounded up to whole 32-row blocks
static constexpr int ink2_head_share(int64_t vocab, int nc) {
    return (int) ((vocab + 32 * nc - 1) / (32 * nc) * 32);
}

// the tensor `name`'s slice for card of nc, false when replicated. ne1 flattens dims past ne0
static inline bool ink2_tp_spec(const char * name, int card, int nc, int64_t ne1, qz_slice & s) {
    s.rows.clear();
    s.cols.clear();
    const int c = card;
    if (strcmp(name, "output.weight") == 0) {
        const int share = ink2_head_share(ne1, nc);
        for (int i = 0; i < share; i++) {
            const int r = share * c + i;
            s.rows.push_back(r < ne1 ? r : QZ_SLICE_ZERO);
        }
        return true;
    }
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * suf = strchr(name + 4, '.');
    if (!suf) return false;
    suf++;
    auto is = [&](const char * x) { return strcmp(suf, x) == 0; };
    if (is("attn_q.weight") || is("attn_gate.weight")) {             // 24 heads x 128
        ink2_range(s.rows, 3072 / nc * c, 3072 / nc);
    } else if (is("attn_k.weight") || is("attn_v.weight") ||          // 8 kv heads x 128
               is("attn_k_conv.weight") || is("attn_v_conv.weight")) {
        ink2_range(s.rows, 1024 / nc * c, 1024 / nc);
    } else if (is("attn_k_conv_proj.weight") || is("attn_v_conv_proj.weight")) {   // 8 kv heads x 4 taps
        ink2_range(s.rows, 32 / nc * c, 32 / nc);
    } else if (is("attn_rel.weight")) {                                // 24 heads x 16
        ink2_range(s.rows, 384 / nc * c, 384 / nc);
    } else if (is("attn_sink_proj.weight") || is("attn_q_clip.weight") || is("attn_rel_clip.weight")) {
        ink2_range(s.rows, 24 / nc * c, 24 / nc);                      // per q head
    } else if (is("attn_output.weight")) {                             // contraction: 96 blocks of 32
        ink2_range(s.cols, 96 / nc * c, 96 / nc);
    } else if (is("ffn_gate_up_exps.weight")) {                        // per expert: gate 1152 | up 1152
        const int ne = (int) (ne1 / 2304), fl = 1152 / nc;
        for (int e = 0; e < ne; e++) {
            ink2_range(s.rows, e * 2304 + fl * c, fl);
            ink2_range(s.rows, e * 2304 + 1152 + fl * c, fl);
        }
    } else if (is("ffn_down_exps.weight")) {                           // contraction: 36 blocks of 32
        ink2_range(s.cols, 36 / nc * c, 36 / nc);
    } else {
        return false;
    }
    return true;
}

// The DSpark drafter's slices (ink2_dspark.h). Card c owns q heads 24c/nc.., kv heads
// 8c/nc.. for attention and context K / V, FFN rows 9216c/nc.. and 1/nc of the
// contractions of attn_output (96 blocks) and ffn_down (288); the rest is replicated.
static inline bool ink2_ds_tp_spec(const char * name, int card, int nc, int64_t, qz_slice & s) {
    s.rows.clear();
    s.cols.clear();
    const int c = card;
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * suf = strchr(name + 4, '.');
    if (!suf) return false;
    suf++;
    auto is = [&](const char * x) { return strcmp(suf, x) == 0; };
    if (is("attn_q.weight")) ink2_range(s.rows, 3072 / nc * c, 3072 / nc);
    else if (is("attn_k.weight") || is("attn_v.weight") || is("attn_k_ctx.weight") || is("attn_v_ctx.weight"))
        ink2_range(s.rows, 1024 / nc * c, 1024 / nc);
    else if (is("attn_output.weight")) ink2_range(s.cols, 96 / nc * c, 96 / nc);
    else if (is("ffn_gate.weight") || is("ffn_up.weight")) ink2_range(s.rows, 9216 / nc * c, 9216 / nc);
    else if (is("ffn_down.weight")) ink2_range(s.cols, 288 / nc * c, 288 / nc);
    else return false;
    return true;
}

// the loader's hooks: this build's card count
static inline bool ink2_tp_slice(const char * name, int card, int64_t ne1, qz_slice & s) {
    return ink2_tp_spec(name, card, QZ_NC, ne1, s);
}
static inline bool ink2_ds_tp_slice(const char * name, int card, int64_t ne1, qz_slice & s) {
    return ink2_ds_tp_spec(name, card, QZ_NC, ne1, s);
}
