// TP2 per-card weight slices: card c owns q heads 12c.. and kv heads 4c.. (whole
// GQA groups), half of every expert's FF (gate rows 576c.., up rows 1152 + 576c..,
// down contraction blocks 18c..) and head rows [V/2 c, +V/2); the rest is replicated.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "loader.h"   // qz_slice

static inline void ink2_range(std::vector<int32_t> & v, int a, int n) {
    for (int i = 0; i < n; i++) v.push_back(a + i);
}

// slice of tensor `name` for card; false: replicated. ne1: every dim past ne0, flattened
static inline bool ink2_tp2_spec(const char * name, int card, int64_t ne1, qz_slice & s) {
    s.rows.clear();
    s.cols.clear();
    const int c = card;
    if (strcmp(name, "output.weight") == 0) {
        ink2_range(s.rows, (int) (ne1 / 2) * c, (int) (ne1 / 2));
        return true;
    }
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * suf = strchr(name + 4, '.');
    if (!suf) return false;
    suf++;
    auto is = [&](const char * x) { return strcmp(suf, x) == 0; };
    if (is("attn_q.weight") || is("attn_gate.weight")) {             // 24 heads x 128
        ink2_range(s.rows, 1536 * c, 1536);
    } else if (is("attn_k.weight") || is("attn_v.weight") ||          // 8 kv heads x 128
               is("attn_k_conv.weight") || is("attn_v_conv.weight")) {
        ink2_range(s.rows, 512 * c, 512);
    } else if (is("attn_k_conv_proj.weight") || is("attn_v_conv_proj.weight")) {   // 8 kv heads x 4 taps
        ink2_range(s.rows, 16 * c, 16);
    } else if (is("attn_rel.weight")) {                                // 24 heads x 16
        ink2_range(s.rows, 192 * c, 192);
    } else if (is("attn_sink_proj.weight") || is("attn_q_clip.weight") || is("attn_rel_clip.weight")) {
        ink2_range(s.rows, 12 * c, 12);                                // per q head
    } else if (is("attn_output.weight")) {                             // contraction: 96 blocks of 32
        ink2_range(s.cols, 48 * c, 48);
    } else if (is("ffn_gate_up_exps.weight")) {                        // per expert: gate 1152 | up 1152
        const int ne = (int) (ne1 / 2304);
        for (int e = 0; e < ne; e++) {
            ink2_range(s.rows, e * 2304 + 576 * c, 576);
            ink2_range(s.rows, e * 2304 + 1152 + 576 * c, 576);
        }
    } else if (is("ffn_down_exps.weight")) {                           // contraction: 36 blocks of 32
        ink2_range(s.cols, 18 * c, 18);
    } else {
        return false;
    }
    return true;
}

// the DSpark drafter's TP2 slices (ink2_dspark.h): card c owns q heads 12c.., kv heads
// 4c.. (attention and context K / V), FFN rows 4608c.. and the matching contraction
// halves of attn_output (48 of 96 blocks) and ffn_down (144 of 288); the rest is replicated
static inline bool ink2_ds_tp2_spec(const char * name, int card, int64_t, qz_slice & s) {
    s.rows.clear();
    s.cols.clear();
    const int c = card;
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * suf = strchr(name + 4, '.');
    if (!suf) return false;
    suf++;
    auto is = [&](const char * x) { return strcmp(suf, x) == 0; };
    if (is("attn_q.weight")) ink2_range(s.rows, 1536 * c, 1536);
    else if (is("attn_k.weight") || is("attn_v.weight") || is("attn_k_ctx.weight") || is("attn_v_ctx.weight"))
        ink2_range(s.rows, 512 * c, 512);
    else if (is("attn_output.weight")) ink2_range(s.cols, 48 * c, 48);
    else if (is("ffn_gate.weight") || is("ffn_up.weight")) ink2_range(s.rows, 4608 * c, 4608);
    else if (is("ffn_down.weight")) ink2_range(s.cols, 144 * c, 144);
    else return false;
    return true;
}
