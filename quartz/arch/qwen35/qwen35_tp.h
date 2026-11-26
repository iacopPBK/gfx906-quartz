// Per-card weight slices for nc cards (2 or 4). Card c owns k heads [16c/nc, +16/nc) with
// their q rows and conv channels, the v heads tiled on them (v head h belongs to k head
// h % 16, so card c's v heads are (j / KL) * 16 + KL c + j % KL), q heads from 24c/nc,
// kv heads from 4c/nc, FFN rows [17408c/nc, +17408/nc) and head rows [V c/nc, +V/nc).
// The rest is replicated.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "loader.h"   // qz_slice

// local v head j of card c, nc cards
static inline int qz_tp_vhead(int card, int j, int nc) {
    const int kl = 16 / nc;
    return (j / kl) * 16 + kl * card + j % kl;
}

static inline void qz_tp_range(std::vector<int32_t> & v, int a, int n) {
    for (int i = 0; i < n; i++) v.push_back(a + i);
}

// slice of tensor `name` for card of nc. Returns false for a replicated tensor
static inline bool qz_tp_spec(const char * name, int card, int nc, int64_t ne1, qz_slice & s) {
    s.rows.clear();
    s.cols.clear();
    const int c = card, kl = 16 / nc, vl = 48 / nc;
    if (strcmp(name, "output.weight") == 0) {
        qz_tp_range(s.rows, (int) (ne1 / nc) * c, (int) (ne1 / nc));
        return true;
    }
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * suf = strchr(name + 4, '.');
    if (!suf) return false;
    suf++;
    auto is = [&](const char * x) { return strcmp(suf, x) == 0; };
    if (is("attn_qkv.weight") || is("ssm_conv1d.weight")) {     // rows = conv channels: q | k | v
        qz_tp_range(s.rows, kl * 128 * c, kl * 128);
        qz_tp_range(s.rows, 2048 + kl * 128 * c, kl * 128);
        for (int j = 0; j < vl; j++) qz_tp_range(s.rows, 4096 + qz_tp_vhead(c, j, nc) * 128, 128);
    } else if (is("attn_gate.weight")) {
        for (int j = 0; j < vl; j++) qz_tp_range(s.rows, qz_tp_vhead(c, j, nc) * 128, 128);
    } else if (is("ssm_beta.weight") || is("ssm_alpha.weight")) {
        for (int j = 0; j < vl; j++) s.rows.push_back(qz_tp_vhead(c, j, nc));
    } else if (is("ssm_dt.bias") || is("ssm_a")) {
        for (int j = 0; j < vl; j++) s.cols.push_back(qz_tp_vhead(c, j, nc));
    } else if (is("ssm_out.weight")) {
        for (int j = 0; j < vl; j++) qz_tp_range(s.cols, qz_tp_vhead(c, j, nc) * 4, 4);
    } else if (is("attn_q.weight")) {                          // 24 heads x (256 q + 256 gate)
        qz_tp_range(s.rows, 12288 / nc * c, 12288 / nc);
    } else if (is("attn_k.weight") || is("attn_v.weight")) {   // 4 kv heads x 256
        qz_tp_range(s.rows, 1024 / nc * c, 1024 / nc);
    } else if (is("attn_output.weight")) {                     // contraction: 192 blocks of 32
        qz_tp_range(s.cols, 192 / nc * c, 192 / nc);
    } else if (is("ffn_gate.weight") || is("ffn_up.weight")) {
        qz_tp_range(s.rows, 17408 / nc * c, 17408 / nc);
    } else if (is("ffn_down.weight")) {                        // contraction: 544 blocks of 32
        qz_tp_range(s.cols, 544 / nc * c, 544 / nc);
    } else {
        return false;
    }
    return true;
}

// the loader's hook: this build's card count
static inline bool qz_tp_slice(const char * name, int card, int64_t ne1, qz_slice & s) {
    return qz_tp_spec(name, card, QZ_NC, ne1, s);
}
