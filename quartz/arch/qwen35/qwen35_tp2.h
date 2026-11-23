// TP2 per-card weight slices: card c owns v heads (j/8)*16 + 8c + j%8, q heads
// 12c.., kv heads 2c, 2c+1, FFN rows [8704c, +8704), head rows [V/2 c, +V/2).
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "loader.h"   // qz_slice


static inline int qz_tp2_vhead(int card, int j) { return (j / 8) * 16 + 8 * card + j % 8; }

static inline void qz_tp2_range(std::vector<int32_t> & v, int a, int n) {
    for (int i = 0; i < n; i++) v.push_back(a + i);
}

// slice of tensor `name` for card; returns false for a replicated tensor
static inline bool qz_tp2_spec(const char * name, int card, int64_t ne1, qz_slice & s) {
    s.rows.clear();
    s.cols.clear();
    const int c = card;
    if (strcmp(name, "output.weight") == 0) {
        qz_tp2_range(s.rows, (int) (ne1 / 2) * c, (int) (ne1 / 2));
        return true;
    }
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * suf = strchr(name + 4, '.');
    if (!suf) return false;
    suf++;
    auto is = [&](const char * x) { return strcmp(suf, x) == 0; };
    if (is("attn_qkv.weight") || is("ssm_conv1d.weight")) {     // rows = conv channels
        qz_tp2_range(s.rows, 1024 * c, 1024);
        qz_tp2_range(s.rows, 2048 + 1024 * c, 1024);
        for (int j = 0; j < 24; j++) qz_tp2_range(s.rows, 4096 + qz_tp2_vhead(c, j) * 128, 128);
    } else if (is("attn_gate.weight")) {
        for (int j = 0; j < 24; j++) qz_tp2_range(s.rows, qz_tp2_vhead(c, j) * 128, 128);
    } else if (is("ssm_beta.weight") || is("ssm_alpha.weight")) {
        for (int j = 0; j < 24; j++) s.rows.push_back(qz_tp2_vhead(c, j));
    } else if (is("ssm_dt.bias") || is("ssm_a")) {
        for (int j = 0; j < 24; j++) s.cols.push_back(qz_tp2_vhead(c, j));
    } else if (is("ssm_out.weight")) {
        for (int j = 0; j < 24; j++) qz_tp2_range(s.cols, qz_tp2_vhead(c, j) * 4, 4);
    } else if (is("attn_q.weight")) {
        qz_tp2_range(s.rows, 6144 * c, 6144);
    } else if (is("attn_k.weight") || is("attn_v.weight")) {
        qz_tp2_range(s.rows, 512 * c, 512);
    } else if (is("attn_output.weight")) {
        qz_tp2_range(s.cols, 96 * c, 96);
    } else if (is("ffn_gate.weight") || is("ffn_up.weight")) {
        qz_tp2_range(s.rows, 8704 * c, 8704);
    } else if (is("ffn_down.weight")) {
        qz_tp2_range(s.cols, 272 * c, 272);
    } else {
        return false;
    }
    return true;
}
