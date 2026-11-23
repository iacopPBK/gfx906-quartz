// host only: hparams from the GGUF (shared headers first, then the arch in namespace q35)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"

namespace q35 {

#include "qwen35_spec.h"

#include <cstdio>
#include <cstring>

// Parse "blk.<n>.<suffix>". Returns false if not a layer tensor.
static bool split_blk(const char * name, int & il, const char *& suffix) {
    if (strncmp(name, "blk.", 4) != 0) return false;
    il = atoi(name + 4);
    const char * dot = strchr(name + 4, '.');
    if (!dot) return false;
    suffix = dot + 1;
    return il >= 0;
}

bool qz_hparams::load(const qz_gguf & g) {
    if (!g.ctx) return false;

    n_vocab   = (int) g.u32("qwen35.vocab_size");
    d_model   = (int) g.u32("qwen35.embedding_length");
    n_ff      = (int) g.u32("qwen35.feed_forward_length");
    n_head    = (int) g.u32("qwen35.attention.head_count");
    n_kv      = (int) g.u32("qwen35.attention.head_count_kv");
    head_dim  = (int) g.u32("qwen35.attention.key_length");
    ssm_state = (int) g.u32("qwen35.ssm.state_size");
    ssm_group = (int) g.u32("qwen35.ssm.group_count");
    ssm_rank  = (int) g.u32("qwen35.ssm.time_step_rank");
    ssm_conv  = (int) g.u32("qwen35.ssm.conv_kernel");
    if (d_model <= 0 || head_dim <= 0) return false;
    if (ssm_conv <= 0) ssm_conv = 4;

    kv_dim = n_kv * head_dim;
    q_dim  = n_head * head_dim * 2;   // q heads carry 2x key dim, see below

    const int64_t nt = gguf_get_n_tensors(g.ctx);

    // layer kinds from the tensor set: attn_output, ssm_out or nextn.*
    int max_layer = -1;
    for (int64_t t = 0; t < nt; t++) {
        const char * name = gguf_get_tensor_name(g.ctx, t);
        int il; const char * suffix;
        if (!split_blk(name, il, suffix)) continue;
        if (il > max_layer) max_layer = il;
        if (layer_kind.size() <= (size_t) il) layer_kind.resize(il + 1, 99);
        if (strncmp(suffix, "nextn", 5) == 0)         layer_kind[il] = QZ_LAYER_MTP;
        else if (strncmp(suffix, "attn_output", 11) == 0) layer_kind[il] = QZ_LAYER_ATTN;
        else if (strncmp(suffix, "ssm_out", 7) == 0)  layer_kind[il] = QZ_LAYER_RECR;
    }
    if (max_layer < 0) return false;

    n_attn = n_recr = n_mtp = 0;
    for (int il = 0; il <= max_layer; il++) {
        switch (layer_kind[il]) {
            case QZ_LAYER_ATTN: n_attn++; break;
            case QZ_LAYER_RECR: n_recr++; break;
            case QZ_LAYER_MTP:  n_mtp++;  break;
            default: break;   // no weight-bearing tensor: dead layer id
        }
    }
    // trunk layers are contiguous from 0, the nextn block sits on top
    n_layer = n_attn + n_recr;
    if (n_layer <= 0) return false;

    // vocab size and attn_q's output dim come from the tensors (the GGUF
    // has no vocab key; some exports drop attn_q's 2x)
    for (int64_t t = 0; t < nt; t++) {
        const char * name = gguf_get_tensor_name(g.ctx, t);
        const int64_t ne1 = gguf_get_tensor_ne(g.ctx, t)[1];
        if (n_vocab <= 0 && strcmp(name, "token_embd.weight") == 0) n_vocab = (int) ne1;
        const size_t nlen = strlen(name);
        if (nlen > 10 && strcmp(name + nlen - 10, "attn_q.weight") == 0 &&
            strncmp(name, "blk.", 4) == 0) {
            int il; const char * suffix;
            if (split_blk(name, il, suffix) && layer_kind[il] == QZ_LAYER_ATTN) {
                q_dim = (int) ne1;
            }
        }
    }
    return n_vocab > 0;
}

}   // namespace q35
