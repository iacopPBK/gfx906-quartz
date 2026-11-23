// Inkling-2-Mini (inkling2_loop): the shape quartz's engine is built for, checked
// against the GGUF keys and every tensor (docs/inkling2.md).
#pragma once

constexpr int D = 2304, NL = 24, HQ = 24, HKV = 8, HD = 128, NE = 128, NE_USED = 8, FFE = 1152;
constexpr int WIN = 1024, DREL = 16, REL_EXT = 1024, NCONV = 4, ARB = 6;
constexpr int LOOP0 = 6, LOOP1 = 18, NPASS = 3, GRANK = 64, NCTRL = 4;
constexpr int NAPP = NL + (NPASS - 1) * (LOOP1 - LOOP0);   // layer applications (state slots) per token: 48
constexpr int VOCAB = 201024, VOCAB_REAL = 200058;
constexpr int32_t END_MESSAGE = 200010;                     // <|end_message|>: closes every chat message
constexpr int32_t IMG_PAD = 200054;                         // <|unused_200054|>: the template's image placeholder (after <|content_image|>)
constexpr int CONV_CH = 2 * HKV * HD + 2 * D;               // conv history channels per slot: K | V | attn post | ffn post

inline bool is_full(int il) { return il % 6 == 5; }         // layers 5, 11, 17, 23
inline bool is_loop(int il) { return il >= LOOP0 && il < LOOP1; }
inline int  slot_phys(int s) { return s < NL ? s : LOOP0 + (s - NL) % (LOOP1 - LOOP0); }

struct hparams {
    float eps = 1e-6f, qk_clip = 0, rel_clip = 0, tau_alpha = 0, route_scale = 0;
    float situ_beta = 0, situ_lin_beta = 0, softcap = 0;
    int tau_floor = 0;
    int32_t eos = -1;
};

// the GGUF keys quartz depends on; false (with a message per mismatch) for another shape
static inline bool parse(const qz_gguf & g, hparams & h) {
    bool ok = true;
    auto need = [&](const char * k, int64_t want, int64_t got) {
        if (got != want) { fprintf(stderr, "inkling2: %s is %" PRId64 ", quartz supports %" PRId64 "\n", k, got, want); ok = false; }
    };
    auto u = [&](const char * k, int64_t want) { std::string s = std::string("inkling2_loop.") + k; need(s.c_str(), want, g.key(s.c_str()) < 0 ? -1 : (int64_t) g.u32(s.c_str())); };
    auto f = [&](const char * k) { std::string s = std::string("inkling2_loop.") + k;
        if (g.key(s.c_str()) < 0) { fprintf(stderr, "inkling2: missing %s\n", s.c_str()); ok = false; return 0.0f; }
        return g.f32(s.c_str()); };
    u("block_count", NL);
    u("embedding_length", D);
    u("feed_forward_length", FFE);
    u("expert_feed_forward_length", FFE);
    u("expert_count", NE);
    u("expert_used_count", NE_USED);
    u("attention.key_length", HD);
    u("attention.value_length", HD);
    u("attention.key_length_swa", HD);
    u("attention.value_length_swa", HD);
    u("attention.sliding_window", WIN);
    u("attention.relative_length", DREL);
    u("attention.relative_extent", REL_EXT);
    u("shortconv.l_cache", NCONV);
    u("attn_res.block_size", ARB);
    u("loop.start", LOOP0);
    u("loop.end", LOOP1);
    u("loop.count", NPASS);
    u("loop.gate_rank", GRANK);
    u("rope.dimension_count", 0);
    u("vocab_size", VOCAB);
    u("vocab_size_unpadded", VOCAB_REAL);
    if (g.flag("inkling2_loop.loop.varying")) { fprintf(stderr, "inkling2: loop.varying is not supported\n"); ok = false; }
    const std::vector<int64_t> hq = g.ints("inkling2_loop.attention.head_count"), hkv = g.ints("inkling2_loop.attention.head_count_kv"),
                               swa = g.ints("inkling2_loop.attention.sliding_window_pattern");
    if ((int) hq.size() != NL || (int) hkv.size() != NL || (int) swa.size() != NL) {
        fprintf(stderr, "inkling2: head_count / head_count_kv / sliding_window_pattern must have %d entries\n", NL); ok = false;
    } else {
        for (int il = 0; il < NL; il++) {
            if (hq[il] != HQ || hkv[il] != HKV) { fprintf(stderr, "inkling2: layer %d has %" PRId64 "/%" PRId64 " heads, quartz supports %d/%d\n", il, hq[il], hkv[il], HQ, HKV); ok = false; }
            if ((swa[il] != 0) == is_full(il)) { fprintf(stderr, "inkling2: layer %d sliding-window flag differs from quartz's full layers 5/11/17/23\n", il); ok = false; }
        }
    }
    h.eps = f("attention.layer_norm_rms_epsilon");
    h.qk_clip = f("attention.qk_clip");
    h.rel_clip = f("attention.relative_clip");
    h.tau_floor = (int) g.u32("inkling2_loop.attention.log_scale_floor");
    h.tau_alpha = f("attention.log_scale_alpha");
    h.route_scale = f("expert_weights_scale");
    h.situ_beta = f("activation.situ_beta");
    h.situ_lin_beta = f("activation.situ_linear_beta");
    h.softcap = f("final_logit_softcapping");
    h.eos = (int32_t) g.u32("tokenizer.ggml.eos_token_id", (uint32_t) -1);
    if (h.tau_floor <= 0) { fprintf(stderr, "inkling2: missing attention.log_scale_floor\n"); ok = false; }
    return ok;
}

// every tensor's dims and type (Q8_0 matrices, F32 vectors and tables), and no others
static inline bool check_tensors(const qz_gguf & g) {
    bool ok = true;
    int n = 0;
    char nm[96];
    auto t = [&](const char * name, std::initializer_list<int64_t> dims, ggml_type ty) { ok &= g.shape(name, dims, ty); n++; };
    auto b = [&](int il, const char * suf, std::initializer_list<int64_t> dims, ggml_type ty) {
        snprintf(nm, sizeof(nm), "blk.%d.%s", il, suf); t(nm, dims, ty);
    };
    const ggml_type Q = GGML_TYPE_Q8_0, F = GGML_TYPE_F32;
    t("token_embd.weight", {D, VOCAB}, Q);
    t("token_embd_norm.weight", {D}, F);
    t("output_norm.weight", {D}, F);
    t("output_res_query.weight", {D}, F);
    t("output.weight", {D, VOCAB}, Q);
    for (int il = 0; il < NL; il++) {
        b(il, "attn_q.weight", {D, HQ * HD}, Q);
        b(il, "attn_gate.weight", {D, HQ * HD}, Q);
        b(il, "attn_k.weight", {D, HKV * HD}, Q);
        b(il, "attn_v.weight", {D, HKV * HD}, Q);
        b(il, "attn_k_conv.weight", {NCONV, HKV * HD}, F);
        b(il, "attn_v_conv.weight", {NCONV, HKV * HD}, F);
        b(il, "attn_output.weight", {HQ * HD, D}, Q);
        b(il, "attn_post_conv.weight", {NCONV, D}, F);
        b(il, "attn_q_clip.weight", {HD, HQ}, F);
        b(il, "attn_rel.weight", {D, HQ * DREL}, Q);
        b(il, "attn_rel_bias.weight", {DREL, REL_EXT}, F);
        b(il, "attn_rel_clip.weight", {DREL, HQ}, F);
        b(il, "attn_sink_proj.weight", {D, HQ}, Q);
        b(il, "ffn_norm.weight", {D}, F);
        b(il, "ffn_res_query.weight", {D}, F);
        b(il, "ffn_post_conv.weight", {NCONV, D}, F);
        b(il, "ffn_gate_inp.weight", {D, NE}, F);
        b(il, "exp_probs_b.bias", {NE}, F);
        b(il, "ffn_gate_up_exps.weight", {D, 2 * FFE, NE}, Q);
        b(il, "ffn_down_exps.weight", {FFE, D, NE}, Q);
        if (is_full(il)) {
            b(il, "attn_q_inp_norm.weight", {D}, F);
            b(il, "attn_k_inp_norm.weight", {D}, F);
            b(il, "attn_v_inp_norm.weight", {D}, F);
            b(il, "attn_k_conv_proj.weight", {D, HKV * NCONV}, Q);
            b(il, "attn_v_conv_proj.weight", {D, HKV * NCONV}, Q);
        } else {
            b(il, "attn_norm.weight", {D}, F);
        }
        if (il > 0) {
            if (is_full(il)) b(il, "attn_res_query.weight", {D, 3}, F);
            else             b(il, "attn_res_query.weight", {D}, F);
        }
        if (is_loop(il)) {
            for (const char * c : {"q", "k", "v", "ffn"}) {
                char s[64];
                snprintf(s, sizeof(s), "loop_%s_ctrl_down.weight", c);   b(il, s, {2 * D, GRANK}, F);
                snprintf(s, sizeof(s), "loop_%s_ctrl_gain.weight", c);   b(il, s, {GRANK, D}, F);
                snprintf(s, sizeof(s), "loop_%s_ctrl_forget.weight", c); b(il, s, {GRANK, D}, F);
            }
            b(il, "loop_msg_query.weight", {D, NCTRL}, F);
        }
    }
    if (gguf_get_n_tensors(g.ctx) != n) {
        fprintf(stderr, "inkling2: the file has %" PRId64 " tensors, quartz knows %d\n", gguf_get_n_tensors(g.ctx), n);
        ok = false;
    }
    return ok;
}
