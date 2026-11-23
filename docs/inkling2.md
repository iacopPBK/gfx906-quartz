# Inkling-2-Mini (`inkling2_loop`): the forward pass

The spec quartz's `inkling2` engine is built from. Source: mainline llama.cpp
`src/models/inkling2.cpp` and `src/models/inkling2-loop.cpp` at e300919cc
(2026-11-22), the reference quartz is checked against. Names in `code` are GGUF
tensor names without the `blk.N.` prefix or the `.weight` suffix. Shapes are
in ggml order (`[ne0, ne1]`; a matrix `[in, out]` maps `in` to `out`).

## Constants (GGUF keys `inkling2_loop.*`)

| | value | key |
|---|---|---|
| d | 2304 | `embedding_length` |
| physical layers | 24 | `block_count` |
| q heads / kv heads / head dim | 24 / 8 / 128 | `attention.head_count{,_kv}`, `key_length` |
| experts / used / expert FF | 128 / 8 / 1152 | `expert_count`, `expert_used_count`, `expert_feed_forward_length` |
| route scale | 8 | `expert_weights_scale` |
| SITU β / linear β | 4 / 25 | `activation.situ_beta`, `activation.situ_linear_beta` |
| SWA window | 1024; layers 5, 11, 17, 23 are full attention | `attention.sliding_window{,_pattern}` |
| relative dim / extent | 16 / 1024 | `attention.relative_length`, `relative_extent` |
| q/k clip, r clip | 22.627 (= 16√2), 8 | `attention.qk_clip`, `relative_clip` |
| τ floor / α | 8192 / 0.1 | `attention.log_scale_floor`, `log_scale_alpha` |
| short conv taps | 4 (3 history rows) | `shortconv.l_cache` |
| attn-res block | 6 | `attn_res.block_size` |
| loop core / passes / gate rank | layers [6, 18) / 3 / 64 | `loop.start`, `loop.end`, `loop.count`, `loop.gate_rank` |
| rms eps | 1e-6 everywhere | `attention.layer_norm_rms_epsilon` |
| logit softcap | 30 | `final_logit_softcapping` |
| vocab | 201024 rows, logits cut to 200058 | `vocab_size{,_unpadded}` |

No RoPE (`rope.dimension_count` 0): position enters only through the relative
bias, τ and the short convs.

## Driver: 48 layer applications per token

```
x = rmsnorm(token_embd[tok]) * token_embd_norm        # image rows skip this norm
blocks = []
for il in 0..5:                 x = layer(il, phys=il, x, blocks)
prelude, prelude_blocks = x, blocks
for pass in 0..2:
    if pass > 0:
        bank     = blocks[len(prelude_blocks) + 1 :]  # the previous pass's blocks after its first
        bank_out = x                                  # and its output
        x, blocks = prelude, prelude_blocks           # every pass restarts from the prelude
    for ip in 6..17:
        slot = ip if pass == 0 else 24 + (pass-1)*12 + (ip-6)
        if pass == 0: x = layer(slot, ip, x, blocks)
        else:         x = layer(slot, ip, x, blocks, msg = mix(bank, bank_out, loop_msg_query[ip]))
for il in 18..23:               x = layer(il, il, x, blocks)    # on the last pass's x and blocks
logits = output(x, blocks)
```

- Weights are physical (24 layers). State is per slot: there are 48 slots, each
  with its own KV and its own conv history. Slots 24..35 are pass 1 of layers
  6..17, slots 36..47 are pass 2.
- `msg` is computed once per pass from the previous pass and holds 4 rows per
  token (`loop_msg_query` `[2304, 4]`): one each for the Q, K, V and FFN
  controllers. Every core layer of the pass uses its own query, so the message
  differs per layer.
- Block bookkeeping: a layer whose `ip % 6 == 0` (0, 6, 12, 18) pushes its
  input onto `blocks` and starts a fresh residual (see below). The prelude
  therefore leaves one block, `blocks[0]` = the embedded input. Each pass adds
  `blocks[1]` = its layer-6 input and `blocks[2]` = its layer-12 input. The
  bank that pass p + 1 reads is pass p's layer-12 input block and its layer-17
  output. Layer 18 then pushes the last pass's layer-17 output as `blocks[3]`.

## `mix(blocks, cur, query)`: attention over depth

```
S  = [blocks..., cur]                  # n = len(blocks) + 1 states per token
Sn = rmsnorm(S)                        # no gain, eps 1e-6
for each query column q_j (query is [2304, m]):
    w   = softmax_n(Sn · q_j / sqrt(2304))
    out_j = sum_n w_n * Sn_n           # the mix of the *normed* states
```

It returns m rows per token. m = 1 except for `attn_res_query` on full layers
(m = 3: separate Q, K, V inputs) and `loop_msg_query` (m = 4).

## `control(cur, msg, c)`: loop controller c ∈ {q, k, v, ffn}

```
g      = [rmsnorm(cur); rmsnorm(msg)] · loop_c_ctrl_down        # [4608] -> [64]
gain   = tanh(g · loop_c_ctrl_gain)                             # [64] -> [2304]
forget = tanh(g · loop_c_ctrl_forget)
return (1 - forget) * cur + gain * msg
```

## `layer(slot, ip, x, blocks, msg?)`

```
inpSA = x
in    = mix(blocks, x, attn_res_query) if attn_res_query else x   # layer 0 has none
inQ, inK, inV = in (SWA: the same row; full: rows 0, 1, 2)
if msg:     inQ,inK,inV = control(in*, msg[q|k|v], q|k|v)          # SWA inputs diverge here
SWA:   inQ,inK,inV = rmsnorm(.) * attn_norm   (once, shared, when there is no msg)
full:  rmsnorm(inQ) * attn_q_inp_norm, inK * attn_k_inp_norm, inV * attn_v_inp_norm

Q = clip(inQ · attn_q per head (128) * attn_q_clip, 22.63)          # [24 x 128]
K = clip(conv(inK · attn_k, attn_k_conv, dyn_k), 22.63)            # [8 x 128]
V = rmsnorm(conv(inV · attn_v, attn_v_conv, dyn_v))                # per head, no gain
R = clip(inQ · attn_rel per head (16) * attn_rel_clip, 8)          # [24 x 16]
full layers: Q *= τ, R *= τ,  τ = 1 + 0.1 * ln(max(1, (pos + 1) / 8192))
t = inQ · attn_sink_proj                                           # [24] sink logit per head

score(h, j) = Q_h · K_kv(h),j / sqrt(128) + R_h · B[:, d(j)] - t_h
    d(j) = pos - pos_j if 0 <= that < 1024 else 1024; B = [attn_rel_bias | 0] ([16, 1025])
    so beyond the extent (full layers only) the bias is just -t_h
softmax over j and one extra zero logit (the sink), causal; SWA: the last 1024 positions
o = Σ softmax * V, then o *= sigmoid(inQ · attn_gate)               # [24 x 128] = 3072
a = conv(o · attn_output, attn_post_conv)                           # [3072] -> [2304]

x = a if ip % 6 == 0 (and blocks.push(inpSA)) else inpSA + a
f = mix(blocks, x, ffn_res_query)
if msg: f = control(f, msg[ffn], ffn)
f = rmsnorm(f) * ffn_norm
x = x + conv(moe(f), ffn_post_conv)
```

- `clip(v, s, c)`: `v *= s` (per channel, when given), then v is scaled down so
  its L2 norm is at most c (`factor = min(1, c / (|v| + 1e-6))`), per head.
- `conv(u, w, dyn)`: a causal depthwise conv over the last 4 positions, plus a
  residual. `w` is `[4, C]`, and the oldest tap is first (ggml `ssm_conv`). On
  full layers, K and V add dynamic taps: `dyn = inK · attn_k_conv_proj`
  `[2304, 32]` = 4 taps × 8 kv heads. Tap `lag` multiplies the input `lag`
  positions back, the same scalar for every channel of the head. The result
  is `(u + conv) / sqrt(2·4 + 1)` with dynamic taps and `(u + conv) / sqrt(5)`
  without.
- Conv history: 3 rows per slot of 6656 channels, laid out as K (1024) | V
  (1024) | attn post (2304) | ffn post (2304).
- `moe(f)`:
  - Routing: `logits = f · ffn_gate_inp` (F32, `[2304, 128]`), `p = sigmoid(logits)`.
  - Selection: the top 8 by `p + exp_probs_b`.
  - Weights: the selected `p` (without the bias), normalised to sum 1, times 8.
  - Experts: `ffn_gate_up_exps` `[2304, 2304, 128]` gives each expert its gate
    (rows 0..1151) then its up (rows 1152..2303). The activation is
    `situ = 4·tanh(gate/4)·sigmoid(gate) · 25·tanh(up/25)`, followed by
    `ffn_down_exps` `[1152, 2304, 128]`. There is no shared expert.

## `output(x, blocks)`

```
h = rmsnorm(mix(blocks, x, output_res_query)) * output_norm
logits = 30 * tanh((h · output) / 30), first 200058 ids
```

## State per sequence (what the prefix cache and decode must carry)

| State | Size | Kind |
|---|---|---|
| full-attention K/V: layers 5, 11, 17, 23 → slots 5, 11, 17, 23, 29, 35, 41, 47 | 8 × ctx × 2 × 8 × 128 | positional |
| SWA K/V: the other 40 slots | 40 × 1024 × 2 × 8 × 128 | window ring |
| conv history: 48 slots | 48 × 3 × 6656 f32 | fixed |

Restoring a prefix-cache snapshot therefore needs the 40 SWA rings and the conv
history in the blob list, or a re-prefill of the last 1024 rows.

## quartz's TP2 layout (`quartz/arch/inkling2/ink2_tp2.h`)

Card c ∈ {0, 1} holds:

| Tensor | Card c's part |
|---|---|
| `attn_q`, `attn_gate` | q heads 12c .. 12c + 11 (rows 1536c, 1536 rows) |
| `attn_k`, `attn_v`, `attn_k_conv`, `attn_v_conv` | kv heads 4c .. 4c + 3 (whole GQA groups of 3 q heads) |
| `attn_k_conv_proj`, `attn_v_conv_proj` | the dynamic taps of those 4 kv heads |
| `attn_rel`, `attn_sink_proj`, `attn_q_clip`, `attn_rel_clip` | the rows of its 12 q heads |
| `attn_output` | half of the contraction (Q8 blocks 48c .. 48c + 47) |
| `ffn_gate_up_exps` | per expert, gate rows 576c .. and up rows 1152 + 576c .. (576 each) |
| `ffn_down_exps` | half of the contraction (Q8 blocks 18c .. 18c + 17) |
| `output` | vocabulary rows [V/2 · c, V/2 · (c + 1)) |

Everything else (embeddings, norms, controllers, routers, mix queries, conv
taps of the post convs) is on both cards. The attention output and the
expert down projection produce partial sums, which the cards exchange and
add twice per layer application: after the attention and after the experts.
The post convs and the residual then run on both cards on the full row. Each
card keeps the K/V, and the K/V conv history, of its own 4 kv heads. The logits are split by vocabulary, and the pick merges
the two cards' top-K.

