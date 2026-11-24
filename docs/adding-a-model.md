# Adding a model

quartz runs one model per process. Each architecture has its own engine written
for that model on two MI50s: a persistent decode kernel, a batched prefill and
the speculative decoding that fits the model. There is no generic graph. The
architectures share the plumbing around the engine, so a new model adds a
directory and a few table entries and leaves the others alone.

## Supported architectures

| `general.architecture` | Directory | Model | Features |
|---|---|---|---|
| `qwen35` | `quartz/arch/qwen35/` | Qwen3.8-27B Q8_0 or a fine-tune, with the nextn block | TP2, MTP and DSpine, prefix cache, `qwen3vl` images |
| `inkling2_loop` | `quartz/arch/inkling2/` | Inkling-2-Mini Q8_0 | TP2, DSpark, prefix cache, `inkling2v` images |

## Layout

| Path | What | Changed per arch |
|---|---|---|
| `quartz/capi.h`, `quartz/capi.cpp` | The C API. `qz_create` reads `general.architecture` and calls the arch's create function from `qz_archs[]` | one table entry |
| `quartz/arch.h` | `qz_ctx`, the interface an arch implements, and `qz_create_opts` | one create declaration |
| `quartz/arch/<arch>/` | The arch: engine, kernels, prefill, TP split, drafter, memory check, `qz_ctx` subclass | the arch only |
| `quartz/core/` | Host code with no arch knowledge: `devices.h` (pair selection), `cards.h` (peer access, P2P inboxes, residency check), `sampler.h` (host sampling), `prefix_cache.h` (snapshot slots, chunked prefill driver) | — |
| `quartz/kernels/` | Device building blocks templated on the arch's widths: `dev.h` (address hints, int8 dot, DPP, wave sums, Q8 rounding), `gemv.h`, `xchg.h` (inbox exchange), `pf_gemm.h` (prefill GEMM and quant), `topk.h` | — |
| `quartz/loader.{h,cpp}`, `quartz/gguf.h`, `quartz/gguf_lite.h`, `quartz/repack.h` | GGUF reading and the Q8_0 repack; the arch supplies the TP slice callback | — |
| `quartz/mem.h`, `quartz/preflight.h` | Per-card memory ledger; the peer-link check before the load | — |
| `quartz/tokapi.cpp` | Byte-level BPE. `TOK_PRES[]` maps `tokenizer.ggml.pre` to its regexes | one row per new pre-tokenizer |
| `quartz/vision/` | Image encoders, one namespace per `clip.projector_type`; `common.h` and `resize.h` are shared | one encoder per projector |
| `quartz/server/quartz_server.py` | The OpenAI server, arch-neutral | — |
| `quartz/server/dialects.py` | Per-arch chat dialect: reasoning, content and tool-call parsing | one class per arch |

## Steps

1. **Directory and namespace.** Create `quartz/arch/<arch>/` with a prelude
   header (see `qwen35/q35_prelude.h`) that includes every system and shared
   header at global scope. Each `.cpp` of the arch includes the prelude, opens
   `namespace <ns> {`, then includes the arch's own headers. Shared headers
   are never included inside the namespace. Two arches can then both define
   `qz_engine`, `qz_cycle` and so on in one `libquartz.so`. Non-template
   `__global__` kernels in `quartz/kernels/` are `static` for the same reason.
2. **`qz_ctx`.** Derive from `qz_ctx` and implement `info`, `max_ctx`,
   `generate` and `abort`; the vision calls are optional. `info` fills every
   field of `qz_model_info`: `arch` (the GGUF string), `name`, `n_vocab`,
   `max_ctx`, `img_pad` (the image placeholder id, or -1), `vision` and `dspine`.
3. **Register.** Declare `qz_ctx * qz_<arch>_create(const char *, const qz_create_opts &)`
   in `quartz/arch.h`, define it outside the namespace as a forwarder, and add
   it to `qz_archs[]` in `quartz/capi.cpp`. The Makefile picks up
   `quartz/arch/*/*.cpp` and `*.h`.
4. **Weights.** Load with `qz_loader` and a `slice` callback for the arch's TP
   split (see `qwen35_tp2.h`). Before loading, check each tensor's shape and
   type with `qz_gguf::shape(name, {ne0, ne1, ...}, type)` (ggml order), and
   read the hparams with `qz_gguf::u32`, `f32`, `flag` and `ints`.
5. **Memory check.** Estimate the per-card need from the GGUF and the context
   before the load, and refuse with the largest `--max-ctx` that fits (see
   `qwen35_preflight.h`). Call `qz_link_check()` after it.
6. **Prefix cache.** Name the arch's fixed-size state as `qz_blob`s (a live
   buffer and a mid-chunk staging buffer per card). Call `qz_pcache::prefill`
   with the message-boundary token and the hooks: `reset` (fresh state),
   `restored` (what else to clear after a restore), `chunk` (prefill rows
   `a .. b-1`, stage at `mid`), `sync`, and optionally `valid` (reject a
   snapshot). Positional K/V is never copied: rows below a restored position
   are still valid. A ring-buffer KV (sliding window) must be a blob, or
   `valid` must reject snapshots whose rows the ring has overwritten.
7. **Tokenizer.** For a new `tokenizer.ggml.pre`, add its regex list to
   `TOK_PRES[]`, copied from llama.cpp's `llama-vocab.cpp`. Check it against
   llama.cpp's `test-tokenizer-0` on a mixed-script text: the ids must match.
8. **Server dialect.** Add a `Dialect` subclass to `quartz/server/dialects.py`
   and register it in `DIALECTS` under the GGUF arch string. Set
   `special_pieces = True` when the model's structure tokens are CONTROL
   tokens, which otherwise decode to nothing.
9. **Docs.** Add a row to the table above, the model to the README, and its
   files and memory to `docs/requirements.md`.

## Validation

- **New arch.** Compare greedy tokens and per-layer hidden states with
  llama.cpp on the same GGUF. The first divergences should come only at
  near-tied logits.
- **Shared code.** A change must leave every arch bit-identical:
  - The device ISA of every kernel TU is unchanged: `hipcc --cuda-device-only -S`,
    compared per function after normalising labels, comments and name prefixes.
  - A token A/B of the two `libquartz.so` builds gives identical output:
    greedy and seeded sampling, each drafter, short and long prompts, and a
    repeated prompt for the prefix cache, comparing cached row counts too.
