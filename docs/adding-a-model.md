# Adding a model

quartz runs one model per process. Each architecture brings its own engine,
designed for that model on two MI50s: a persistent decode megakernel, a
batched prefill, and whatever speculation fits the model. It does not use a
generic graph. What architectures share is the plumbing around the engine, so
a new model adds a directory and some table entries and leaves the existing
architectures alone.

## Supported architectures

| `general.architecture` | Directory | Model | Notes |
|---|---|---|---|
| `qwen35` | `quartz/arch/qwen35/` | Swift-1.5-Qwen3.8-27B Q8_0 (with the nextn block) | TP2, MTP and DSpine, prefix cache, qwen3vl image input |

## Layout

| Path | What | Who changes it |
|---|---|---|
| `quartz/capi.h`, `quartz/capi.cpp` | the C API; `qz_create` reads `general.architecture` and calls the arch's create function from `qz_archs[]` | one table entry per arch |
| `quartz/arch.h` | `qz_ctx`, the interface an arch implements; `qz_create_opts` | the create declaration per arch |
| `quartz/arch/<arch>/` | the arch: engine, kernels, prefill, TP split, drafter, `qz_ctx` subclass | the arch only |
| `quartz/core/` | host code with no arch knowledge: `sampler.h` (host sampling chain), `cards.h` (peer access, P2P inbox buffers, residency check), `prefix_cache.h` (snapshot slots and the chunked prefill driver) | shared |
| `quartz/kernels/` | device building blocks templated on the arch's widths: Q8 GEMV, the inbox exchange, row ops, prefill GEMM / quant, top-k | shared |
| `quartz/loader.{h,cpp}`, `quartz/gguf.h`, `quartz/gguf_lite.h` | GGUF reading and the Q8_0 plane repack; the arch supplies the TP slice callback | shared |
| `quartz/tokapi.cpp` | byte-level BPE; `TOK_PRES[]` maps `tokenizer.ggml.pre` to its regexes | one row per new pre-tokenizer |
| `quartz/vision/<projector>.{h,cpp}` | image encoders, one namespace per `clip.projector_type`; an arch's `vision_attach` picks one | one encoder per projector |
| `quartz/server/quartz_server.py` | the OpenAI server; arch-neutral | — |
| `quartz/server/dialects.py` | per-arch chat dialect: reasoning / content / tool-call parsing, streamed and whole | one class per arch |

## Steps

1. **Directory and namespace.** Create `quartz/arch/<arch>/` with a prelude header
   (see `qwen35/q35_prelude.h`). The prelude includes every system and shared
   quartz header at global scope. Each `.cpp` of the arch includes the prelude
   first, opens `namespace <ns> {`, and only then includes the arch's own
   headers. Arch code must not include a shared header inside its namespace.
   Two arches can then both define `qz_engine`, `qz_cycle` and friends in one
   `libquartz.so` without symbol collisions. Non-template `__global__` kernels in
   `quartz/kernels/` are `static` for the same reason: every arch TU gets its own
   copy.
2. **`qz_ctx`.** Derive from `qz_ctx` and implement `info`, `max_ctx`,
   `generate` and `abort`; the vision calls are optional. In `info`, fill
   `arch` (the GGUF string), `name`, `n_vocab`, `max_ctx`, and `img_pad` (the
   image placeholder id, or -1).
3. **Register.** Declare `qz_ctx * qz_<arch>_create(const char *, const qz_create_opts &)`
   in `quartz/arch.h`, define it outside the namespace as a forwarder, and add it
   to `qz_archs[]` in `quartz/capi.cpp`. The Makefile picks up
   `quartz/arch/*/*.cpp` and `quartz/arch/*/*.h` on its own.
4. **Weights.** Load with `qz_loader`, using a `slice` callback for the
   arch's TP split (see `qwen35_tp2.h`). Check every tensor's file shape with
   `qz_loader::shape(name, {ne0, ne1, ...})` (ggml order) before trusting
   the hparams.
5. **Prefix cache.** Name the arch's fixed-size recurrent state as
   `qz_blob`s: a live buffer and a mid-chunk staging buffer per card. Call
   `qz_pcache::prefill` with the arch's message-boundary token and hooks:
   `reset` for fresh state, `restored` for whatever else must be cleared after
   a restore, `chunk` to prefill rows `a .. b-1` and stage at `mid`, and
   `sync`. Positional state (KV arenas) is never copied, since the rows below
   a restored position are still valid. An arch whose KV is a ring (sliding
   window) must also list that ring as a blob, or snapshot only at positions
   the ring can serve.
6. **Tokenizer.** If `tokenizer.ggml.pre` is new, add its regex list to
   `TOK_PRES[]`, copied verbatim from llama.cpp's `llama-vocab.cpp` for that
   pre type. Check the result against llama.cpp's `test-tokenizer-0` on a
   mixed-script text: the token ids must match exactly.
7. **Server dialect.** Add a `Dialect` subclass to `quartz/server/dialects.py`
   and register it in `DIALECTS` under the GGUF arch string. Set
   `special_pieces = True` when the model's structure tokens are CONTROL
   tokens: they decode to nothing otherwise. The chat template runs as-is
   (jinja2, with HF's `{% generation %}` and `tojson` options handled in
   `Engine._load_template`).
8. **Docs.** Add a row to the table above and the arch's hardware, file and
   memory requirements to `docs/requirements.md`.

## Validation

- **Correctness vs a reference.** For a new arch, compare greedy tokens and
  per-layer hidden states with llama.cpp on the same GGUF. Decode is greedy
  repeatable, so the reference is exact up to float reduction order: first
  divergences should come only at near-tied logits.
- **Refactors of shared code.** These must leave every existing arch
  bit-identical. Check two things:
  - The device ISA of every kernel TU is unchanged. Use
    `hipcc --cuda-device-only -S`, then compare per function after
    normalising labels, comments and namespace or template name prefixes.
  - A token A/B of the two `libquartz.so` builds through ctypes gives identical
    output: greedy and seeded-sampled, each drafter, short and long prompts,
    with a repeated prompt to exercise the prefix cache. Compare the cached
    row counts as well as the tokens.

  The A/B tooling and data stay outside this repo.
