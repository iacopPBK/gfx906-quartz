# Requirements

What quartz requires from the machine and the model files, and what it assumes
about them. Build and start commands are in the [README](../README.md); flags,
defaults and environment variables are in
[configuration.md](configuration.md).

## What you supply

| | Item | Needed for |
|---|---|---|
| GPU | exactly 2× AMD MI50/MI60 (gfx906) 32 GB, with PCIe peer access | all |
| Toolchain | ROCm: `hipcc`, HIP runtime, rocBLAS; a C++17 compiler | build, run |
| Model (`--model`) | Qwen3.8-27B (or a fine-tune) as **Q8_0** GGUF, **with the nextn (MTP) block**, ~28 GB; or Inkling-2-Mini as **Q8_0** GGUF, ~28 GB | all |
| Drafter (`--dspine`) | Qwen: DSpine draft GGUF, ~2.1 GB. Inkling: DSpark draft GGUF, ~0.4 GB | decode |
| Projector (`--mmproj`) | `qwen3vl_merger` vision GGUF, ~0.9 GB (Qwen only) | images |
| Runtime | Python 3 with `jinja2` | server |

## Hardware

- **Exactly two gfx906 cards, 32 GB each, with PCIe peer access (P2P)
  between them.** The model is split across the pair and the cards exchange
  data directly at every layer, so there is no single-card mode, no mode for
  three or more cards, and no fallback without P2P. With more than two cards
  visible the server uses one pair; `QZ_DEVICES=a,b` picks which.
- **Memory, Qwen3.8-27B.** The default context (262144 tokens) with DSpine uses about
  **59 GB of the ~69 GB** the two cards report: weights 30.4, KV cache 18.2,
  DSpine 7.6, buffers 3.2. Without DSpine it is about 52 GB, and a smaller
  `--max-ctx` shrinks the KV cache. The prefix cache adds about 1.2 GB of
  pinned host RAM. The server prints the full ledger at start.
  If less than 256 MB stays free after the load, batched prefill is turned
  off, and with it DSpine and image input (the start log says so).
- **Memory, Inkling-2-Mini.** The weights take about half of the 28 GB file
  per card. What grows with the context is small: the 8 full-attention
  slots' K/V take 16 KB per token per card (4.3 GB at 262144), and the 40
  sliding-window rings take a fixed 0.7 GB. The DSpark drafter is split over
  both cards like the target; its context cache adds 6 KB per token per card
  (1.6 GB at 262144). At the default context the cards hold 20.6 GB each, and
  about 2 GB more each with DSpark. The start-up check is in
  `arch/inkling2/ink2_preflight.h`.
- **gfx906 only** (MI50, MI60). The kernels use gfx906 instructions and assume
  its 60 CUs and 64 KB of LDS; load aborts on a card where they do not fit.
- **P2P setup.** The server sets `HSA_FORCE_FINE_GRAIN_PCIE=1` and
  `GPU_MAX_HW_QUEUES=8` unless already set. If the cards have no peer access,
  see [Troubleshooting](troubleshooting.md#cards).

## Build

- `make` links `build/bin/libquartz.so`, the server's default `--lib`, with
  `hipcc` and C++17. rocBLAS is used by the vision encoder only; there is no
  cmake, no submodule and no llama.cpp tree.
- `hipcc` comes from `PATH`, else `$(ROCM_PATH)/bin/hipcc`
  (`ROCM_PATH` defaults to `/opt/rocm`). At run time `libamdhip64` and
  `librocblas` must resolve through the system loader.

## Models

Only the GGUFs in the README's model tables are supported, and the target and the
draft must be **Q8_0** (the projector is the F16 file). This is a hard limit,
not a tested-only list: the kernels read Q8_0 weights and nothing else, so
Q6_K, Q4_K_M and other quantisations are refused at load with the reason. Any fine-tune of Qwen3.8-27B works if it is Q8_0 and keeps
the nextn (MTP) block.

- Qwen: the target must have the MTP block: decode always drafts with MTP or
  DSpine, there is no plain token-by-token path.
- Qwen: the DSpine draft must come from the same base model, since it reads the
  target's embeddings.
- Qwen: images are resized to at most 4096 tokens and always decode on MTP.
- Inkling: the GGUF's `general.architecture` must be `inkling2_loop`, and its
  tensors must match [inkling2.md](inkling2.md) (checked at load). Without a
  drafter, decode runs one token per step. The DSpark draft reads the target's
  embeddings and features, so it must match the target. No image input.

## Server

- Chat uses the template embedded in the GGUF.
- `127.0.0.1:8090`, no authentication (`--host` changes it), model id `quartz`
  for Qwen, `quartz-<model name>` for other models.
- One request at a time; `--max-ctx` at least 1024.
- Logs go to stderr; `QZ_TRACE=<dir>` adds per-request traces, read with
  `quartz/server/quartz_trace.py`.

## Licensing

quartz is MIT licensed ([LICENSE](../LICENSE)). `third_party/llama-unicode` is MIT with its
notice kept; `third_party/stb/stb_image.h` is public domain.
