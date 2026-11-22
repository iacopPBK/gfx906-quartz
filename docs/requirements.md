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
| Model (`--model`) | Qwen3.8-27B (or a fine-tune) as **Q8_0** GGUF, **with the nextn (MTP) block**, ~28 GB | all |
| Drafter (`--dspine`) | DSpine draft GGUF, ~2.1 GB | decode |
| Projector (`--mmproj`) | `qwen3vl_merger` vision GGUF, ~0.9 GB | images |
| Runtime | Python 3 with `jinja2` | server |

## Hardware

- **Exactly two gfx906 cards, 32 GB each, with PCIe peer access (P2P)
  between them.** The model is split across the pair and the cards exchange
  data directly at every layer, so there is no single-card mode, no mode for
  three or more cards, and no fallback without P2P. With more than two cards
  visible the server uses one pair; `QZ_DEVICES=a,b` picks which.
- **Memory.** The default context (262144 tokens) with DSpine uses about
  **59 GB of the ~69 GB** the two cards report: weights 30.4, KV cache 18.2,
  DSpine 7.6, buffers 3.2. Without DSpine it is about 52 GB, and a smaller
  `--max-ctx` shrinks the KV cache. The prefix cache adds about 1.2 GB of
  pinned host RAM. The server prints the full ledger at start.
  If less than 256 MB stays free after the load, batched prefill is turned
  off, and with it DSpine and image input (the start log says so).
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

Only the GGUFs in the table above are supported, and the target and the
draft must be **Q8_0** (the projector is the F16 file). This is a hard limit,
not a tested-only list: the kernels read Q8_0 weights and nothing else, so
Q6_K, Q4_K_M and other quantisations are refused at load with the reason. Any fine-tune of Qwen3.8-27B works if it is Q8_0 and keeps
the nextn (MTP) block.

- The target must have the MTP block: decode always drafts with MTP or
  DSpine, there is no plain token-by-token path.
- The DSpine draft must come from the same base model, since it reads the
  target's embeddings.
- Images are resized to at most 4096 tokens and always decode on MTP.

## Server

- Chat uses the template embedded in the GGUF.
- `127.0.0.1:8090`, no authentication (`--host` changes it), model id `quartz`.
- One request at a time; `--max-ctx` at least 1024.
- Logs go to stderr; `QZ_TRACE=<dir>` adds per-request traces, read with
  `quartz/server/quartz_trace.py`.

## Licensing

quartz is MIT licensed ([LICENSE](../LICENSE)). `third_party/llama-unicode` is MIT with its
notice kept; `third_party/stb/stb_image.h` is public domain.
