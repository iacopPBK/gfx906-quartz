# Requirements

What quartz requires from the machine and the model files, and what it assumes
about them. Build and start commands are in the [README](../README.md); flags,
defaults and environment variables are in
[configuration.md](configuration.md).

## What you supply

| | Item | Needed for |
|---|---|---|
| GPU | 2× AMD MI50 (gfx906), 32 GB, PCIe peer access | all |
| Toolchain | ROCm: `hipcc`, HIP runtime, rocBLAS; a C++17 compiler | build, run |
| Model (`--model`) | Qwen3.8-27B Q8_0 GGUF **with the nextn (MTP) block**, ~28 GB | all |
| Drafter (`--dspine`) | DSpine draft GGUF, ~2.1 GB | decode |
| Projector (`--mmproj`) | `qwen3vl_merger` vision GGUF, ~0.9 GB | images |
| Runtime | Python 3 with `jinja2` | server |

## Hardware

- **Two gfx906 cards** with peer access between them. The server selects them
  among the visible devices, so nothing has to be exported; `QZ_DEVICES` picks
  a different pair when more than two are visible.
- **gfx906 only.** The build targets `--offload-arch=gfx906`;
  `quartz/dspine.h` carries `v_dot4_i32_i8` inline asm.
- **60 CUs, 64 KB LDS.** Every persistent kernel launches 60 blocks × 512
  threads and synchronizes through a grid barrier. Load checks the occupancy of
  each specialization and aborts ("grid barrier would hang") when
  `blocks_per_cu × CUs < 60`; extra CUs pass and stay unused.
- **32 GB per card.** The default context takes 29.7 GB of 34.3 per card:
  weights 15.2, KV 9.1, DSpine 3.8, scratch 0.6. The 8 prefix-cache snapshots
  live in pinned host memory, ~75 MB per slot per card. The server prints the
  per-card ledger at start.
- **Prefill headroom.** Below the 2048-row prefill buffers plus 256 MB free on
  either card, batched prefill is turned off, and with it DSpine and image
  input: `--dspine` falls back to MTP with a log line, while a DSpine
  *initialization* failure aborts the load.
- **P2P over PCIe** requires `HSA_FORCE_FINE_GRAIN_PCIE=1`; the server sets it
  and `GPU_MAX_HW_QUEUES=8` unless already set.

## Build

- `make` links `build/bin/libquartz.so`, the server's default `--lib`, with
  `hipcc` and C++17. rocBLAS is used by the vision encoder only; there is no
  cmake, no submodule and no llama.cpp tree.
- `hipcc` comes from `PATH`, else `$(ROCM_PATH)/bin/hipcc`
  (`ROCM_PATH` defaults to `/opt/rocm`). At run time `libamdhip64` and
  `librocblas` must resolve through the system loader.

## Models

Those three GGUFs are the only supported ones: their geometry is fixed and
checked at load, so another architecture, quantisation or vocabulary is refused
with the reason. The target must carry the nextn (MTP) block — decode drafts
with it or verifies with DSpine, there is no plain autoregressive path — and the
draft must come from the same base, since it reads the target's embeddings.
Images are resized to at most 4096 tokens and always decode on MTP.

## Server

- Chat uses the template embedded in the GGUF.
- `127.0.0.1:8090`, no authentication (`--host` changes it), model id `quartz`.
- One request at a time; `--max-ctx` at least 1024.
- Logs go to stderr; `QZ_TRACE=<dir>` adds per-request traces, read with
  `quartz/server/quartz_trace.py`.

## Licensing

quartz is MIT licensed ([LICENSE](../LICENSE)). `third_party/llama-unicode` is MIT with its
notice kept; `third_party/stb/stb_image.h` is public domain.
