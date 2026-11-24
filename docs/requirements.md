# Requirements

What quartz needs from the machine and the model files. Build and start
commands are in the [README](../README.md), flags and environment variables in
[configuration.md](configuration.md).

## What you supply

| | Item | Needed for |
|---|---|---|
| GPU | exactly 2× AMD MI50/MI60 (gfx906) 32 GB; PCIe peer access recommended | all |
| Toolchain | ROCm (`hipcc`, HIP runtime, rocBLAS), a C++17 compiler | build, run |
| Model (`--model`) | Qwen3.8-27B or a fine-tune as Q8_0 GGUF with the nextn (MTP) block, ~28 GB; or Inkling-2-Mini as Q8_0 GGUF, ~28 GB | all |
| Drafter (`--dspine`) | Qwen: DSpine draft GGUF, ~2.1 GB. Inkling: DSpark draft GGUF, ~0.4 GB | speculative decode |
| Projector (`--mmproj`) | Qwen: `qwen3vl_merger` GGUF, ~0.9 GB. Inkling: `inkling2v` GGUF, ~0.6 GB | images |
| Runtime | Python 3 with `jinja2` | server |

## Hardware

- **Two gfx906 cards, 32 GB each.** Every layer is split across the pair and the
  cards write into each other's memory, so there is no single-card mode and no
  mode for more cards. With more than two cards visible, quartz uses one pair
  (`QZ_DEVICES`).
- **P2P is recommended, not required.** With peer access the cards write into
  each other's VRAM. Without it they exchange through pinned host memory, at the
  same addresses, with the same arithmetic and the same greedy output, only
  slower: prefill about 10-25 % and decode 5-10 %. `QZ_LINK=auto` (the default)
  takes a peer pair when there is one and host memory otherwise; `p2p` and
  `host` force a mode ([configuration](configuration.md)).
- **gfx906 only.** The kernels use gfx906 instructions and need every block of
  a 60-block grid resident at once; the load fails on a card where they do not fit.
- **P2P setup.** The server sets `HSA_FORCE_FINE_GRAIN_PCIE=1` and
  `GPU_MAX_HW_QUEUES=8` unless already set; the host link does not need either.
  If you expect peer access and the start says `link host (no P2P)`, see
  [Troubleshooting](troubleshooting.md#cards).

## Memory

The server checks before the load that the requested context fits on both
cards, and prints a per-card ledger after it.

**Qwen3.8-27B** at the default context (262144) with DSpine: about 59 GB of the
~69 GB the two cards report (weights 30.4, KV cache 18.2, DSpine 7.6, buffers
3.2). Without DSpine, about 52 GB. A smaller `--max-ctx` shrinks the KV cache.
The prefix cache takes about 1.2 GB of pinned host RAM. If less than 256 MB is
free after the load, batched prefill is off, and with it DSpine and image input.

**Inkling-2-Mini**, per card: the weights take about half of the file. The 8
full-attention slots' K/V take 16 KB per token (4.3 GB at 262144), the 40
sliding-window rings a fixed 0.7 GB. The DSpark drafter is split over both
cards; its context cache adds 6 KB per token (1.6 GB at 262144). At the
default context: 20.6 GB per card, with DSpark 22.4 GB on card 0 and 22.2 GB on
card 1. The vision encoder adds up to 1.8 GB on card 0. The prefix cache takes
about 0.1 GB of pinned host RAM.

Without P2P none of the above changes, but the exchange takes a stretch of
pinned host RAM: a few tens of MB in decode, and up to about 0.25 GB while
prefill runs (the bulk copies are staged through it).

## Build

- `make` builds `build/bin/libquartz.so` with `hipcc`. rocBLAS is used by the
  vision encoders only. There is no cmake, no submodule and no llama.cpp tree.
- `hipcc` comes from `PATH`, else `$(ROCM_PATH)/bin/hipcc` (`ROCM_PATH`
  defaults to `/opt/rocm`). At run time `libamdhip64` and `librocblas` must
  resolve through the system loader.

## Models

Only the models in the README are supported. Target and draft must be Q8_0
(the projector is F16): the kernels read no other weight format, and other
quantisations are refused at load.

- Qwen: the target must have the MTP block. Decode always drafts, with MTP or
  DSpine; there is no token-by-token path.
- Qwen: the DSpine draft must come from the same base model; it reads the
  target's embeddings.
- Qwen: images are resized to 8..4096 tokens and decode on MTP.
- Inkling: `general.architecture` must be `inkling2_loop` and the tensors must
  match [inkling2.md](inkling2.md); both are checked at load. Without a
  drafter, decode runs one token per step. The DSpark draft reads the target's
  embeddings and features, so it must match the target.
- Inkling: images keep their size, padded to a multiple of 40 px, and are
  scaled down only above 4096 tokens.

## Licensing

quartz is MIT licensed ([LICENSE](../LICENSE)). `third_party/llama-unicode` is
MIT with its notice kept; `third_party/stb/stb_image.h` is public domain.
