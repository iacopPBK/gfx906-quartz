# Architecture

How quartz turns a request into tokens, for Qwen3.8-27B. The code is in
`quartz/`: the model's own engine in `arch/qwen35/`, the parts every model
shares in `core/`, `kernels/` and the top level (see
[Adding a model](adding-a-model.md)). File names below are relative to
`quartz/` and point at the part that does each step.

## The model on two cards

Qwen3.8-27B has 64 layers: 16 full-attention layers and 48 recurrent
(DeltaNet) layers, plus one MTP layer that predicts the next tokens.

The weights are split in half across the two cards (tensor parallel, `arch/qwen35/qwen35_tp2.h`):
each card holds half of the attention heads, half of the FFN rows and half of
the vocabulary head. After each layer the cards swap their partial results by
writing straight into each other's memory over PCIe. This is why both cards
must be gfx906 with peer access; `core/devices.h` picks the pair at start.

## A request, step by step

1. **Template.** The server renders the chat with the template stored in the
   GGUF and tokenizes it (`server/quartz_server.py`, `tokapi.cpp`).
2. **Prefix cache.** If the prompt starts like a recent one, quartz resumes
   from the deepest saved point instead of starting over (`core/prefix_cache.h`).
   Attention K/V stay on the cards; the recurrent state is saved in up to 8
   snapshots in host memory, taken at chunk ends and at message boundaries.
3. **Prefill.** The rest of the prompt runs as batched int8 GEMMs in chunks of
   2048 tokens (`arch/qwen35/prefill.h`). Prompts under 16 tokens skip this and go straight
   to decode.
4. **Decode.** One persistent kernel per card runs the whole model for a few
   rows at a time and checks drafted tokens on the device
   (`arch/qwen35/qwen35_cycle.hip`, `arch/qwen35/engine.h`). Every block of the kernel stays resident
   and synchronizes through a grid barrier, which is why the GPU must have 60
   CUs.
5. **Images.** With `--mmproj`, images are encoded on card 0 (`vision/qwen3vl.cpp`)
   and their rows replace the image placeholder tokens in the prompt.

## Drafters

quartz never decodes one token at a time: a drafter proposes several tokens
and the target model verifies them in one pass. Only tokens the target agrees
with are kept, so the output is the same as plain decoding (exactly for greedy,
in distribution for sampling).

| Drafter | What it is | Rows per round |
|:---|:---|:---|
| **MTP** | The target's own MTP layer | 3 or 4, chosen adaptively (`--slot-t`) |
| **DSpine** | A separate small draft model that reads the target's features from 5 layers | 8, or 12 after a run of good rounds |

DSpine is faster on code and structured text, MTP on prose. With `--dspine`,
the **drafter switch** watches how many tokens each round keeps and moves
between the two mid-answer (`arch/qwen35/qwen35_model.cpp`, `QZ_SW_*`). Image prompts always
decode on MTP.

## Memory

At start quartz checks that the context fits on both cards and that the peer
link works before loading anything (`arch/qwen35/qwen35_preflight.h`,
`preflight.h`), then prints a per-card
ledger of what it allocated (`mem.h`). The default context (262144 tokens)
takes about 59 GB in total with DSpine (half on each card).
