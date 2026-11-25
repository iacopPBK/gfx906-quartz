# Architecture

How quartz turns a request into tokens. Paths are relative to `quartz/`: each
model's engine is in `arch/<arch>/`, the shared parts in `core/`, `kernels/` and
the top level (see [Adding a model](adding-a-model.md)).

## Qwen3.8-27B

64 layers: 16 full-attention layers and 48 recurrent (DeltaNet) layers, plus
one MTP layer that drafts the next tokens.

### Two cards

The weights are split in half across the cards (`arch/qwen35/qwen35_tp2.h`):
each card holds half the attention heads, half the FFN rows and half the
vocabulary head. After each layer the cards write their partial results
straight into each other's memory over PCIe. `core/devices.h` picks the pair.

Without peer access (`QZ_LINK=host`, or `auto` on a pair that has none) the
mailboxes the cards hand each other — the exchange buffers, DSpine's verify
slots, DSpark's handshake — live in pinned host memory instead, and bulk copies
are staged through it (`core/link.h`). Both cards reach a mailbox through the
same pointer, so the exchange kernels run unchanged: same arithmetic, same
greedy output, and the extra cost is the PCIe trip.

Either way, a card only ever writes to the other card's memory: no kernel loads
or read-modify-writes across the link. Some boards (Intel client root ports)
forward peer writes but not peer reads, and a peer read there takes both cards
down until a reboot. Partials go into the reader's own memory by a store or a
peer copy, and every wait polls a local flag.

### A request

1. **Template.** The server renders the chat with the template from the GGUF
   and tokenizes it (`server/quartz_server.py`, `tokapi.cpp`).
2. **Prefix cache.** If the prompt starts like a recent one, decoding resumes
   from the deepest saved point (`core/prefix_cache.h`). Attention K/V stay on
   the cards; the recurrent state is saved in up to 8 snapshots in host
   memory, at chunk ends and at message boundaries.
3. **Prefill.** The rest of the prompt runs as int8 GEMMs in 2048-token chunks
   (`arch/qwen35/prefill.h`). Text prompts under 16 tokens go straight to decode.
4. **Decode.** One persistent kernel per card runs the whole model for a few
   rows per round and checks the drafted tokens on the device
   (`arch/qwen35/qwen35_cycle.hip`, `arch/qwen35/engine.h`). Its blocks
   synchronize through a grid barrier, so all of them must be resident at once.
5. **Images.** With `--mmproj`, images are encoded on both cards
   (`vision/qwen3vl.cpp`) and their rows replace the image placeholder tokens.

### Drafters

Decode always drafts: the target verifies several proposed tokens in one pass
and keeps those it agrees with, so the output equals plain decoding (exactly
for greedy, in distribution for sampling).

| Drafter | What it is | Rows per round |
|:---|:---|:---|
| **MTP** | The target's own MTP layer | 3 or 4, chosen adaptively (`--slot-t`) |
| **DSpine** | A separate draft model that reads the target's features from 5 layers | 8, or 12 after a run of good rounds |

DSpine is faster on code and structured text, MTP on prose. With `--dspine`,
the drafter switch follows the tokens kept per round and moves between the two
mid-answer (`arch/qwen35/qwen35_model.cpp`). Image prompts decode on MTP.

### Memory

Before the load, quartz checks that the context fits on both cards and that the
link between them carries the exchange (`arch/qwen35/qwen35_preflight.h`,
`preflight.h`). After it, it prints a per-card ledger of what it allocated
(`mem.h`). Numbers are in [requirements](requirements.md#memory).

## Inkling-2-Mini

`arch/inkling2/` follows the same request path with its own engine. The
forward pass is in [inkling2.md](inkling2.md).

- **Two cards.** Each card holds half the query and KV heads, half of every
  expert's FFN rows and half the vocabulary head (`ink2_tp2.h`). Partial sums
  cross over PCIe twice per layer application.
- **Prefill** runs in 2048-token chunks with a grouped GEMM over the selected
  experts (`ink2_pf.h`).
- **Prefix cache.** Snapshots hold the conv histories. The sliding-window K/V
  rings are 8192 rows, longer than the 1024-row window, and a snapshot is used
  only while its window rows are still in the ring.
- **Decode** is one persistent kernel per card per token (`ink2_cycle.h`).
  With a DSpark draft, each round drafts up to 7 tokens and a second persistent
  kernel verifies them with the anchor as up to 8 rows (`ink2_verify.h`). The
  round's width follows the recent acceptance and the measured round times.
  A row's token is picked as a single step would pick it. Greedy keeps a draft
  only when it equals the row's pick. Under a sampler with a temperature the
  drafter draws its drafts from its own distribution and a row accepts with
  probability `min(1, p / q)` or takes a draw from the residual `max(0, p - q)`,
  so the output keeps the sampler's distribution. The plain step and the verify
  share the attention body (`ink2_flash.h`). The drafter (`ink2_dspark.h`) is
  split over both cards like the target. The switch runs single steps while
  they are faster than the rounds.
- **Images** are encoded on card 0 (`vision/inkling2v.cpp`). Their rows
  replace the image placeholder in the prefill, without the embedding norm.
  Image prompts decode like text, DSpark included.
- **Memory** is checked before the load by `ink2_preflight.h`.
