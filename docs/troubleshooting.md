# Troubleshooting

Start-up problems, in the order the server meets them. Messages go to stderr
with a time stamp; the lines before an `error:` line give the engine's reason.

## Before the load

The server checks these first and exits with code 2.

| Message | Cause and fix |
|---|---|
| ``… libquartz.so not found: run `make` first`` | The library is not built, or `--lib` points elsewhere. |
| `--model … No such file or directory` | Wrong path to the GGUF (same for `--dspine`, `--mmproj`). |
| `… is not a GGUF file (is the download complete?)` | A partial or wrong download. Compare the size with the model page. |
| `cannot listen on 127.0.0.1:8090 … Address already in use` | Another server is running. Stop it or pass `--port`. |

## During the load

| Message | Cause and fix |
|---|---|
| `error: engine load failed …` | The engine refused the setup; the lines above say why. Usually a Qwen target without the MTP block, an unsupported model, or not enough free memory. |
| `RuntimeError: projector load failed: <path>` | `--mmproj` is not the model's projector (`qwen3vl_merger` for Qwen, `inkling2v` for Inkling), or there is no memory left for it. |
| `card N (HIP n) has X of 34.3 GB free and this start needs about Y GB` | Not enough free VRAM. The next line gives the largest `--max-ctx` that fits and says whether another process holds memory. Lower `--max-ctx` or drop `--dspine`. `QZ_NO_PREFLIGHT=1` loads anyway. |
| `HIP 0 and HIP 1 are not a usable pair (…); usable pairs: …` | With `QZ_LINK=p2p`: the two cards are not both gfx906 or have no peer access. Drop `QZ_LINK`, so the exchange goes through host memory, or pick a listed pair with `QZ_DEVICES=a,b`. |
| `peer check: … did not finish` | Nothing was written across the link in 10 s, so the kernel may still be stuck and quartz stops. With `QZ_LINK` unset or `auto` the server restarts once with `QZ_LINK=host`; with `QZ_LINK=p2p` this is the end. Check the PCIe topology and the IOMMU. |
| `peer check: … words arrived wrong` | Data written by one card into the other did not arrive intact. With `auto` the engine moves the exchange to host memory by itself and says so; `QZ_LINK=p2p` refuses. Check the PCIe topology, the IOMMU and peer-to-peer settings. |
| `link check: … did not finish` or `… arrived wrong` | The same two failures for the host link, when there is no peer access: the cards reach pinned host memory badly. A platform problem: look for IOMMU faults in `dmesg`. |
| `host mailbox (card N, …): HIP a maps it at …; HIP b at …` | `qz_xalloc` cannot give both cards one address for a pinned host mailbox, which the exchange needs. Same platform problem as the row above; `QZ_LINK=p2p` never asks for it. |

These lines are normal: `memory per card: …` (the memory check's estimate),
`peer link ok: …` or `link host (no P2P): …` (which link the engine took), and
`devices: N of M are gfx906, using HIP …`.

## Cards

- quartz needs exactly two gfx906 cards with 32 GB each. Peer access between them
  makes it faster but is not required. 16 GB cards, a single card and three or
  more cards are not supported.
- If peer access is expected but the start takes the host link, check the BIOS
  (Above 4G decoding / large BAR), the IOMMU (off, or `iommu=pt` on the kernel
  command line), and the slots (both cards on the same CPU).
  `rocm-smi --showtopo` shows whether the cards reach each other.

## The link

`QZ_LINK` picks how the cards reach each other: `auto` (the default) takes a peer
pair when there is one and pinned host memory otherwise, `p2p` and `host` force
one of the two ([configuration](configuration.md)). The start log says which was
taken and the start-up summary repeats it (`… | link host (no P2P) | …`):

- `peer link ok: HIP 0 -> 1 … GB/s` — peer access, the fast path.
- `link host (no P2P): … GB/s, flag round trip … us` — pinned host memory. The
  same arithmetic and the same greedy output, roughly 5-20 % slower.
- `start: the peer check hung, so the peer link cannot be used: restarting once
  with QZ_LINK=host, …` — the server's own line, after a start that gave up on a
  hanging peer check in `auto` mode. One restart, never a loop.

## While running

| Symptom | Cause and fix |
|---|---|
| Image requests fail | Images need `--mmproj` and must be base64 `data:` URLs. |
| The drafter is not used | `--dspine` is missing, or (Qwen) batched prefill is off for lack of memory; the start log says so. |
| Decode on prose is slower with the drafter | Drafts are accepted less on prose. The drafter switch, on by default, moves away from the drafter when that happens; check that `QZ_NO_SWITCH` is not set. |
| Two sampled runs of the same request and seed differ | With Inkling's DSpark, the rows per round and the drafter switch follow measured timing, and with them the sampler's stream. Greedy runs do not differ, and neither link mode changes this. `QZ_DS_ROWS=8 QZ_NO_SWITCH=1` pins it. |
