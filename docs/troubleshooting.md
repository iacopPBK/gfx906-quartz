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
| `HIP 0 and HIP 1 are not a usable pair (…); usable pairs: …` | The two cards are not both gfx906 or have no peer access. Pick a listed pair with `QZ_DEVICES=a,b`, or see [Cards](#cards). |
| `peer check: … did not finish` or `… words arrived wrong` | Data written by one card into the other did not arrive intact. Check the PCIe topology and the IOMMU and peer-to-peer settings. |

These lines are normal: `memory per card: …` (the memory check's estimate),
`peer link ok: …`, and `devices: N of M are gfx906, using HIP …`.

## Cards

- quartz needs exactly two gfx906 cards with 32 GB each and peer access
  between them. 16 GB cards, a single card and three or more cards are not
  supported.
- Without peer access, check the BIOS (Above 4G decoding / large BAR), the
  IOMMU (off, or `iommu=pt` on the kernel command line), and the slots (both
  cards on the same CPU). `rocm-smi --showtopo` shows whether the cards reach
  each other.

## While running

| Symptom | Cause and fix |
|---|---|
| Image requests fail | Images need `--mmproj` and must be base64 `data:` URLs. |
| The drafter is not used | `--dspine` is missing, or (Qwen) batched prefill is off for lack of memory; the start log says so. |
| Decode on prose is slower with the drafter | Drafts are accepted less on prose. The drafter switch, on by default, moves away from the drafter when that happens; check that `QZ_NO_SWITCH` is not set. |
