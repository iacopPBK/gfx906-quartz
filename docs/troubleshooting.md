# Troubleshooting

Start-up problems, in the order the server meets them. Every message below is
printed on stderr with a time stamp; the lines before an `error:` line carry the
engine's own reason.

## Before the load starts

The server checks these in a second and exits with code 2.

| Message | Cause and fix |
|---|---|
| `libquartz.so not found: run make first` | The library is not built, or `--lib` points elsewhere: run `make`. |
| `--model … No such file or directory` | Wrong path to the GGUF (same for `--dspine`, `--mmproj`). |
| `… is not a GGUF file (is the download complete?)` | A partial or wrong download: check the size against the model page. |
| `cannot listen on 127.0.0.1:8090 … Address already in use` | Another server is running: stop it or pass `--port`. |

## During the load

| Symptom | Cause and fix |
|---|---|
| `error: engine load failed … (the lines above say why …)` | The engine refused the setup. Read the lines above it: usually a Qwen target has no MTP (nextn) block, the model is not the supported geometry, or the cards lack free memory. |
| `error: projector load failed` | The `--mmproj` file is not the `qwen3vl_merger` projector, or there is no memory left for it. |
| `card N … has X of 34.3 GB free and this start needs about Y GB` | Not enough free VRAM, checked before the weights load. The line under it names the largest `--max-ctx` that fits, and says when another process already holds memory. For Qwen, the default context with DSpine needs about 59 GB across the two cards; lower `--max-ctx` or drop `--dspine`. `QZ_NO_PREFLIGHT=1` loads anyway. |
| `HIP 0 and HIP 1 are not a usable pair (no peer access …); usable pairs: none` | The cards cannot reach each other's memory over PCIe, so quartz cannot run on them. See [Cards](#cards) below. |
| `peer check: … did not finish` or `… words arrived wrong` | A kernel on one card wrote into the other card's memory and the data did not arrive intact. quartz cannot run on that pair: check the PCIe topology and the platform's IOMMU and peer-to-peer settings. |
| `memory per card: weights … = Y GB; free …` and `peer link ok: … GB/s` | Not errors: the start-up checks passed. The first line is the estimate the memory check used. |
| `devices: 2 of 3 are gfx906, using HIP 0,1` | Not an error: the server kept the gfx906 pair and left the other card out. With more than two gfx906 cards, pick the pair with `QZ_DEVICES=a,b`. |

## Cards

- quartz needs **exactly two gfx906 cards with 32 GB each and peer access
  (P2P) between them**. 16 GB cards, a single card, and setups of three or more
  cards working together are not supported, and there is no mode without P2P.
- If there is no peer access, the usual places to look are the BIOS (enable
  Above 4G decoding / large BAR), the IOMMU (off, or `iommu=pt` on the kernel
  command line), and the slots: both cards should hang off the same CPU.
  `rocm-smi --showtopo` shows whether the pair can reach each other.
- The server sets `HSA_FORCE_FINE_GRAIN_PCIE=1` and `GPU_MAX_HW_QUEUES=8` unless
  you already set them.

## While running

| Symptom | Cause and fix |
|---|---|
| Image requests fail | Images need `--mmproj`, and must be base64 `data:` URLs in the request. |
| `DSpine` is not used | `--dspine` is missing, or the prefill headroom check turned batched prefill off; the start log says so. |
| Decode is slower on long prose than plain MTP | Expected with DSpine alone. By default the drafter switch moves to MTP when acceptance drops; `QZ_NO_SWITCH=1` turns that off. |
