# Configuration

Every flag and environment variable, with its default.

## Build

| Variable | Default | Effect |
|---|---|---|
| `BIN` | `build/bin` | Output directory |
| `ROCM_PATH` | `/opt/rocm` | Where to look for `hipcc` when it is not on `PATH` |
| `HIPCC` | `hipcc` from `PATH` | Compiler |

## Server

| Flag | Default | Effect |
|---|---|---|
| `--model` | required | Target GGUF |
| `--dspine` | unset | Draft GGUF: DSpine for Qwen, DSpark for Inkling |
| `--mmproj` | unset | Vision projector (`qwen3vl_merger` for Qwen, `inkling2v` for Inkling); enables image input |
| `--lib` | `build/bin/libquartz.so` | Engine library |
| `--host`, `--port` | `127.0.0.1`, `8090` | Bind address; no authentication |
| `--max-ctx` | `262144` | Context length, at least 1024. Memory per context is in [requirements](requirements.md#memory) |
| `--slot-t` | `0` | Qwen MTP rows per round: 3, 4, or 0 for adaptive |
| `--slack` | `1` | Qwen adaptive depth slack |
| `--temp` | `1.0` | Default temperature; 0 is greedy |
| `--top-k`, `--top-p`, `--min-p` | `20`, `0.95`, `0` | Default sampling; top-k 1..64 |
| `--reasoning-effort` | `medium` | Default reasoning effort: `xhigh`, `medium`, `low`, or `template` for the template's own default |
| `--progress` | `0` | Log decode progress every N seconds |

## Environment

| Variable | Default | Effect |
|---|---|---|
| `QZ_DEVICES` | `0,1` | The two cards to use, as HIP ordinals. On a machine with other GPUs the server first hides the non-gfx906 cards, so ordinals count gfx906 cards only |
| `QZ_LINK` | `auto` | How the two cards reach each other: `auto` takes a peer pair when there is one and pinned host memory otherwise; `p2p` refuses unless the pair has peer access; `host` forces host memory and never enables peer access. Same kernels and same greedy output either way, slower over host memory |
| `QZ_NO_PREFLIGHT` | unset | Load even when the memory check says the context does not fit |
| `QZ_TRACE` | unset | Directory for one JSON file per request, read with `quartz/server/quartz_trace.py` |
| `QZ_NO_SWITCH` | unset | `1` keeps the drafter for the whole answer. By default, with `--dspine`, Qwen switches between DSpine and MTP on acceptance, and Inkling between DSpark rounds and single steps on measured speed |
| `QZ_SW_LO`, `QZ_SW_HI`, `QZ_SW_WIN` | Qwen: `3.8`, `3.4`, `128` | Leave DSpine below `lo`, leave MTP at or above `hi` tokens per round, averaged over `win` tokens. Each threshold is on its drafter's own scale, so `lo > hi` is intended |
| `QZ_SW_LO`, `QZ_SW_WIN` | Inkling: `2.2`, `24` | DSpark rounds are timed over `win` tokens against the single step. Before the single step has been timed, decode leaves the rounds below `lo` tokens per round |
| `QZ_DS_ROWS` | unset | Inkling: `2`..`8` fixes the DSpark round's rows, the anchor and its drafts. Unset, each round picks its width from the recent acceptance and the measured round times |
| `QZ_IMG_MIN_TOKENS`, `QZ_IMG_MAX_TOKENS` | `8`, `4096` | Fewest and most tokens an image resizes to; each token takes a context row. Inkling reads only the maximum |
| `QZ_START_RETRIES` | `3` | Start attempts when the server crashes or hangs before it is ready; `0` runs without the supervisor. A peer check that hangs is its own case: one restart with `QZ_LINK=host`, in the default mode only |
| `QZ_START_TIMEOUT` | `180` | Seconds to wait for each start attempt |

The server sets `QZ_DSPINE` and `QZ_MMPROJ` from `--dspine` and `--mmproj`, and
`HSA_FORCE_FINE_GRAIN_PCIE=1` and `GPU_MAX_HW_QUEUES=8` unless they are set. The
host link does not depend on them (its mailboxes ask for coherent pinned memory
themselves); they are kept for the peer link.

## Log

One line per finished request:

```
17:44:55 1402 tok 103.5 t/s 5.90/rnd dspine | prompt 28055, 26324 cached, 3.4 s
```

Output tokens, decode speed, tokens per round, decode mode (`mtp` or `dspine`
for Qwen, `plain` or `dspark` for Inkling), prompt tokens, tokens taken from the
prefix cache, prefill time, and the finish reason when it is not `stop`.

On a machine with GPUs other than gfx906, a `devices:` line at start names the
gfx906 cards the server kept.
