# Configuration

Every switch and its default. Measured on 2× MI50 with the pi model and DSpine,
`--max-ctx 32768`, a 2.7k-token prompt; speeds vary ±3%.

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
| `--dspine` | unset | DSpine draft GGUF; decode uses it, images stay on MTP |
| `--mmproj` | unset | `qwen3vl_merger` projector; enables image input |
| `--lib` | `build/bin/libquartz.so` | Engine library |
| `--host`, `--port` | `127.0.0.1`, `8090` | Bind address, no authentication |
| `--max-ctx` | `262144` | Context length, at least 1024; 262144 takes 29.7 GB per card |
| `--slot-t` | `0` | Rows per cycle: 3, 4, or 0 adaptive |
| `--slack` | `1` | Adaptive depth slack |
| `--temp` | `1.0` | Default temperature, 0 is greedy |
| `--top-k`, `--top-p`, `--min-p` | `20`, `0.95`, `0` | Default sampling; top-k 1..64 |
| `--reasoning-effort` | `medium` | `xhigh`, `medium`, `low`, or `template` |
| `--progress` | `0` | Log decode progress every N seconds |

The server sets
`HSA_FORCE_FINE_GRAIN_PCIE=1` and `GPU_MAX_HW_QUEUES=8` unless they are set.

## Environment

| Variable | Default | Effect |
|---|---|---|
| `QZ_NO_PREFLIGHT` | unset | Load even when the start-up memory check says the context does not fit |
| `QZ_DEVICES` | `0,1` | Which two visible gfx906 cards to use, as HIP ordinals |
| `QZ_TRACE` | unset | Directory for one JSON per request, read with `quartz/server/quartz_trace.py` |
| `QZ_DSPINE` | unset | Draft GGUF path; the server sets it from `--dspine` |
| `QZ_NO_SWITCH` | unset | Set to `1` to keep DSpine for the whole answer. By default, with `--dspine`, decode switches drafters mid-answer on acceptance |
| `QZ_SW_LO`, `QZ_SW_HI`, `QZ_SW_WIN` | `3.8`, `3.4`, `128` | Leave DSpine below, leave MTP at or above this many tokens per round; window averaged. Each on its drafter's own scale, so `lo > hi` is intended |
| `QZ_IMG_MIN_TOKENS`, `QZ_IMG_MAX_TOKENS` | `8`, `4096` | Fewest and most tokens an image resizes to; each token takes a context row |

## Log

One line per finished request, plus engine and error lines:

```
17:44:55 1402 tok 103.5 t/s 5.90/rnd dspine | prompt 28055, 26324 cached, 3.4 s
```

Output tokens, decode speed, tokens per round, drafter, prompt tokens and how
many came from the prefix cache, prefill time, finish reason when it is not
`stop`.

A machine with cards of more than one architecture also gets one `devices:`
line at start, naming the two it chose.

## Speed

| | Greedy | T 1.0, top-k 20, top-p 0.95 |
|---|---|---|
| MTP | 93 tok/s | 77.8 tok/s |
| DSpine | 95 tok/s | 74.8 tok/s |

With DSpine on, the default drafter switch carries 38 mixed prompts of 800 tokens in 322 s,
against 361 s always-DSpine (`QZ_NO_SWITCH=1`) and 348 s always-MTP (greedy).
