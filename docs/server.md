# Server

`quartz/server/quartz_server.py` is an OpenAI-compatible HTTP server. It runs
one generation at a time; concurrent requests wait.

## Endpoints

| Method | Path | |
|:---|:---|:---|
| `POST` | `/v1/chat/completions` | Chat, with tools, images and streaming |
| `POST` | `/v1/completions` | Plain text, or a list of token ids as `prompt` |
| `GET` | `/v1/models` | The loaded model: id `quartz` for Qwen, `quartz-<general.name>` otherwise |
| `GET` | `/health`, `/v1/health` | Liveness |

## Request fields

| Field | Notes |
|:---|:---|
| `messages`, `prompt` | Chat messages, or text / token ids for `/v1/completions` |
| `max_tokens`, `max_completion_tokens` | Default 4096 for chat, 16 for completions; capped by the context |
| `temperature`, `top_k`, `top_p`, `min_p`, `seed` | Sampling. `temperature 0` or `top_k 1` is greedy; otherwise `top_k` must be 1..64 |
| `stop` | String or list of strings |
| `stream` | Server-sent events, OpenAI format |
| `tools` | Rendered by the chat template; calls come back as `tool_calls` |
| `reasoning_effort` | Qwen: `minimal` / `low` → `low`, `medium`, `high` / `xhigh` / `max` → `xhigh`. Other models: passed to the template unchanged (Inkling: `none` .. `max`). `none` turns thinking off: the answer opens directly. Default from `--reasoning-effort` |
| `chat_template_kwargs` | Passed to the template, e.g. `{"enable_thinking": false}` (thinking off, on every model); a `reasoning_effort` here wins |
| `ignore_eos` | Keep generating past the end-of-answer token |
| `return_tokens` | Add the generated token ids to the response as `tokens` |

Accepted but not applied: `presence_penalty`, `frequency_penalty`,
`repeat_penalty`, `typical_p`, `dry_multiplier`, `xtc_probability`,
`logit_bias`, `grammar`, `json_schema`, `response_format`. A non-neutral value
adds a `quartz_note` to the response.

## Responses

- Reasoning comes back in `reasoning_content`, separate from `content`.
- Tool calls are parsed from the output into `tool_calls`, with
  `finish_reason: "tool_calls"`. Each architecture has its own parser
  (`quartz/server/dialects.py`).
- Images must be base64 `data:` URLs and need `--mmproj`; otherwise HTTP 400.
- `timings` follows llama-server: tokens from the prefix cache, prompt and
  decode counts, times and speeds.
- Bad requests get HTTP 400. If the engine fails mid-answer, a normal request
  gets HTTP 500 and a streaming one an `error` event before `[DONE]`.

## Chat template

The template stored in the target GGUF, run with jinja2. Hugging Face
`{% generation %}` tags are treated as plain blocks.

## Traces

`QZ_TRACE=<dir>` writes one JSON file per request: the request, the rendered
prompt, the raw output with special tokens, the token ids, and flags for
answers that look wrong (cut short, thinking never closed, empty content,
unparsed tool call).

```bash
python3 quartz/server/quartz_trace.py <dir>          # list requests, * = flagged
python3 quartz/server/quartz_trace.py <file.json>    # one request
```
