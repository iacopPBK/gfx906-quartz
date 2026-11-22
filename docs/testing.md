# Testing

quartz has no test suite. A change is checked by comparing the new build with
the previous one on the same prompts.

## Greedy A/B

Greedy decoding is deterministic, so two builds that compute the same thing
produce the same token ids.

1. Build the old version and keep its library:
   `cp build/bin/libquartz.so /tmp/libquartz-old.so`.
2. Build the new version.
3. Run the same prompts through each build with `temperature 0` and
   `"return_tokens": true`, starting the server once per build:

```bash
python3 quartz/server/quartz_server.py --model <target.gguf> --lib /tmp/libquartz-old.so
python3 quartz/server/quartz_server.py --model <target.gguf>
```

4. Compare the `tokens` arrays. They must be identical.

Keep it short: a few prompts (one short, one of 1-2k tokens, and the same one
again to hit the prefix cache), about 32 generated tokens each. Run it with
MTP, and again with `--dspine` if the change touches DSpine.

**One expected difference:** DSpine's 12-row verify rounds can split from the
8-row ones at near-ties, where two tokens have almost the same score. A
difference that starts at such a token is not a bug.

## Speed and quality

The numbers in the README were measured like this:

- **Prefill:** pp2048 at depth 0 and 16384, median of three runs, against
  llama.cpp's `llama-bench` with the same settings.
- **Decode:** 400-token answers per text type, sampled with a fixed seed and
  thinking off, read from the server's per-request log line.
- **Perplexity:** wikitext-2, 8 windows of 2048 tokens, second half of each
  window scored, against `llama-perplexity`. The quartz side used a temporary
  hook that is not in this tree.
