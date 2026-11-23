# Testing

quartz has no test suite. A change is checked by comparing the new build with
the previous one on the same prompts.

## Greedy A/B

Greedy decoding is deterministic: two builds that compute the same thing
produce the same token ids.

1. Build the old version and keep its library:
   `cp build/bin/libquartz.so /tmp/libquartz-old.so`.
2. Build the new version.
3. Start the server once per build and send the same prompts with
   `temperature 0` and `"return_tokens": true`:

```bash
python3 quartz/server/quartz_server.py --model <target.gguf> --lib /tmp/libquartz-old.so
python3 quartz/server/quartz_server.py --model <target.gguf>
```

4. Compare the `tokens` arrays. They must be identical.

Use a few prompts (one short, one of 1-2k tokens, and the same one again to hit
the prefix cache), about 32 generated tokens each. Run each model the change
touches, without and with `--dspine`; for Inkling, the DSpark rounds give the
same tokens as single steps, so both runs must match each other too.

Qwen DSpine's 12-row rounds can differ from the 8-row ones at near-ties, where
two tokens have almost the same score. A difference that starts at such a
token is not a bug.

## Changes to shared code

A change in `quartz/core`, `quartz/kernels` or the loader must leave every
architecture bit-identical. See [Adding a model](adding-a-model.md#validation).
