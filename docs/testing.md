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
touches, without and with `--dspine`; for Inkling, a greedy run gives the same
tokens with the DSpark rounds and as single steps, so both runs must match each
other too. A sampled run keeps the sampler's distribution rather than the token
sequence, so the two runs of it are not comparable token for token.

Qwen DSpine's 12-row rounds can differ from the 8-row ones at near-ties, where
two tokens have almost the same score. A difference that starts at such a
token is not a bug.

## Changes to shared code

A change in `quartz/core`, `quartz/kernels` or the loader must leave every
architecture bit-identical. See [Adding a model](adding-a-model.md#validation).

## Four cards on two GPUs

`QZ_TP=4 QZ_VIRT=2` with the `make tp4` library runs the 4-card path on two GPUs
(two cards per GPU). It checks correctness, not speed: cards on one GPU write to
each other through local memory, so the PCIe link between four real GPUs is not
exercised.

On some hosts, one GPU writing to several peers at once drops P2P to a fraction of a GB/s.
The exchanges therefore write to one peer per step: two hops for the decode
sums, and one copy stream per card for the prefill partials. `make tp4 XCHECK=1`
builds `build/tp4chk`, which checks this rule under emulation
(`core/xcheck.h`). After each request it prints, per code site and card, every
store to a second peer within one step (between system fences, or after the
thread's own flag), and every copy that started while the same card's copy to
another peer was still running on another stream. Normal builds compile none of
it.

```bash
make tp4 XCHECK=1
QZ_TP=4 QZ_VIRT=2 LIB=build/tp4chk/libquartz.so python3 your_test.py 2>&1 | grep XCHECK
```
