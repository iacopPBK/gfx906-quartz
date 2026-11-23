// inkling2 start-up memory check, before the weight load (as qwen35_preflight.h):
// the memory each card needs for the requested context against what it has free.
// Included inside namespace ink2, after ink2_kernels.h and ink2_dspark.h.

// what grows with the context, per card: the 8 full-attention slots' f16 K / V
// (ctx rows), the 40 SWA slots' rings (PF_RING rows) and the conv histories;
// on card 0 with the drafter, its 3 layers' f16 context K / V
inline double ink2_ctx_bytes(int ctx, int card, bool ds) {
    double b = 0;
    for (int s = 0; s < NAPP; s++) b += 2.0 * (is_full(slot_phys(s)) ? ctx : PF_RING) * KVL * 2;
    b += (double) NAPP * 3 * CHL * 4;
    if (ds && card == 0) b += 2.0 * DS_NL * ctx * DS_KV * 2;
    return b;
}
// the rest: the activations (VR rows), the batched prefill's buffers, the HIP runtime's
// share; the drafter's weights and work buffers (card 0; card 1 holds its head fold
// buffers); 256 MB left free on top. Measured 2026-11-23 at 262144: 20.6 GB per card,
// with DSpark 24.3 / 20.6 (this estimate: 20.9, with DSpark 24.8 / 21.0)
constexpr double INK_PF_FIXED = 1.5e9, INK_PF_DS0 = 0.6e9, INK_PF_DS1 = 0.1e9, INK_PF_HEADROOM = 0.25e9;

// 0: fits (or cannot be sized here: the load reports the model problem); 1: refused
inline int ink2_vram_check(const qz_gguf & g, int ctx, const char * dspark) {
    const bool ds = dspark != nullptr;
    double need[2], free_b[2], total_b[2], w[2];
    for (int c = 0; c < 2; c++) {
        qz_loader L;
        L.slice = ink2_tp2_spec;
        if (!L.plan(g, c)) return 0;
        size_t fr = 0, tot = 0;
        if (hipSetDevice(qz_dev_of(c)) != hipSuccess || hipMemGetInfo(&fr, &tot) != hipSuccess) return 0;
        w[c] = (double) L.need_bytes; free_b[c] = (double) fr; total_b[c] = (double) tot;
        need[c] = w[c] + ink2_ctx_bytes(ctx, c, ds) + INK_PF_FIXED + INK_PF_HEADROOM + (ds ? (c == 0 ? INK_PF_DS0 : INK_PF_DS1) : 0.0);
    }
    (void) hipSetDevice(qz_dev_of(0));
    fprintf(stderr, "inkling2: memory per card: weights %.1f + context %.1f (ctx %d)%s + buffers %.1f = %.1f / %.1f GB; free %.1f and %.1f GB\n",
            w[0] / 1e9, ink2_ctx_bytes(ctx, 0, ds) / 1e9, ctx, ds ? " with DSpark" : "", (INK_PF_FIXED + INK_PF_HEADROOM) / 1e9,
            need[0] / 1e9, need[1] / 1e9, free_b[0] / 1e9, free_b[1] / 1e9);
    int bad = -1;
    for (int c = 0; c < 2; c++)
        if (need[c] > free_b[c] && (bad < 0 || need[c] - free_b[c] > need[bad] - free_b[bad])) bad = c;
    if (bad < 0) return 0;
    int fit = 0;                                   // the largest context that fits on both cards
    for (int lo = 1024, hi = ctx; lo <= hi; ) {
        const int mid = lo + (hi - lo) / 2;
        bool ok = true;
        for (int c = 0; c < 2; c++) ok = ok && need[c] - ink2_ctx_bytes(ctx, c, ds) + ink2_ctx_bytes(mid, c, ds) <= free_b[c];
        if (ok) { fit = mid; lo = mid + 1; } else hi = mid - 1;
    }
    fit = fit / 1024 * 1024;
    fprintf(stderr, "inkling2: card %d (HIP %d) has %.1f of %.1f GB free and this start needs about %.1f GB\n",
            bad, qz_dev_of(bad), free_b[bad] / 1e9, total_b[bad] / 1e9, need[bad] / 1e9);
    if (total_b[bad] - free_b[bad] > 1.0e9)
        fprintf(stderr, "        %.1f GB of it is already in use: another process (an old server?) may hold it\n",
                (total_b[bad] - free_b[bad]) / 1e9);
    if (fit >= 1024) fprintf(stderr, "        --max-ctx %d fits\n", fit);
    else fprintf(stderr, "        even --max-ctx 1024 does not fit: free the card\n");
    if (getenv("QZ_NO_PREFLIGHT")) {
        fprintf(stderr, "inkling2: QZ_NO_PREFLIGHT set: loading anyway\n");
        return 0;
    }
    return 1;
}
