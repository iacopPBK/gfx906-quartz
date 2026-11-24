// inkling2 start-up memory check, run before the weight load (as qwen35_preflight.h). It
// compares the memory each card needs for the requested context with what the card has
// free. Included inside namespace ink2, after ink2_kernels.h and ink2_dspark.h.

// what grows with the context, per card: the 8 full-attention slots' f16 K / V (ctx rows),
// the 40 SWA slots' rings (PF_RING rows) and the conv histories. With the drafter, also its
// 3 layers' f16 context K / V for the card's 4 kv heads.
inline double ink2_ctx_bytes(int ctx, int card, bool ds) {
    double b = 0;
    for (int s = 0; s < NAPP; s++) b += 2.0 * (is_full(slot_phys(s)) ? ctx : PF_RING) * KVL * 2;
    b += (double) NAPP * 3 * CHL * 4;
    if (ds) b += 2.0 * DS_NL * ctx * DS_KVL * 2;
    return b;
}
// the fixed part: the activations (VR rows), the batched prefill's buffers and the HIP
// runtime's share. INK_PF_DS0 / DS1 cover the drafter's work buffers (the head fold on both
// cards, the Markov chain on card 0); its weight slices are sized from the GGUF instead.
// 256 MB of headroom on top. Measured at ctx 262144: 20.6 GB per card, this estimate 20.9.
constexpr double INK_PF_FIXED = 1.5e9, INK_PF_DS0 = 0.25e9, INK_PF_DS1 = 0.2e9, INK_PF_HEADROOM = 0.25e9;

// host mode only: each card's copy of the peer's two prefill partial slots (ink2_pf.h
// PARTP), 2 x PF_M x D floats. Under p2p the conv reads the peer's VRAM and these do not
// exist, so the estimate is unchanged there.
inline double ink2_partp_bytes() {
    return qz_link == QZ_LINK_HOST ? 2.0 * PF_M * D * 4 : 0.0;
}

// the vision encoder (QZ_MMPROJ, set by the server's --mmproj) on card 0. Covers the weight
// file and its activations at the 4096-token image limit.
constexpr double INK_PF_VIS = 1.15e9;
inline double ink2_vis_bytes() {
    const char * p = getenv("QZ_MMPROJ");
    if (!p) return 0;
    FILE * f = fopen(p, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    const double b = (double) ftell(f);
    fclose(f);
    return b + INK_PF_VIS;
}

// the drafter's weights on card c (TP2 slices, the Markov chain on card 0), 0 if unreadable
inline double ink2_ds_weight_bytes(const char * path, int c) {
    qz_gguf g;
    if (!path || !g.open(path)) return 0;
    qz_loader L;
    L.slice = ink2_ds_tp2_spec;
    if (c == 1) L.skip = ink2_dspark::chain_only;
    return L.plan(g, c) ? (double) L.need_bytes : 0;
}

// 0 when it fits or cannot be sized here (the load reports the problem), 1 when refused
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
        need[c] = w[c] + ink2_ctx_bytes(ctx, c, ds) + INK_PF_FIXED + INK_PF_HEADROOM + ink2_partp_bytes() +
                  (ds ? ink2_ds_weight_bytes(dspark, c) + (c == 0 ? INK_PF_DS0 : INK_PF_DS1) : 0.0) +
                  (c == 0 ? ink2_vis_bytes() : 0.0);
    }
    (void) hipSetDevice(qz_dev_of(0));
    fprintf(stderr, "inkling2: memory per card: weights %.1f + context %.1f (ctx %d)%s + buffers %.1f = %.1f / %.1f GB; free %.1f and %.1f GB\n",
            w[0] / 1e9, ink2_ctx_bytes(ctx, 0, ds) / 1e9, ctx, ds ? " with DSpark" : "",
            (INK_PF_FIXED + INK_PF_HEADROOM + ink2_partp_bytes()) / 1e9,
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
