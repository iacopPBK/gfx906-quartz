// qwen35 start-up memory check, run before the weight load: compares the memory each
// card needs for the requested context with what it has free.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

#include "core/devices.h"
#include "engine.h"
#include "loader.h"

// Per-card ledger: DSpine takes 3.8 GB at any context, and the cycle buffers, the
// snapshot staging, the drafter switch and the HIP runtime share about 1 GB. The
// batched prefill wants 256 MB left free.
constexpr double QZ_PF_DSPINE = 3.85e9, QZ_PF_FIXED = 1.0e9, QZ_PF_HEADROOM = 0.25e9;
// With --mmproj both cards grow by 1.1-1.2 GB at load and one by another ~0.3 GB once
// an image is encoded; the projector is 0.93 GB, so budget its file size + 1 GB.
constexpr double QZ_PF_MMPROJ_EXTRA = 1.0e9;

inline double qz_file_bytes(const char * p) {
    struct stat st;
    return p && stat(p, &st) == 0 ? (double) st.st_size : 0.0;
}

// 0: fits or cannot be sized here (init reports it); 1: refused
inline int qz_vram_check(const char * model, int max_ctx, int steps, size_t scratch, const char * dspine, const char * mmproj) {
    qz_gguf g;
    if (!g.open(model)) return 0;
    double weights[QZ_NC] = {}, free_b[QZ_NC] = {}, total_b[QZ_NC] = {};
    for (int c = 0; c < QZ_NC; c++) {
        qz_loader L;
        L.slice = qz_tp_slice;
        if (!L.plan(g, c)) return 0;
        weights[c] = (double) L.need_bytes;
        size_t fr = 0, tot = 0;
        if (hipSetDevice(qz_dev_of(c)) != hipSuccess || hipMemGetInfo(&fr, &tot) != hipSuccess) return 0;
        free_b[c] = (double) fr; total_b[c] = (double) tot;
    }
    (void) hipSetDevice(qz_dev_of(0));
    const double kv = (double) qz_kv_bytes(steps), ds = dspine ? QZ_PF_DSPINE : 0.0;
    const double mm = mmproj ? qz_file_bytes(mmproj) + QZ_PF_MMPROJ_EXTRA : 0.0;
    const double buf = (double) scratch + QZ_PF_FIXED + QZ_PF_HEADROOM;
    double need[QZ_NC];
    for (int c = 0; c < QZ_NC; c++) need[c] = weights[c] + kv + ds + mm + buf;
    char extra[64] = "";
    if (ds > 0) snprintf(extra, sizeof extra, " + DSpine %.1f", ds / 1e9);
    if (mm > 0) snprintf(extra + strlen(extra), sizeof extra - strlen(extra), " + vision %.1f", mm / 1e9);
    // GPU g carries cards g, g + nphys, ..: their needs add up against its free memory
    const int ng = qz_nphys;
    auto gneed = [&](int g, double kv_m) {
        double t = 0;
        for (int c = g; c < QZ_NC; c += ng) t += need[c] - kv + kv_m;
        return t;
    };
    if (ng == 2 && QZ_NC == 2)
        fprintf(stderr, "quartz: memory per card: weights %.1f + KV %.1f (ctx %d)%s + buffers %.1f = %.1f GB; free %.1f and %.1f GB\n",
                weights[0] / 1e9, kv / 1e9, max_ctx, extra, buf / 1e9, need[0] / 1e9, free_b[0] / 1e9, free_b[1] / 1e9);
    else {
        fprintf(stderr, "quartz: memory per card: weights %.1f + KV %.1f (ctx %d)%s + buffers %.1f = %.1f GB; per GPU",
                weights[0] / 1e9, kv / 1e9, max_ctx, extra, buf / 1e9, need[0] / 1e9);
        for (int g = 0; g < ng; g++) fprintf(stderr, " HIP %d %.1f / %.1f free", qz_dev_of(g), gneed(g, kv) / 1e9, free_b[g] / 1e9);
        fprintf(stderr, "\n");
    }
    int bad = -1;
    for (int g = 0; g < ng; g++)
        if (gneed(g, kv) > free_b[g] && (bad < 0 || gneed(g, kv) - free_b[g] > gneed(bad, kv) - free_b[bad])) bad = g;
    if (bad < 0) return 0;

    // the largest context whose K/V fits next to everything else on both cards
    int fit = 0;
    for (int lo = 1024, hi = max_ctx; lo <= hi; ) {
        const int mid = lo + (hi - lo) / 2;
        const double kv_m = (double) qz_kv_bytes(mid + 2 * QZ_TMAX);
        bool ok = true;
        for (int g = 0; g < ng; g++) ok = ok && gneed(g, kv_m) <= free_b[g];
        if (ok) { fit = mid; lo = mid + 1; } else hi = mid - 1;
    }
    fit = fit / 1024 * 1024;
    if (ng == QZ_NC)
        fprintf(stderr, "quartz: card %d (HIP %d) has %.1f of %.1f GB free and this start needs about %.1f GB\n",
                bad, qz_dev_of(bad), free_b[bad] / 1e9, total_b[bad] / 1e9, need[bad] / 1e9);
    else
        fprintf(stderr, "quartz: HIP %d (cards %d..) has %.1f of %.1f GB free and this start needs about %.1f GB there\n",
                qz_dev_of(bad), bad, free_b[bad] / 1e9, total_b[bad] / 1e9, gneed(bad, kv) / 1e9);
    if (total_b[bad] - free_b[bad] > 1.0e9)
        fprintf(stderr, "        %.1f GB of it is already in use: another process (an old server?) may hold it\n",
                (total_b[bad] - free_b[bad]) / 1e9);
    if (fit >= 1024) fprintf(stderr, "        --max-ctx %d fits%s\n", fit, ds > 0 ? ", or drop --dspine for more context" : "");
    else fprintf(stderr, "        even --max-ctx 1024 does not fit%s\n", ds > 0 ? ": free the card or drop --dspine" : ": free the card");
    if (getenv("QZ_NO_PREFLIGHT")) {
        fprintf(stderr, "quartz: QZ_NO_PREFLIGHT set: loading anyway\n");
        return 0;
    }
    return 1;
}
