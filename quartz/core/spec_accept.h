// One verify row of speculative sampling (Leviathan et al., "Fast Inference from
// Transformers via Speculative Decoding"): the draft d was drawn from the proposal q, the
// target's filtered distribution over the row's n candidate ids is p, and the emitted
// token is d with probability min(1, p(d) / q(d)), else a draw from the residual
// max(0, p - q) over p's support. Every row of a round that does this keeps the output
// exactly the sampler's own distribution: q only has to be a distribution over the vocab
// with q > 0 wherever it can sample, and it may be as far from p as it likes.
#pragma once

#include <cmath>

struct qz_spec_out {
    int id;             // the row's emitted token
    bool accepted;      // true: it is the draft, false: a residual draw
};

// ids[0 .. n-1]: the row's candidate ids (the sampler chain's candidates, so p is exactly
// the distribution sample_row would draw from). pw: their unnormalized weights. d: the
// draft, qd = q(d), qv[i] = q(ids[i]), u and u2: two draws in [0, 1).
// Accept when u * qd < p(d) (so p(d) == 0 always rejects and q(d) == 0 always accepts);
// otherwise draw from max(0, p - q), and from p itself when that residual has no mass
// (which happens when q = p on every candidate, or on rounding).
inline qz_spec_out qz_spec_row(const int * ids, const float * pw, int n, int d, double qd,
                               const double * qv, double u, double u2) {
    double S = 0.0;
    for (int i = 0; i < n; i++) S += (double) pw[i];
    double pd = 0.0;
    for (int i = 0; i < n; i++) if (ids[i] == d) { pd = (double) pw[i] / S; break; }
    if (u * qd < pd) return { d, true };
    double R = 0.0;
    for (int i = 0; i < n; i++) {
        const double r = (double) pw[i] / S - qv[i];
        if (r > 0.0) R += r;
    }
    if (R > 0.0) {
        const double tgt = R * u2;
        double run = 0.0;
        int last = ids[0];
        for (int i = 0; i < n; i++) {
            const double r = (double) pw[i] / S - qv[i];
            if (r <= 0.0) continue;
            last = ids[i];
            run += r;
            if (run >= tgt) return { ids[i], false };
        }
        return { last, false };     // rounding past the end of the walk
    }
    const double tgt = S * u2;      // the cumulative walk sample_row does
    double run = 0.0;
    for (int i = 0; i < n; i++) {
        run += (double) pw[i];
        if (run >= tgt) return { ids[i], false };
    }
    return { ids[n - 1], false };
}
