// Host sampler: llama.cpp's sampler chain over the exact per-row top-K candidates of
// both cards (kernels/topk.h), plus the C API spec conversion.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "capi.h"
#include "core/devices.h"   // QZ_NC_MAX
#include "kernels/topk.h"

// Sampled decode: llama.cpp's sampler chain, with order codes 2 top-k, 3 top-p, 4 min-p
// and 7 temperature. A draft is accepted only if it equals the sample, exactly.
struct qz_sampling {
    float temp = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    int min_keep = 0;
    uint32_t seed = 0;
    int n_order = 0;
    int order[8] = {};
};

struct qz_sampler {
    const qz_sampling * smp = nullptr;   // null: greedy
    std::mt19937 rng;
    int ncard = 2;                       // cards whose candidates tk[c] holds
    std::vector<float2> tk[QZ_NC_MAX];           // per card [rows][QZ_TOPK_NB][QZ_TOPK] (logit, id bits)
    std::vector<std::pair<float, int>> cand;
    void set(const qz_sampling * s) { smp = s; if (s) rng.seed(s->seed); }
    int sample_row(int t);
    // Row t's chain: ids in cand, unnormalized weights in w, n candidates returned.
    int row_chain(int t, std::vector<float> & w);
};

// C spec into qz_sampling. False when the engine cannot sample this chain exactly.
inline bool qz_sampling_from_c(const qz_sampling_c * smp, qz_sampling & ms) {
    bool ok = smp->top_k >= 1 && smp->top_k <= QZ_TOPK && smp->n_order >= 1 && smp->n_order <= 8;
    bool seen_k = false;
    for (int i = 0; ok && i < smp->n_order; i++) {
        const int o = smp->order[i];
        ok = o == 2 || o == 3 || o == 4 || o == 7;
        if (o == 2) seen_k = true;
        else if (!seen_k) ok = false;   // top-k must cut first
    }
    if (!ok) return false;
    ms.temp = smp->temp; ms.top_k = smp->top_k; ms.top_p = smp->top_p; ms.min_p = smp->min_p;
    ms.min_keep = smp->min_keep; ms.seed = smp->seed; ms.n_order = smp->n_order;
    for (int i = 0; i < smp->n_order; i++) ms.order[i] = smp->order[i];
    return true;
}

inline int qz_sampler::row_chain(int t, std::vector<float> & w) {
    const qz_sampling & sp = *smp;
    cand.clear();
    for (int c = 0; c < ncard; c++) {
        const float2 * e = tk[c].data() + (size_t) t * QZ_TOPK_NB * QZ_TOPK;
        for (int j = 0; j < QZ_TOPK_NB * QZ_TOPK; j++) {
            int id;
            memcpy(&id, &e[j].y, 4);
            if (id >= 0) cand.push_back({ e[j].x, id });
        }
    }
    auto desc = [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    size_t n = std::min<size_t>(QZ_TOPK, cand.size());
    std::partial_sort(cand.begin(), cand.begin() + n, cand.end(), desc);
    cand.resize(n);
    // The chain, applied as llama-sampler.cpp does on a sorted candidate array.
    std::vector<float> & p = w;
    for (int k = 0; k < sp.n_order && n > 0; k++) {
        switch (sp.order[k]) {
            case 2:   // top-k
                if (sp.top_k > 0) n = std::min<size_t>(n, (size_t) sp.top_k);
                break;
            case 3:   // top-p: softmax over the current candidates, cumulative cut
                if (sp.top_p < 1.0f) {
                    const float max_l = cand[0].first;
                    float sum = 0.0f;
                    p.resize(n);
                    for (size_t i = 0; i < n; i++) { p[i] = expf(cand[i].first - max_l); sum += p[i]; }
                    float cum = 0.0f;
                    size_t last = n;
                    for (size_t i = 0; i < n; i++) {
                        cum += p[i] / sum;
                        if (cum >= sp.top_p && i + 1 >= (size_t) sp.min_keep) { last = i + 1; break; }
                    }
                    n = last;
                }
                break;
            case 4:   // min-p: logit >= max + log(p) (sorted path)
                if (sp.min_p > 0.0f) {
                    const float min_logit = cand[0].first + logf(sp.min_p);
                    size_t i = 1;
                    for (; i < n; i++) if (cand[i].first < min_logit && i >= (size_t) sp.min_keep) break;
                    n = i;
                }
                break;
            case 7:   // temperature
                if (sp.temp <= 0.0f) n = 1;
                else for (size_t i = 0; i < n; i++) cand[i].first /= sp.temp;
                break;
            default:
                break;
        }
    }
    if (n <= 1) { w.clear(); return 1; }
    const float max_l = cand[0].first;
    p.resize(n);
    for (size_t i = 0; i < n; i++) p[i] = expf(cand[i].first - max_l);
    return (int) n;
}

inline int qz_sampler::sample_row(int t) {
    // Softmax in float with a double sum, one uniform draw, then a cumulative pick.
    std::vector<float> p;
    const size_t n = (size_t) row_chain(t, p);
    std::uniform_real_distribution<double> u01(0.0f, 1.0f);
    if (n <= 1) { (void) u01(rng); return cand[0].second; }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += p[i];
    const double tgt = sum * u01(rng);
    double run = 0.0;
    for (size_t i = 0; i < n; i++) {
        run += p[i];
        if (run >= tgt) return cand[i].second;
    }
    return cand[n - 1].second;
}
