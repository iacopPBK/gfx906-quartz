// Prefix cache: snapshots of an arch's fixed-size state at prompt positions, in pinned
// host memory. An arch names the state as blobs, a live and a mid-chunk staging buffer
// per card. KV arenas are positional and not copied: rows below the restored position
// are still valid in the arena.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "mem.h"

struct qz_blob {
    void * live[2] = {};    // per card: the state the engine runs on
    void * stage[2] = {};   // per card: the prefill's mid-chunk copy (null: none)
    size_t bytes = 0;       // per card
};

class qz_pcache {
public:
    struct slot {
        int pos = -1;                  // state after prompt row pos - 1 (-1: empty)
        unsigned long use = 0;
        std::vector<void *> host[2];   // per card, one buffer per blob
    };
    std::vector<qz_blob> blobs;
    std::vector<slot> slots;
    std::vector<uint64_t> keys;        // the prompt keys the engine state was built from
    unsigned long clock = 0;
    int head = -1;                     // the first message's end: kept while prompts match it
    int ncard = 2;
    hipStream_t st[2] = {};

    // n slots, every blob's host buffers allocated up front
    void init(const std::vector<qz_blob> & b, int n, hipStream_t s0, hipStream_t s1) {
        blobs = b; st[0] = s0; st[1] = s1;
        slots.resize(n);
        for (auto & sn : slots) alloc(sn);
    }

    // The deepest slot whose keys 0 .. pos match k, with pos <= n - 1. Slots past the
    // matching prefix, and slots that `valid` rejects, are dropped.
    slot * match(const std::vector<uint64_t> & k, int n, const std::function<bool(int)> & valid = {}) {
        int L = 0;
        while (L < (int) keys.size() && L < n && keys[L] == k[L]) L++;
        slot * use = nullptr;
        for (auto & sn : slots) {
            if (sn.pos < 0) continue;
            if (sn.pos + 1 > L) { sn.pos = -1; continue; }   // its tokens differ from this prompt's
            if (valid && !valid(sn.pos)) { sn.pos = -1; continue; }
            if (sn.pos <= n - 1 && (!use || sn.pos > use->pos)) use = &sn;
        }
        return use;
    }
    void forget_all() { for (auto & sn : slots) sn.pos = -1; }
    // the engine state no longer matches any snapshot
    void drop() { forget_all(); keys.clear(); head = -1; }

    // a slot for position pos: the one already there, a free one, or a victim (never `keep`)
    slot * pick(int pos, const slot * keep = nullptr) {
        slot * best = nullptr;
        for (auto & sn : slots) if (sn.pos == pos) best = &sn;          // same position: refresh in place
        if (!best) for (auto & sn : slots) if (sn.pos < 0) { best = &sn; break; }
        if (!best) best = victim(keep);
        alloc(*best);
        best->pos = -1;
        return best;
    }
    // Live state into a new slot for pos, async on the cards' streams.
    void take(int pos, const slot * keep = nullptr) {
        slot * best = pick(pos, keep);
        for (int i = 0; i < ncard; i++) {
            chk(hipSetDevice(qz_dev_of(i)));
            for (size_t j = 0; j < blobs.size(); j++)
                chk(hipMemcpyAsync(best->host[i][j], blobs[j].live[i], blobs[j].bytes, hipMemcpyDeviceToHost, st[i]));
        }
        chk(hipSetDevice(qz_dev_of(0)));
        best->pos = pos;
        best->use = ++clock;
    }
    // Staged mid-chunk state into slot sn, picked before the chunk ran.
    void take_staged(slot * sn, int pos) {
        for (int i = 0; i < ncard; i++) {
            chk(hipSetDevice(qz_dev_of(i)));
            for (size_t j = 0; j < blobs.size(); j++)
                chk(hipMemcpyAsync(sn->host[i][j], blobs[j].stage[i], blobs[j].bytes, hipMemcpyDeviceToHost, st[i]));
        }
        chk(hipSetDevice(qz_dev_of(0)));
        sn->pos = pos;
        sn->use = ++clock;
    }
    // Slot into live state, async. The caller resets the rest and syncs.
    void restore(slot & sn) {
        for (int i = 0; i < ncard; i++) {
            chk(hipSetDevice(qz_dev_of(i)));
            for (size_t j = 0; j < blobs.size(); j++)
                chk(hipMemcpyAsync(blobs[j].live[i], sn.host[i][j], blobs[j].bytes, hipMemcpyHostToDevice, st[i]));
        }
        chk(hipSetDevice(qz_dev_of(0)));
        sn.use = ++clock;
    }

    // the arch's side of a cached prefill
    struct hooks {
        std::function<void()> reset;      // no snapshot matched: fresh state
        std::function<void()> restored;   // blobs queued back: reset the rest, then sync
        // Prompt rows a .. b-1, where b is the next row. With mid >= 0, also stage the
        // state after row a + mid - 1 into the stage buffers. Nonzero: failed.
        std::function<int(int a, int b, int mid)> chunk;
        std::function<void()> sync;       // both cards idle (progress timing)
        // Optional: whether a snapshot at pos still fits the positional state, since a
        // ring-buffered window may have been overwritten.
        std::function<bool(int pos)> valid;
    };
    // Prefill prompt rows up to P - 2 from the deepest matching snapshot, in chunks, with
    // a snapshot at each chunk end, at the end of the first message (the second `boundary`
    // token) and at the last boundary mid-chunk, or `tail` rows before the end when the
    // chunk has no boundary. Returns the cached row count, -1 on failure.
    int prefill(const int32_t * p, int P, const std::vector<uint64_t> & k, int32_t boundary,
                int chunk, int tail, const hooks & h) {
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        slot * use = match(k, P, h.valid);
        int start = 0;
        if (use) { restore(*use); h.restored(); start = use->pos; }
        else { h.reset(); forget_all(); }
        keys = k;
        head = -1;
        for (int q = 1; q < P - 1; q++)
            if (p[q] == boundary) { head = q; break; }
        int cut = P - 1 - tail;
        for (int q = P - 2; q > start; q--)
            if (p[q] == boundary) { cut = q; break; }
        const bool prog = P - 1 - start > chunk;
        auto tc = t0;
        for (int a = start; a < P - 1;) {
            int b = std::min(a + chunk, P - 1);
            if (head > a && head < b) b = head;
            slot * mid = cut > a && cut < b ? pick(cut, use) : nullptr;
            if (h.chunk(a, b, mid ? cut - a : -1)) { drop(); return -1; }
            if (mid) take_staged(mid, cut);   // stream order: before the next chunk rewrites the stage
            take(b, mid);
            if (prog) {
                h.sync();
                const auto t = clk::now();
                const double ms = std::chrono::duration<double, std::milli>(t - tc).count();
                const double tot = std::chrono::duration<double, std::milli>(t - t0).count();
                qz_stamp(stderr);
                fprintf(stderr, "  prefill %d/%d %.0f t/s %.1f s\n", b, P - 1, 1000.0 * (b - a) / ms, tot / 1000.0);
                tc = t;
            }
            a = b;
        }
        return start;
    }

private:
    static void chk(hipError_t e) {
        if (e != hipSuccess) { fprintf(stderr, "prefix cache: HIP error %s\n", hipGetErrorString(e)); exit(1); }
    }
    void alloc(slot & sn) {
        for (int i = 0; i < ncard; i++) {
            if (sn.host[i].size() == blobs.size()) continue;
            sn.host[i].assign(blobs.size(), nullptr);
            for (size_t j = 0; j < blobs.size(); j++)
                chk(hipHostMalloc(&sn.host[i][j], blobs[j].bytes, hipHostMallocPortable));
        }
    }
    // Evict the snapshot whose loss costs least: the one with the smallest gap between its
    // neighbours relative to the next position. Never the two newest, `keep` or the head.
    // When no candidate is left, the least recently used.
    slot * victim(const slot * keep) {
        std::vector<slot *> live;
        for (auto & sn : slots) if (sn.pos >= 0) live.push_back(&sn);
        std::sort(live.begin(), live.end(), [](const slot * x, const slot * y) { return x->pos < y->pos; });
        slot * best = nullptr;
        double best_cost = 0;
        const int n = (int) live.size();
        for (int i = 0; i + 2 < n; i++) {
            slot * sn = live[i];
            if (sn == keep || sn->pos == head) continue;
            const int prev = i > 0 ? live[i - 1]->pos : 0, next = live[i + 1]->pos;
            const double cost = (double) (next - prev) / next;
            if (!best || cost < best_cost) { best = sn; best_cost = cost; }
        }
        if (!best) for (auto & sn : slots) if (&sn != keep && (!best || sn.use < best->use)) best = &sn;
        return best;
    }
};
