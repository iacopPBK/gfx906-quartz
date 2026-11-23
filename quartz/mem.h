// Device memory ledger (per card, tagged by the innermost qz_mem_scope) and an
// arena that packs buffers under QZ_ARENA_BIG into 256 MB chunks.
#pragma once

#include <hip/hip_runtime.h>

#include "core/devices.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// "HH:MM:SS " in local time: the engine's routine lines start like the server's own
static inline void qz_stamp(FILE * f) {
    const time_t t = time(nullptr);
    struct tm tm;
    localtime_r(&t, &tm);
    fprintf(f, "%02d:%02d:%02d ", tm.tm_hour, tm.tm_min, tm.tm_sec);
}

#define QZ_MEM_CARDS qz_dev_max     // devices.h: the ledger is per HIP ordinal

struct qz_mem_ledger {
    struct rec { int card; size_t bytes; std::string tag; };
    std::mutex mu;
    std::unordered_map<void *, rec> live;
    std::map<std::string, size_t> by[QZ_MEM_CARDS];
    size_t tracked[QZ_MEM_CARDS] = {};
    long long base[QZ_MEM_CARDS] = { -1, -1, -1, -1, -1, -1, -1, -1 };   // used bytes before the engine's first buffer
};
inline qz_mem_ledger & qz_mem() { static qz_mem_ledger l; return l; }

#define QZ_ARENA_CHUNK ((size_t) 256 << 20)
#define QZ_ARENA_BIG   ((size_t) 64 << 20)
#define QZ_ARENA_ALIGN ((size_t) 256)
#define QZ_ARENA_GUARD ((size_t) 64 << 10)   // never handed out: a read just past a chunk's last buffer stays mapped

struct qz_arena {
    struct chunk { char * base; std::map<size_t, size_t> live; };   // offset -> size
    std::mutex mu;
    std::vector<chunk *> chunks[QZ_MEM_CARDS];
    std::unordered_map<void *, chunk *> owner;
};
inline qz_arena & qz_arena_get() { static qz_arena a; return a; }

// the first gap of `a` bytes in one of the card's chunks, or a new chunk
inline hipError_t qz_arena_alloc(void ** p, size_t bytes, int dev) {
    qz_arena & A = qz_arena_get();
    const size_t a = (bytes + QZ_ARENA_ALIGN - 1) / QZ_ARENA_ALIGN * QZ_ARENA_ALIGN;
    std::lock_guard<std::mutex> g(A.mu);
    for (qz_arena::chunk * c : A.chunks[dev]) {
        size_t at = 0;
        for (const auto & kv : c->live) {
            if (kv.first >= at + a) break;
            at = kv.first + kv.second;
        }
        if (at + a <= QZ_ARENA_CHUNK) {
            c->live[at] = a;
            *p = c->base + at;
            A.owner[*p] = c;
            return hipSuccess;
        }
    }
    void * base = nullptr;
    const hipError_t e = hipMalloc(&base, QZ_ARENA_CHUNK + QZ_ARENA_GUARD);
    if (e != hipSuccess) return e;
    qz_arena::chunk * c = new qz_arena::chunk{ (char *) base, {} };
    A.chunks[dev].push_back(c);
    c->live[0] = a;
    *p = base;
    A.owner[*p] = c;
    return hipSuccess;
}

// true if p was the arena's (then released after a device sync, its chunk
// too if empty)
inline bool qz_arena_free(void * p) {
    qz_arena & A = qz_arena_get();
    std::lock_guard<std::mutex> g(A.mu);
    auto it = A.owner.find(p);
    if (it == A.owner.end()) return false;
    (void) hipDeviceSynchronize();
    qz_arena::chunk * c = it->second;
    A.owner.erase(it);
    c->live.erase((size_t) ((char *) p - c->base));
    if (c->live.empty()) {
        for (auto & v : A.chunks)
            for (size_t i = 0; i < v.size(); i++)
                if (v[i] == c) { v.erase(v.begin() + i); break; }
        (void) hipFree(c->base);
        delete c;
    }
    return true;
}

inline const char *& qz_mem_tag() { static thread_local const char * t = nullptr; return t; }

// tags this thread's allocations until the end of the scope (nests)
struct qz_mem_scope {
    const char * prev;
    explicit qz_mem_scope(const char * tag) : prev(qz_mem_tag()) { qz_mem_tag() = tag; }
    ~qz_mem_scope() { qz_mem_tag() = prev; }
    qz_mem_scope(const qz_mem_scope &) = delete;
    qz_mem_scope & operator=(const qz_mem_scope &) = delete;
};

inline void qz_mem_add(void * p, size_t bytes, const char * sub) {
    if (!p) return;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= QZ_MEM_CARDS) dev = 0;
    std::string tag = qz_mem_tag() ? qz_mem_tag() : "untagged";
    if (sub && *sub) tag += std::string("/") + sub;
    qz_mem_ledger & L = qz_mem();
    std::lock_guard<std::mutex> g(L.mu);
    L.live[p] = { dev, bytes, tag };
    L.by[dev][tag] += bytes;
    L.tracked[dev] += bytes;
}

template <class T>
inline hipError_t qz_dmalloc(T ** p, size_t bytes, const char * sub = nullptr) {
    int dev = 0;
    hipError_t e;
    if (bytes && bytes < QZ_ARENA_BIG && hipGetDevice(&dev) == hipSuccess && dev >= 0 && dev < QZ_MEM_CARDS)
        e = qz_arena_alloc((void **) p, bytes, dev);
    else
        e = hipMalloc((void **) p, bytes);
    if (e == hipSuccess) qz_mem_add((void *) *p, bytes, sub);
    return e;
}

// hipDeviceMallocUncached pages (the cross-card inboxes: a peer's writes
// must land in HBM, not behind this card's L2)
template <class T>
inline hipError_t qz_dmalloc_uncached(T ** p, size_t bytes, const char * sub = nullptr) {
    const hipError_t e = hipExtMallocWithFlags((void **) p, bytes, hipDeviceMallocUncached);
    if (e == hipSuccess) qz_mem_add((void *) *p, bytes, sub);
    return e;
}

inline hipError_t qz_dfree(void * p) {
    if (!p) return hipSuccess;
    {
        qz_mem_ledger & L = qz_mem();
        std::lock_guard<std::mutex> g(L.mu);
        auto it = L.live.find(p);
        if (it != L.live.end()) {
            const qz_mem_ledger::rec & r = it->second;
            L.by[r.card][r.tag] -= r.bytes;
            if (!L.by[r.card][r.tag]) L.by[r.card].erase(r.tag);
            L.tracked[r.card] -= r.bytes;
            L.live.erase(it);
        }
    }
    if (qz_arena_free(p)) return hipSuccess;
    return hipFree(p);
}

// the cards' used memory before the engine allocates, so the report can tell
// the baseline from untracked growth
inline void qz_mem_baseline(int n_cards) {
    qz_mem_ledger & L = qz_mem();
    int dev0 = 0;
    (void) hipGetDevice(&dev0);
    for (int c = 0; c < n_cards && c < 2; c++) {
        size_t fr = 0, tot = 0;
        const int d = qz_dev_of(c);
        if (d >= QZ_MEM_CARDS) continue;
        if (hipSetDevice(d) == hipSuccess && hipMemGetInfo(&fr, &tot) == hipSuccess) L.base[d] = (long long) (tot - fr);
    }
    (void) hipSetDevice(dev0);
}

// one line per card: used / total, free, the four largest tag groups and the
// rest
inline void qz_mem_report(FILE * f, int n_cards, double min_gb = 0.05) {
    qz_mem_ledger & L = qz_mem();
    int dev0 = 0;
    (void) hipGetDevice(&dev0);
    for (int c = 0; c < n_cards && c < 2; c++) {
        size_t fr = 0, tot = 0;
        const int d = qz_dev_of(c);
        if (d >= QZ_MEM_CARDS) continue;
        if (hipSetDevice(d) != hipSuccess || hipMemGetInfo(&fr, &tot) != hipSuccess) continue;
        // tags grouped by their first word (target/weights + target/head -> target)
        std::map<std::string, size_t> grp;
        size_t tracked = 0;
        {
            std::lock_guard<std::mutex> g(L.mu);
            tracked = L.tracked[d];
            for (const auto & kv : L.by[d]) grp[kv.first.substr(0, kv.first.find_first_of("/ "))] += kv.second;
        }
        std::vector<std::pair<std::string, size_t>> v(grp.begin(), grp.end());
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
        const size_t used = tot - fr;
        size_t rest = used > tracked ? used - tracked : 0;   // the runtime's own share
        qz_stamp(f);
        if (d == c) fprintf(f, "card %d %.1f/%.1f GB (%.1f free):", c, used / 1e9, tot / 1e9, fr / 1e9);
        else fprintf(f, "card %d (HIP %d) %.1f/%.1f GB (%.1f free):", c, d, used / 1e9, tot / 1e9, fr / 1e9);
        for (size_t i = 0; i < v.size(); i++) {
            if (i < 4 && v[i].second / 1e9 >= min_gb) fprintf(f, " %s %.1f,", v[i].first.c_str(), v[i].second / 1e9);
            else rest += v[i].second;
        }
        fprintf(f, " rest %.1f\n", rest / 1e9);
    }
    (void) hipSetDevice(dev0);
}
