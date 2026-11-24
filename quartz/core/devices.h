// The two cards the engine runs on, as HIP device ordinals. qz_dev_select() runs once
// before any allocation: QZ_DEVICES=a,b when set, else devices 0 and 1. HIP device 0
// must be gfx906 whatever the pair is. This build has gfx906 code objects only and
// the runtime resolves kernels through device 0, so another architecture in slot 0 fails
// every launch ("invalid device function") even when the work runs elsewhere.
//
// QZ_LINK=auto|p2p|host says how the two cards reach each other, and is resolved here
// with them: peer access (P2P), or pinned host memory carrying the exchange (host), for
// a pair HIP cannot put peer access on. core/link.h allocates and copies for the mode.
// p2p keeps the old behaviour: no peer pair is refused.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

inline constexpr int qz_dev_max = 8;     // the ledger's table size (QZ_MEM_CARDS)
inline int qz_dev[2] = { 0, 1 };          // card index -> HIP ordinal
inline bool qz_dev_chosen = false;

enum qz_link_mode { QZ_LINK_P2P, QZ_LINK_HOST };
inline qz_link_mode qz_link = QZ_LINK_P2P;   // the mode the pair runs in; set below
inline bool qz_link_auto = false;             // QZ_LINK=auto: a failed peer check falls back to host

inline int qz_dev_of(int card) { return qz_dev[card]; }

inline bool qz_dev_gfx906(const hipDeviceProp_t & p) {
    return std::strstr(p.gcnArchName, "gfx906") != nullptr;
}

// capability only: nothing is enabled here
inline bool qz_dev_peer(int a, int b) {
    int x = 0, y = 0;
    return hipDeviceCanAccessPeer(&x, a, b) == hipSuccess && x &&
           hipDeviceCanAccessPeer(&y, b, a) == hipSuccess && y;
}

inline const char * qz_dev_env() {
    const char * v = getenv("HIP_VISIBLE_DEVICES");
    return v ? v : getenv("ROCR_VISIBLE_DEVICES");
}

inline void qz_dev_list(const std::vector<hipDeviceProp_t> & p) {
    for (size_t i = 0; i < p.size(); i++)
        fprintf(stderr, "  HIP %zu  %-28s %s\n", i, p[i].name, p[i].gcnArchName);
}

inline std::string qz_dev_why(const std::vector<hipDeviceProp_t> & p, int a, int b, bool need_peer = true) {
    char buf[256];
    if (!qz_dev_gfx906(p[(size_t) a])) snprintf(buf, sizeof buf, "HIP %d is %s, not gfx906", a, p[(size_t) a].gcnArchName);
    else if (!qz_dev_gfx906(p[(size_t) b])) snprintf(buf, sizeof buf, "HIP %d is %s, not gfx906", b, p[(size_t) b].gcnArchName);
    else if (need_peer && !qz_dev_peer(a, b)) snprintf(buf, sizeof buf, "no peer access between HIP %d and HIP %d", a, b);
    else snprintf(buf, sizeof buf, "HIP %d and HIP %d are not usable", a, b);
    return buf;
}

// QZ_LINK: 0 auto, 1 p2p, 2 host, -1 for anything else
inline int qz_dev_link_want() {
    const char * v = getenv("QZ_LINK");
    if (!v || !*v || !std::strcmp(v, "auto")) return 0;
    if (!std::strcmp(v, "p2p")) return 1;
    if (!std::strcmp(v, "host")) return 2;
    return -1;
}

// the pair a host link runs on: 0 and 1 when HIP 1 is gfx906 too, else the first gfx906
// card past 0 (HIP 0 is gfx906 by the time this is asked). False when there is none.
inline bool qz_dev_host_pair(const std::vector<hipDeviceProp_t> & p, int * a, int * b) {
    *a = 0; *b = -1;
    for (int j = 1; j < (int) p.size(); j++)
        if (qz_dev_gfx906(p[(size_t) j])) { *b = j; return true; }
    return false;
}

inline int qz_dev_select() {
    if (qz_dev_chosen) return 0;
    qz_dev_chosen = true;

    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) { fprintf(stderr, "quartz: cannot enumerate HIP devices\n"); return 1; }
    std::vector<hipDeviceProp_t> p((size_t) (n > 0 ? n : 1));
    for (int i = 0; i < n; i++)
        if (hipGetDeviceProperties(&p[(size_t) i], i) != hipSuccess) {
            fprintf(stderr, "quartz: cannot query HIP device %d\n", i);
            return 1;
        }

    if (n >= 1 && !qz_dev_gfx906(p[0])) {
        fprintf(stderr, "quartz: HIP device 0 is %s and this build has gfx906 code objects only;\n"
                        "        the runtime resolves kernels through device 0, so nothing can run.\n"
                        "        Put a gfx906 card first, or unset the visible set.\n", p[0].gcnArchName);
        qz_dev_list(p);
        return 1;
    }
    if (n < 2) {
        const char * env = qz_dev_env();
        if (env) fprintf(stderr, "quartz: the engine needs two GPUs, %d visible (visible set %s)\n", n, env);
        else fprintf(stderr, "quartz: the engine needs two GPUs, %d visible\n", n);
        qz_dev_list(p);
        return 1;
    }

    const int lw = qz_dev_link_want();
    if (lw < 0) {
        fprintf(stderr, "quartz: QZ_LINK=%s is not auto, p2p or host\n", getenv("QZ_LINK"));
        return 1;
    }
    const bool want_p2p = lw == 1, want_host = lw == 2;   // neither: auto, decided by what the cards give
    qz_link_auto = lw == 0;

    const char * spec = getenv("QZ_DEVICES");
    if (spec) {                                // explicit pair wins
        int a = -1, b = -1;
        const int hi = n < qz_dev_max ? n - 1 : qz_dev_max - 1;
        if (sscanf(spec, "%d,%d", &a, &b) != 2 || a < 0 || b < 0 || a > hi || b > hi || a == b) {
            fprintf(stderr, "quartz: QZ_DEVICES=%s is not a pair of distinct ordinals in 0..%d\n", spec, hi);
            qz_dev_list(p);
            return 1;
        }
        if (!qz_dev_gfx906(p[(size_t) a]) || !qz_dev_gfx906(p[(size_t) b])) {
            fprintf(stderr, "quartz: QZ_DEVICES=%s: %s\n", spec, qz_dev_why(p, a, b, !want_host).c_str());
            qz_dev_list(p);
            return 1;
        }
        // host mode never asks the runtime about peer access; p2p insists on it; auto
        // takes whatever the pair has
        bool peer = false;
        if (!want_host) {
            peer = qz_dev_peer(a, b);
            if (!peer && want_p2p) {
                fprintf(stderr, "quartz: QZ_DEVICES=%s: %s\n", spec, qz_dev_why(p, a, b).c_str());
                qz_dev_list(p);
                return 1;
            }
        }
        qz_dev[0] = a; qz_dev[1] = b;
        if (want_host || !peer) {
            qz_link = QZ_LINK_HOST;
            fprintf(stderr, "quartz: card 0 = HIP %d, card 1 = HIP %d (QZ_DEVICES=%s), %s: pinned host memory carries the exchange\n",
                    a, b, spec, want_host ? "QZ_LINK=host" : "no P2P");
        } else {
            fprintf(stderr, "quartz: card 0 = HIP %d, card 1 = HIP %d (QZ_DEVICES=%s)\n", a, b, spec);
        }
        return 0;
    }

    if (!want_host && qz_dev_gfx906(p[1]) && qz_dev_peer(0, 1)) return 0;

    if (want_p2p) {
        // 0 and 1 are not a usable pair: list the pairs that are and let the caller pick
        fprintf(stderr, "quartz: HIP 0 and HIP 1 are not a usable pair (%s); usable pairs:", qz_dev_why(p, 0, 1).c_str());
        int any = 0;
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
                if (qz_dev_gfx906(p[(size_t) a]) && qz_dev_gfx906(p[(size_t) b]) && qz_dev_peer(a, b)) {
                    fprintf(stderr, " %d,%d", a, b);
                    any++;
                }
        fprintf(stderr, any ? " - pick one with QZ_DEVICES=a,b; the pair must be gfx906 and HIP 0 must be too\n"
                            : " none\n");
        qz_dev_list(p);
        return 1;
    }

    // auto: another peer pair beats a host link (HIP 0 must stay gfx906, checked above)
    if (!want_host)
        for (int a = 0; a < n && a < qz_dev_max; a++)
            for (int b = a + 1; b < n && b < qz_dev_max; b++)
                if (qz_dev_gfx906(p[(size_t) a]) && qz_dev_gfx906(p[(size_t) b]) && qz_dev_peer(a, b)) {
                    qz_dev[0] = a; qz_dev[1] = b;
                    fprintf(stderr, "quartz: card 0 = HIP %d, card 1 = HIP %d (HIP 0 and HIP 1 have no peer access; set QZ_DEVICES to choose)\n", a, b);
                    return 0;
                }

    // no peer pair at all (or host mode, which does not look for one): two gfx906
    // cards, the exchange over pinned host memory
    int a = 0, b = -1;
    if (!qz_dev_host_pair(p, &a, &b)) {
        fprintf(stderr, "quartz: no second gfx906 card to link with HIP 0, %d visible\n", n);
        qz_dev_list(p);
        return 1;
    }
    qz_dev[0] = a; qz_dev[1] = b;
    qz_link = QZ_LINK_HOST;
    fprintf(stderr, "quartz: card 0 = HIP %d, card 1 = HIP %d, %s: pinned host memory carries the exchange\n",
            a, b, want_host ? "QZ_LINK=host" : "no P2P");
    return 0;
}
