// The cards the engine runs on, as HIP device ordinals. qz_dev_select() runs once before
// any allocation. QZ_TP=2|4 is the card count (2 by default; the arch code is built for
// one count, QZ_NC). The cards sit on QZ_TP / QZ_VIRT GPUs: QZ_DEVICES=a,b (or a,b,c,d)
// when set, else devices 0, 1, ... With QZ_VIRT=2 each GPU carries two virtual cards on
// CU-masked streams over alternate CUs (qz_card_stream), card c on GPU c % (that count);
// this emulates 4 cards on 2 GPUs. HIP device 0
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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

inline constexpr int qz_dev_max = 8;     // the ledger's table size (QZ_MEM_CARDS)
inline constexpr int QZ_NC_MAX = 4;      // the most cards a tensor-parallel group has (shared arrays' size)
inline int qz_dev[QZ_NC_MAX] = { 0, 1 }; // card index -> HIP ordinal
inline bool qz_dev_chosen = false;
inline int qz_ncard = 2;                 // cards in the group (QZ_TP)
inline int qz_virt = 1;                  // cards per GPU (QZ_VIRT)
inline int qz_nphys = 2;                 // GPUs: qz_ncard / qz_virt

enum qz_link_mode { QZ_LINK_P2P, QZ_LINK_HOST };
inline qz_link_mode qz_link = QZ_LINK_P2P;   // the mode the pair runs in; set below
inline bool qz_link_auto = false;             // QZ_LINK=auto: a failed peer check falls back to host

inline int qz_dev_of(int card) { return qz_dev[card]; }
// card c's place among the virtual cards on its GPU, 0 .. qz_virt - 1
inline int qz_card_slot(int card) { return card / qz_nphys; }
// two cards on one GPU: their exchange is local, no peer access between them
inline bool qz_card_same_gpu(int a, int b) { return qz_dev[a] == qz_dev[b]; }

// card c's CUs: all of its GPU's, or every qz_virt-th starting at its slot (an alternating
// mask; a contiguous half leaves one CU unusable and the grid barrier hangs). m holds 64
// bits. Returns the CU count, 0 on error. The current device need not be the card's.
inline int qz_card_mask(int card, uint32_t m[2]) {
    int n = 0;
    if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, qz_dev_of(card)) != hipSuccess || n <= 0) return 0;
    if (n > 64) n = 64;
    m[0] = m[1] = 0;
    int k = 0;
    for (int cu = 0; cu < n; cu++)
        if (cu % qz_virt == qz_card_slot(card)) { m[cu / 32] |= 1u << (cu % 32); k++; }
    return k;
}
// the CUs a card's persistent grid may use: one block per CU
inline int qz_card_ncu(int card) { uint32_t m[2]; return qz_card_mask(card, m); }
// a stream for card c's work, on the current device (the card's): non-blocking when the
// card has the whole GPU, else CU-masked to the card's CUs
inline hipError_t qz_card_stream(int card, hipStream_t * st) {
    if (qz_virt == 1) return hipStreamCreateWithFlags(st, hipStreamNonBlocking);
    uint32_t m[2];
    if (!qz_card_mask(card, m)) return hipErrorInvalidDevice;
    return hipExtStreamCreateWithCUMask(st, 2, m);
}

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

inline int qz_dev_select_gpus();
inline int qz_dev_select() {
    if (qz_dev_chosen) return 0;
    qz_dev_chosen = true;
    const char * tp = getenv("QZ_TP"), * vi = getenv("QZ_VIRT");
    qz_ncard = tp && *tp ? atoi(tp) : 2;
    qz_virt = vi && *vi ? atoi(vi) : 1;
    if ((qz_ncard != 2 && qz_ncard != 4) || (qz_virt != 1 && qz_virt != 2) || qz_ncard / qz_virt < 2) {
        fprintf(stderr, "quartz: QZ_TP=%s QZ_VIRT=%s: the cards are 2 or 4, on 2 or 4 GPUs (QZ_VIRT 1 or 2)\n",
                tp ? tp : "2", vi ? vi : "1");
        return 1;
    }
    qz_nphys = qz_ncard / qz_virt;
    // two queues per stream would be shared otherwise: each virtual card's persistent kernel
    // needs its own hardware queue (set before HIP starts; an explicit value is kept)
    if (qz_virt > 1) setenv("GPU_MAX_HW_QUEUES", "8", 0);
    const int rc = qz_dev_select_gpus();
    if (rc) return rc;
    if (qz_virt > 1 && qz_link != QZ_LINK_P2P) {
        fprintf(stderr, "quartz: QZ_VIRT=%d needs peer access between the GPUs (the host link is two cards only)\n", qz_virt);
        return 1;
    }
    int ph[QZ_NC_MAX];
    for (int g = 0; g < qz_nphys; g++) ph[g] = qz_dev[g];
    for (int c = 0; c < qz_ncard; c++) qz_dev[c] = ph[c % qz_nphys];
    if (qz_ncard > 2 || qz_virt > 1) {
        fprintf(stderr, "quartz: %d cards on %d GPUs:", qz_ncard, qz_nphys);
        for (int c = 0; c < qz_ncard; c++) fprintf(stderr, " card %d = HIP %d%s", c, qz_dev[c], qz_virt > 1 ? (qz_card_slot(c) ? " (odd CUs)" : " (even CUs)") : "");
        fprintf(stderr, "\n");
    }
    return 0;
}

// four GPUs: QZ_DEVICES=a,b,c,d or 0,1,2,3, all gfx906, peer access between every pair
inline int qz_dev_select4(const std::vector<hipDeviceProp_t> & p, int n) {
    int d[4] = { 0, 1, 2, 3 };
    const char * spec = getenv("QZ_DEVICES");
    const int hi = n < qz_dev_max ? n - 1 : qz_dev_max - 1;
    if (spec && sscanf(spec, "%d,%d,%d,%d", &d[0], &d[1], &d[2], &d[3]) != 4) {
        fprintf(stderr, "quartz: QZ_DEVICES=%s: four cards need four ordinals\n", spec);
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        if (d[i] < 0 || d[i] > hi) { fprintf(stderr, "quartz: HIP %d is not visible (0..%d)\n", d[i], hi); qz_dev_list(p); return 1; }
        if (!qz_dev_gfx906(p[(size_t) d[i]])) { fprintf(stderr, "quartz: HIP %d is %s, not gfx906\n", d[i], p[(size_t) d[i]].gcnArchName); return 1; }
        for (int j = 0; j < i; j++) {
            if (d[i] == d[j]) { fprintf(stderr, "quartz: QZ_DEVICES lists HIP %d twice\n", d[i]); return 1; }
            if (!qz_dev_peer(d[i], d[j])) {
                fprintf(stderr, "quartz: no peer access between HIP %d and HIP %d; four cards need it on every pair\n", d[j], d[i]);
                qz_dev_list(p);
                return 1;
            }
        }
    }
    if (qz_dev_link_want() == 2) { fprintf(stderr, "quartz: QZ_LINK=host is two cards only\n"); return 1; }
    for (int i = 0; i < 4; i++) qz_dev[i] = d[i];
    qz_link = QZ_LINK_P2P;
    return 0;
}

// the GPUs (qz_nphys of them) into qz_dev[0 ..], with the link mode
inline int qz_dev_select_gpus() {

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
    if (n >= qz_nphys && qz_nphys == 4) return qz_dev_select4(p, n);
    if (n < qz_nphys) {
        const char * env = qz_dev_env();
        if (env) fprintf(stderr, "quartz: the engine needs %d GPUs, %d visible (visible set %s)\n", qz_nphys, n, env);
        else fprintf(stderr, "quartz: the engine needs %d GPUs, %d visible\n", qz_nphys, n);
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
