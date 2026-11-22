#include "dspine_block.h"
#include "devices.h"
#include "engine.h"
#include "prefill.h"
// the verify at two widths (8: the bit-exact production rows; 12: copy-heavy
// rounds), one TU each in its own namespace over shared 12-row replay rings
#undef D
#undef FF
#undef G
#undef CCH
#undef HV
#undef S
#undef AQ
#undef AKV
#undef HQ
#undef HKV
#undef HD
#undef BLKPOS
#include <initializer_list>
#include <type_traits>
#include "flash9.h"
namespace mv8 {
#define MV_T 8
#define MV_TR 12
#include "dspine_verify.h"
#undef MV_T
#undef MV_TR
}
namespace mv12 {
#define MV_T 12
#define MV_TR 12
#include "dspine_verify.h"
#undef MV_T
#undef MV_TR
}
#include "dspine_run.h"
#include "capi.h"
#include "preflight.h"
#include "vision.h"

#include <algorithm>
#include <cstring>
#include <deque>

constexpr int32_t QZ_IM_START = 248045;   
constexpr int32_t QZ_IMG_PAD = 248056;
struct qz_snap { int pos = -1; unsigned long use = 0; float * dn[2] = {}, * cs[2] = {}; };
struct qz_ctx {
    qz_engine eng;
    qz_prefill pf;
    bool pf_ok = false;
    static constexpr int pf_min = 16;   // shorter prompts decode on the cycle's forced rows
    static constexpr int tail = 64;
    int head = -1;                
    std::vector<qz_snap> snaps;
    float * sdn[2] = {}, * scs[2] = {};   
    std::vector<uint64_t> ckeys;
    unsigned long clock = 0;
    int last_cached = 0;
    struct img { uint64_t hash = 0; int id = 0, nx = 0, ny = 0; unsigned long use = 0; std::vector<float> rows; };
    qz_vision * vis = nullptr;
    std::vector<img> imgs;
    int img_next = 0;
    float * img_dev[2] = {};
    size_t img_cap = 0;

    qz_dsrun ds;
    // drafter switch (on with DSpine unless QZ_NO_SWITCH): DSpine <-> MTP mid-answer on the rounds' acceptance
    struct {
        bool on = false;
        float lo = 3.8f, hi = 3.4f;       // leave DSpine below lo tok/round, MTP at hi and above
        int win = 128, flush = 512;       // window tokens; MTP feature rows per DSpine ring fill
        int cap = 1024;                   // staged MTP entries per flush
        float * hn[2] = {};               // staged entry inputs [cap + 12][QZ_D] per card
        std::vector<int32_t> tok;         // staged entry j: position p0 + j
        int p0 = 0;
    } sw;
};

static size_t qz_dn_bytes(const qz_engine & e) { return (size_t) e.hp.n_recr * g2_HV * QZ_S * QZ_S * 4; }
static size_t qz_cs_bytes(const qz_engine & e) { return (size_t) e.hp.n_recr * 3 * g2_CCH * 4; }
static void qz_snap_alloc(qz_ctx * c, qz_snap & sn) {
    for (int i = 0; i < 2; i++)
        if (!sn.dn[i]) {
            CHECK(hipHostMalloc((void **) &sn.dn[i], qz_dn_bytes(c->eng), hipHostMallocPortable));
            CHECK(hipHostMalloc((void **) &sn.cs[i], qz_cs_bytes(c->eng), hipHostMallocPortable));
        }
}


static qz_snap * qz_snap_victim(qz_ctx * c, const qz_snap * keep) {
    std::vector<qz_snap *> live;
    for (auto & sn : c->snaps) if (sn.pos >= 0) live.push_back(&sn);
    std::sort(live.begin(), live.end(), [](const qz_snap * x, const qz_snap * y) { return x->pos < y->pos; });
    qz_snap * best = nullptr;
    double best_cost = 0;
    const int n = (int) live.size();
    for (int i = 0; i + 2 < n; i++) {
        qz_snap * sn = live[i];
        if (sn == keep || sn->pos == c->head) continue;
        const int prev = i > 0 ? live[i - 1]->pos : 0, next = live[i + 1]->pos;
        const double cost = (double) (next - prev) / next;
        if (!best || cost < best_cost) { best = sn; best_cost = cost; }
    }
    if (!best) for (auto & sn : c->snaps) if (&sn != keep && (!best || sn.use < best->use)) best = &sn;
    return best;
}


static qz_snap * qz_snap_slot(qz_ctx * c, int pos, const qz_snap * keep = nullptr) {
    qz_snap * best = nullptr;
    for (auto & sn : c->snaps) if (sn.pos == pos) best = &sn;          // same position: refresh in place
    if (!best) for (auto & sn : c->snaps) if (sn.pos < 0) { best = &sn; break; }
    if (!best) best = qz_snap_victim(c, keep);
    qz_snap_alloc(c, *best);
    best->pos = -1;
    return best;
}

static void qz_snap_take(qz_ctx * c, int pos, const qz_snap * keep = nullptr) {
    qz_snap * best = qz_snap_slot(c, pos, keep);
    for (int i = 0; i < 2; i++) {
        qz_card & K = c->eng.C[i];
        CHECK(hipSetDevice(qz_dev_of(i)));
        CHECK(hipMemcpyAsync(best->dn[i], K.dn_all, qz_dn_bytes(c->eng), hipMemcpyDeviceToHost, K.st));
        CHECK(hipMemcpyAsync(best->cs[i], K.cs_all, qz_cs_bytes(c->eng), hipMemcpyDeviceToHost, K.st));
    }
    CHECK(hipSetDevice(qz_dev_of(0)));
    best->pos = pos;
    best->use = ++c->clock;
}

static void qz_snap_restore(qz_ctx * c, qz_snap & sn) {
    for (int i = 0; i < 2; i++) {
        qz_card & K = c->eng.C[i];
        CHECK(hipSetDevice(qz_dev_of(i)));
        CHECK(hipMemcpyAsync(K.dn_all, sn.dn[i], qz_dn_bytes(c->eng), hipMemcpyHostToDevice, K.st));
        CHECK(hipMemcpyAsync(K.cs_all, sn.cs[i], qz_cs_bytes(c->eng), hipMemcpyHostToDevice, K.st));
        CHECK(hipMemsetAsync(K.ring_all, 0, (size_t) c->eng.hp.n_recr * 2 * g2_RING_PAR * 4, K.st));
        CHECK(hipMemsetAsync(K.ctl_d, 0, 8 * sizeof(int), K.st));
    }
    c->eng.sync2();
    sn.use = ++c->clock;
}

static void qz_cache_drop(qz_ctx * c) {   // the engine state no longer matches any snapshot's K / V
    for (auto & sn : c->snaps) sn.pos = -1;
    c->ckeys.clear();
    c->head = -1;
}

static void qz_pf_setup(qz_ctx * c) {
    // + headroom; the buffers themselves come from the engine's scratch when it holds them
    const size_t need = (c->eng.C[0].scr_bytes >= qz_prefill::bytes(2048) ? 0 : qz_prefill::bytes(2048)) + ((size_t) 256 << 20);
    for (int i = 0; i < 2; i++) {
        size_t fr = 0, tot = 0;
        CHECK(hipSetDevice(qz_dev_of(i)));
        CHECK(hipMemGetInfo(&fr, &tot));
        if (fr < need) {
            fprintf(stderr, "quartz: batched prefill off: card %d has %zu MiB free, needs %zu\n", i, fr >> 20, need >> 20);
            CHECK(hipSetDevice(qz_dev_of(0)));
            return;
        }
    }
    CHECK(hipSetDevice(qz_dev_of(0)));
    c->pf_ok = c->pf.init(c->eng, 2048) == 0;
    // one slot cannot hold a chunk's mid-chunk snapshot and its end one at once
    c->snaps.resize(8);
    // the snapshots up front in pinned host memory (~75 MB per slot per card),
    // the mid-chunk staging on each card
    qz_mem_scope mem("snapshot staging");
    if (c->pf_ok) {
        for (auto & sn : c->snaps) qz_snap_alloc(c, sn);
        for (int i = 0; i < 2; i++) {
            CHECK(hipSetDevice(qz_dev_of(i)));
            CHECK(qz_dmalloc(&c->sdn[i], qz_dn_bytes(c->eng))); CHECK(qz_dmalloc(&c->scs[i], qz_cs_bytes(c->eng)));
        }
    }
    CHECK(hipSetDevice(qz_dev_of(0)));
}

// a prompt position's prefix-cache key: its token, or for an image row the
// image's content hash and the row (the image tokens all share one id)
static std::vector<uint64_t> qz_keys(const std::vector<int32_t> & p, const qz_mm_view * mm, const std::vector<uint64_t> * row_hash) {
    std::vector<uint64_t> k(p.size());
    for (size_t q = 0; q < p.size(); q++) {
        const int r = mm ? mm->img((int) q) : -1;
        k[q] = r < 0 ? (uint64_t) (uint32_t) p[q] : (*row_hash)[r];
    }
    return k;
}

// per-round tokens over the last `win` tokens of a segment
struct qz_win {
    std::deque<int> r;
    int sum = 0, seg = 0;
    void add(int n, int win) { r.push_back(n); sum += n; seg += n; while (sum - r.front() >= win) { sum -= r.front(); r.pop_front(); } }
    float avg() const { return (float) sum / (float) r.size(); }
};

// decode from pos (anchor at pos) switching drafters: DSpine segments
// stage MTP entries, MTP segments capture DSpine features; each switch
// recomputes the uncommitted rows on the new path
static int qz_switch_run(qz_ctx * c, int pos, int32_t anchor, int max_new, const std::vector<int32_t> & s,
                         qz_emit_cb cb, void * user, qz_gen_stats * st) {
    auto & w = c->sw;
    qz_engine & e = c->eng; qz_dsrun & ds = c->ds; qz_prefill & pf = c->pf;
    using clk = std::chrono::steady_clock;
    const int fcap = pf.Mmax;
    qz_gen_stats tot, g;
    tot.dspine = 1;
    int left = max_new, rc = 0, pc = pos, fl = pos, nsw = 0;
    int seg_tok[2] = {}, seg_rnd[2] = {};   // [0] DSpine, [1] MTP
    double cu_ms[2] = {};   // catch-up into [0] DSpine, [1] MTP
    bool on_ds = true;
    std::vector<int32_t> pend;
    qz_win win;
    w.tok.clear();
    auto stage_flush = [&]() {
        if (w.tok.empty()) return;
        pf.mtp_rows(e, w.tok.data(), w.hn, w.p0, (int) w.tok.size());
        w.p0 += (int) w.tok.size();
        w.tok.clear();
    };
    auto feat_fill = [&](int a, int b) {   // cycle-captured features of positions a .. b-1
        while (a < b) {
            const int r0 = a % fcap, n = std::min(b - a, fcap - r0);
            ds.fill_prefill(pf.feat + (size_t) r0 * QZ_D, fcap, a, n);
            a += n;
        }
    };
    ds.sw_hook = [&](int p, int n, int W, const int32_t * toks, const int * out) {
        if (w.tok.empty()) w.p0 = p + 1;
        for (int i = 0; i < 2; i++) {   // entries p+1 .. p+n: the token there, row j's normed hidden
            CHECK(hipSetDevice(qz_dev_of(i)));
            const float * h = W > 8 ? ds.mw[i].Ah.h_nextn : ds.mv[i].Ah.h_nextn;
            CHECK(hipMemcpy(w.hn[i] + w.tok.size() * QZ_D, h, (size_t) n * QZ_D * 4, hipMemcpyDeviceToDevice));
        }
        CHECK(hipSetDevice(qz_dev_of(0)));
        for (int j = 0; j < n; j++) w.tok.push_back(j + 1 < n ? toks[j + 1] : out[n - 1]);
        if ((int) w.tok.size() > w.cap - 12) stage_flush();
        win.add(n, w.win);
        seg_tok[0] += n; seg_rnd[0]++;
        return win.seg >= w.win && win.avg() < w.lo;
    };
    e.sw_hook = [&](int p, int n) {
        if (p - fl >= w.flush) { feat_fill(fl, p); fl = p; }
        win.add(n, w.win);
        seg_tok[1] += n; seg_rnd[1]++;
        return win.seg >= w.win && win.avg() >= w.hi;
    };
    e.C[0].A.fcap = pf.feat;
    e.C[0].A.fcap_cap = fcap;
    for (;;) {
        win = qz_win();
        int r;
        if (on_ds) {
            r = ds.generate(e, pc, anchor, left, s, (qz_emit_fn) cb, user, &g);
        } else {
            for (int i = 0; i < 2; i++) { CHECK(hipSetDevice(qz_dev_of(i))); CHECK(hipMemset(e.C[i].ctl_d, 0, 8 * sizeof(int))); }
            CHECK(hipSetDevice(qz_dev_of(0)));
            r = e.generate(pend, left, s, (qz_emit_fn) cb, user, &g, pc);   // <= 3 rows: one forced chunk, as a request's first
        }
        tot.n_gen += g.n_gen; tot.decode_ms += g.decode_ms + (on_ds ? 0 : g.prefill_ms);
        tot.decode_cycles += g.decode_cycles; tot.decode_tok += g.decode_tok; tot.wide_cycles += g.wide_cycles;
        left -= g.n_gen;
        if (r < 0 || g.stop != 4) { tot.stop = g.stop; rc = r; break; }
        const auto t = clk::now();
        if (on_ds) {   // -> MTP: entries up to the uncommitted rows, which are redone
            pend = ds.sw_pend; pc = ds.sw_pc;
            stage_flush();
            fl = pc;
        } else {       // -> DSpine: its context up to the anchor, then the uncommitted rows forced
            pend = e.sw_pend; pc = e.sw_pc;
            feat_fill(fl, pc + (int) pend.size() - 1);
            ds.forced = pend;
            anchor = pend[0];
            w.tok.clear();
        }
        cu_ms[on_ds] += std::chrono::duration<double, std::milli>(clk::now() - t).count();
        on_ds = !on_ds;
        nsw++;
    }
    ds.forced.clear();
    ds.sw_hook = nullptr; e.sw_hook = nullptr;
    e.C[0].A.fcap = nullptr;
    qz_stamp(stderr);
    fprintf(stderr, "  switch: %d, catch-up %.1f ms to dspine, %.1f ms to mtp | dspine %d tok %.2f/rnd | mtp %d tok %.2f/rnd\n", nsw, cu_ms[0], cu_ms[1],
            seg_tok[0], seg_rnd[0] ? (double) seg_tok[0] / seg_rnd[0] : 0.0, seg_tok[1], seg_rnd[1] ? (double) seg_tok[1] / seg_rnd[1] : 0.0);
    if (st) *st = tot;
    return rc;
}

// prompt through the batched prefill + decode, or the engine's own path; rc as
// generate. mm / row_hash: the prompt's images (null: text only)
static int qz_run(qz_ctx * c, const std::vector<int32_t> & p, int max_new, const std::vector<int32_t> & s,
                  qz_emit_cb cb, void * user, qz_gen_stats * st,
                  const qz_mm_view * mm = nullptr, const std::vector<uint64_t> * row_hash = nullptr) {
    const int P = (int) p.size();
    c->last_cached = 0;
    c->eng.set_rope_shift(mm ? mm->shift : 0);
    if (c->ds.on) c->ds.set_rope_shift(mm ? mm->shift : 0);
    if (!c->pf_ok || (P < c->pf_min && !mm) || P + max_new + 2 * QZ_TMAX > c->eng.steps) {
        qz_cache_drop(c);   // the engine's own path resets the arena
        if (mm) { if (st) st->stop = -1; return -1; }   // image rows need the batched prefill
        return c->eng.generate(p, max_new, s, (qz_emit_fn) cb, user, st);
    }
    const auto t0 = std::chrono::steady_clock::now();
    // prompts with images decode on MTP (DSpine is exact there too, but it
    // accepts too little on image descriptions to beat MTP)
    const bool ds_on = c->ds.on && !mm;
    // the deepest snapshot whose prompt keys 0 .. pos match (pos <= P - 1)
    const std::vector<uint64_t> keys = qz_keys(p, mm, row_hash);
    int L = 0;
    while (L < (int) c->ckeys.size() && L < P && c->ckeys[L] == keys[L]) L++;
    qz_snap * use = nullptr;
    for (auto & sn : c->snaps) {
        if (sn.pos < 0) continue;
        if (sn.pos + 1 > L) { sn.pos = -1; continue; }   // its tokens differ from this prompt's
        if (sn.pos <= P - 1 && (!use || sn.pos > use->pos)) use = &sn;
    }
    int start = 0;
    if (use) { qz_snap_restore(c, *use); start = use->pos; }
    else { c->eng.reset_state(); for (auto & sn : c->snaps) sn.pos = -1; }
    c->ckeys = keys;
    // the first message's end: the second <|im_start|> (a snapshot there is
    // pinned while prompts keep matching it)
    c->head = -1;
    for (int q = 1; q < P - 1; q++)
        if (p[q] == QZ_IM_START) { c->head = q; break; }
    // chunks up to P - 2, a snapshot at each chunk end, at the first message's
    // end and mid-chunk at the last <|im_start|> (else c->tail before the end)
    const int chunk = 2048;
    int cut = P - 1 - c->tail;
    for (int q = P - 2; q > start; q--)
        if (p[q] == QZ_IM_START) { cut = q; break; }
    // progress per chunk on stderr when the prefill spans more than one chunk
    const bool pf_prog = P - 1 - start > chunk;
    auto tc = t0;
    for (int a = start; a < P - 1;) {
        int b = std::min(a + chunk, P - 1);
        if (c->head > a && c->head < b) b = c->head;
        qz_snap * mid = cut > a && cut < b ? qz_snap_slot(c, cut, use) : nullptr;
        float * md[2] = { mid ? c->sdn[0] : nullptr, mid ? c->sdn[1] : nullptr }, * mc[2] = { mid ? c->scs[0] : nullptr, mid ? c->scs[1] : nullptr };
        if (c->pf.run(c->eng, p.data() + a, a, b - a, p[b], mid ? cut - a : -1, md, mc, mm)) { qz_cache_drop(c); if (st) st->stop = -1; return -1; }
        if (ds_on) { c->eng.sync2(); c->ds.fill_prefill(c->pf.feat, c->pf.Mmax, a, b - a); }
        if (mid) {   // the staged mid-chunk state to its host slot (stream order: before the next chunk rewrites it)
            for (int i = 0; i < 2; i++) {
                CHECK(hipSetDevice(qz_dev_of(i)));
                CHECK(hipMemcpyAsync(mid->dn[i], c->sdn[i], qz_dn_bytes(c->eng), hipMemcpyDeviceToHost, c->eng.C[i].st));
                CHECK(hipMemcpyAsync(mid->cs[i], c->scs[i], qz_cs_bytes(c->eng), hipMemcpyDeviceToHost, c->eng.C[i].st));
            }
            CHECK(hipSetDevice(qz_dev_of(0)));
            mid->pos = cut; mid->use = ++c->clock;
        }
        qz_snap_take(c, b, mid);
        if (pf_prog) {
            c->eng.sync2();
            const auto t = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t - tc).count();
            const double tot = std::chrono::duration<double, std::milli>(t - t0).count();
            qz_stamp(stderr);
            fprintf(stderr, "  prefill %d/%d %.0f t/s %.1f s\n",
                    b, P - 1, 1000.0 * (b - a) / ms, tot / 1000.0);
            tc = t;
        }
        a = b;
    }
    c->last_cached = start;
    const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // greedy and sampled requests alike on DSpine (sampled: drafts from the
    // drafter's calibrated q, exact accept / residual)
    const int rc = ds_on && c->sw.on
        ? qz_switch_run(c, P - 1, p[P - 1], max_new, s, cb, user, st)
        : ds_on
        ? c->ds.generate(c->eng, P - 1, p[P - 1], max_new, s, (qz_emit_fn) cb, user, st)
        : c->eng.generate(std::vector<int32_t>{ p[P - 1] }, max_new, s, (qz_emit_fn) cb, user, st, P - 1);
    if (st) { st->prefill_ms += pf_ms; st->n_prompt = P; }
    return rc;
}

extern "C" {

qz_ctx * qz_create(const char * model, int max_ctx, int slot_t, int adapt_slack) {
    if (!model || max_ctx < 1024 || (slot_t != 0 && slot_t != 3 && slot_t != 4) || adapt_slack < 0) return nullptr;
    if (qz_dev_select()) return nullptr;
    qz_mem_scope mem("engine");
    qz_mem_baseline(2);
    // steps = arena positions: prompt + generated + the cycle / MTP overhang
    const int steps = max_ctx + 2 * QZ_TMAX;
    // the scratch the cycle's partials share with the batched prefill and the DSpine verify's partials
    const size_t scratch = std::max(qz_prefill::bytes(2048), mv12::mv_partial_bytes());
    // before the weights: memory for this context on both cards, then the peer link
    if (qz_vram_check(model, max_ctx, steps, scratch, getenv("QZ_DSPINE"), getenv("QZ_MMPROJ")) || qz_p2p_check())
        return nullptr;
    qz_ctx * c = new qz_ctx();
    c->eng.scratch_min = scratch;
    if (c->eng.init(model, steps, slot_t, adapt_slack)) {
        delete c;      // device buffers of a failed init are not released (one engine per process)
        return nullptr;
    }
    qz_pf_setup(c);
    if (const char * dsp = getenv("QZ_DSPINE")) {
        if (!c->pf_ok) fprintf(stderr, "quartz: DSpine off: it needs the batched prefill\n");
        else if (c->pf.feat_init() || c->ds.init(c->eng, dsp)) {
            fprintf(stderr, "quartz: DSpine init failed (%s)\n", dsp);
            delete c;
            return nullptr;
        }
    }
    const char * no_sw = getenv("QZ_NO_SWITCH");
    if (c->ds.on && !(no_sw && *no_sw && strcmp(no_sw, "0") != 0)) {
        auto & w = c->sw;
        if (const char * v = getenv("QZ_SW_LO")) w.lo = (float) atof(v);
        if (const char * v = getenv("QZ_SW_HI")) w.hi = (float) atof(v);
        if (const char * v = getenv("QZ_SW_WIN")) w.win = atoi(v);
        qz_mem_scope mem("drafter switch");
        for (int i = 0; i < 2; i++) { CHECK(hipSetDevice(qz_dev_of(i))); CHECK(qz_dmalloc(&w.hn[i], (size_t) (w.cap + 12) * QZ_D * 4)); }
        CHECK(hipSetDevice(qz_dev_of(0)));
        w.on = true;
        fprintf(stderr, "quartz: drafter switch on: leave DSpine < %.2f, MTP >= %.2f tok/round, window %d\n", w.lo, w.hi, w.win);
    }
    // per card: used / free, the ledger's tags, what the runtime holds beyond them
    qz_mem_report(stderr, 2);
    return c;
}

// qz_sampling_c -> the engine's spec; false if the chain is not one the engine samples exactly
static bool qz_sampling_from_c(const qz_sampling_c * smp, qz_sampling & ms) {
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

int qz_max_ctx(const qz_ctx * c) { return c ? c->eng.steps - 2 * QZ_TMAX : 0; }

static void qz_stats_out(const qz_ctx * c, const qz_gen_stats & st, qz_stats * out) {
    if (!out) return;
    out->prefill_ms = st.prefill_ms;
    out->decode_ms = st.decode_ms;
    out->n_prompt = st.n_prompt;
    out->n_gen = st.n_gen;
    out->prefill_cycles = st.prefill_cycles;
    out->decode_cycles = st.decode_cycles;
    out->decode_tok = st.decode_tok;
    out->stop = st.stop;
    out->n_cached = c->last_cached;
    out->dspine = st.dspine;
    out->wide_cycles = st.wide_cycles;
}

int qz_generate(qz_ctx * c, const int32_t * prompt, int n_prompt, int max_new,
                const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out) {
    return qz_generate_mm(c, prompt, n_prompt, nullptr, 0, max_new, stop, n_stop, cb, user, out, nullptr);
}

// qz_generate with sampled decode (smp = NULL: greedy, as qz_generate)
int qz_generate_s(qz_ctx * c, const int32_t * prompt, int n_prompt, int max_new,
                  const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out,
                  const qz_sampling_c * smp) {
    return qz_generate_mm(c, prompt, n_prompt, nullptr, 0, max_new, stop, n_stop, cb, user, out, smp);
}

int qz_vision_attach(qz_ctx * c, const char * mmproj) {
    if (!c || !mmproj || c->vis) return -1;
    if (!c->pf_ok) { fprintf(stderr, "quartz: images need the batched prefill\n"); return -1; }
    // both cards, half the encoder each
    c->vis = qz_vision_load(mmproj);
    if (!c->vis) return -1;
    if (qz_vision_n_embd(c->vis) != QZ_D) {
        fprintf(stderr, "quartz: projector width %d != model width %d\n", qz_vision_n_embd(c->vis), QZ_D);
        qz_vision_free(c->vis); c->vis = nullptr; return -1;
    }
    return 0;
}

int qz_image_pad_token(void) { return QZ_IMG_PAD; }

static uint64_t qz_fnv(const void * b, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= ((const unsigned char *) b)[i]; h *= 1099511628211ull; }
    return h;
}

int qz_image_add(qz_ctx * c, const void * bytes, size_t n, int * nx, int * ny) {
    if (!c || !c->vis || !bytes || !n) return -1;
    const uint64_t h = qz_fnv(bytes, n) ^ (uint64_t) n;
    for (auto & im : c->imgs)
        if (im.hash == h) {
            im.use = ++c->clock;
            if (nx) *nx = im.nx;
            if (ny) *ny = im.ny;
            return im.id;
        }
    int gx = 0, gy = 0;
    const int nt = qz_vision_encode(c->vis, bytes, n, &gx, &gy);
    if (nt <= 0) return -1;
    const int cap = 8;
    if ((int) c->imgs.size() >= cap) {
        auto lru = std::min_element(c->imgs.begin(), c->imgs.end(),
                                    [](const qz_ctx::img & a, const qz_ctx::img & b) { return a.use < b.use; });
        c->imgs.erase(lru);
    }
    qz_ctx::img im;
    im.hash = h; im.id = c->img_next++; im.nx = gx; im.ny = gy; im.use = ++c->clock;
    im.rows.resize((size_t) nt * QZ_D);
    if (qz_vision_copy_rows(c->vis, im.rows.data()) != nt) return -1;
    c->imgs.push_back(std::move(im));
    if (nx) *nx = gx;
    if (ny) *ny = gy;
    return c->imgs.back().id;
}

double qz_image_last_ms(const qz_ctx * c) { return c && c->vis ? qz_vision_last_ms(c->vis) : 0.0; }

int qz_generate_mm(qz_ctx * c, const int32_t * prompt, int n_prompt, const int * images, int n_images,
                   int max_new, const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out,
                   const qz_sampling_c * smp) {
    if (!c || !prompt || n_prompt < 1 || n_images < 0 || (n_images && !images)) return -1;
    qz_sampling ms;
    if (smp && !qz_sampling_from_c(smp, ms)) return -1;
    std::vector<int32_t> p(prompt, prompt + n_prompt);
    std::vector<int32_t> s;
    if (stop && n_stop > 0) s.assign(stop, stop + n_stop);
    const int P = n_prompt;

    // the image rows: each run of <|image_pad|> takes the next image, nx * ny long
    qz_mm_view mm;
    std::vector<int> irow, rpos;
    std::vector<uint64_t> row_hash;
    bool has_img = false;
    for (int q = 0; q < P; q++) has_img |= p[q] == QZ_IMG_PAD;
    if (has_img || n_images) {
        if (p[P - 1] == QZ_IMG_PAD) return -1;
        std::vector<const qz_ctx::img *> im(n_images);
        size_t R = 0;
        for (int k = 0; k < n_images; k++) {
            for (auto & e : c->imgs) if (e.id == images[k]) im[k] = &e;
            if (!im[k]) { fprintf(stderr, "quartz: image %d not in the cache\n", images[k]); return -1; }
            R += (size_t) im[k]->nx * im[k]->ny;
        }
        irow.assign(P, -1);
        rpos.resize((size_t) 3 * P);
        row_hash.resize(R);
        int r = 0, k = 0, row = 0;   // rope position, image, row
        for (int q = 0; q < P;) {
            if (p[q] != QZ_IMG_PAD) { rpos[3 * q] = rpos[3 * q + 1] = rpos[3 * q + 2] = r++; q++; continue; }
            int e = q;
            while (e < P && p[e] == QZ_IMG_PAD) e++;
            if (k >= n_images || e - q != im[k]->nx * im[k]->ny) {
                fprintf(stderr, "quartz: image run of %d tokens at %d does not match image %d\n", e - q, q, k);
                return -1;
            }
            const int nx = im[k]->nx, ny = im[k]->ny;
            for (int i = 0; i < e - q; i++) {
                irow[q + i] = row + i;
                rpos[3 * (q + i)] = r; rpos[3 * (q + i) + 1] = r + i / nx; rpos[3 * (q + i) + 2] = r + i % nx;
                uint64_t x = im[k]->hash + 0x9e3779b97f4a7c15ull * (uint64_t) (i + 1);
                x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29;
                row_hash[row + i] = x | (1ull << 63);   // never a token id
            }
            r += std::max(nx, ny);
            row += e - q;
            q = e;
            k++;
        }
        if (k != n_images) { fprintf(stderr, "quartz: %d images for %d image runs\n", n_images, k); return -1; }
        if (R > c->img_cap) {
            for (int i = 0; i < 2; i++) {
                CHECK(hipSetDevice(qz_dev_of(i)));
                if (c->img_dev[i]) (void) qz_dfree(c->img_dev[i]);
                c->img_dev[i] = nullptr;
                qz_mem_scope mem("images");
                if (qz_dmalloc(&c->img_dev[i], R * QZ_D * sizeof(float)) != hipSuccess) { c->img_cap = 0; return -1; }
            }
            c->img_cap = R;
        }
        for (int i = 0; i < 2; i++) {
            CHECK(hipSetDevice(qz_dev_of(i)));
            size_t off = 0;
            for (int j = 0; j < n_images; j++) {
                const size_t b = (size_t) im[j]->nx * im[j]->ny * QZ_D;
                CHECK(hipMemcpy(c->img_dev[i] + off, im[j]->rows.data(), b * sizeof(float), hipMemcpyHostToDevice));
                off += b;
            }
        }
        mm.irow = irow.data(); mm.rpos = rpos.data(); mm.n_pos = P; mm.shift = r - P;
        mm.rows[0] = c->img_dev[0]; mm.rows[1] = c->img_dev[1];
    }

    c->eng.set_sampling(smp ? &ms : nullptr);
    qz_gen_stats st;
    const int rc = qz_run(c, p, max_new, s, cb, user, &st, mm.n_pos ? &mm : nullptr, mm.n_pos ? &row_hash : nullptr);
    c->eng.set_sampling(nullptr);
    c->eng.set_rope_shift(0);
    if (c->ds.on) c->ds.set_rope_shift(0);
    qz_stats_out(c, st, out);
    return rc;
}

void qz_abort(qz_ctx * c) { if (c) c->eng.abort_flag = 1; }

void qz_destroy(qz_ctx * c) { if (c && c->vis) qz_vision_free(c->vis); delete c; }   // device memory is released at process exit

}
