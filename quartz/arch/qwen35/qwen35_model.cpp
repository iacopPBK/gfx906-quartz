// qwen35: the C API model (q35_ctx) over the engine, prefill, DSpine and vision.
#include "q35_prelude.h"

namespace q35 {

#include "dspine_block.h"
#include "core/devices.h"
#include "engine.h"
#include "prefill.h"
// the verify at two widths, MV_T 8 for the bit-exact production rows and 12 for the
// copy-heavy rounds. One TU each, in its own namespace over shared 12-row replay rings.
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
#include "qwen35_preflight.h"

#include <algorithm>
#include <cstring>
#include <deque>

constexpr int32_t QZ_IM_START = 248045;   
constexpr int32_t QZ_IMG_PAD = 248056;
// Qwen3.8-27B Q8_0 on TP2: engine, batched prefill, prefix cache, DSpine and vision
struct q35_ctx : qz_ctx {
    qz_engine eng;
    std::string name;             // GGUF general.name
    qz_prefill pf;
    bool pf_ok = false;
    static constexpr int pf_min = 16;   // prompts shorter than this decode on the cycle's forced rows
    static constexpr int tail = 64;
    qz_pcache pc;                 // blobs: the GDN state dn, the conv state cs
    float * sdn[QZ_NC] = {}, * scs[QZ_NC] = {};   // mid-chunk staging of dn and cs
    int last_cached = 0;
    struct img { uint64_t hash = 0; int id = 0, nx = 0, ny = 0; unsigned long use = 0; std::vector<float> rows; };
    qwen3vl::qz_vision * vis = nullptr;
    std::vector<img> imgs;
    unsigned long img_clock = 0;
    int img_next = 0;
    float * img_dev[QZ_NC] = {};
    size_t img_cap = 0;

    qz_dsrun ds;
    // drafter switch, on with DSpine unless QZ_NO_SWITCH: DSpine or MTP, mid-answer
    struct {
        bool on = false;
        float lo = 3.8f, hi = 3.4f;       // leave DSpine below lo tok/round, MTP at hi and above
        int win = 128, flush = 512;       // window tokens, and MTP feature rows per DSpine ring fill
        int cap = 1024;                   // staged MTP entries per flush
        float * hn[QZ_NC] = {};               // staged entry inputs [cap + 12][QZ_D] per card
        std::vector<int32_t> tok;         // staged entry j: position p0 + j
        int p0 = 0;
    } sw;

    ~q35_ctx() override;
    void info(qz_model_info * out) const override;
    int  max_ctx() const override { return eng.steps - 2 * QZ_TMAX; }
    int  generate(const int32_t * prompt, int n_prompt, const int * images, int n_images,
                  int max_new, const int32_t * stop, int n_stop, qz_emit_cb cb, void * user,
                  qz_stats * out, const qz_sampling_c * smp) override;
    void abort() override { eng.abort_flag = 1; }
    int    vision_attach(const char * mmproj) override;
    int    image_add(const void * bytes, size_t n, int * nx, int * ny) override;
    double image_last_ms() const override;
};

static size_t qz_dn_bytes(const qz_engine & e) { return (size_t) e.hp.n_recr * g2_HV * QZ_S * QZ_S * 4; }
static size_t qz_cs_bytes(const qz_engine & e) { return (size_t) e.hp.n_recr * 3 * g2_CCH * 4; }
static void qz_cache_drop(q35_ctx * c) { c->pc.drop(); }   // the engine state has left every snapshot behind, so drop the cache

static void qz_pf_setup(q35_ctx * c) {
    // plus headroom; the buffers come from the engine's scratch when it holds them
    const size_t need = (c->eng.C[0].scr_bytes >= qz_prefill::bytes(2048) ? 0 : qz_prefill::bytes(2048)) + ((size_t) 256 << 20);
    for (int i = 0; i < QZ_NC; i++) {
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
    // the snapshots sit in pinned host memory up front, about 75 MB per slot per card.
    // One slot cannot hold a chunk's mid-chunk snapshot and its end snapshot at once.
    qz_mem_scope mem("snapshot staging");
    if (c->pf_ok) {
        qz_blob dn, cs;
        dn.bytes = qz_dn_bytes(c->eng); cs.bytes = qz_cs_bytes(c->eng);
        for (int i = 0; i < QZ_NC; i++) {
            CHECK(hipSetDevice(qz_dev_of(i)));
            CHECK(qz_dmalloc(&c->sdn[i], dn.bytes)); CHECK(qz_dmalloc(&c->scs[i], cs.bytes));
            dn.live[i] = c->eng.C[i].dn_all; dn.stage[i] = c->sdn[i];
            cs.live[i] = c->eng.C[i].cs_all; cs.stage[i] = c->scs[i];
        }
        CHECK(hipSetDevice(qz_dev_of(0)));
        c->pc.init({ dn, cs }, 8, c->eng.C[0].st, c->eng.C[1].st);
    }
    CHECK(hipSetDevice(qz_dev_of(0)));
}

// a prompt position's prefix-cache key: its token, or for an image row the image's
// content hash mixed with the row, since the image tokens all share one id
static std::vector<uint64_t> qz_keys(const std::vector<int32_t> & p, const qz_mm_view * mm, const std::vector<uint64_t> * row_hash) {
    std::vector<uint64_t> k(p.size());
    for (size_t q = 0; q < p.size(); q++) {
        const int r = mm ? mm->img((int) q) : -1;
        k[q] = r < 0 ? (uint64_t) (uint32_t) p[q] : (*row_hash)[r];
    }
    return k;
}

// per-round tokens over the last win tokens of a segment
struct qz_win {
    std::deque<int> r;
    int sum = 0, seg = 0;
    void add(int n, int win) { r.push_back(n); sum += n; seg += n; while (sum - r.front() >= win) { sum -= r.front(); r.pop_front(); } }
    float avg() const { return (float) sum / (float) r.size(); }
};

// decode from pos, the anchor, switching drafters. DSpine segments stage MTP entries,
// MTP segments capture DSpine features. A switch recomputes the uncommitted rows.
static int qz_switch_run(q35_ctx * c, int pos, int32_t anchor, int max_new, const std::vector<int32_t> & s,
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
        for (int i = 0; i < QZ_NC; i++) {   // entries p+1 .. p+n: the token there and row j's normed hidden
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
            for (int i = 0; i < QZ_NC; i++) { CHECK(hipSetDevice(qz_dev_of(i))); CHECK(hipMemset(e.C[i].ctl_d, 0, 8 * sizeof(int))); }
            CHECK(hipSetDevice(qz_dev_of(0)));
            r = e.generate(pend, left, s, (qz_emit_fn) cb, user, &g, pc);   // <= 3 rows: one forced chunk, like a request's first
        }
        tot.n_gen += g.n_gen; tot.decode_ms += g.decode_ms + (on_ds ? 0 : g.prefill_ms);
        tot.decode_cycles += g.decode_cycles; tot.decode_tok += g.decode_tok; tot.wide_cycles += g.wide_cycles;
        left -= g.n_gen;
        if (r < 0 || g.stop != 4) { tot.stop = g.stop; rc = r; break; }
        const auto t = clk::now();
        if (on_ds) {   // to MTP: entries up to the uncommitted rows, which are redone
            pend = ds.sw_pend; pc = ds.sw_pc;
            stage_flush();
            fl = pc;
        } else {       // to DSpine: context to the anchor, then the uncommitted rows forced
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

// prompt through the batched prefill and decode, or the engine's own path. rc as in
// generate. mm and row_hash carry the prompt's images; null means text only.
static int qz_run(q35_ctx * c, const std::vector<int32_t> & p, int max_new, const std::vector<int32_t> & s,
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
    // prompts with images decode on MTP. DSpine is exact there too, but it accepts too
    // little on image descriptions to beat MTP.
    const bool ds_on = c->ds.on && !mm;
    qz_pcache::hooks h;
    h.reset = [&] { c->eng.reset_state(); };
    h.restored = [&] {   // the decode rings and control words start clean
        for (int i = 0; i < QZ_NC; i++) {
            qz_card & K = c->eng.C[i];
            CHECK(hipSetDevice(qz_dev_of(i)));
            CHECK(hipMemsetAsync(K.ring_all, 0, (size_t) c->eng.hp.n_recr * 2 * g2_RING_PAR * 4, K.st));
            CHECK(hipMemsetAsync(K.ctl_d, 0, 8 * sizeof(int), K.st));
        }
        c->eng.sync2();
    };
    h.chunk = [&](int a, int b, int mid) {
        float * md[QZ_NC] = {}, * mc[QZ_NC] = {};
        if (mid >= 0)
            for (int i = 0; i < QZ_NC; i++) { md[i] = c->sdn[i]; mc[i] = c->scs[i]; }
        if (c->pf.run(c->eng, p.data() + a, a, b - a, p[b], mid, md, mc, mm)) return -1;
        if (ds_on) { c->eng.sync2(); c->ds.fill_prefill(c->pf.feat, c->pf.Mmax, a, b - a); }
        return 0;
    };
    h.sync = [&] { c->eng.sync2(); };
    // message boundary: <|im_start|>
    const int start = c->pc.prefill(p.data(), P, qz_keys(p, mm, row_hash), QZ_IM_START, 2048, c->tail, h);
    if (start < 0) { if (st) st->stop = -1; return -1; }
    c->last_cached = start;
    const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // greedy and sampled requests both run on DSpine. A sampled one drafts from the
    // drafter's calibrated q, with exact accept and residual.
    const int rc = ds_on && c->sw.on
        ? qz_switch_run(c, P - 1, p[P - 1], max_new, s, cb, user, st)
        : ds_on
        ? c->ds.generate(c->eng, P - 1, p[P - 1], max_new, s, (qz_emit_fn) cb, user, st)
        : c->eng.generate(std::vector<int32_t>{ p[P - 1] }, max_new, s, (qz_emit_fn) cb, user, st, P - 1);
    if (st) { st->prefill_ms += pf_ms; st->n_prompt = P; }
    return rc;
}


static void qz_stats_out(const q35_ctx * c, const qz_gen_stats & st, qz_stats * out) {
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

int q35_ctx::vision_attach(const char * mmproj) {
    q35_ctx * c = this;
    if (c->vis) return -1;
    if (!c->pf_ok) { fprintf(stderr, "quartz: images need the batched prefill\n"); return -1; }
    // both cards, half the encoder each
    c->vis = qwen3vl::qz_vision_load(mmproj);
    if (!c->vis) return -1;
    if (qwen3vl::qz_vision_n_embd(c->vis) != QZ_D) {
        fprintf(stderr, "quartz: projector width %d != model width %d\n", qwen3vl::qz_vision_n_embd(c->vis), QZ_D);
        qwen3vl::qz_vision_free(c->vis); c->vis = nullptr; return -1;
    }
    return 0;
}

static uint64_t qz_fnv(const void * b, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= ((const unsigned char *) b)[i]; h *= 1099511628211ull; }
    return h;
}

int q35_ctx::image_add(const void * bytes, size_t n, int * nx, int * ny) {
    q35_ctx * c = this;
    if (!c->vis) return -1;
    const uint64_t h = qz_fnv(bytes, n) ^ (uint64_t) n;
    for (auto & im : c->imgs)
        if (im.hash == h) {
            im.use = ++c->img_clock;
            if (nx) *nx = im.nx;
            if (ny) *ny = im.ny;
            return im.id;
        }
    int gx = 0, gy = 0;
    const int nt = qwen3vl::qz_vision_encode(c->vis, bytes, n, &gx, &gy);
    if (nt <= 0) return -1;
    const int cap = 8;
    if ((int) c->imgs.size() >= cap) {
        auto lru = std::min_element(c->imgs.begin(), c->imgs.end(),
                                    [](const q35_ctx::img & a, const q35_ctx::img & b) { return a.use < b.use; });
        c->imgs.erase(lru);
    }
    q35_ctx::img im;
    im.hash = h; im.id = c->img_next++; im.nx = gx; im.ny = gy; im.use = ++c->img_clock;
    im.rows.resize((size_t) nt * QZ_D);
    if (qwen3vl::qz_vision_copy_rows(c->vis, im.rows.data()) != nt) return -1;
    c->imgs.push_back(std::move(im));
    if (nx) *nx = gx;
    if (ny) *ny = gy;
    return c->imgs.back().id;
}

double q35_ctx::image_last_ms() const { return vis ? qwen3vl::qz_vision_last_ms(vis) : 0.0; }

int q35_ctx::generate(const int32_t * prompt, int n_prompt, const int * images, int n_images,
                      int max_new, const int32_t * stop, int n_stop, qz_emit_cb cb, void * user, qz_stats * out,
                      const qz_sampling_c * smp) {
    q35_ctx * c = this;
    if (!prompt || n_prompt < 1 || n_images < 0 || (n_images && !images)) return -1;
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
        std::vector<const q35_ctx::img *> im(n_images);
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
            for (int i = 0; i < QZ_NC; i++) {
                CHECK(hipSetDevice(qz_dev_of(i)));
                if (c->img_dev[i]) (void) qz_dfree(c->img_dev[i]);
                c->img_dev[i] = nullptr;
                qz_mem_scope mem("images");
                if (qz_dmalloc(&c->img_dev[i], R * QZ_D * sizeof(float)) != hipSuccess) { c->img_cap = 0; return -1; }
            }
            c->img_cap = R;
        }
        for (int i = 0; i < QZ_NC; i++) {
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

q35_ctx::~q35_ctx() { if (vis) qwen3vl::qz_vision_free(vis); }   // device memory is released at process exit

void q35_ctx::info(qz_model_info * out) const {
    snprintf(out->arch, sizeof(out->arch), "qwen35");
    snprintf(out->name, sizeof(out->name), "%s", name.c_str());
    out->n_vocab = eng.n_vocab;
    out->max_ctx = max_ctx();
    out->img_pad = QZ_IMG_PAD;
    out->vision = vis != nullptr;
    out->dspine = ds.on;
}

static qz_ctx * create(const char * model, const qz_create_opts & o) {
    const int max_ctx = o.max_ctx, slot_t = o.slot_t, adapt_slack = o.adapt_slack;
    if ( max_ctx < 1024 || (slot_t != 0 && slot_t != 3 && slot_t != 4) || adapt_slack < 0) return nullptr;
    qz_mem_scope mem("engine");
    qz_mem_baseline(2);
    // steps, the arena positions: prompt, generated tokens and the cycle / MTP overhang
    const int steps = max_ctx + 2 * QZ_TMAX;
    // the scratch the cycle's partials share with batched prefill and the DSpine verify
    const size_t scratch = std::max(qz_prefill::bytes(2048), mv12::mv_partial_bytes());
    // before the weights: memory for this context on both cards, then the peer link
    if (qz_vram_check(model, max_ctx, steps, scratch, getenv("QZ_DSPINE"), getenv("QZ_MMPROJ")) || qz_link_check())
        return nullptr;
    q35_ctx * c = new q35_ctx();
    c->eng.scratch_min = scratch;
    if (c->eng.init(model, steps, slot_t, adapt_slack)) {
        delete c;      // device buffers of a failed init are not released (one engine per process)
        return nullptr;
    }
    {
        qz_gguf g;
        if (g.open(model)) c->name = g.str("general.name");
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
        for (int i = 0; i < QZ_NC; i++) { CHECK(hipSetDevice(qz_dev_of(i))); CHECK(qz_dmalloc(&w.hn[i], (size_t) (w.cap + 12) * QZ_D * 4)); }
        CHECK(hipSetDevice(qz_dev_of(0)));
        w.on = true;
        fprintf(stderr, "quartz: drafter switch on: leave DSpine < %.2f, MTP >= %.2f tok/round, window %d\n", w.lo, w.hi, w.win);
    }
    // per card: used and free, the ledger's tags, what the runtime holds beyond them
    qz_mem_report(stderr, 2);
    return c;
}

}   // namespace q35

qz_ctx * qz_qwen35_create(const char * model, const qz_create_opts & o) { return q35::create(model, o); }
