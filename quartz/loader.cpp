// Model loader: plans placement, allocates, then streams and repacks each tensor.
#include "loader.h"
#include "repack.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return false; } } while (0)

bool qz_loader::plan(const qz_gguf & g, int card) {
    if (!g.ctx) return false;
    const uint64_t data_off = gguf_get_data_offset(g.ctx);

    const int64_t nt = gguf_get_n_tensors(g.ctx);
    ts.resize(nt);
    raw_off.resize(nt);
    need_bytes = 0;
    for (int64_t t = 0; t < nt; t++) {
        ts[t].name = gguf_get_tensor_name(g.ctx, t);
        ts[t].type = gguf_get_tensor_type(g.ctx, t);
        const int64_t * ne = gguf_get_tensor_ne(g.ctx, t);
        for (int d = 0; d < GGML_MAX_DIMS; d++) ts[t].ne[d] = ne[d] > 0 ? ne[d] : 1;
        ts[t].ne0 = ne[0];
        ts[t].ne1 = 1;                      // rows: dims past ne0, conv weights are 4-D
        for (int d = 1; d < GGML_MAX_DIMS; d++) ts[t].ne1 *= ne[d] > 0 ? ne[d] : 1;
        raw_off[t] = data_off + gguf_get_tensor_offset(g.ctx, t);
        ts[t].src_ne0 = ts[t].ne0;
        ts[t].src_ne1 = ts[t].ne1;
        if (card >= 0 && slice && slice(ts[t].name.c_str(), card, ts[t].ne1, ts[t].slice)) {
            if (ts[t].type != GGML_TYPE_Q8_0 && ts[t].type != GGML_TYPE_F32) {
                fprintf(stderr, "qz_loader: cannot slice %s (type)\n", ts[t].name.c_str()); return false;
            }
            const int64_t cu = ts[t].type == GGML_TYPE_Q8_0 ? 32 : 1;
            if (!ts[t].slice.cols.empty()) ts[t].ne0 = (int64_t) ts[t].slice.cols.size() * cu;
            if (!ts[t].slice.rows.empty()) ts[t].ne1 = (int64_t) ts[t].slice.rows.size();
            for (int32_t r : ts[t].slice.rows) if (r < 0 || r >= ts[t].src_ne1) { fprintf(stderr, "qz_loader: bad row map %s\n", ts[t].name.c_str()); return false; }
            for (int32_t cc : ts[t].slice.cols) if (cc < 0 || cc * cu >= ts[t].src_ne0) { fprintf(stderr, "qz_loader: bad col map %s\n", ts[t].name.c_str()); return false; }
        }
        ts[t].vstride = ts[t].ne0;
        ts[t].shared = skip && skip(ts[t].name.c_str());
        if (!ts[t].shared) need_bytes += ts[t].bytes();
    }
    return true;
}

bool qz_loader::load(const char * path, const qz_gguf & g, int dev, int card) {
    if (!g.ctx) return false;
    CHECK(hipSetDevice(dev));
    size_t total_b = 0;
    CHECK(hipMemGetInfo(&free_before, &total_b));
    if (!plan(g, card)) return false;
    const int64_t nt = (int64_t) ts.size();

    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "qz_loader: cannot open %s\n", path); return false; }
    {   // every tensor must end inside the file, so a truncated download fails here
        fseek(f, 0, SEEK_END);
        const uint64_t flen = (uint64_t) ftell(f);
        for (int64_t t = 0; t < nt; t++) {
            const double bs = (double) ggml_type_size(ts[t].type) / ggml_blck_size(ts[t].type);
            const uint64_t end = raw_off[t] + (uint64_t) ((double) ts[t].src_ne0 * ts[t].src_ne1 * bs);
            if (end > flen) {
                fprintf(stderr, "qz_loader: %s ends at byte %" PRIu64 ", past the end of %s (%" PRIu64 " bytes): truncated file?\n",
                        ts[t].name.c_str(), end, path, flen);
                fclose(f); return false;
            }
        }
    }
    // sliced tensors gather their rows and blocks from a read-only map
    const uint8_t * map = nullptr;
    size_t map_len = 0;
    if (card >= 0) {
        fseek(f, 0, SEEK_END);
        map_len = (size_t) ftell(f);
        map = (const uint8_t *) mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fileno(f), 0);
        if (map == MAP_FAILED) { fprintf(stderr, "qz_loader: mmap failed\n"); fclose(f); return false; }
        madvise((void *) map, map_len, MADV_SEQUENTIAL);
    }
    if ((size_t) need_bytes > free_before) { fprintf(stderr, "qz_loader: weights do not fit\n"); if (map) munmap((void *) map, map_len); fclose(f); return false; }

    // allocate big-first so the pool fragments cleanly
    std::vector<int64_t> order(nt);
    for (int64_t i = 0; i < nt; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return ts[a].bytes() > ts[b].bytes();
    });
    for (int64_t oi : order) {
        qz_dev_tensor & t = ts[oi];
        if (t.shared) continue;
        const char * role = t.name == "token_embd.weight" ? "embd" : t.name == "output.weight" ? "head" : "weights";
        if (t.type == GGML_TYPE_Q8_0) {
            CHECK(qz_dmalloc(&t.vals, (size_t)(t.vstride * t.ne1), role));
            CHECK(qz_dmalloc(&t.scal, (size_t)(t.ne0 * t.ne1 / 16), role));
        } else {
            CHECK(qz_dmalloc(&t.vals, (size_t) t.bytes(), role));
        }
    }

    // Stream and repack through two pinned buffers. A chunk is repacked into one
    // buffer while the other's async copy runs, which avoids small pageable copies.
    const auto t0 = std::chrono::steady_clock::now();
    const size_t CHUNK = (size_t) 64 << 20;
    uint8_t * pin[2] = { nullptr, nullptr };
    hipEvent_t ev[2] = { nullptr, nullptr };
    hipStream_t st = nullptr;
    CHECK(hipHostMalloc((void **) &pin[0], CHUNK, hipHostMallocDefault));
    CHECK(hipHostMalloc((void **) &pin[1], CHUNK, hipHostMallocDefault));
    CHECK(hipEventCreateWithFlags(&ev[0], hipEventDisableTiming));
    CHECK(hipEventCreateWithFlags(&ev[1], hipEventDisableTiming));
    CHECK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    bool ok = true;
    int cur = 0;
    // hand out the next pinned buffer once its previous copy has landed
    auto take = [&]() -> uint8_t * {
        cur ^= 1;
        if (hipEventSynchronize(ev[cur]) != hipSuccess) ok = false;
        return pin[cur];
    };
    auto submit = [&](void * dev_dst, const void * src, size_t n) {
        if (n && hipMemcpyAsync(dev_dst, src, n, hipMemcpyHostToDevice, st) != hipSuccess) ok = false;
    };
    // both events start signaled, so the first take() does not wait
    CHECK(hipEventRecord(ev[0], st));
    CHECK(hipEventRecord(ev[1], st));

    std::vector<uint8_t> raw;
    for (int64_t t = 0; t < nt && ok; t++) {
        qz_dev_tensor & d = ts[t];
        if (d.shared) continue;
        const int64_t K = d.ne0, N = d.ne1, VS = d.vstride;
        const bool sliced = !d.slice.rows.empty() || !d.slice.cols.empty();
        if (sliced && d.type == GGML_TYPE_F32) {
            // small tensors (conv1d, dt_bias, ssm_a): one staging buffer
            const float * src = (const float *)(map + raw_off[t]);
            if ((size_t) d.bytes() > CHUNK) { fprintf(stderr, "f32 slice too big %s\n", d.name.c_str()); ok = false; break; }
            float * buf = (float *) take();
            for (int64_t r = 0; r < N; r++) {
                const int64_t sr = d.slice.rows.empty() ? r : d.slice.rows[r];
                for (int64_t k = 0; k < K; k++) {
                    const int64_t sc = d.slice.cols.empty() ? k : d.slice.cols[k];
                    buf[r * K + k] = src[sr * d.src_ne0 + sc];
                }
            }
            submit(d.vals, buf, (size_t) d.bytes());
            if (hipEventRecord(ev[cur], st) != hipSuccess) ok = false;
            continue;
        }
        if (sliced) {
            // Q8_0: gather the selected 34-byte blocks of each selected row, then
            // repack exactly as the whole-model path does
            const int64_t src_raw = (d.src_ne0 / 32) * 34;
            const int64_t row_raw = (K / 32) * 34;
            const int64_t batch = std::max<int64_t>(1, std::min<int64_t>(N, (int64_t)(CHUNK / (size_t)(VS + K / 16))));
            std::vector<uint8_t> rr((size_t) row_raw);
            for (int64_t r = 0; r < N && ok; r += batch) {
                const int64_t nr = std::min(batch, N - r);
                uint8_t * buf = take();
                uint32_t * vals = (uint32_t *) buf;
                uint16_t * scal = (uint16_t *)(buf + nr * VS);
                if (VS != K) memset(buf, 0, (size_t)(nr * VS));
                for (int64_t i = 0; i < nr; i++) {
                    const int64_t sr = d.slice.rows.empty() ? r + i : d.slice.rows[r + i];
                    const uint8_t * srow = map + raw_off[t] + sr * src_raw;
                    const uint8_t * use = srow;
                    if (!d.slice.cols.empty()) {
                        for (size_t b = 0; b < d.slice.cols.size(); b++)
                            memcpy(rr.data() + b * 34, srow + (size_t) d.slice.cols[b] * 34, 34);
                        use = rr.data();
                    }
                    qz_repack_row_q8(use, (int) K, vals + i * (VS / 4), scal + i * (K / 32));
                }
                submit((char *) d.vals + r * VS, vals, (size_t)(nr * VS));
                submit((char *) d.scal + r * K / 16, scal, (size_t)(nr * K / 16));
                if (hipEventRecord(ev[cur], st) != hipSuccess) ok = false;
            }
            continue;
        }
        if (d.type != GGML_TYPE_Q8_0) {
            const size_t nb = (size_t) d.bytes();
            fseek(f, (long) raw_off[t], SEEK_SET);
            for (size_t o = 0; o < nb && ok; o += CHUNK) {
                const size_t n = std::min(CHUNK, nb - o);
                uint8_t * buf = take();
                if (fread(buf, 1, n, f) != n) { fprintf(stderr, "short read %s\n", d.name.c_str()); ok = false; break; }
                submit((char *) d.vals + o, buf, n);
                if (hipEventRecord(ev[cur], st) != hipSuccess) ok = false;
            }
            continue;
        }
        // Q8_0: K bytes per row in the values plane, K/16 in the scale plane
        const int64_t row_raw = (K / 32) * 34;
        const int64_t batch = std::max<int64_t>(1, std::min<int64_t>(N, (int64_t)(CHUNK / (size_t)(VS + K / 16))));
        raw.resize((size_t)(batch * row_raw));
        for (int64_t r = 0; r < N && ok; r += batch) {
            const int64_t nr = std::min(batch, N - r);
            fseek(f, (long)(raw_off[t] + r * row_raw), SEEK_SET);
            if (fread(raw.data(), 1, (size_t)(nr * row_raw), f) != (size_t)(nr * row_raw)) {
                fprintf(stderr, "short read %s row %" PRId64 "\n", d.name.c_str(), r); ok = false; break;
            }
            uint8_t * buf = take();
            uint32_t * vals = (uint32_t *) buf;
            uint16_t * scal = (uint16_t *)(buf + nr * VS);
            if (VS != K) memset(buf, 0, (size_t)(nr * VS));
            for (int64_t i = 0; i < nr; i++)
                qz_repack_row_q8(raw.data() + i * row_raw, (int) K, vals + i * (VS / 4), scal + i * (K / 32));
            submit((char *) d.vals + r * VS, vals, (size_t)(nr * VS));
            submit((char *) d.scal + r * K / 16, scal, (size_t)(nr * K / 16));
            if (hipEventRecord(ev[cur], st) != hipSuccess) ok = false;
        }
    }
    const hipError_t es = hipStreamSynchronize(st);
    (void) hipStreamDestroy(st);
    (void) hipEventDestroy(ev[0]);
    (void) hipEventDestroy(ev[1]);
    (void) hipHostFree(pin[0]);
    (void) hipHostFree(pin[1]);
    if (map) munmap((void *) map, map_len);
    if (!ok || es != hipSuccess) { fprintf(stderr, "qz_loader: streaming failed\n"); fclose(f); return false; }
    CHECK(hipDeviceSynchronize());
    load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(hipMemGetInfo(&free_after, &total_b));
    fclose(f);
    return true;
}

int64_t qz_loader::find(const char * name) const {
    for (size_t i = 0; i < ts.size(); i++)
        if (ts[i].name == name) return (int64_t) i;
    return -1;
}
