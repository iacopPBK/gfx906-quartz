// inkling2v image encoder (see inkling2v.h), as llama.cpp's mtmd inkling2v graph:
// patches of 20 px (the image at its own size, padded to a multiple of 40 with
// -1 before the normalization), + the 32 x 32 position table resized bilinearly
// (half-pixel centers) to the patch grid, a CLS row first, pre-LayerNorm, 23
// pre-norm layers (LayerNorm, attention with q / k 2D RoPE: dims 0..31 by the
// column, 32..63 by the row, 1-based, CLS at 0; layer scales on both branches;
// GELU-erf FFN), then the CLS row dropped, 2 x 2 patches concatenated
// ((dy, dx) = (0,0), (0,1), (1,0), (1,1)) and the 2-layer GELU-erf MLP.
// rocBLAS f16 GEMMs with f32 accumulation; attention as two batched GEMMs per
// query chunk with an f32 softmax between. One card (the encoder is 0.3 B).
#include "inkling2v.h"
#include "loader.h"
#include "core/devices.h"
#include "gguf.h"
#include "mem.h"
#include "resize.h"
#include "kernels/dev.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#include "stb/stb_image.h"   // the implementation is in qwen3vl.cpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


namespace inkling2v {

#include "common.h"

static constexpr int V_D = 1024, V_H = 16, V_HD = 64, V_PATCH = 20, V_MERGE = 2, V_GRID = 32, V_K = 3 * V_PATCH * V_PATCH;
static constexpr int V_FF = 4096, V_MM = 4 * V_D;
static constexpr float V_THETA = 10000.f;
static constexpr size_t V_SBUF = (size_t) 64 << 20;     // attention scores per chunk (floats)

// ------------------------------------------------------------- kernels ----

// pixels (w0 x h0, RGB) -> patch rows [N][1200] f16, patch n = py * pw + px, columns
// (c, ky, kx) as the conv kernel [1024][3][20][20]; outside the image: -1 (the
// reference pads the 0..255 pixels with -1, then normalizes)
__global__ void __launch_bounds__(256) v_patchify(const unsigned char * __restrict__ rgb, int w0, int h0, int pw,
                                                  __half * __restrict__ P) {
    const int n = blockIdx.x, py = n / pw, px = n % pw;
    for (int j = threadIdx.x; j < V_K; j += 256) {
        const int c = j / (V_PATCH * V_PATCH), ky = (j / V_PATCH) % V_PATCH, kx = j % V_PATCH;
        const int y = py * V_PATCH + ky, x = px * V_PATCH + kx;
        const float u = y < h0 && x < w0 ? (float) rgb[((size_t) y * w0 + x) * 3 + c] : -1.f;
        P[(size_t) n * V_K + j] = __float2half((u - 127.5f) / 127.5f);
    }
}

// row 0: CLS + its position; rows 1 .. N: += the position table at the patch,
// bilinear as ggml_interpolate (half-pixel centers, clamped) when the grid is not 32 x 32
__global__ void __launch_bounds__(256) v_pos(float * __restrict__ x, const float * __restrict__ cls,
                                             const float * __restrict__ cls_pos, const float * __restrict__ grid,
                                             int pw, int ph) {
    const int r = blockIdx.x;
    float * xr = x + (size_t) r * V_D;
    if (r == 0) {
        for (int d = threadIdx.x; d < V_D; d += 256) xr[d] = cls[d] + cls_pos[d];
        return;
    }
    const int n = r - 1, py = n / pw, px = n % pw;
    const float sf0 = (float) pw / V_GRID, sf1 = (float) ph / V_GRID;
    const float fy = ((float) py + 0.5f) / sf1 - 0.5f, fx = ((float) px + 0.5f) / sf0 - 0.5f;
    int y0 = (int) floorf(fy), x0 = (int) floorf(fx);
    int y1 = y0 + 1, x1 = x0 + 1;
    y0 = max(0, min(y0, V_GRID - 1)); y1 = max(0, min(y1, V_GRID - 1));
    x0 = max(0, min(x0, V_GRID - 1)); x1 = max(0, min(x1, V_GRID - 1));
    const float dy = fminf(fmaxf(fy - (float) y0, 0.f), 1.f), dx = fminf(fmaxf(fx - (float) x0, 0.f), 1.f);
    const float * a = grid + (size_t) (y0 * V_GRID + x0) * V_D, * b = grid + (size_t) (y0 * V_GRID + x1) * V_D;
    const float * c = grid + (size_t) (y1 * V_GRID + x0) * V_D, * e = grid + (size_t) (y1 * V_GRID + x1) * V_D;
    for (int d = threadIdx.x; d < V_D; d += 256)
        xr[d] += a[d] * (1 - dx) * (1 - dy) + b[d] * dx * (1 - dy) + c[d] * (1 - dx) * dy + e[d] * dx * dy;
}

// LayerNorm of one f32 row of V_D (weight + bias); out f32 (may alias x) or f16
template <typename T>
__global__ void __launch_bounds__(256) v_ln(const float * __restrict__ x, const float * __restrict__ w,
                                            const float * __restrict__ b, float eps, T * __restrict__ out) {
    constexpr int PT = V_D / 256;
    __shared__ float red[4];
    const float * r = x + (size_t) blockIdx.x * V_D;
    float v[PT], s = 0.f;
    #pragma unroll
    for (int i = 0; i < PT; i++) { v[i] = r[threadIdx.x + i * 256]; s += v[i]; }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) s += __shfl_xor(s, off, 64);
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = s;
    __syncthreads();
    const float mean = (red[0] + red[1] + red[2] + red[3]) / V_D;
    __syncthreads();
    float q = 0.f;
    #pragma unroll
    for (int i = 0; i < PT; i++) { v[i] -= mean; q += v[i] * v[i]; }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) q += __shfl_xor(q, off, 64);
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = q;
    __syncthreads();
    const float inv = 1.f / sqrtf((red[0] + red[1] + red[2] + red[3]) / V_D + eps);
    #pragma unroll
    for (int i = 0; i < PT; i++) {
        const int d = threadIdx.x + i * 256;
        out[(size_t) blockIdx.x * V_D + d] = (T) (v[i] * inv * w[d] + b[d]);
    }
}

// q / k / v (f32 [N][V_D] each, + bias) -> f16; q and k with the 2D RoPE (ggml mode 0:
// adjacent pairs): dims 0..31 at the column position, 32..63 at the row position
__global__ void __launch_bounds__(256) v_qkv(const float * __restrict__ y, size_t plane, const float * __restrict__ qb,
                                             const float * __restrict__ kb, const float * __restrict__ vb, int pw,
                                             __half * __restrict__ Q, __half * __restrict__ K, __half * __restrict__ V) {
    const int r = blockIdx.x;
    const int pwr = r == 0 ? 0 : (r - 1) % pw + 1, phr = r == 0 ? 0 : (r - 1) / pw + 1;
    const float * yq = y + (size_t) r * V_D, * yk = yq + plane, * yv = yq + 2 * plane;
    const size_t o = (size_t) r * V_D;
    for (int idx = threadIdx.x; idx < V_D / 2; idx += 256) {      // one pair per thread
        const int h = idx / (V_HD / 2), j = idx % (V_HD / 2);     // pair j of head h
        const int half = j / 16, i = j % 16;                      // half 0: column, 1: row; pair i of it
        const float p = (float) (half == 0 ? pwr : phr);
        const float ang = p * powf(V_THETA, -2.f * (float) i / 32.f);
        const float cs = cosf(ang), sn = sinf(ang);
        const int d0 = h * V_HD + 2 * j;
        const float q0 = yq[d0] + qb[d0], q1 = yq[d0 + 1] + qb[d0 + 1];
        const float k0 = yk[d0] + kb[d0], k1 = yk[d0 + 1] + kb[d0 + 1];
        Q[o + d0] = __float2half(q0 * cs - q1 * sn); Q[o + d0 + 1] = __float2half(q0 * sn + q1 * cs);
        K[o + d0] = __float2half(k0 * cs - k1 * sn); K[o + d0 + 1] = __float2half(k0 * sn + k1 * cs);
        V[o + d0] = __float2half(yv[d0] + vb[d0]); V[o + d0 + 1] = __float2half(yv[d0 + 1] + vb[d0 + 1]);
    }
}

// rows of n scores (already scaled) -> softmax as f16
__global__ void __launch_bounds__(256) v_softmax(const float * __restrict__ s, int n, __half * __restrict__ p) {
    __shared__ float red[4];
    const float * r = s + (size_t) blockIdx.x * n;
    __half * o = p + (size_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += 256) m = fmaxf(m, r[j]);
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, 64));
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float l = 0.f;
    for (int j = threadIdx.x; j < n; j += 256) l += expf(r[j] - m);
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) l += __shfl_xor(l, off, 64);
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = l;
    __syncthreads();
    const float il = 1.f / (red[0] + red[1] + red[2] + red[3]);
    for (int j = threadIdx.x; j < n; j += 256) o[j] = __float2half(expf(r[j] - m) * il);
}

// x += ls * (y + b)  (b, ls per column of V_D; either may be null)
__global__ void __launch_bounds__(256) v_resid(float * __restrict__ x, const float * __restrict__ y, const float * __restrict__ b,
                                               const float * __restrict__ ls, size_t n) {
    const size_t i = (size_t) blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int d = (int) (i % V_D);
    float v = y[i] + (b ? b[d] : 0.f);
    if (ls) v *= ls[d];
    x[i] += v;
}

__device__ __forceinline__ float v_gelu_erf(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }

// out = gelu_erf(y + b) as f16 (b may be null); grid (d / 256, rows)
__global__ void __launch_bounds__(256) v_bias_gelu(const float * __restrict__ y, const float * __restrict__ b,
                                                   int d, __half * __restrict__ out) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= d) return;
    const size_t o = (size_t) blockIdx.y * d + i;
    out[o] = __float2half(v_gelu_erf(y[o] + (b ? b[i] : 0.f)));
}

// the patch rows (x rows 1 ..) of each 2 x 2 block concatenated -> [T][4 V_D] f16,
// token t = by * (pw / 2) + bx, part (dy, dx) at (2 dy + dx) V_D
__global__ void __launch_bounds__(256) v_merge(const float * __restrict__ x, int pw, __half * __restrict__ M) {
    const int t = blockIdx.x, bw = pw / V_MERGE, by = t / bw, bx = t % bw;
    for (int e = threadIdx.x; e < V_MM; e += 256) {
        const int part = e / V_D, d = e % V_D, dy = part >> 1, dx = part & 1;
        const size_t src = 1 + (size_t) (V_MERGE * by + dy) * pw + V_MERGE * bx + dx;
        M[(size_t) t * V_MM + e] = __float2half(x[src * V_D + d]);
    }
}

// --------------------------------------------------------------- host ----

namespace {
struct vlayer {
    const __half * q_w, * k_w, * v_w, * o_w, * up_w, * down_w;
    const float * q_b, * k_b, * v_b, * o_b, * up_b, * down_b, * ln1_w, * ln1_b, * ln2_w, * ln2_b, * ls1, * ls2;
};
}   // namespace

struct qz_vision {
    int dev = -1;
    hipStream_t st = nullptr;
    rocblas_handle blas = nullptr;
    qz_loader ld;
    std::vector<vlayer> L;
    const __half * patch_w = nullptr, * mm1_w = nullptr, * mm2_w = nullptr;
    const float * cls = nullptr, * cls_pos = nullptr, * pos = nullptr, * pre_w = nullptr, * pre_b = nullptr,
                * mm1_b = nullptr, * mm2_b = nullptr;
    float eps = 1e-5f;
    int n_out = 0, max_tok = 4096, np_max = 0;
    unsigned char * rgb = nullptr;
    size_t rgb_cap = 0;
    __half * P = nullptr, * h = nullptr, * Q = nullptr, * K = nullptr, * V = nullptr, * O = nullptr, * G = nullptr, * Pa = nullptr;
    float * x = nullptr, * y = nullptr, * S = nullptr, * out = nullptr;
    int n_tok = 0;
    double ms = 0;
};

// the projector type: clip.projector_type, or clip.vision.projector_type (newer converters)
static std::string proj_type(const qz_gguf & g) {
    const std::string t = g.str("clip.projector_type");
    return t.empty() ? g.str("clip.vision.projector_type") : t;
}

bool qz_vision_is(const char * path) {
    qz_gguf g;
    return path && g.open(path) && proj_type(g) == "inkling2v";
}

void qz_vision_free(qz_vision * v) {
    if (!v) return;
    if (v->dev >= 0 && hipSetDevice(v->dev) == hipSuccess) {
        if (v->st) (void) hipStreamSynchronize(v->st);
        for (void * p : { (void *) v->rgb, (void *) v->P, (void *) v->h, (void *) v->Q, (void *) v->K, (void *) v->V,
                          (void *) v->O, (void *) v->G, (void *) v->Pa, (void *) v->x, (void *) v->y, (void *) v->S, (void *) v->out })
            if (p) (void) qz_dfree(p);
        for (auto & t : v->ld.ts) if (!t.shared && t.vals) (void) qz_dfree(t.vals);
        if (v->blas) rocblas_destroy_handle(v->blas);
        if (v->st) (void) hipStreamDestroy(v->st);
    }
    delete v;
}

qz_vision * qz_vision_load(const char * path) {
    if (qz_dev_select()) return nullptr;
    qz_gguf g;
    if (!g.open(path)) { fprintf(stderr, "quartz_vision: cannot open %s\n", path); return nullptr; }
    if (proj_type(g) != "inkling2v" || g.u32("clip.vision.embedding_length") != V_D ||
        g.u32("clip.vision.attention.head_count") != V_H || g.u32("clip.vision.patch_size") != V_PATCH ||
        g.u32("clip.vision.spatial_merge_size", 2) != V_MERGE || g.u32("clip.vision.feed_forward_length") != V_FF) {
        fprintf(stderr, "quartz_vision: %s is not a supported inkling2v projector\n", path); return nullptr;
    }
    const int nl = (int) g.u32("clip.vision.block_count");
    qz_vision * v = new qz_vision;
    v->dev = qz_dev_of(0);
    v->eps = g.f32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
    if (getenv("QZ_IMG_MAX_TOKENS")) v->max_tok = std::max(1, atoi(getenv("QZ_IMG_MAX_TOKENS")));
    v->np_max = v->max_tok * V_MERGE * V_MERGE + 1;
    auto fail = [&](const char * what) { fprintf(stderr, "quartz_vision: %s\n", what); qz_vision_free(v); return (qz_vision *) nullptr; };
    {
        qz_mem_scope mem("vision");
        if (hipSetDevice(v->dev) != hipSuccess || !v->ld.load(path, g, v->dev)) return fail("weights failed to load");
    }
    auto get = [&](const char * n, ggml_type t, int64_t ne0, int64_t nel) -> const void * {
        const qz_dev_tensor * x = v->ld.get(n);
        if (!x || x->type != t || x->ne0 != ne0 || x->ne0 * x->ne1 != nel) return nullptr;
        return x->vals;
    };
    auto f16 = [&](const char * n, int64_t k, int64_t m) { return (const __half *) get(n, GGML_TYPE_F16, k, k * m); };
    auto f32 = [&](const char * n, int64_t len) { return (const float *) get(n, GGML_TYPE_F32, len, len); };
    v->patch_w = (const __half *) get("v.patch_embd.weight", GGML_TYPE_F16, V_PATCH, (int64_t) V_K * V_D);
    v->cls = f32("v.class_embd", V_D); v->cls_pos = f32("v.class_position_embd.weight", V_D);
    v->pos = (const float *) get("v.position_embd.weight", GGML_TYPE_F32, V_D, (int64_t) V_GRID * V_GRID * V_D);
    v->pre_w = f32("v.pre_ln.weight", V_D); v->pre_b = f32("v.pre_ln.bias", V_D);
    v->mm1_w = f16("mm.1.weight", V_MM, V_MM); v->mm1_b = f32("mm.1.bias", V_MM);   // biases: optional
    const qz_dev_tensor * m2 = v->ld.get("mm.2.weight");
    if (!m2 || m2->type != GGML_TYPE_F16 || m2->ne0 != V_MM) return fail("missing mm.2.weight");
    v->n_out = (int) m2->ne1;
    v->mm2_w = (const __half *) m2->vals;
    v->mm2_b = f32("mm.2.bias", v->n_out);
    if (!v->patch_w || !v->cls || !v->cls_pos || !v->pos || !v->pre_w || !v->pre_b || !v->mm1_w)
        return fail("missing or unexpected tensors");
    char nm[64];
    for (int l = 0; l < nl; l++) {
        auto n = [&](const char * s) { snprintf(nm, sizeof nm, "v.blk.%d.%s", l, s); return nm; };
        vlayer y{};
        y.q_w = f16(n("attn_q.weight"), V_D, V_D); y.k_w = f16(n("attn_k.weight"), V_D, V_D); y.v_w = f16(n("attn_v.weight"), V_D, V_D);
        y.o_w = f16(n("attn_out.weight"), V_D, V_D); y.up_w = f16(n("ffn_up.weight"), V_D, V_FF); y.down_w = f16(n("ffn_down.weight"), V_FF, V_D);
        y.q_b = f32(n("attn_q.bias"), V_D); y.k_b = f32(n("attn_k.bias"), V_D); y.v_b = f32(n("attn_v.bias"), V_D);
        y.o_b = f32(n("attn_out.bias"), V_D); y.up_b = f32(n("ffn_up.bias"), V_FF); y.down_b = f32(n("ffn_down.bias"), V_D);
        y.ln1_w = f32(n("ln1.weight"), V_D); y.ln1_b = f32(n("ln1.bias"), V_D); y.ln2_w = f32(n("ln2.weight"), V_D); y.ln2_b = f32(n("ln2.bias"), V_D);
        y.ls1 = f32(n("ls1.weight"), V_D); y.ls2 = f32(n("ls2.weight"), V_D);
        if (!y.q_w || !y.k_w || !y.v_w || !y.o_w || !y.up_w || !y.down_w || !y.q_b || !y.k_b || !y.v_b || !y.o_b ||
            !y.up_b || !y.down_b || !y.ln1_w || !y.ln1_b || !y.ln2_w || !y.ln2_b || !y.ls1 || !y.ls2)
            return fail("missing layer tensors");
        v->L.push_back(y);
    }
    const size_t np = v->np_max;
    bool ok = hipStreamCreateWithFlags(&v->st, hipStreamNonBlocking) == hipSuccess;
    {
        qz_mem_scope mem("vision");
        ok = ok && qz_dmalloc(&v->P, np * V_K * 2, "act") == hipSuccess
                && qz_dmalloc(&v->x, np * V_D * 4, "act") == hipSuccess
                && qz_dmalloc(&v->h, np * V_D * 2, "act") == hipSuccess
                && qz_dmalloc(&v->y, np * V_FF * 4, "act") == hipSuccess
                && qz_dmalloc(&v->G, np * V_FF * 2, "act") == hipSuccess
                && qz_dmalloc(&v->Q, np * V_D * 2, "act") == hipSuccess
                && qz_dmalloc(&v->K, np * V_D * 2, "act") == hipSuccess
                && qz_dmalloc(&v->V, np * V_D * 2, "act") == hipSuccess
                && qz_dmalloc(&v->O, np * V_D * 2, "act") == hipSuccess
                && qz_dmalloc(&v->S, V_SBUF * 4, "attention") == hipSuccess
                && qz_dmalloc(&v->Pa, V_SBUF * 2, "attention") == hipSuccess
                && qz_dmalloc(&v->out, (size_t) v->max_tok * v->n_out * 4, "out") == hipSuccess;
    }
    ok = ok && rocblas_create_handle(&v->blas) == rocblas_status_success && rocblas_set_stream(v->blas, v->st) == rocblas_status_success;
    if (!ok) return fail("scratch allocation failed");
    return v;
}

int qz_vision_n_embd(const qz_vision * v) { return v->n_out; }
double qz_vision_last_ms(const qz_vision * v) { return v->ms; }

int qz_vision_copy_rows(const qz_vision * v, float * host) {
    VCHK(hipSetDevice(v->dev));
    VCHK(hipMemcpy(host, v->out, (size_t) v->n_tok * v->n_out * 4, hipMemcpyDeviceToHost));
    return v->n_tok;
}

int qz_vision_encode_rgb(qz_vision * v, const unsigned char * rgb, int w, int h, int * nx_out, int * ny_out) {
    if (w <= 0 || h <= 0) return -1;
    const auto t0 = std::chrono::steady_clock::now();
    const int al = V_PATCH * V_MERGE;
    auto up = [&](int a) { return (a + al - 1) / al * al; };
    // the image at its own size; past the token limit scaled down (calc_size_preserved_ratio,
    // align 40, Pillow bicubic), as the reference
    std::vector<unsigned char> img;
    int w0 = w, h0 = h;
    const int64_t maxp = (int64_t) v->max_tok * al * al;
    if ((int64_t) up(w) * up(h) > maxp) {
        auto rnd = [&](float x) { return (int) std::round(x / (float) al) * al; };
        auto fl = [&](float x) { return (int) std::floor(x / (float) al) * al; };
        int wb = std::max(al, rnd((float) w)), hb = std::max(al, rnd((float) h));
        if ((int64_t) wb * hb > maxp) {
            const float beta = std::sqrt((float) h * w / (float) maxp);
            hb = std::max(al, fl(h / beta)); wb = std::max(al, fl(w / beta));
        }
        qz_img::resize_bicubic(rgb, w, h, img, wb, hb);
        w0 = wb; h0 = hb;
    } else {
        img.assign(rgb, rgb + (size_t) w * h * 3);
    }
    const int W = up(w0), H = up(h0), pw = W / V_PATCH, ph = H / V_PATCH, Np = pw * ph, N = Np + 1;
    const int nx = pw / V_MERGE, ny = ph / V_MERGE, T = nx * ny;
    if (T > v->max_tok) { fprintf(stderr, "quartz_vision: %dx%d tokens over the limit\n", nx, ny); return -1; }
    VCHK(hipSetDevice(v->dev));
    hipStream_t s = v->st;
    if (img.size() > v->rgb_cap) {
        if (v->rgb) (void) qz_dfree(v->rgb);
        v->rgb = nullptr;
        qz_mem_scope mem("vision");
        VCHK(qz_dmalloc(&v->rgb, img.size(), "image"));
        v->rgb_cap = img.size();
    }
    VCHK(hipMemcpyAsync(v->rgb, img.data(), img.size(), hipMemcpyHostToDevice, s));
    v_patchify<<<Np, 256, 0, s>>>(v->rgb, w0, h0, pw, v->P);
    if (vgemm(v->blas, v->patch_w, v->P, v->x + V_D, false, V_D, Np, V_K)) return -1;
    v_pos<<<N, 256, 0, s>>>(v->x, v->cls, v->cls_pos, v->pos, pw, ph);
    v_ln<float><<<N, 256, 0, s>>>(v->x, v->pre_w, v->pre_b, v->eps, v->x);
    const size_t nd = (size_t) N * V_D;
    const int qc = (int) std::min<size_t>(N, V_SBUF / ((size_t) V_H * N));       // queries per attention chunk
    const float scale = 1.f / std::sqrt((float) V_HD), one = 1.f, zero = 0.f;
    for (const vlayer & l : v->L) {
        v_ln<__half><<<N, 256, 0, s>>>(v->x, l.ln1_w, l.ln1_b, v->eps, v->h);
        if (vgemm(v->blas, l.q_w, v->h, v->y, false, V_D, N, V_D) || vgemm(v->blas, l.k_w, v->h, v->y + nd, false, V_D, N, V_D) ||
            vgemm(v->blas, l.v_w, v->h, v->y + 2 * nd, false, V_D, N, V_D)) return -1;
        v_qkv<<<N, 256, 0, s>>>(v->y, nd, l.q_b, l.k_b, l.v_b, pw, v->Q, v->K, v->V);
        for (int q0 = 0; q0 < N; q0 += qc) {
            const int nq = std::min(qc, N - q0);
            // S[h][i][j] = scale q_i . k_j  (column-major: keys x queries per head)
            VBLAS(rocblas_gemm_strided_batched_ex(v->blas, rocblas_operation_transpose, rocblas_operation_none, N, nq, V_HD, &scale,
                                                  v->K, rocblas_datatype_f16_r, V_D, V_HD,
                                                  v->Q + (size_t) q0 * V_D, rocblas_datatype_f16_r, V_D, V_HD, &zero,
                                                  v->S, rocblas_datatype_f32_r, N, (rocblas_stride) nq * N,
                                                  v->S, rocblas_datatype_f32_r, N, (rocblas_stride) nq * N, V_H,
                                                  rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
            v_softmax<<<V_H * nq, 256, 0, s>>>(v->S, N, v->Pa);
            // O[i][h 64 + d] = sum_j P[h][i][j] V[j][h 64 + d]
            VBLAS(rocblas_gemm_strided_batched_ex(v->blas, rocblas_operation_none, rocblas_operation_none, V_HD, nq, N, &one,
                                                  v->V, rocblas_datatype_f16_r, V_D, V_HD,
                                                  v->Pa, rocblas_datatype_f16_r, N, (rocblas_stride) nq * N, &zero,
                                                  v->O + (size_t) q0 * V_D, rocblas_datatype_f16_r, V_D, V_HD,
                                                  v->O + (size_t) q0 * V_D, rocblas_datatype_f16_r, V_D, V_HD, V_H,
                                                  rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
        }
        if (vgemm(v->blas, l.o_w, v->O, v->y, false, V_D, N, V_D)) return -1;
        v_resid<<<(nd + 255) / 256, 256, 0, s>>>(v->x, v->y, l.o_b, l.ls1, nd);
        v_ln<__half><<<N, 256, 0, s>>>(v->x, l.ln2_w, l.ln2_b, v->eps, v->h);
        if (vgemm(v->blas, l.up_w, v->h, v->y, false, V_FF, N, V_D)) return -1;
        v_bias_gelu<<<dim3(V_FF / 256, N), 256, 0, s>>>(v->y, l.up_b, V_FF, v->G);
        if (vgemm(v->blas, l.down_w, v->G, v->y, false, V_D, N, V_FF)) return -1;
        v_resid<<<(nd + 255) / 256, 256, 0, s>>>(v->x, v->y, l.down_b, l.ls2, nd);
    }
    v_merge<<<T, 256, 0, s>>>(v->x, pw, v->h);
    if (vgemm(v->blas, v->mm1_w, v->h, v->y, false, V_MM, T, V_MM)) return -1;
    v_bias_gelu<<<dim3(V_MM / 256, T), 256, 0, s>>>(v->y, v->mm1_b, V_MM, v->G);
    if (vgemm(v->blas, v->mm2_w, v->G, v->out, false, v->n_out, T, V_MM)) return -1;
    if (v->mm2_b) v_bias<<<dim3((v->n_out + 255) / 256, T), 256, 0, s>>>(v->out, v->mm2_b, v->n_out);
    VCHK(hipGetLastError());
    VCHK(hipStreamSynchronize(s));
    v->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    v->n_tok = T;
    if (nx_out) *nx_out = nx;
    if (ny_out) *ny_out = ny;
    return T;
}

}   // namespace inkling2v
