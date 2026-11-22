// qwen3vl_merger image encoder (see vision.h): rocBLAS f16 GEMMs with f32
// accumulation, v_flash attention.
#include "vision.h"
#include "loader.h"
#include "devices.h"
#include "gguf.h"
#include "mem.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define VCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "quartz_vision: HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define VBLAS(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { \
    fprintf(stderr, "quartz_vision: rocBLAS error %s at %s:%d\n", rocblas_status_to_string(s_), __FILE__, __LINE__); return -1; } } while (0)

static constexpr int V_D = 1152, V_H = 16, V_HD = 72, V_PATCH = 16, V_GRID = 48, V_K = 3 * V_PATCH * V_PATCH;

// ------------------------------------------------------------- kernels ----

// pixels -> patch rows [N][768] f16 in the 2x2 block order (ty, tx, dy, dx);
// columns (c, ky, kx) as the conv kernel [1152][3][16][16]
__global__ void __launch_bounds__(256) v_patchify(const unsigned char * __restrict__ rgb, int W, __half * __restrict__ P) {
    const int n = blockIdx.x, bpr = W / 32;
    const int b = n >> 2, ty = b / bpr, tx = b % bpr;
    const int py = 2 * ty + ((n >> 1) & 1), px = 2 * tx + (n & 1);
    for (int j = threadIdx.x; j < V_K; j += 256) {
        const int c = j >> 8, ky = (j >> 4) & 15, kx = j & 15;
        const float u = (float) rgb[((size_t) (py * V_PATCH + ky) * W + px * V_PATCH + kx) * 3 + c] / 255.0f;
        P[(size_t) n * V_K + j] = __float2half((u - 0.5f) / 0.5f);
    }
}

// x[n] += patch bias + the 48 x 48 position grid, bilinear (align corners) at
// the patch's (py, px) of a ph x pw patch grid
__global__ void __launch_bounds__(256) v_pos(float * __restrict__ x, const float * __restrict__ bias,
                                             const float * __restrict__ grid, int pw, int ph) {
    const int n = blockIdx.x, bpr = pw / 2;
    const int b = n >> 2, ty = b / bpr, tx = b % bpr;
    const int py = 2 * ty + ((n >> 1) & 1), px = 2 * tx + (n & 1);
    const float sy = ph > 1 ? (float) py * ((float) (V_GRID - 1) / (float) (ph - 1)) : 0.f;
    const float sx = pw > 1 ? (float) px * ((float) (V_GRID - 1) / (float) (pw - 1)) : 0.f;
    const int y0 = min((int) floorf(sy), V_GRID - 1), x0 = min((int) floorf(sx), V_GRID - 1);
    const int y1 = min(y0 + 1, V_GRID - 1), x1 = min(x0 + 1, V_GRID - 1);
    const float fy = sy - (float) y0, fx = sx - (float) x0;
    const float * g00 = grid + (size_t) (y0 * V_GRID + x0) * V_D, * g01 = grid + (size_t) (y0 * V_GRID + x1) * V_D;
    const float * g10 = grid + (size_t) (y1 * V_GRID + x0) * V_D, * g11 = grid + (size_t) (y1 * V_GRID + x1) * V_D;
    for (int d = threadIdx.x; d < V_D; d += 256) {
        const float a = g00[d] * (1.f - fx) + g01[d] * fx;
        const float c = g10[d] * (1.f - fx) + g11[d] * fx;
        x[(size_t) n * V_D + d] += bias[d] + a * (1.f - fy) + c * fy;
    }
}

// LayerNorm (eps 1e-6, weight + bias) of one f32 row of V_D -> f16
__global__ void __launch_bounds__(128) v_ln(const float * __restrict__ x, const float * __restrict__ w,
                                            const float * __restrict__ b, __half * __restrict__ out) {
    constexpr int PT = V_D / 128;
    __shared__ float red[2];
    const float * r = x + (size_t) blockIdx.x * V_D;
    float v[PT], s = 0.f;
    #pragma unroll
    for (int i = 0; i < PT; i++) { v[i] = r[threadIdx.x + i * 128]; s += v[i]; }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) s += __shfl_xor(s, off, 64);
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = s;
    __syncthreads();
    const float mean = (red[0] + red[1]) / V_D;
    __syncthreads();
    float q = 0.f;
    #pragma unroll
    for (int i = 0; i < PT; i++) { v[i] -= mean; q += v[i] * v[i]; }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) q += __shfl_xor(q, off, 64);
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = q;
    __syncthreads();
    const float inv = 1.f / sqrtf((red[0] + red[1]) / V_D + 1e-6f);
    #pragma unroll
    for (int i = 0; i < PT; i++) {
        const int d = threadIdx.x + i * 128;
        out[(size_t) blockIdx.x * V_D + d] = __float2half(v[i] * inv * w[d] + b[d]);
    }
}

// qkv + bias -> Q, K (2D vision rope) as f16 [nh][N][V_HD], V as f16 key pairs
// [nh][N / 2][V_HD][2]
__global__ void __launch_bounds__(256) v_qkv(const float * __restrict__ qkv, const float * __restrict__ bias,
                                             int N, int pw, int nh, __half * __restrict__ Q, __half * __restrict__ K,
                                             __half * __restrict__ V) {
    const int hd = nh * V_HD;
    const int n = blockIdx.x, bpr = pw / 2;
    const int b = n >> 2, ty = b / bpr, tx = b % bpr;
    const int py = 2 * ty + ((n >> 1) & 1), px = 2 * tx + (n & 1);
    const float * r = qkv + (size_t) n * 3 * hd;
    for (int idx = threadIdx.x; idx < nh * 36; idx += 256) {
        const int h = idx / 36, i = idx % 36;
        const int k = i < 18 ? i : i - 18;
        const float ang = (float) (i < 18 ? py : px) * powf(10000.f, -(float) (2 * k) / 36.f);
        const float cs = cosf(ang), sn = sinf(ang);
        const size_t o = ((size_t) h * N + n) * V_HD;
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            const int base = t * hd + h * V_HD;
            const float a = r[base + i] + bias[base + i], c = r[base + i + 36] + bias[base + i + 36];
            __half * dst = t == 0 ? Q : K;
            dst[o + i]      = __float2half(a * cs - c * sn);
            dst[o + i + 36] = __float2half(a * sn + c * cs);
        }
        const int vb = 2 * hd + h * V_HD;
        const size_t ov = (((size_t) h * (N / 2) + n / 2) * V_HD + i) * 2 + (n & 1);
        V[ov]      = __float2half(r[vb + i] + bias[vb + i]);
        V[ov + 72] = __float2half(r[vb + i + 36] + bias[vb + i + 36]);
    }
}

typedef _Float16 vf_h2 __attribute__((ext_vector_type(2)));
#define VF_QKP 2                  // QK: K reads issued this many steps ahead
// ViT flash attention: non-causal, head dim 72, f16 Q / K / V, f32 softmax;
// block = 128 queries of one head, 4 waves, key tiles of 16.
#define VF_QB 128
#define VF_KT 16
#define VF_KS 40                  // K row stride, dwords (9 chunks of 8 halves + a zero chunk)
#define VF_VS 96                  // V pair row stride, dwords: 8 dim groups of 12 (9 used: no bank conflicts)
#define VF_PS (VF_QB + 4)         // P pair row stride, dwords

__device__ __forceinline__ float vf_dpp1(float v) {   // lane ^ 1
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));
}
__device__ __forceinline__ float vf_dpp2(float v) {   // lane ^ 2
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false));
}
__device__ __forceinline__ uint32_t vf_pack(float a, float b) {
    return __builtin_bit_cast(uint32_t, vf_h2{ (_Float16) a, (_Float16) b });
}

__global__ void __launch_bounds__(256, 2) v_flash(const __half * __restrict__ Q, const __half * __restrict__ K,
                                                  const uint32_t * __restrict__ Vp, int N, float sl2,
                                                  __half * __restrict__ O, int ldo) {
    __shared__ __attribute__((aligned(16))) uint32_t Ks[VF_KT * VF_KS];
    __shared__ __attribute__((aligned(16))) uint32_t Vs[VF_KT / 2 * VF_VS];
    __shared__ __attribute__((aligned(16))) uint32_t Ps[VF_KT / 2 * VF_PS];
    __shared__ __attribute__((aligned(16))) float cr[VF_QB], ls[VF_QB];
    const int h = blockIdx.y, t = threadIdx.x, w = t >> 6, lane = t & 63;
    const int q0 = blockIdx.x * VF_QB, NP = (N + 1) / 2;
    const __half * Qh = Q + (size_t) h * N * V_HD, * Kh = K + (size_t) h * N * V_HD;
    const uint32_t * Vh = Vp + (size_t) h * NP * V_HD;
    if (t < VF_KT * 4) Ks[(t >> 2) * VF_KS + 36 + (t & 3)] = 0;
    const int hf = lane & 1, kg = (lane >> 1) & 1, qp = lane >> 2;
    uint32_t qv[2][20];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int m = q0 + w * 32 + qp + 16 * r;
        const uint32_t * src = (const uint32_t *) (Qh + (size_t) (m < N ? m : 0) * V_HD);
#pragma unroll
        for (int i = 0; i < 20; i++) {
            const int d2 = hf * 20 + i;
            const uint32_t x = __builtin_nontemporal_load(src + (d2 < 36 ? d2 : 35));   // unconditional: no branch per element
            qv[r][i] = m < N && d2 < 36 ? x : 0u;
        }
    }
    const int qq = lane >> 3, dg = lane & 7;
    float mx = -INFINITY, l = 0.f;   // query hf of the lane's pair
    float o[4][9];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int d = 0; d < 9; d++) o[i][d] = 0.f;
    // tile items: 144 K uint4 (16 rows x 9), then 144 V uint4 (8 pairs x 18); thread t: items t, t + 256
    uint4 nb[2];
    auto load = [&](int kt) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int it = t + u * 256;
            nb[u] = make_uint4(0, 0, 0, 0);
            if (it < 144) {
                const int row = kt + it / 9;
                if (row < N) nb[u] = ((const uint4 *) (Kh + (size_t) row * V_HD))[it % 9];
            } else if (it < 288) {
                const int v = it - 144, pr = kt / 2 + v / 18;
                if (pr < NP) nb[u] = ((const uint4 *) (Vh + (size_t) pr * V_HD))[v % 18];
            }
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int it = t + u * 256;
            if (it < 144) {
                *(uint4 *) &Ks[(it / 9) * VF_KS + (it % 9) * 4] = nb[u];
            } else if (it < 288) {
                const int v = it - 144, pr = v / 18, d0 = (v % 18) * 4;
                const uint32_t x[4] = { nb[u].x, nb[u].y, nb[u].z, nb[u].w };
#pragma unroll
                for (int e = 0; e < 4; e++) { const int d = d0 + e; Vs[pr * VF_VS + (d / 9) * 12 + d % 9] = x[e]; }
            }
        }
    };
#pragma unroll 1
    for (int kt = 0; kt < N; kt += VF_KT) {
        load(kt);
        __syncthreads();
        store();
        __syncthreads();
        float s[2][8];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int j = 0; j < 8; j++) s[r][j] = 0.f;
        {   // software-pipelined: step it = (key j, chunk c); the read of step it + VF_QKP is in flight
            constexpr int NS = 40;
            auto kaddr = [&](int it) { return (const uint4 *) &Ks[(2 * (it / 5) + kg) * VF_KS + (hf * 5 + it % 5) * 4]; };
            uint4 kb[VF_QKP + 1];
#pragma unroll
            for (int it = 0; it < VF_QKP; it++) kb[it] = *kaddr(it);
#pragma unroll
            for (int it = 0; it < NS; it++) {
                if (it + VF_QKP < NS) kb[(it + VF_QKP) % (VF_QKP + 1)] = *kaddr(it + VF_QKP);
                const uint4 k4 = kb[it % (VF_QKP + 1)];
                const int j = it / 5, c = it % 5;
                const uint32_t kk[4] = { k4.x, k4.y, k4.z, k4.w };
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const vf_h2 k2 = __builtin_bit_cast(vf_h2, kk[u]);
                    s[0][j] = __builtin_amdgcn_fdot2(__builtin_bit_cast(vf_h2, qv[0][c * 4 + u]), k2, s[0][j], false);
                    s[1][j] = __builtin_amdgcn_fdot2(__builtin_bit_cast(vf_h2, qv[1][c * 4 + u]), k2, s[1][j], false);
                }
                asm volatile("" : "+v"(s[0][j]), "+v"(s[1][j]));   // both queries' dots of this step stay together
            }
        }
        // softmax: after the dim-half sum, lane hf takes query hf (its 8 keys; the kg partner has the other 8)
        uint32_t mh = 0u - (uint32_t) hf, msk = 0u - (uint32_t) kg;
        asm volatile("" : "+v"(mh), "+v"(msk));
        float x[8], tm = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float a0 = s[0][j] + vf_dpp1(s[0][j]), a1 = s[1][j] + vf_dpp1(s[1][j]);
            const float a = __uint_as_float((__float_as_uint(a0) & ~mh) | (__float_as_uint(a1) & mh));
            x[j] = kt + 2 * j + kg < N ? a * sl2 : -INFINITY;
            tm = fmaxf(tm, x[j]);
        }
        tm = fmaxf(tm, vf_dpp2(tm));
        const float nm = fmaxf(mx, tm);
        const float corr = __builtin_amdgcn_exp2f(mx - nm);
        float ps = 0.f, p[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            p[j] = (float) (_Float16) __builtin_amdgcn_exp2f(x[j] - nm);
            ps += p[j];
        }
        ps += vf_dpp2(ps);
        l = l * corr + ps;
        mx = nm;
        if (kg == 0) cr[w * 32 + qp + 16 * hf] = corr;
#pragma unroll
        for (int i = 0; i < 4; i++) {   // pairs kg * 4 + i of query hf: keys 2j from the kg = 0 lane, 2j + 1 from kg = 1
            const uint32_t pi = __float_as_uint(p[i]), p4 = __float_as_uint(p[4 + i]);
            const uint32_t mine = (pi & ~msk) | (p4 & msk), send = (p4 & ~msk) | (pi & msk);
            const uint32_t recv = (uint32_t) __builtin_amdgcn_mov_dpp((int) send, 0x4E, 0xF, 0xF, false);
            const uint32_t lo = (mine & ~msk) | (recv & msk), hi = (recv & ~msk) | (mine & msk);
            Ps[(kg * 4 + i) * VF_PS + w * 32 + qp + 16 * hf] = vf_pack(__uint_as_float(lo), __uint_as_float(hi));
        }
        // P and cr are this wave's own rows: a wave-level fence orders the LDS writes before the reads
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        {   // PV
            const float4 c4 = *(const float4 *) &cr[w * 32 + qq * 4];
            const float cc[4] = { c4.x, c4.y, c4.z, c4.w };
            if (__any(cc[0] != 1.f || cc[1] != 1.f || cc[2] != 1.f || cc[3] != 1.f))
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int d = 0; d < 9; d++) o[i][d] *= cc[i];
#pragma unroll 4
            for (int j = 0; j < VF_KT / 2; j++) {
                const uint4 pp = *(const uint4 *) &Ps[j * VF_PS + w * 32 + qq * 4];
                const vf_h2 p2[4] = { __builtin_bit_cast(vf_h2, pp.x), __builtin_bit_cast(vf_h2, pp.y),
                                      __builtin_bit_cast(vf_h2, pp.z), __builtin_bit_cast(vf_h2, pp.w) };
                const uint32_t * vr = &Vs[j * VF_VS + dg * 12];
                const uint4 va = *(const uint4 *) vr, vb = *(const uint4 *) (vr + 4);
                const uint32_t vv[9] = { va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w, vr[8] };
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int d = 0; d < 9; d++) o[i][d] = __builtin_amdgcn_fdot2(p2[i], __builtin_bit_cast(vf_h2, vv[d]), o[i][d], false);
            }
        }
    }
    if (kg == 0) ls[w * 32 + qp + 16 * hf] = l;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = q0 + w * 32 + qq * 4 + i;
        if (m >= N) continue;
        const float il = 1.f / ls[w * 32 + qq * 4 + i];
        __half * dst = O + (size_t) m * ldo + h * V_HD + dg * 9;
#pragma unroll
        for (int d = 0; d < 9; d++) dst[d] = __float2half(o[i][d] * il);
    }
}

// two-card sum: x += (p0 + p1) + b[i % V_D], p0 / p1 card 0's / card 1's partial
// on both cards (the same order: the residual stays identical)
__global__ void __launch_bounds__(256) v_add2(float * __restrict__ x, const float * __restrict__ p0,
                                              const float * __restrict__ p1, const float * __restrict__ b, size_t n) {
    const size_t i = (size_t) blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] = x[i] + ((p0[i] + p1[i]) + b[i % V_D]);
}

// x[n][d] += b[d]
__global__ void __launch_bounds__(256) v_bias(float * __restrict__ x, const float * __restrict__ b, int d) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < d) x[(size_t) blockIdx.y * d + i] += b[i];
}

// out = gelu(y + b), tanh form, f16
__global__ void __launch_bounds__(256) v_bias_gelu(const float * __restrict__ y, const float * __restrict__ b,
                                                   int d, __half * __restrict__ out) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= d) return;
    const size_t o = (size_t) blockIdx.y * d + i;
    const float v = y[o] + b[i];
    out[o] = __float2half(0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))));
}

// --------------------------------------------------------------- host ----

namespace {

struct vlayer {
    const __half * qkv_w, * out_w, * up_w, * down_w;
    const float * qkv_b, * out_b, * up_b, * down_b, * ln1_w, * ln1_b, * ln2_w, * ln2_b;
};

// Pillow-compatible separable resampling, bicubic (a = -0.5), 22-bit fixed
// point: mtmd-image.cpp's img_tool::resize_pillow without its other filters
void resize_bicubic(const unsigned char * src, int sw, int sh, std::vector<unsigned char> & dst, int dw, int dh) {
    const int PB = 22;
    auto filt = [](double x) {
        constexpr double a = -0.5;
        if (x < 0.0) x = -x;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    auto clip8 = [](int v) -> unsigned char { return (unsigned char) (v < 0 ? 0 : v > 255 ? 255 : v); };
    auto weights = [&](int in, int out, std::vector<int> & bounds, std::vector<int32_t> & w) {
        double scale = (double) in / out, fs = scale < 1.0 ? 1.0 : scale;
        const double support = 2.0 * fs;
        const int ks = (int) std::ceil(support) * 2 + 1;
        std::vector<double> pw((size_t) out * ks);
        bounds.resize((size_t) out * 2);
        for (int xx = 0; xx < out; xx++) {
            const double center = (xx + 0.5) * scale, ss = 1.0 / fs;
            double ww = 0.0;
            int xmin = (int) (center - support + 0.5); if (xmin < 0) xmin = 0;
            int xmax = (int) (center + support + 0.5); if (xmax > in) xmax = in;
            xmax -= xmin;
            int x = 0;
            for (; x < xmax; x++) { const double v = filt((x + xmin - center + 0.5) * ss); pw[(size_t) xx * ks + x] = v; ww += v; }
            for (x = 0; x < xmax; x++) if (ww != 0.0) pw[(size_t) xx * ks + x] /= ww;
            for (; x < ks; x++) pw[(size_t) xx * ks + x] = 0;
            bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
        }
        w.resize((size_t) out * ks);
        const double fx = std::ldexp(1.0, PB);
        for (size_t i = 0; i < w.size(); i++) w[i] = (int32_t) (pw[i] * fx + (pw[i] < 0 ? -0.5 : 0.5));
        return ks;
    };
    std::vector<unsigned char> cur(src, src + (size_t) sw * sh * 3);
    int cw = sw;
    if (dw != sw) {
        std::vector<int> bd; std::vector<int32_t> k;
        const int ks = weights(sw, dw, bd, k);
        std::vector<unsigned char> o((size_t) dw * sh * 3);
        for (int yy = 0; yy < sh; yy++)
            for (int xx = 0; xx < dw; xx++) {
                const int xmin = bd[xx * 2], xc = bd[xx * 2 + 1];
                const int32_t * kk = &k[(size_t) xx * ks];
                const unsigned char * p = &cur[((size_t) yy * cw + xmin) * 3];
                int32_t s0 = 1 << (PB - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xc; x++, p += 3) { s0 += p[0] * kk[x]; s1 += p[1] * kk[x]; s2 += p[2] * kk[x]; }
                unsigned char * q = &o[((size_t) yy * dw + xx) * 3];
                q[0] = clip8(s0 >> PB); q[1] = clip8(s1 >> PB); q[2] = clip8(s2 >> PB);
            }
        cur.swap(o); cw = dw;
    }
    if (dh != sh) {
        std::vector<int> bd; std::vector<int32_t> k;
        const int ks = weights(sh, dh, bd, k);
        const size_t re = (size_t) cw * 3;
        std::vector<unsigned char> o(re * dh);
        std::vector<int32_t> acc(re);
        for (int yy = 0; yy < dh; yy++) {
            const int ymin = bd[yy * 2], yc = bd[yy * 2 + 1];
            const int32_t * kk = &k[(size_t) yy * ks];
            std::fill(acc.begin(), acc.end(), 1 << (PB - 1));
            for (int y = 0; y < yc; y++) {
                const unsigned char * r = &cur[(size_t) (ymin + y) * re];
                for (size_t i = 0; i < re; i++) acc[i] += r[i] * kk[y];
            }
            for (size_t i = 0; i < re; i++) o[(size_t) yy * re + i] = clip8(acc[i] >> PB);
        }
        cur.swap(o);
    }
    dst.swap(cur);
}

} // namespace

// one card's half: heads [8c, 8c + 8), FFN rows [c ff/2, (c + 1) ff/2); out /
// down run over the matching input halves and the partials meet in an exchange
struct vcard {
    int dev = -1;
    hipStream_t st = nullptr;
    hipEvent_t ev = nullptr;
    rocblas_handle blas = nullptr;
    std::vector<vlayer> L;                 // this card's weight slices
    std::vector<void *> own;               // slices / copies this card allocated
    const __half * patch_w0 = nullptr, * patch_w1 = nullptr;
    const float * patch_b = nullptr, * pos = nullptr;
    // scratch, sized for np_max patches
    unsigned char * rgb = nullptr;
    size_t rgb_cap = 0;
    __half * P = nullptr, * h = nullptr, * Q = nullptr, * K = nullptr, * V = nullptr, * O = nullptr, * G = nullptr;
    float * x = nullptr, * y = nullptr, * part = nullptr, * inbox[2] = {};   // inbox: the peer's partials (uncached)
};

struct qz_vision {
    int xk = 0;                            // inbox slot of the next exchange
    vcard C[2];
    qz_loader ld;                          // the file's tensors on card 0
    int n_out = 0, min_tok = 8, max_tok = 4096, np_max = 0, ff = 0;
    // merger (card 0)
    const __half * mm0_w = nullptr, * mm2_w = nullptr;
    const float * post_w = nullptr, * post_b = nullptr, * mm0_b = nullptr, * mm2_b = nullptr;
    float * out = nullptr;
    int n_tok = 0;
    double ms = 0;
};

// Y[n][m] (+)= X[n][k] . W[m][k]^T, W f16 [m][k], X f16, Y f32 or f16
static int vgemm(rocblas_handle blas, const __half * W, const __half * X, void * Y, bool y16, int m, int n, int k, float beta) {
    const float alpha = 1.f;
    const rocblas_datatype yt = y16 ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
    VBLAS(rocblas_gemm_ex(blas, rocblas_operation_transpose, rocblas_operation_none, m, n, k, &alpha,
                          W, rocblas_datatype_f16_r, k, X, rocblas_datatype_f16_r, k, &beta,
                          Y, yt, m, Y, yt, m, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    return 0;
}

static void grid_of(const qz_vision * v, int w, int h, int * nx, int * ny) {
    // calc_size_preserved_ratio (mtmd-image.cpp): align 32, round half even, min / max pixels
    const int f = 32;
    const int minp = v->min_tok * 1024, maxp = v->max_tok * 1024;
    auto rnd = [&](float x) { return (int) std::nearbyint(x / (float) f) * f; };
    auto cl = [&](float x) { return (int) std::ceil(x / (float) f) * f; };
    auto fl = [&](float x) { return (int) std::floor(x / (float) f) * f; };
    int wb = std::max(f, rnd((float) w)), hb = std::max(f, rnd((float) h));
    if ((int64_t) hb * wb > maxp) {
        const float beta = std::sqrt((float) h * w / maxp);
        hb = std::max(f, fl(h / beta)); wb = std::max(f, fl(w / beta));
    } else if ((int64_t) hb * wb < minp) {
        const float beta = std::sqrt((float) minp / ((float) h * w));
        hb = cl(h * beta); wb = cl(w * beta);
    }
    *nx = wb / f; *ny = hb / f;
}

extern "C" {

qz_vision * qz_vision_load(const char * path) {
    if (qz_dev_select()) return nullptr;
    const int dev = qz_dev_of(0), dev1 = qz_dev_of(1);
    qz_gguf g;
    if (!g.open(path)) { fprintf(stderr, "quartz_vision: cannot open %s\n", path); return nullptr; }
    if (g.str("clip.projector_type") != "qwen3vl_merger" || g.u32("clip.vision.embedding_length") != V_D ||
        g.u32("clip.vision.attention.head_count") != V_H || g.u32("clip.vision.patch_size") != V_PATCH ||
        g.u32("clip.vision.spatial_merge_size", 2) != 2) {
        fprintf(stderr, "quartz_vision: %s is not a supported qwen3vl_merger projector\n", path); return nullptr;
    }
    const int nl = (int) g.u32("clip.vision.block_count");
    qz_vision * v = new qz_vision;
    v->C[0].dev = dev;
    v->C[1].dev = dev1;
    if (getenv("QZ_IMG_MIN_TOKENS")) v->min_tok = std::max(1, atoi(getenv("QZ_IMG_MIN_TOKENS")));
    if (getenv("QZ_IMG_MAX_TOKENS")) v->max_tok = std::max(v->min_tok, atoi(getenv("QZ_IMG_MAX_TOKENS")));
    v->np_max = v->max_tok * 4;
    auto fail = [&](const char * what) { fprintf(stderr, "quartz_vision: %s\n", what); qz_vision_free(v); return (qz_vision *) nullptr; };
    // the layer matrices are sliced from the file (host) onto each card, not loaded whole
    static const char * const BIG[6] = { "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight" };
    v->ld.skip = [](const char * n) {
            if (strncmp(n, "v.blk.", 6)) return false;
            const char * d = strchr(n + 6, '.');
            for (const char * b : BIG) if (d && !strcmp(d + 1, b)) return true;
            return false;
        };
    {
        qz_mem_scope mem("vision");
        if (hipSetDevice(dev) != hipSuccess || !v->ld.load(path, g, dev)) return fail("weights failed to load");
    }
    auto tix = [&](const char * n, ggml_type t) { const int64_t i = v->ld.find(n); return i >= 0 && v->ld.ts[i].type == t ? i : -1; };
    auto f16 = [&](const char * n) { const int64_t i = tix(n, GGML_TYPE_F16); return i >= 0 ? (const __half *) v->ld.ts[i].vals : nullptr; };
    auto f32 = [&](const char * n) { const int64_t i = tix(n, GGML_TYPE_F32); return i >= 0 ? (const float *) v->ld.ts[i].vals : nullptr; };
    vcard & c0 = v->C[0];
    c0.patch_w0 = f16("v.patch_embd.weight"); c0.patch_w1 = f16("v.patch_embd.weight.1"); c0.patch_b = f32("v.patch_embd.bias");
    c0.pos = f32("v.position_embd.weight"); v->post_w = f32("v.post_ln.weight"); v->post_b = f32("v.post_ln.bias");
    v->mm0_w = f16("mm.0.weight"); v->mm0_b = f32("mm.0.bias"); v->mm2_w = f16("mm.2.weight"); v->mm2_b = f32("mm.2.bias");
    const int64_t i2 = v->ld.find("mm.2.weight"), iu = v->ld.find("v.blk.0.ffn_up.weight");
    if (!c0.patch_w0 || !c0.patch_w1 || !c0.patch_b || !c0.pos || !v->post_w || !v->post_b || !v->mm0_w || !v->mm0_b ||
        !v->mm2_w || !v->mm2_b || i2 < 0 || v->ld.ts[i2].ne0 != 4 * V_D ||
        v->ld.ts[v->ld.find("v.position_embd.weight")].ne1 != V_GRID * V_GRID) return fail("missing or unexpected tensors");
    v->n_out = (int) v->ld.ts[i2].ne1;
    if (iu < 0) return fail("missing layer tensors");
    v->ff = (int) v->ld.ts[iu].ne1;
    if (v->ff % 2) return fail("odd FFN width: no two-card split");
    char nm[64];
    const int nc = 2, nh = V_H / nc, hd = nh * V_HD, ffc = v->ff / nc;
    for (int l = 0; l < nl; l++) {
        auto nmf = [&](const char * s) { snprintf(nm, sizeof nm, "v.blk.%d.%s", l, s); return nm; };
        vlayer y{};
        y.out_b = f32(nmf("attn_out.bias")); y.down_b = f32(nmf("ffn_down.bias"));
        y.ln1_w = f32(nmf("ln1.weight")); y.ln1_b = f32(nmf("ln1.bias")); y.ln2_w = f32(nmf("ln2.weight")); y.ln2_b = f32(nmf("ln2.bias"));
        bool big_ok = true;
        for (int b = 0; b < 6; b++) big_ok = big_ok && tix(nmf(BIG[b]), b == 1 || b == 4 ? GGML_TYPE_F32 : GGML_TYPE_F16) >= 0;
        if (!big_ok || !y.out_b || !y.down_b || !y.ln1_w || !y.ln1_b || !y.ln2_w || !y.ln2_b) return fail("missing layer tensors");
        c0.L.push_back(y);
    }
    {                                      // the per-card slices
        for (int c = 0; c < 2; c++) {          // peer access both ways (the engine may have enabled it already)
            if (hipSetDevice(v->C[c].dev) != hipSuccess) return fail("bad device");
            const hipError_t e = hipDeviceEnablePeerAccess(v->C[1 - c].dev, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) return fail("no peer access between the cards");
            (void) hipGetLastError();
        }
        FILE * fp = fopen(path, "rb");
        if (!fp) return fail("cannot reopen the file");
        bool ok = true;
        auto al = [&](vcard & C, size_t bytes) -> void * {
            void * p = nullptr;
            if (!ok || hipSetDevice(C.dev) != hipSuccess) { ok = false; return nullptr; }
            qz_mem_scope mem("vision");
            if (qz_dmalloc(&p, bytes, "weights") != hipSuccess) { ok = false; return nullptr; }
            C.own.push_back(p);
            return p;
        };
        auto cp = [&](void * d, const void * src, size_t bytes) { ok = ok && hipMemcpy(d, src, bytes, hipMemcpyDefault) == hipSuccess; };
        auto cp2 = [&](void * d, size_t dp, const void * src, size_t sp, size_t w, size_t rows) {
            ok = ok && hipMemcpy2D(d, dp, src, sp, w, rows, hipMemcpyDefault) == hipSuccess;
        };
        vcard & c1 = v->C[1];
        auto dup = [&](const float * src, size_t n) -> const float * {   // a small f32 vector of card 0 on card 1
            float * d = (float *) al(c1, n * 4);
            if (d) cp(d, src, n * 4);
            return d;
        };
        std::vector<char> hb[6];
        for (int l = 0; l < nl && ok; l++) {
            for (int b = 0; b < 6 && ok; b++) {    // this layer's matrices from the file
                snprintf(nm, sizeof nm, "v.blk.%d.%s", l, BIG[b]);
                const int64_t i = v->ld.find(nm);
                hb[b].resize((size_t) v->ld.ts[i].bytes());
                ok = fseeko(fp, (off_t) v->ld.raw_off[i], SEEK_SET) == 0 && fread(hb[b].data(), 1, hb[b].size(), fp) == hb[b].size();
            }
            const __half * qw = (const __half *) hb[0].data(), * ow = (const __half *) hb[2].data(),
                         * uw = (const __half *) hb[3].data(), * dw = (const __half *) hb[5].data();
            const float * qb = (const float *) hb[1].data(), * ub = (const float *) hb[4].data();
            for (int c = 0; c < 2 && ok; c++) {
                vcard & C = v->C[c];
                if (c == 1) {
                    const vlayer & f = c0.L[l];
                    vlayer y{};
                    y.out_b = dup(f.out_b, V_D); y.down_b = dup(f.down_b, V_D);
                    y.ln1_w = dup(f.ln1_w, V_D); y.ln1_b = dup(f.ln1_b, V_D); y.ln2_w = dup(f.ln2_w, V_D); y.ln2_b = dup(f.ln2_b, V_D);
                    C.L.push_back(y);
                }
                vlayer & y = C.L[l];
                __half * qkv = (__half *) al(C, (size_t) 3 * hd * V_D * 2), * o = (__half *) al(C, (size_t) V_D * hd * 2),
                       * u = (__half *) al(C, (size_t) ffc * V_D * 2), * d = (__half *) al(C, (size_t) V_D * ffc * 2);
                float * qkvb = (float *) al(C, (size_t) 3 * hd * 4), * ubs = (float *) al(C, (size_t) ffc * 4);
                if (!ok) break;
                for (int t = 0; t < 3; t++) {      // Q, K, V rows of this card's heads
                    const size_t r0 = (size_t) t * V_D + (size_t) c * hd;
                    cp(qkv + (size_t) t * hd * V_D, qw + r0 * V_D, (size_t) hd * V_D * 2);
                    cp(qkvb + (size_t) t * hd, qb + r0, (size_t) hd * 4);
                }
                cp2(o, (size_t) hd * 2, ow + (size_t) c * hd, (size_t) V_D * 2, (size_t) hd * 2, V_D);
                cp(u, uw + (size_t) c * ffc * V_D, (size_t) ffc * V_D * 2);
                cp(ubs, ub + (size_t) c * ffc, (size_t) ffc * 4);
                cp2(d, (size_t) ffc * 2, dw + (size_t) c * ffc, (size_t) v->ff * 2, (size_t) ffc * 2, V_D);
                y.qkv_w = qkv; y.qkv_b = qkvb; y.out_w = o; y.up_w = u; y.up_b = ubs; y.down_w = d;
            }
        }
        fclose(fp);
        if (ok) {
            __half * w0 = (__half *) al(c1, (size_t) V_D * V_K * 2), * w1 = (__half *) al(c1, (size_t) V_D * V_K * 2);
            if (ok) { cp(w0, c0.patch_w0, (size_t) V_D * V_K * 2); cp(w1, c0.patch_w1, (size_t) V_D * V_K * 2); }
            c1.patch_w0 = w0; c1.patch_w1 = w1;
            c1.patch_b = dup(c0.patch_b, V_D);
            c1.pos = dup(c0.pos, (size_t) V_GRID * V_GRID * V_D);
        }
        if (!ok) return fail("weight slices failed");
    }
    const size_t np = v->np_max, wide = std::max<size_t>(std::max<size_t>(3 * hd, ffc), V_D);   // merger: T x 4 V_D = np x V_D
    bool ok = true;
    for (int c = 0; c < nc && ok; c++) {
        vcard & C = v->C[c];
        ok = hipSetDevice(C.dev) == hipSuccess && hipStreamCreateWithFlags(&C.st, hipStreamNonBlocking) == hipSuccess
          && hipEventCreateWithFlags(&C.ev, hipEventDisableTiming) == hipSuccess;
        qz_mem_scope mem("vision");
        ok = ok && qz_dmalloc(&C.P, np * V_K * 2, "act") == hipSuccess
                && qz_dmalloc(&C.x, np * V_D * 4, "act") == hipSuccess
                && qz_dmalloc(&C.h, np * V_D * 2, "act") == hipSuccess
                && qz_dmalloc(&C.y, np * wide * 4, "act") == hipSuccess
                && qz_dmalloc(&C.G, np * wide * 2, "act") == hipSuccess
                && qz_dmalloc(&C.Q, np * hd * 2, "act") == hipSuccess
                && qz_dmalloc(&C.K, np * hd * 2, "act") == hipSuccess
                && qz_dmalloc(&C.V, np * hd * 2, "act") == hipSuccess
                && qz_dmalloc(&C.O, np * hd * 2, "act") == hipSuccess;
        ok = ok && qz_dmalloc(&C.part, np * V_D * 4, "exchange") == hipSuccess
                    && qz_dmalloc_uncached(&C.inbox[0], np * V_D * 4, "exchange") == hipSuccess
                    && qz_dmalloc_uncached(&C.inbox[1], np * V_D * 4, "exchange") == hipSuccess;
        if (c == 0) ok = ok && qz_dmalloc(&v->out, (size_t) v->max_tok * v->n_out * 4, "out") == hipSuccess;
        ok = ok && rocblas_create_handle(&C.blas) == rocblas_status_success && rocblas_set_stream(C.blas, C.st) == rocblas_status_success;
    }
    if (!ok) return fail("scratch allocation failed");
    (void) hipSetDevice(dev);
    return v;
}

int qz_vision_n_embd(const qz_vision * v) { return v->n_out; }
double qz_vision_last_ms(const qz_vision * v) { return v->ms; }

int qz_vision_copy_rows(const qz_vision * v, float * host) {
    VCHK(hipSetDevice(v->C[0].dev));
    VCHK(hipMemcpy(host, v->out, (size_t) v->n_tok * v->n_out * 4, hipMemcpyDeviceToHost));
    return v->n_tok;
}

int qz_vision_encode_rgb(qz_vision * v, const unsigned char * rgb, int w, int h, int * nx_out, int * ny_out) {
    if (w <= 0 || h <= 0) return -1;
    int nx, ny;
    grid_of(v, w, h, &nx, &ny);
    if (nx * ny > v->max_tok) { fprintf(stderr, "quartz_vision: %dx%d tokens over the limit\n", nx, ny); return -1; }
    const auto t0 = std::chrono::steady_clock::now();
    const int W = nx * 32, H = ny * 32, pw = W / V_PATCH, ph = H / V_PATCH, N = pw * ph, T = N / 4;
    std::vector<unsigned char> img;
    if (W != w || H != h) resize_bicubic(rgb, w, h, img, W, H);
    else img.assign(rgb, rgb + (size_t) w * h * 3);
    const int nc = 2, nh = V_H / nc, hd = nh * V_HD, ffc = v->ff / nc;
    const float sl2 = 1.4426950408889634f / std::sqrt((float) V_HD);   // softmax scale, base 2
    int err = 0;
    auto each = [&](auto && fn) {
        for (int c = 0; c < nc && !err; c++) {
            vcard & C = v->C[c];
            if (hipSetDevice(C.dev) != hipSuccess) { err = 1; return; }
            err = fn(c, C, C.st);
        }
    };
    // two cards: the partials in part -> x += (p0 + p1) + b on both
    auto exchange = [&](auto bias) {
        each([&](int c, vcard & C, hipStream_t s) {
            VCHK(hipMemcpyPeerAsync(v->C[1 - c].inbox[v->xk], v->C[1 - c].dev, C.part, C.dev, (size_t) N * V_D * 4, s));
            VCHK(hipEventRecord(C.ev, s));
            return 0;
        });
        each([&](int c, vcard & C, hipStream_t s) {
            VCHK(hipStreamWaitEvent(s, v->C[1 - c].ev, 0));
            const float * in = C.inbox[v->xk];
            v_add2<<<((size_t) N * V_D + 255) / 256, 256, 0, s>>>(C.x, c == 0 ? C.part : in, c == 0 ? in : C.part, bias(C), (size_t) N * V_D);
            return 0;
        });
        v->xk ^= 1;
    };
    each([&](int, vcard & C, hipStream_t s) {
        if (img.size() > C.rgb_cap) {
            if (C.rgb) (void) qz_dfree(C.rgb);
            qz_mem_scope mem("vision");
            VCHK(qz_dmalloc(&C.rgb, img.size(), "image"));
            C.rgb_cap = img.size();
        }
        VCHK(hipMemcpyAsync(C.rgb, img.data(), img.size(), hipMemcpyHostToDevice, s));
        // patch embedding (on each card): both temporal kernels see the still image
        v_patchify<<<N, 256, 0, s>>>(C.rgb, W, C.P);
        if (vgemm(C.blas, C.patch_w0, C.P, C.x, false, V_D, N, V_K, 0.f) || vgemm(C.blas, C.patch_w1, C.P, C.x, false, V_D, N, V_K, 1.f)) return -1;
        v_pos<<<N, 256, 0, s>>>(C.x, C.patch_b, C.pos, pw, ph);
        return 0;
    });
    for (size_t il = 0; il < v->C[0].L.size() && !err; il++) {
        each([&](int, vcard & C, hipStream_t s) {
            const vlayer & l = C.L[il];
            v_ln<<<N, 128, 0, s>>>(C.x, l.ln1_w, l.ln1_b, C.h);
            if (vgemm(C.blas, l.qkv_w, C.h, C.y, false, 3 * hd, N, V_D, 0.f)) return -1;
            v_qkv<<<N, 256, 0, s>>>(C.y, l.qkv_b, N, pw, nh, C.Q, C.K, C.V);
            v_flash<<<dim3((N + VF_QB - 1) / VF_QB, nh), 256, 0, s>>>(C.Q, C.K, (const uint32_t *) C.V, N, sl2, C.O, hd);
            if (vgemm(C.blas, l.out_w, C.O, C.part, false, V_D, N, hd, 0.f)) return -1;
            return 0;
        });
        exchange([&](vcard & C) { return C.L[il].out_b; });
        each([&](int, vcard & C, hipStream_t s) {
            const vlayer & l = C.L[il];
            v_ln<<<N, 128, 0, s>>>(C.x, l.ln2_w, l.ln2_b, C.h);
            if (vgemm(C.blas, l.up_w, C.h, C.y, false, ffc, N, V_D, 0.f)) return -1;
            v_bias_gelu<<<dim3((ffc + 255) / 256, N), 256, 0, s>>>(C.y, l.up_b, ffc, C.G);
            if (vgemm(C.blas, l.down_w, C.G, C.part, false, V_D, N, ffc, 0.f)) return -1;
            return 0;
        });
        exchange([&](vcard & C) { return C.L[il].down_b; });
    }
    if (err) return -1;
    // merger (card 0): post LN, 4 patches (one 2x2 block) per row, mm.0 + GELU, mm.2
    vcard & C = v->C[0];
    hipStream_t s = C.st;
    VCHK(hipSetDevice(C.dev));
    v_ln<<<N, 128, 0, s>>>(C.x, v->post_w, v->post_b, C.h);
    if (vgemm(C.blas, v->mm0_w, C.h, C.y, false, 4 * V_D, T, 4 * V_D, 0.f)) return -1;
    v_bias_gelu<<<dim3((4 * V_D + 255) / 256, T), 256, 0, s>>>(C.y, v->mm0_b, 4 * V_D, C.G);
    if (vgemm(C.blas, v->mm2_w, C.G, v->out, false, v->n_out, T, 4 * V_D, 0.f)) return -1;
    v_bias<<<dim3((v->n_out + 255) / 256, T), 256, 0, s>>>(v->out, v->mm2_b, v->n_out);
    VCHK(hipGetLastError());
    for (int c = nc - 1; c >= 0; c--) { VCHK(hipSetDevice(v->C[c].dev)); VCHK(hipStreamSynchronize(v->C[c].st)); }
    v->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    v->n_tok = T;
    if (nx_out) *nx_out = nx;
    if (ny_out) *ny_out = ny;
    return T;
}

int qz_vision_encode(qz_vision * v, const void * bytes, size_t n, int * nx, int * ny) {
    int w = 0, h = 0, c = 0;
    unsigned char * px = stbi_load_from_memory((const stbi_uc *) bytes, (int) n, &w, &h, &c, 3);
    if (!px) { fprintf(stderr, "quartz_vision: cannot decode the image (%s)\n", stbi_failure_reason()); return -1; }
    const int r = qz_vision_encode_rgb(v, px, w, h, nx, ny);
    stbi_image_free(px);
    return r;
}

void qz_vision_free(qz_vision * v) {
    if (!v) return;
    for (int c = 1; c >= 0; c--) {
        vcard & C = v->C[c];
        if (C.dev < 0 || hipSetDevice(C.dev) != hipSuccess) continue;
        if (C.st) (void) hipStreamSynchronize(C.st);
        for (void * p : { (void *) C.rgb, (void *) C.P, (void *) C.x, (void *) C.h, (void *) C.y, (void *) C.G, (void *) C.Q,
                          (void *) C.K, (void *) C.V, (void *) C.O, (void *) C.part, (void *) C.inbox[0], (void *) C.inbox[1],
                          c == 0 ? (void *) v->out : nullptr })
            if (p) (void) qz_dfree(p);
        for (void * p : C.own) (void) qz_dfree(p);
        if (c == 0) for (auto & t : v->ld.ts) if (!t.shared && t.vals) (void) qz_dfree(t.vals);
        if (C.blas) rocblas_destroy_handle(C.blas);
        if (C.ev) (void) hipEventDestroy(C.ev);
        if (C.st) (void) hipStreamDestroy(C.st);
    }
    delete v;
}

} // extern "C"
