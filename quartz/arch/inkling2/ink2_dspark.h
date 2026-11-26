// inkling2 DSpark drafter: the GGUF "dflash" arch with the DSpark recurrent Markov head.
// Three DFlash layers run DS_BS = 8 rows, the anchor token's normed target embedding and
// 7 mask rows. The rows attend non-causally to each other and to a per-layer context K / V
// cache. A position enters that cache through 4 target features: the residual entering
// applications 5, 23 and 41 and the head's normed input. A layer mixes them by
// softmax(dspark_fuse) then RMSNorm (dflare_norm; dspark_norm for the Markov context g).
// The Markov head chains the round's drafts. joint_proj maps
// [state | markov_w1[prev] | hidden_i | g of the anchor - 1] to gate, cand and latent. A
// draft's logits are the target head on hidden_i + hidden_out(latent), one 7-row head
// GEMV plus fold . [latent; 1]. fold = head . [hidden_out | bias] is built at load, per
// card, over its vocab half.
// The layers split over the two cards as the target's do (ink2_ds_tp2_spec). Card c owns
// 12 q heads, 4 kv heads with their context K / V cache, and half the FFN. The attn_output
// and ffn_down partials are exchanged and summed on both cards, card 0's first, so the
// residual stays replicated.
// The Markov steps run as one device-side chain with no host round trip. Card 0's joint
// kernel waits for the peer's argmax through peer memory flags, and card 1's fold waits for
// card 0's lat1 the same way. Only the round's tokens come back.
// Included after ink2_engine.h / ink2_pf.h (same TU).
#include "core/link.h"           // qz_xcopy / qz_xdst: the partials and the feature planes

constexpr int DS_BS = 8, DS_NL = 3, DS_R = 256, DS_FF = 9216, DS_Q = 3072, DS_KV = 1024, DS_NQ = 24, DS_NKV = 8;
constexpr int DS_NQL = DS_NQ / 2, DS_NKVL = DS_NKV / 2, DS_QL = DS_Q / 2, DS_KVL = DS_KV / 2, DS_FFL = DS_FF / 2;   // per card
constexpr int DS_NT = 256;                     // the drafter GEMV's block (4 waves, 4 rows each)
constexpr float DS_EPS = 1e-6f, DS_THETA = 1e6f;
constexpr long long DS_SPIN = 2000000000ll;    // clock64 ticks a cross-card wait may spin

// ---- the Markov chain's cross-card handshake, over peer memory at system scope
__device__ __forceinline__ void ds_flag(unsigned int * p, unsigned int seq) {
    __threadfence_system();
    __hip_atomic_store(p, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
// wait for the peer's flag >= seq. Past DS_SPIN it sets err and returns.
__device__ __forceinline__ void ds_wait(unsigned int * p, unsigned int seq, int * err) {
    const long long t0 = clock64();
    for (;;) {
        const unsigned int v = __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if ((int) (v - seq) >= 0) return;
        if (err && __hip_atomic_load(err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) return;
        if (clock64() - t0 > DS_SPIN) { if (err) atomicCAS(err, 0, 1); return; }
    }
}

// ---- kernels
// out rows = RMSNorm(sum_k a_k tap_k) * w, taps [row][4][D] with row stride tstride
static __global__ __launch_bounds__(256) void k_ds_mixnorm(const float * taps, size_t tstride, float4 a, const float * w,
                                                           float * out, int ostride, float eps) {
    __shared__ float red[16];
    const float * t = taps + (size_t) blockIdx.x * tstride;
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        v[k] = a.x * t[i] + a.y * t[D + i] + a.z * t[2 * D + i] + a.w * t[3 * D + i];
        ss[0] += v[k] * v[k];
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        out[(size_t) blockIdx.x * ostride + i] = v[k] * r * w[i];
    }
}
// the same rows, written straight to int8 with per-32 scales (row stride D)
static __global__ __launch_bounds__(256) void k_ds_mixq(const float * taps, size_t tstride, float4 a, const float * w,
                                                        unsigned char * q, float * qs, float eps) {
    __shared__ float red[16];
    const float * t = taps + (size_t) blockIdx.x * tstride;
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        v[k] = a.x * t[i] + a.y * t[D + i] + a.z * t[2 * D + i] + a.w * t[3 * D + i];
        ss[0] += v[k] * v[k];
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        quant32(v[k] * r * w[i], q + (size_t) blockIdx.x * D, qs + (size_t) blockIdx.x * (D / 32), i);
    }
}
// batched prefill mix + RMSNorm into grouped int8 planes [D/128][m][128] for k_pf_dgemm
static __global__ __launch_bounds__(256) void k_ds_mixq_pf(const float * taps, size_t tstride, float4 a, const float * w,
                                                           int8_t * q, float * qs, int m, float eps) {
    __shared__ float red[16];
    const int row = blockIdx.x;
    const float * t = taps + (size_t) row * tstride;
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        v[k] = a.x * t[i] + a.y * t[D + i] + a.z * t[2 * D + i] + a.w * t[3 * D + i];
        ss[0] += v[k] * v[k];
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        pf_quant_unit(v[k] * r * w[i], i, row, m, q, qs, threadIdx.x & 31);
    }
}
// rows of x (row stride K) to int8 with per-32 scales. When w is non-null, each row is
// RMSNormed by it first. With y, the normed f32 rows are also written.
static __global__ __launch_bounds__(256) void k_ds_quant(const float * x, const float * w, int K, unsigned char * q, float * qs,
                                                         float * y, float eps) {
    __shared__ float red[16];
    const float * xr = x + (size_t) blockIdx.x * K;
    float r = 1.f;
    if (w) {
        float ss[1] = {0.f};
        for (int i = threadIdx.x; i < K; i += 256) ss[0] += xr[i] * xr[i];
        bsum<1, false>(ss, red, 4);
        r = rsqrtf(ss[0] / K + eps);
    }
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = w ? xr[i] * r * w[i] : xr[i];
        if (y) y[(size_t) blockIdx.x * K + i] = v;
        quant32(v, q + (size_t) blockIdx.x * K, qs + (size_t) blockIdx.x * (K / 32), i);
    }
}
// x += a + b, the two cards' partials with card 0's first on both cards, then the row's
// int8 planes from RMSNorm(x) * w. One kernel does the residual add and the quantization.
static __global__ __launch_bounds__(256) void k_ds_add_quant(float * x, const float * a, const float * b, const float * w, int K,
                                                             unsigned char * q, float * qs, float eps) {
    __shared__ float red[16];
    float * xr = x + (size_t) blockIdx.x * K;
    const float * ar = a + (size_t) blockIdx.x * K, * br = b + (size_t) blockIdx.x * K;
    float ss[1] = {0.f};
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = xr[i] + (ar[i] + br[i]);
        xr[i] = v;
        ss[0] += v * v;
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / K + eps);
    for (int i = threadIdx.x; i < K; i += 256) quant32(xr[i] * r * w[i], q + (size_t) blockIdx.x * K, qs + (size_t) blockIdx.x * (K / 32), i);
}
// g = silu(g) * u, then its int8 planes.
static __global__ __launch_bounds__(256) void k_ds_swiglu_quant(float * g, const float * u, int K, unsigned char * q, float * qs) {
    float * gr = g + (size_t) blockIdx.y * K;
    const float * ur = u + (size_t) blockIdx.y * K;
    {
        const int i = blockIdx.x * 256 + threadIdx.x;
        const float x = gr[i];
        const float v = x / (1.f + expf(-x)) * ur[i];
        gr[i] = v;
        quant32(v, q + (size_t) blockIdx.y * K, qs + (size_t) blockIdx.y * (K / 32), i);
    }
}
// T-row Q8_0 GEMV, y = [T][N], 4 rows per wave. Accumulates like ink2_verify's v_core_acc
// (units past 64 spread over the wave), then sums with the v_tsum butterfly.
template <int T, int NT = DS_NT>
static __global__ __launch_bounds__(NT) void k_ds_gemv(const unsigned char * aq, const float * as, int K,
                                                       const char * wv, const __half * ws, int N, float * y) {
    extern __shared__ char lds[];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tid = threadIdx.x;
    const int n_u = K / 32;
    unsigned char * la = (unsigned char *) lds;                 // [T][K]
    float * ls = (float *) (lds + (size_t) T * K);              // [T][n_u]
    for (int i = tid; i < T * K / 16; i += NT) ((uint4 *) la)[i] = ((const uint4 *) aq)[i];
    for (int i = tid; i < T * n_u; i += NT) ls[i] = as[i];
    __syncthreads();
    const int S = 4;
    constexpr int TP = T <= 1 ? 1 : (T <= 2 ? 2 : (T <= 4 ? 4 : 8)), NV = TP * 4;
    static_assert(T <= TP, "T <= 8");
    for (int g = blockIdx.x; g * (NT / 64 * S) < N; g += gridDim.x) {
        const int r0 = g * (NT / 64 * S) + wave * S;
        float acc[T][4];
        v_core_acc<T, 4>(la, ls, K, n_u, n_u, wv, ws, K, n_u, N, r0, lane, acc);
        float fv[NV];
        #pragma unroll
        for (int t = 0; t < TP; t++)
            #pragma unroll
            for (int q = 0; q < 4; q++) fv[t * 4 + q] = t < T ? acc[t < T ? t : 0][q] : 0.f;
        const float val = v_tsum<NV>(fv, lane);
        const int kv = v_tsum_k<NV>(lane), vt = kv / 4, vs = kv % 4;
        if ((lane & (64 / NV - 1)) == 0 && vt < T && r0 + vs < N) y[(size_t) vt * N + r0 + vs] = val;
    }
}
// Batched 2- or 3-projection GEMV sharing one activation staging in LDS (q|k|v and gate|up)
struct ds_gjob { const char * wv; const __half * ws; float * y; int N, g1; };
struct ds_gjobs { ds_gjob j[3]; int n; };
template <int T, int NT = DS_NT>
static __global__ __launch_bounds__(NT) void k_ds_gemv_batch(const unsigned char * aq, const float * as, int K,
                                                             ds_gjobs J) {
    extern __shared__ char lds[];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tid = threadIdx.x;
    const int n_u = K / 32;
    unsigned char * la = (unsigned char *) lds;
    float * ls = (float *) (lds + (size_t) T * K);
    for (int i = tid; i < T * K / 16; i += NT) ((uint4 *) la)[i] = ((const uint4 *) aq)[i];
    for (int i = tid; i < T * n_u; i += NT) ls[i] = as[i];
    __syncthreads();
    const int S = 4, total_g = J.j[J.n - 1].g1;
    constexpr int TP = T <= 1 ? 1 : (T <= 2 ? 2 : (T <= 4 ? 4 : 8)), NV = TP * 4;
    for (int b = blockIdx.x; b < total_g; b += gridDim.x) {
        int k = 0;
        while (k + 1 < J.n && b >= J.j[k].g1) k++;
        const ds_gjob & j = J.j[k];
        const int g = b - (k ? J.j[k - 1].g1 : 0);
        const int r0 = g * (NT / 64 * S) + wave * S;
        float acc[T][4];
        v_core_acc<T, 4>(la, ls, K, n_u, n_u, j.wv, j.ws, K, n_u, j.N, r0, lane, acc);
        float fv[NV];
        #pragma unroll
        for (int t = 0; t < TP; t++)
            #pragma unroll
            for (int q = 0; q < 4; q++) fv[t * 4 + q] = t < T ? acc[t < T ? t : 0][q] : 0.f;
        const float val = v_tsum<NV>(fv, lane);
        const int kv = v_tsum_k<NV>(lane), vt = kv / 4, vs = kv % 4;
        if ((lane & (64 / NV - 1)) == 0 && vt < T && r0 + vs < j.N) j.y[(size_t) vt * j.N + r0 + vs] = val;
    }
}
// the same walk for a K too wide to stage (the FFN down projection, T = 8, K = 9216). The
// activations come from HBM / L2, 4 waves of 2 rows per block, N / 8 blocks to fill the 60
// CUs. A row's value does not depend on the rows per wave.
template <int T>
static __global__ __launch_bounds__(256) void k_ds_gemv_wide(const unsigned char * aq, const float * as, int K,
                                                            const char * wv, const __half * ws, int N, float * y) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    constexpr int S = 2;
    const int r0 = blockIdx.x * 4 * S + wave * S;
    float acc[T][S];
    qz_gemv_rows_t<T, S>(aq, as, K, wv, ws, N, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int t = 0; t < T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++)
                if (r0 + s < N) y[(size_t) t * N + r0 + s] = acc[t][s];
}
// per (row, head) RMSNorm by w over 128 lanes, then NeoX rope at pos0 + row. x is
// [rows][nh][128]; head blockIdx.x < nq reads q, the rest read k.
static __global__ __launch_bounds__(128) void k_ds_hnorm_rope(float * xq, const float * wq, int nq, float * xk, const float * wk,
                                                              int nk, int pos0, float eps) {
    __shared__ float red[4], s[HD];
    const bool isq = (int) blockIdx.x < nq;
    float * x = isq ? xq : xk;
    const float * w = isq ? wq : wk;
    const int nh = isq ? nq : nk;
    float * xr = x + ((size_t) blockIdx.y * nh + (isq ? blockIdx.x : blockIdx.x - nq)) * HD;
    const int e = threadIdx.x;
    const float v = xr[e];
    float ss = v * v;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 64);
    if ((e & 63) == 0) red[e >> 6] = ss;
    __syncthreads();
    const float r = rsqrtf((red[0] + red[1]) / HD + eps);
    s[e] = v * r * w[e];
    __syncthreads();
    if (e < HD / 2) {
        const float inv = powf(DS_THETA, -2.f * e / HD);
        const float ang = (float) (pos0 + (int) blockIdx.y) * inv;
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float x0 = s[e], x1 = s[e + HD / 2];
        xr[e] = x0 * cs - x1 * sn;
        xr[e + HD / 2] = x0 * sn + x1 * cs;
    }
}
// f32 rows [m][DS_KVL] of this card's kv heads into the f16 cache at rows pos0 ..
static __global__ void k_ds_store(const float * k, const float * v, __half * ck, __half * cv, int pos0, int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * DS_KVL) return;
    ck[(size_t) pos0 * DS_KVL + i] = __float2half(k[i]);
    cv[(size_t) pos0 * DS_KVL + i] = __float2half(v[i]);
}
// batched context K: per-(row, kv head) RMSNorm, NeoX rope at pos0 + row, and f16 cache store
static __global__ __launch_bounds__(128) void k_ds_knorm_rope_store(const float * xk, const float * wk, __half * ck,
                                                                    int pos0, float eps) {
    __shared__ float red[4], s[HD];
    const int h = blockIdx.x, row = blockIdx.y, e = threadIdx.x;
    const float * xr = xk + ((size_t) row * DS_NKVL + h) * HD;
    const float v = xr[e];
    float ss = v * v;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 64);
    if ((e & 63) == 0) red[e >> 6] = ss;
    __syncthreads();
    const float r = rsqrtf((red[0] + red[1]) / HD + eps);
    s[e] = v * r * wk[e];
    __syncthreads();
    if (e < HD / 2) {
        const float inv = powf(DS_THETA, -2.f * e / HD);
        const float ang = (float) (pos0 + row) * inv;
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float x0 = s[e], x1 = s[e + HD / 2];
        __half * dst = ck + ((size_t) (pos0 + row) * DS_NKVL + h) * HD;
        dst[e] = __float2half(x0 * cs - x1 * sn);
        dst[e + HD / 2] = __float2half(x0 * sn + x1 * cs);
    }
}
// attention of the block's DS_BS query rows, KV head blockIdx.x of this card, over the
// context keys [lo, a) of the f16 cache and the DS_BS block keys in f32, with no mask.
// One block per (KV head, key split): its DS_NQV = 3 q heads x 8 rows query vectors share
// each key tile, which is staged in LDS as f32 so the inner loop is plain f32 FMAs and
// 128-bit LDS reads. Each 8-lane group of a block owns one query vector and 16 of the 128
// dims; the score sums the group's 8 lanes with 3 DPP steps. Every split leaves an
// (m, l, o[128]) partial per query vector, and the last split also folds in the block's
// own DS_BS f32 keys. Only the split count varies with the context length.
constexpr int DS_GQA = DS_NQL / DS_NKVL;                     // 3 q heads per KV head
constexpr int DS_NQV = DS_BS * DS_GQA;                       // 24 query vectors of one KV head
constexpr int DS_NSPL = 60;                                  // key splits per KV head (4 * 60 = 240 blocks = 4/CU)
constexpr int DS_TILE = 12;                                  // cache keys staged per tile
constexpr int DS_ATT_T = DS_NQV * 8;                         // 192 threads: one 8-lane group per query vector
constexpr int DS_PW = HD + 2;
// The LDS tile: one row of DS_TROW floats per key, cut into 8 blocks of 16 dims with 4
// floats of padding. The pad puts the 8 lanes of a group on 8 different 4-bank windows,
// without it they all read banks 0-3 or 16-19 and every tile read is 4-way conflicted.
constexpr int DS_TPAD = 20, DS_TROW = (HD / 16) * DS_TPAD;
__device__ __forceinline__ void ds_st8(float * p, const float * f) {
    float4 * q = (float4 *) p;
    q[0] = make_float4(f[0], f[1], f[2], f[3]); q[1] = make_float4(f[4], f[5], f[6], f[7]);
}
__device__ __forceinline__ void ds_ld16(const float * p, float * f) {
    const float4 * q = (const float4 *) p;
    const float4 a = q[0], b = q[1], c = q[2], d = q[3];
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
    f[8] = c.x; f[9] = c.y; f[10] = c.z; f[11] = c.w; f[12] = d.x; f[13] = d.y; f[14] = d.z; f[15] = d.w;
}
// sum of x over its 8-lane group, in every lane: the quad pair of the two quad_perm
// steps, then row_half_mirror for lane ^ 7
__device__ __forceinline__ float ds_row8(float x) {
    x += qz_dpp<0x4E>(x);     // quad_perm [2,3,0,1]
    x += qz_dpp<0xB1>(x);     // quad_perm [1,0,3,2]
    x += qz_dpp<0x141>(x);    // row_half_mirror
    return x;
}
// one key's contribution to a query vector: the score of the group's 16 dims, then the
// online softmax and the value accumulation (c == 1 leaves o as it is)
__device__ __forceinline__ void ds_attn_key(const float * qq, const float * kf, const float * vf, float * o,
                                            float & m, float & l) {
    float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;         // 4 chains: the lane sum is a 4-way tree
    #pragma unroll
    for (int k = 0; k < 16; k += 4) {
        x0 += qq[k] * kf[k]; x1 += qq[k + 1] * kf[k + 1];
        x2 += qq[k + 2] * kf[k + 2]; x3 += qq[k + 3] * kf[k + 3];
    }
    const float s = ds_row8((x0 + x1) + (x2 + x3));
    const float mn = fmaxf(m, s), c = __builtin_amdgcn_exp2f(m - mn), p = __builtin_amdgcn_exp2f(s - mn);
    l = l * c + p;
    if (c < 1.f)
        #pragma unroll
        for (int k = 0; k < 16; k++) o[k] *= c;
    #pragma unroll
    for (int k = 0; k < 16; k++) o[k] += p * vf[k];
    m = mn;
}
// one tile of DS_TILE cache keys for one query vector: the scores of the whole tile first,
// then a single rescale against the tile's max, then an expf and a value accumulate per
// key. The keys of a tile are independent, unlike a one-key-at-a-time update, so the
// softmax latency does not serialize the tile. Keys past n are masked to -infinity.
__device__ __forceinline__ void ds_attn_tile(const float * qq, const float * sk, const float * sv, int li,
                                             float * o, float & m, float & l, int n) {
    float s[DS_TILE];
    #pragma unroll
    for (int t = 0; t < DS_TILE; t++) {
        float kf[16];
        ds_ld16(sk + t * DS_TROW + li * DS_TPAD, kf);
        float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;        // 4 chains: the lane sum is a 4-way tree
        #pragma unroll
        for (int k = 0; k < 16; k += 4) {
            x0 += qq[k] * kf[k]; x1 += qq[k + 1] * kf[k + 1];
            x2 += qq[k + 2] * kf[k + 2]; x3 += qq[k + 3] * kf[k + 3];
        }
        s[t] = t < n ? ds_row8((x0 + x1) + (x2 + x3)) : -INFINITY;
    }
    float mt = s[0];
    #pragma unroll
    for (int t = 1; t < DS_TILE; t++) mt = fmaxf(mt, s[t]);
    const float mn = fmaxf(m, mt), c = __builtin_amdgcn_exp2f(m - mn);
    l *= c;
    if (c < 1.f)
        #pragma unroll
        for (int k = 0; k < 16; k++) o[k] *= c;
    m = mn;
    #pragma unroll
    for (int t = 0; t < DS_TILE; t++) {
        const float p = __builtin_amdgcn_exp2f(s[t] - mn);
        l += p;
        float vf[16];
        ds_ld16(sv + t * DS_TROW + li * DS_TPAD, vf);
        #pragma unroll
        for (int k = 0; k < 16; k++) o[k] += p * vf[k];
    }
}
static __global__ __launch_bounds__(DS_ATT_T) void k_ds_attn(const float * q, const __half * ck, const __half * cv,
                                                            const float * kb, const float * vb, int lo, int a, int nspl,
                                                            int cks, float * part) {
    __shared__ float sk[DS_TILE * DS_TROW], sv[DS_TILE * DS_TROW];
    const int kh = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x;
    const int grp = tid >> 3, li = tid & 7;                  // query vector, its 16 dims
    const int r = grp % DS_BS, gq = grp / DS_BS, h = kh * DS_GQA + gq;
    const float scale = FL_L2E / sqrtf((float) HD);
    float qq[16], o[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        qq[k] = q[((size_t) r * DS_NQL + h) * HD + li * 16 + k] * scale;
        o[k] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    const int jb = sp * cks, je = min(a - lo, jb + cks);
    // One tile of global loads per thread (DS_TILE * HD / 8 == DS_ATT_T), issued a tile
    // ahead of the LDS staging so the next tile's latency is covered by the compute.
    static_assert(DS_TILE * (HD / 8) == DS_ATT_T, "one uint4 of each cache per thread and tile");
    const int lt = tid / (HD / 8), ldq = (tid % (HD / 8)) * 8;
    const int lds = lt * DS_TROW + (ldq >> 4) * DS_TPAD + (ldq & 15);
    uint4 kc, vc;
    auto fetch = [&](int t0, int n) {
        const size_t so = (size_t) (lo + t0 + min(lt, n - 1)) * DS_KVL + kh * HD + ldq;
        kc = *(const uint4 *) (ck + so); vc = *(const uint4 *) (cv + so);
    };
    if (jb < je) fetch(jb, min(DS_TILE, je - jb));
    for (int t0 = jb; t0 < je; t0 += DS_TILE) {
        const int n = min(DS_TILE, je - t0);
        float f[8];
        __syncthreads();                                     // the previous tile is consumed
        att_h8(kc, f); ds_st8(sk + lds, f);
        att_h8(vc, f); ds_st8(sv + lds, f);
        if (t0 + DS_TILE < je) fetch(t0 + DS_TILE, min(DS_TILE, je - t0 - DS_TILE));
        __syncthreads();
        ds_attn_tile(qq, sk, sv, li, o, m, l, n);
    }
    if (sp == nspl - 1)                                      // the block's own keys, f32, once
        for (int t = 0; t < DS_BS; t++) {
            float kf[16], vf[16];
            ds_ld16(kb + (size_t) t * DS_KVL + kh * HD + li * 16, kf);
            ds_ld16(vb + (size_t) t * DS_KVL + kh * HD + li * 16, vf);
            ds_attn_key(qq, kf, vf, o, m, l);
        }
    float * P = part + (((size_t) kh * DS_NSPL + sp) * DS_NQV + grp) * DS_PW;
    if (li == 0) { P[0] = m; P[1] = l; }
    #pragma unroll
    for (int k = 0; k < 16; k++) P[2 + li * 16 + k] = o[k];
}
// merge the nspl partials of q head blockIdx.x, row blockIdx.y, directly into int8 planes
static __global__ __launch_bounds__(128) void k_ds_attn_merge_quant(const float * part, unsigned char * oq, float * os, int nspl) {
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int kh = h / DS_GQA, grp = (h % DS_GQA) * DS_BS + r;
    const float * B = part + ((size_t) kh * DS_NSPL * DS_NQV + grp) * DS_PW;
    float M = -INFINITY;
    for (int s = 0; s < nspl; s++) M = fmaxf(M, B[(size_t) s * DS_NQV * DS_PW]);
    float L = 0.f, O = 0.f;
    #pragma unroll 1
    for (int w0 = 0; w0 < nspl; w0 += 16) {                // 16 partial loads in flight, folded in order
        float pm[16], pl[16], po[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) {                     // past the end: the last partial, masked below
            const int s = min(w0 + k, nspl - 1);
            const float * P = B + (size_t) s * DS_NQV * DS_PW;
            pm[k] = w0 + k < nspl ? P[0] : -INFINITY;
            pl[k] = P[1]; po[k] = P[2 + d];
        }
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            if (pm[k] == -INFINITY) continue;
            const float e = __builtin_amdgcn_exp2f(pm[k] - M);
            L += pl[k] * e; O += po[k] * e;
        }
    }
    const float v = O / L;
    quant32(v, oq + (size_t) r * DS_QL, os + (size_t) r * (DS_QL / 32), h * HD + d);
}
static __global__ __launch_bounds__(256) void k_ds_embed(const char * ev, const __half * es, int tok, const float * w, float eps,
                                                         float * x, const float * mask) {
    if (blockIdx.x == 0) d_embed(ev, es, tok, w, eps, x, threadIdx.x, true);
    else for (int i = threadIdx.x; i < D; i += 256) x[(size_t) blockIdx.x * D + i] = mask[i];
}
__device__ __forceinline__ float ds_bf16(uint16_t b) { return __uint_as_float((unsigned int) b << 16); }
// joint = W [state | w1[prev] | hid | g] + b, one wave per output row. From step 2 on, prev
// is the argmax both cards agreed on, read from best / xbest once the peer's flag arrives.
// Every block reads it, so no extra launch.
static __global__ __launch_bounds__(512) void k_ds_joint(const uint16_t * W, const float * bias, const float * state,
                                                         const uint16_t * w1, int step, int anc, const float * hid, const float * g,
                                                         float * joint, const unsigned long long * best,
                                                         const unsigned long long * xbest, unsigned int * fl, unsigned int seq,
                                                         int * err, int * out) {
    __shared__ int sprev;
    if (threadIdx.x == 0) {
        int prev = anc;
        if (step > 1) {
            ds_wait(fl, seq, err);
            const unsigned long long b0 = ((volatile unsigned long long *) best)[0];
            const unsigned long long b1 = ((volatile unsigned long long *) xbest)[0];
            prev = (int) ~(unsigned int) ((b0 > b1 ? b0 : b1) & 0xffffffffu);
        }
        if (blockIdx.x == 0 && step > 1) out[step - 2] = prev;
        sprev = prev;
    }
    __syncthreads();
    const int prev = sprev;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    if (r >= 3 * DS_R) return;
    const uint16_t * wr = W + (size_t) r * (2 * DS_R + 2 * D);
    const uint16_t * e = w1 + (size_t) prev * DS_R;
    float acc = 0.f;
    for (int k = lane; k < DS_R; k += 64) acc += ds_bf16(wr[k]) * state[k] + ds_bf16(wr[DS_R + k]) * ds_bf16(e[k]);
    for (int k = lane; k < D; k += 64) acc += ds_bf16(wr[2 * DS_R + k]) * hid[k] + ds_bf16(wr[2 * DS_R + D + k]) * g[k];
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 64);
    if (lane == 0) joint[r] = acc + bias[r];
}
// gate, cand, latent: state = cand + gate * (state - cand) and lat1 = [tanh latent | 1].
static __global__ __launch_bounds__(256) void k_ds_markov(const float * joint, float * state, float * lat1,
                                                         float * lat1_peer, unsigned long long * best, unsigned long long * xbest,
                                                         unsigned int * fl, unsigned int seq) {
    const int k = threadIdx.x;
    if (k == 0) { ((volatile unsigned long long *) best)[0] = 0; ((volatile unsigned long long *) xbest)[0] = 0; }
    const float gate = 1.f / (1.f + expf(-joint[k])), cand = tanhf(joint[DS_R + k]);
    state[k] = cand + gate * (state[k] - cand);
    const float l = tanhf(joint[2 * DS_R + k]);
    lat1[k] = l;
    if (k == 0) lat1[DS_R] = 1.f;
    if (lat1_peer) {
        volatile float * p = (volatile float *) lat1_peer;
        p[k] = l;
        if (k == 0) p[DS_R] = 1.f;
        __syncthreads();
        if (k == 0) ds_flag(fl, seq);
    }
}
// ---- sampled drafts: an exact draw from q = softmax(s * invTq) by Gumbel-max. z = s * invTq
// + G with G = -log(-log(u)), u in (0, 1) from a counter-based hash of the request's seed,
// the round counter, the draft row and the global vocab id. Both cards hash the same key
// for their own ids, so no exchange is needed and a fixed seed repeats a run exactly.
// u takes 23 hash bits + 0.5 over 2^23, i.e. u in [2^-24, 1 - 2^-24]: 24 bits cannot be
// mapped this way in f32, where (2^24 - 1) + 0.5 rounds to 2^24 and u = 1 would make G
// +inf, pinning the draft to whichever id hit that value. The
// grid truncates the Gumbel tails at [-2.81, 16.66], which costs q at most ~exp(-17) of
// its mass per entry.
__device__ __forceinline__ unsigned int ds_fmix(unsigned int x) {           // murmur3's finalizer
    x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float ds_gumbel(unsigned int seed, unsigned int rnd, unsigned int row, unsigned int id) {
    unsigned int h = ds_fmix(seed ^ 0x9e3779b9u);
    h = ds_fmix(h ^ rnd); h = ds_fmix(h ^ row); h = ds_fmix(h ^ id);
    const float u = ((float) (h >> 9) + 0.5f) * (1.f / 8388608.f);          // exact in f32, in (0, 1)
    return -logf(-logf(u));
}
// one draft row's sampler state: q = softmax(logit * invTq) and the (max, sumexp) block
// partials of that row's logsumexp go to part (null: no logsumexp)
struct dsamp { float invTq; unsigned int seed, rnd; float * part; float cap; };   // cap > 0: the target's logit softcap on q's logits
constexpr int DS_LSEB = 128;                   // logsumexp partials per card per row (>= grid)
constexpr int DS_QID = QZ_TOPK + 1;            // a row's candidate ids: p's chain, plus the draft if absent
// draft row logits over this card's head rows: base + fold . lat1, then the argmax into
// best as a packed (orderable value, ~id) max, so ties keep the lower id. WT waits for lat1
// first, on card 1. fold rows are nhp halves apart and 128-byte aligned, walked in j order,
// DS_FE consecutive entries per thread with 8-byte loads. With SMP the packed value is the
// Gumbel-max's z instead of the logit, so the same argmax draws the draft from q, and each
// block also leaves its (max, sumexp) of logit * invTq for the row's logsumexp.
constexpr int DS_FE = 4;
template <bool WT, bool SMP>
static __global__ __launch_bounds__(256) void k_ds_fold_argmax(const float * base, const __half * fold, int nhp, int nvalid,
                                                               const float * lat1, int id0, unsigned long long * best,
                                                               unsigned int * fl, unsigned int seq, int * err,
                                                               dsamp sa, unsigned int grow) {
    __shared__ float sl[DS_R + 1];
    if (WT) {
        if (threadIdx.x == 0) ds_wait(fl, seq, err);
        __syncthreads();
    }
    for (int j = threadIdx.x; j <= DS_R; j += 256) sl[j] = lat1[j];
    __syncthreads();
    const int v0 = (blockIdx.x * 256 + threadIdx.x) * DS_FE;
    unsigned long long key = 0;
    float s[DS_FE];
    #pragma unroll
    for (int e = 0; e < DS_FE; e++) s[e] = 0.f;
    if (v0 < nvalid) {
        const __half * p = fold + v0;
        #pragma unroll
        for (int e = 0; e < DS_FE; e++) s[e] = v0 + e < nvalid ? base[v0 + e] : 0.f;
        #pragma unroll 8
        for (int j = 0; j <= DS_R; j++) {
            const uint2 hv = *(const uint2 *) p;
            const __half2 h01 = *(const __half2 *) &hv.x, h23 = *(const __half2 *) &hv.y;
            const float lj = sl[j];
            s[0] += lj * __low2float(h01);
            s[1] += lj * __high2float(h01);
            s[2] += lj * __low2float(h23);
            s[3] += lj * __high2float(h23);
            p += nhp;
        }
        if (SMP && sa.cap > 0.f)                   // q's logits through the target's softcap, as p's are
            #pragma unroll
            for (int e = 0; e < DS_FE; e++) s[e] = sa.cap * tanhf(s[e] / sa.cap);
        #pragma unroll
        for (int e = 0; e < DS_FE; e++)
            if (v0 + e < nvalid) {
                float v = s[e];
                if (SMP) v = v * sa.invTq + ds_gumbel(sa.seed, sa.rnd, grow, (unsigned int) (id0 + v0 + e));
                unsigned int u = __float_as_uint(v);
                u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                const unsigned long long k = ((unsigned long long) u << 32) | (unsigned int) ~(unsigned int) (id0 + v0 + e);
                key = k > key ? k : key;
            }
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) { const unsigned long long o = __shfl_xor(key, off, 64); key = o > key ? o : key; }
    __shared__ unsigned long long sb[4];
    if ((threadIdx.x & 63) == 0) sb[threadIdx.x >> 6] = key;
    if (SMP) {                                     // this block's (max, sumexp) of logit * invTq
        __shared__ float sm[4], ss[4];
        float m = -INFINITY;
        for (int e = 0; e < DS_FE; e++) if (v0 + e < nvalid) m = fmaxf(m, s[e] * sa.invTq);
        for (int off = 32; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, 64));
        float l = 0.f;
        for (int e = 0; e < DS_FE; e++) if (v0 + e < nvalid) l += expf(s[e] * sa.invTq - m);
        for (int off = 32; off > 0; off >>= 1) l += __shfl_xor(l, off, 64);
        if ((threadIdx.x & 63) == 0) { sm[threadIdx.x >> 6] = m; ss[threadIdx.x >> 6] = l; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bm = sm[0], bl = 0.f;
            for (int i = 1; i < 4; i++) bm = fmaxf(bm, sm[i]);
            for (int i = 0; i < 4; i++) bl += ss[i] * expf(sm[i] - bm);
            sa.part[2 * blockIdx.x] = bm;
            sa.part[2 * blockIdx.x + 1] = bl;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long b = sb[0];
        for (int i = 1; i < 4; i++) b = sb[i] > b ? sb[i] : b;
        atomicMax(best, b);
    }
}
// one draft row's logsumexp from its fold blocks' (max, sum) partials: out[2 r] = max,
// out[2 r + 1] = sum of exp(logit * invTq - max). The host adds the two cards' rows.
static __global__ __launch_bounds__(64) void k_ds_lse_merge(const float * part, int nblk, float * out) {
    if (threadIdx.x) return;
    const float * p = part + (size_t) blockIdx.x * DS_LSEB * 2;
    float m = -INFINITY;
    for (int b = 0; b < nblk; b++) m = fmaxf(m, p[2 * b]);
    float s = 0.f;
    for (int b = 0; b < nblk; b++) s += p[2 * b + 1] * expf(p[2 * b] - m);
    out[2 * blockIdx.x] = m;
    out[2 * blockIdx.x + 1] = s;
}
// The drafter's logit of one candidate id per block: row r of the draft logits is
// base[r] + fold . lat1[r], so this walks the id's fold column. Ids this card does not own
// (id - id0 out of range, -1 for padding) are skipped; the host reads the owner's buffer.
static __global__ __launch_bounds__(64) void k_ds_qids(const float * base, int N, const __half * fold, int nhp, int nvalid,
                                                      int id0, const int * ids, const float * lat1, float * out) {
    const int r = blockIdx.y, v = ids[(size_t) r * gridDim.x + blockIdx.x] - id0;
    if (v < 0 || v >= nvalid) return;
    const __half * p = fold + v;
    const float * lr = lat1 + (size_t) r * (DS_R + 1);
    float acc = 0.f;
    for (int j = threadIdx.x; j <= DS_R; j += 64) acc += lr[j] * __half2float(p[(size_t) j * nhp]);
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 64);
    if (threadIdx.x == 0) out[(size_t) r * gridDim.x + blockIdx.x] = base[(size_t) r * N + v] + acc;
}
// card 1: hand this step's argmax to card 0 by a peer store and a flag, then free the slot
static __global__ __launch_bounds__(64) void k_ds_best_push(unsigned long long * best, unsigned long long * xbest_peer,
                                                            unsigned int * fl_peer, unsigned int seq) {
    if (threadIdx.x != 0) return;
    *(volatile unsigned long long *) xbest_peer = *best;
    ds_flag(fl_peer, seq);
    *best = 0;
}
// the chain's last token, the argmax of its last step, once the peer's push has landed
static __global__ __launch_bounds__(64) void k_ds_last(const unsigned long long * best, const unsigned long long * xbest,
                                                       unsigned int * fl, unsigned int seq, int * err, int * out, int slot) {
    if (threadIdx.x != 0) return;
    ds_wait(fl, seq, err);
    const unsigned long long b0 = ((volatile const unsigned long long *) best)[0];
    const unsigned long long b1 = ((volatile const unsigned long long *) xbest)[0];
    out[slot] = (int) ~(unsigned int) ((b0 > b1 ? b0 : b1) & 0xffffffffu);
}
static __global__ void k_ds_tohalf(const float * y, __half * o, int n, int stride) {
    const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) o[(size_t) r * stride + c] = __float2half(y[(size_t) r * n + c]);
}
// hidden rows 1 .. 7: X += a + b as in k_ds_add_quant, then the Markov input, f32 and int8
static __global__ __launch_bounds__(256) void k_ds_hidden(float * x, const float * a, const float * b, const float * w, int K,
                                                          unsigned char * q, float * qs, float * hf, float eps) {
    __shared__ float red[16];
    float * xr = x + (size_t) (blockIdx.x + 1) * K;
    const float * ar = a + (size_t) (blockIdx.x + 1) * K, * br = b + (size_t) (blockIdx.x + 1) * K;
    float ss[1] = {0.f};
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = xr[i] + (ar[i] + br[i]);
        xr[i] = v;
        ss[0] += v * v;
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / K + eps);
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = xr[i] * r * w[i];
        hf[(size_t) blockIdx.x * K + i] = v;
        quant32(v, q + (size_t) blockIdx.x * K, qs + (size_t) blockIdx.x * (K / 32), i);
    }
}
// columns j0 .. j0 + 7 of hidden_out (bf16 [D][DS_R]) and its bias as 8 f32 rows [8][D]
static __global__ void k_ds_fold_rows(const uint16_t * how, const float * hob, int j0, float * rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 8 * D) return;
    const int t = i / D, d = i % D, j = j0 + t;
    rows[i] = j < DS_R ? ds_bf16(how[(size_t) d * DS_R + j]) : (j == DS_R ? hob[d] : 0.f);
}

struct ink2_dspark {
    struct layer { qw q, k, v, kc, vc, o, gate, up, down; const float * an, * fn, * qn, * kn, * kcn, * dn; };
    // one card's share: weight slices, this card's kv-head context cache, block buffers
    struct card {
        qz_loader ld;
        layer L[DS_NL];
        const float * mask = nullptr, * onorm = nullptr, * hob = nullptr;
        const uint16_t * how = nullptr;
        __half * ck[DS_NL] = {}, * cv[DS_NL] = {};   // [ctx][DS_KVL]
        unsigned char * fq = nullptr; float * fs = nullptr;   // context fill: the 3 layers' mixed rows, int8 [3][m][D]
        float * ckf = nullptr, * cvf = nullptr, * ckf_pf = nullptr;   // context fill: k / v [8][DS_KVL], batched k [3][PF_M][DS_KVL]
        float * X = nullptr, * Y = nullptr, * Yp[2] = {};   // residual [8][D], this card's partial, the peer's (2 slots)
        float * Q = nullptr, * Kb = nullptr, * Vb = nullptr, * O = nullptr, * G = nullptr, * U = nullptr;
        float * hid = nullptr, * part = nullptr;
        unsigned char * aq = nullptr; float * as = nullptr;
        hipEvent_t ev = nullptr;
    };
    ink2_engine * E = nullptr;
    card K[2];
    float4 alpha[DS_NL + 1];
    const float * dsn = nullptr, * jb = nullptr;      // card 0
    const uint16_t * jw = nullptr, * w1 = nullptr;    // card 0
    int ctx = 0;
    int xk = 0;                                // the partials' slot parity
    // card 0: the Markov chain
    float * g = nullptr;                       // the Markov context of the last committed row
    float * tap = nullptr;                     // decode features [4][D] (the cycle writes them)
    float * vtap = nullptr;                    // verify features [VR][4][D] (the verify kernel writes them)
    float * joint = nullptr, * state = nullptr;
    // per card: the draft rows' quantized hidden, base [8][nvalid], fold [(R+1)][nhp], lat1
    unsigned char * hq[2] = {}; float * hs[2] = {}, * base[2] = {}, * lat1[2] = {};
    __half * fold[2] = {};
    int nhp[2] = {};
    unsigned long long * best[2] = {};         // this card's fold target (local, zeroed per step)
    unsigned long long * xbest[2] = {};        // the peer's argmax pushed here (uncached)
    unsigned int * fl[2] = {};                 // the peer's handshake flag (uncached)
    int * out_d = nullptr, * out_h = nullptr;  // the 7 tokens (card 0's)
    // ds_wait's timeout word, one per card: a wait reads and CASes it, so it must be in the
    // waiting card's own memory. In p2p a single card-0 word served both cards, which host
    // mode cannot do -- card 1 would touch card 0's VRAM.
    int * errd[2] = {};
    unsigned int seq = 0;                      // the chain's step counter (monotone)
    // Sampled rounds: each draft row's (max, sumexp) partials and merged logsumexp, the
    // target's candidate ids to evaluate, and their drafter logits.
    float * lses[2] = {};                      // device: [DS_BS][DS_LSEB][2] fold block partials
    float * lsem[2] = {};                      // device: [DS_BS][2] merged per row, per card
    float * lseh[2] = {};                      // pinned: [DS_BS][2]
    int * qid_d[2] = {};                       // device: [DS_BS][DS_QID] candidate ids
    int * qid_h = nullptr;                     // pinned: the same
    float * qs_d[2] = {};                      // device: [DS_BS][DS_QID] drafter logits
    float * qs_h[2] = {};                      // pinned: the same
    unsigned int rnd = 0;                      // the request's round counter (Gumbel key)

    long rounds = 0, accepted = 0;
    static qw Q8(const qz_loader & l, const char * n, int K, int N) {
        const qz_dev_tensor * t = l.get(n);
        if (!t || t->type != GGML_TYPE_Q8_0) { fprintf(stderr, "dspark: missing Q8_0 %s\n", n); exit(1); }
        qw r; r.v = (const char *) t->vals; r.s = (const __half *) t->scal; r.K = (int) t->ne0; r.N = (int) t->ne1;
        if (r.K != K || r.N != N) { fprintf(stderr, "dspark: %s is %d x %d per card, expected %d x %d\n", n, r.K, r.N, K, N); exit(1); }
        return r;
    }
    static const void * Raw(const qz_loader & l, const char * n, ggml_type ty) {
        const qz_dev_tensor * t = l.get(n);
        if (!t || t->type != ty) { fprintf(stderr, "dspark: missing %s (type %d)\n", n, (int) ty); exit(1); }
        return t->vals;
    }
    template <class T> static void al(T ** p, size_t n) { ICHECK(qz_dmalloc((void **) p, n * sizeof(T), "dspark")); }
    // the Markov chain's own tensors live on card 0 only
    static bool chain_only(const char * n) {
        return strcmp(n, "markov_w1.weight") == 0 || strcmp(n, "joint_proj.weight") == 0 || strcmp(n, "joint_proj.bias") == 0 ||
               strcmp(n, "dspark_norm.weight") == 0;
    }

    bool load(const char * path, int ctx_) {
        ctx = ctx_;
        qz_gguf g_;
        if (!g_.open(path) || g_.str("general.architecture") != std::string("dflash")) {
            fprintf(stderr, "dspark: %s is not a dflash (DSpark) GGUF\n", path); return false;
        }
        if (g_.u32("dflash.block_count") != DS_NL || g_.u32("dflash.block_size") != DS_BS || g_.u32("dflash.embedding_length") != D) {
            fprintf(stderr, "dspark: unsupported drafter shape\n"); return false;
        }
        char n[96];
        for (int c = 0; c < 2; c++) {
            card & k = K[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            {
                qz_mem_scope mem("dspark weights");
                k.ld.slice = ink2_ds_tp2_spec;
                if (c == 1) k.ld.skip = chain_only;
                if (!k.ld.load(path, g_, qz_dev_of(c), c)) { fprintf(stderr, "dspark: card %d load failed\n", c); return false; }
            }
            for (int l = 0; l < DS_NL; l++) {
                auto nm = [&](const char * s) { snprintf(n, sizeof(n), "blk.%d.%s.weight", l, s); return n; };
                layer & y = k.L[l];
                y.q = Q8(k.ld, nm("attn_q"), D, DS_QL); y.k = Q8(k.ld, nm("attn_k"), D, DS_KVL); y.v = Q8(k.ld, nm("attn_v"), D, DS_KVL);
                y.kc = Q8(k.ld, nm("attn_k_ctx"), D, DS_KVL); y.vc = Q8(k.ld, nm("attn_v_ctx"), D, DS_KVL);
                y.o = Q8(k.ld, nm("attn_output"), DS_QL, D);
                y.gate = Q8(k.ld, nm("ffn_gate"), D, DS_FFL); y.up = Q8(k.ld, nm("ffn_up"), D, DS_FFL); y.down = Q8(k.ld, nm("ffn_down"), DS_FFL, D);
                y.an = (const float *) Raw(k.ld, nm("attn_norm"), GGML_TYPE_F32); y.fn = (const float *) Raw(k.ld, nm("ffn_norm"), GGML_TYPE_F32);
                y.qn = (const float *) Raw(k.ld, nm("attn_q_norm"), GGML_TYPE_F32); y.kn = (const float *) Raw(k.ld, nm("attn_k_norm"), GGML_TYPE_F32);
                y.kcn = (const float *) Raw(k.ld, nm("attn_k_ctx_norm"), GGML_TYPE_F32); y.dn = (const float *) Raw(k.ld, nm("dflare_norm"), GGML_TYPE_F32);
            }
            k.mask = (const float *) Raw(k.ld, "mask_embd.weight", GGML_TYPE_F32);
            k.onorm = (const float *) Raw(k.ld, "output_norm.weight", GGML_TYPE_F32);
            k.hob = (const float *) Raw(k.ld, "dspark_hidden_out.bias", GGML_TYPE_F32);
            k.how = (const uint16_t *) Raw(k.ld, "dspark_hidden_out.weight", GGML_TYPE_BF16);
        }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        const qz_loader & L0 = K[0].ld;
        dsn = (const float *) Raw(L0, "dspark_norm.weight", GGML_TYPE_F32);
        jb = (const float *) Raw(L0, "joint_proj.bias", GGML_TYPE_F32);
        jw = (const uint16_t *) Raw(L0, "joint_proj.weight", GGML_TYPE_BF16);
        w1 = (const uint16_t *) Raw(L0, "markov_w1.weight", GGML_TYPE_BF16);
        {   // the 4 feature mixes, softmax over each row of dspark_fuse
            float f[16];
            ICHECK(hipMemcpy(f, Raw(L0, "dspark_fuse.weight", GGML_TYPE_F32), sizeof(f), hipMemcpyDeviceToHost));
            for (int r = 0; r <= DS_NL; r++) {
                float mx = f[4 * r], e[4], s = 0;
                for (int k = 1; k < 4; k++) mx = fmaxf(mx, f[4 * r + k]);
                for (int k = 0; k < 4; k++) { e[k] = expf(f[4 * r + k] - mx); s += e[k]; }
                alpha[r] = make_float4(e[0] / s, e[1] / s, e[2] / s, e[3] / s);
            }
        }
        qz_mem_scope mem("dspark state");
        al(&g, D); al(&tap, 4 * D); al(&vtap, (size_t) VR * 4 * D); al(&joint, 3 * DS_R); al(&state, DS_R);
        ICHECK(qz_dmalloc((void **) &out_d, 8 * sizeof(int), "dspark"));
        ICHECK(hipHostMalloc((void **) &out_h, 8 * sizeof(int), 0));
        ICHECK(hipMemset(out_d, 0, 8 * sizeof(int)));
        for (int c = 0; c < 2; c++) {
            card & k = K[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            for (int l = 0; l < DS_NL; l++) { al(&k.ck[l], (size_t) ctx * DS_KVL); al(&k.cv[l], (size_t) ctx * DS_KVL); }
            al(&k.fq, (size_t) DS_NL * PF_M * D); al(&k.fs, (size_t) DS_NL * PF_M * D / 32);
            al(&k.ckf, 8 * DS_KVL); al(&k.cvf, 8 * DS_KVL);
            al(&k.ckf_pf, (size_t) DS_NL * PF_M * DS_KVL);
            al(&k.X, DS_BS * D); al(&k.Y, DS_BS * D); al(&k.Yp[0], DS_BS * D); al(&k.Yp[1], DS_BS * D);
            al(&k.Q, DS_BS * DS_QL); al(&k.Kb, DS_BS * DS_KVL); al(&k.Vb, DS_BS * DS_KVL);
            al(&k.O, DS_BS * DS_QL); al(&k.G, DS_BS * DS_FFL); al(&k.U, DS_BS * DS_FFL); al(&k.hid, DS_BS * D);
            al(&k.part, (size_t) DS_NKVL * DS_NSPL * DS_NQV * DS_PW);
            al(&k.aq, DS_BS * DS_FFL); al(&k.as, DS_BS * DS_FFL / 32);
            ICHECK(hipEventCreateWithFlags(&k.ev, hipEventDisableTiming));
            const qw & H = E->C[c].head;
            nhp[c] = (H.N + 63) & ~63;             // 128-byte aligned fold rows
            al(&hq[c], DS_BS * D); al(&hs[c], DS_BS * D / 32); al(&base[c], (size_t) DS_BS * H.N);
            // The peer's Markov step writes lat1 and fl, and its argmax push writes xbest:
            // the other card's kernel stores into them, so they are mailboxes (uncached VRAM
            // under p2p, pinned host memory under host mode) and qz_xalloc zeroes them.
            // One lat1 slot per draft row: the candidate logits are evaluated after the chain.
            ICHECK(qz_xalloc((void **) &lat1[c], (size_t) DS_BS * (DS_R + 1) * sizeof(float), c, "dspark"));
            al(&fold[c], (size_t) (DS_R + 1) * nhp[c]);
            ICHECK(qz_dmalloc((void **) &best[c], 8, "dspark"));      // this card's fold target only
            ICHECK(hipMemset(best[c], 0, 8));
            ICHECK(qz_xalloc((void **) &xbest[c], 8, c, "dspark"));
            ICHECK(qz_xalloc((void **) &fl[c], 4, c, "dspark"));
            al(&errd[c], 1);
            ICHECK(hipMemset(errd[c], 0, sizeof(int)));
            al(&lses[c], (size_t) DS_BS * DS_LSEB * 2);
            al(&lsem[c], DS_BS * 2);
            ICHECK(hipHostMalloc((void **) &lseh[c], DS_BS * 2 * sizeof(float), 0));
            ICHECK(qz_dmalloc((void **) &qid_d[c], DS_BS * DS_QID * sizeof(int), "dspark"));
            ICHECK(qz_dmalloc((void **) &qs_d[c], DS_BS * DS_QID * sizeof(float), "dspark"));
            ICHECK(hipHostMalloc((void **) &qs_h[c], DS_BS * DS_QID * sizeof(float), 0));
            if (c == 1) ICHECK(hipHostMalloc((void **) &qid_h, DS_BS * DS_QID * sizeof(int), 0));
            // fold = head . [hidden_out | bias], built here 8 columns per GEMV
            float * rows = nullptr, * y = nullptr;
            ICHECK(hipMalloc(&rows, 8 * D * 4)); ICHECK(hipMalloc(&y, (size_t) 8 * H.N * 4));
            for (int j0 = 0; j0 <= DS_R; j0 += 8) {
                k_ds_fold_rows<<<(8 * D + 255) / 256, 256>>>(k.how, k.hob, j0, rows);
                k_ds_quant<<<8, 256>>>(rows, nullptr, D, hq[c], hs[c], nullptr, DS_EPS);
                k_ds_gemv<8><<<240, DS_NT, 8 * D + 8 * (D / 32) * 4>>>(hq[c], hs[c], D, H.v, H.s, H.N, y);
                const int nj = std::min(8, DS_R + 1 - j0);
                k_ds_tohalf<<<dim3((H.N + 255) / 256, nj), 256>>>(y, fold[c] + (size_t) j0 * nhp[c], H.N, nhp[c]);
            }
            ICHECK(hipFuncSetAttribute((const void *) k_ds_gemv<8>, hipFuncAttributeMaxDynamicSharedMemorySize, 49152));
            ICHECK(hipDeviceSynchronize());
            (void) hipFree(rows); (void) hipFree(y);
        }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        E->dtap = tap; E->dtap_head = tap + 3 * D;
        E->vtap = vtap;
        return true;
    }

    // the T-row GEMV. Stages the planes in LDS up to 48 KB (covers ffn_down 8 x 4608 = 41.5 KB).
    void gemv(hipStream_t st, const unsigned char * q, const float * s, const qw & W, float * y, int T) {
        const int lds = T * W.K + T * (W.K / 32) * 4;
        if (lds <= 49152) {
            const int maxb = lds > 32768 ? 60 : 240;
            const int nb = std::min((W.N + 15) / 16, maxb);
            if (T == 8) k_ds_gemv<8><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 7) k_ds_gemv<7><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 6) k_ds_gemv<6><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 5) k_ds_gemv<5><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 4) k_ds_gemv<4><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 3) k_ds_gemv<3><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 2) k_ds_gemv<2><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else k_ds_gemv<1><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
        } else {
            const int nb = (W.N + 7) / 8;
            if (T == 8) k_ds_gemv_wide<8><<<nb, 256, 0, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 7) k_ds_gemv_wide<7><<<nb, 256, 0, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else k_ds_gemv_wide<1><<<nb, 256, 0, st>>>(q, s, W.K, W.v, W.s, W.N, y);
        }
    }
    void gemv_batch8(hipStream_t st, const unsigned char * q, const float * s, int K, std::initializer_list<std::pair<const qw *, float *>> js) {
        ds_gjobs J{};
        int g = 0;
        for (const auto & pr : js) {
            g += (pr.first->N + 15) / 16;
            J.j[J.n++] = ds_gjob{pr.first->v, pr.first->s, pr.second, pr.first->N, g};
        }
        const int lds = 8 * K + 8 * (K / 32) * 4;
        const int nb = std::min(g, 240);
        k_ds_gemv_batch<8><<<nb, DS_NT, lds, st>>>(q, s, K, J);
    }
    // run fn(c, card, stream) on card 0, then card 1, each with its device current
    template <class F> void each(F && fn) {
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); fn(c, K[c], E->C[c].st); }
    }
    // copy Y into the peer's Yp slot and return the slot. The caller sums Y and Yp.
    // qz_xdst: the consumer (the add on the peer's stream, below) is already ordered by the
    // ev handshake at the end of this function, so p2p keeps the single peer copy on this
    // card's stream and adds no event; host mode queues its H2D on the peer's stream, where
    // the consumer runs, and therefore after the peer's previous reader of the slot as well.
    int exchange() {
        const int s = xk;
        xk ^= 1;
        each([&](int c, card & k, hipStream_t st) {
            ICHECK(qz_xcopy(K[1 - c].Yp[s], 1 - c, qz_xdst(E->C[1 - c].st), k.Y, c, st, (size_t) DS_BS * D * 4));
            ICHECK(hipEventRecord(k.ev, st));
        });
        each([&](int c, card &, hipStream_t st) { ICHECK(hipStreamWaitEvent(st, K[1 - c].ev, 0)); });
        return s;
    }
    // card 0's partial first on both cards
    const float * pa(int c, int s) const { return c == 0 ? K[0].Y : K[1].Yp[s]; }
    const float * pb(int c, int s) const { return c == 0 ? K[0].Yp[s] : K[1].Y; }

    // fill context rows p0 .. p0 + m - 1 from their features, which sit on card 0 at row
    // stride tstride floats. g takes the last row. Card 0 mixes and quantizes the rows for
    // the 3 layers and hands the int8 planes to card 1; each card fills its own kv heads.
    void fill(const float * taps, size_t tstride, int p0, int m) {
        if (m > PF_M) { fprintf(stderr, "dspark: fill of %d rows (max %d)\n", m, PF_M); exit(1); }
        const bool batched = m > 8;
        {
            ICHECK(hipSetDevice(qz_dev_of(0)));
            hipStream_t st = E->C[0].st;
            card & k = K[0];
            ICHECK(hipStreamWaitEvent(st, K[1].ev, 0));   // card 1 is done with the last fill's planes
            for (int l = 0; l < DS_NL; l++) {
                if (batched)
                    k_ds_mixq_pf<<<m, 256, 0, st>>>(taps, tstride, alpha[l], k.L[l].dn,
                                                    (int8_t *) (k.fq + (size_t) l * m * D),
                                                    k.fs + (size_t) l * m * (D / 32), m, DS_EPS);
                else
                    k_ds_mixq<<<m, 256, 0, st>>>(taps, tstride, alpha[l], k.L[l].dn,
                                                 k.fq + (size_t) l * m * D, k.fs + (size_t) l * m * (D / 32), DS_EPS);
            }
            k_ds_mixnorm<<<1, 256, 0, st>>>(taps + (size_t) (m - 1) * tstride, tstride, alpha[DS_NL], dsn, g, D, DS_EPS);
            // qz_xdst: card 1's gemvs below are ordered by the k.ev handoff that follows, so
            // p2p stays the two peer copies on card 0's stream; host mode puts the H2Ds on
            // card 1's stream, ahead of those gemvs and after the last fill's readers of the
            // planes, which are on that stream too.
            ICHECK(qz_xcopy(K[1].fq, 1, qz_xdst(E->C[1].st), k.fq, 0, st, (size_t) DS_NL * m * D));
            ICHECK(qz_xcopy(K[1].fs, 1, qz_xdst(E->C[1].st), k.fs, 0, st, (size_t) DS_NL * m * (D / 32) * 4));
            ICHECK(hipEventRecord(k.ev, st));
        }
        each([&](int c, card & k, hipStream_t st) {
            if (c == 1) ICHECK(hipStreamWaitEvent(st, K[0].ev, 0));
            if (batched) {
                dgjobs Jk{}, Jv{};
                Jk.off[0] = Jv.off[0] = 0;
                for (int l = 0; l < DS_NL; l++) {
                    const int8_t * q = (const int8_t *) (k.fq + (size_t) l * m * D);
                    const float * s = k.fs + (size_t) l * m * (D / 32);
                    float * ko = k.ckf_pf + (size_t) l * m * DS_KVL;
                    __half * vo = k.cv[l] + (size_t) p0 * DS_KVL;
                    Jk.j[l] = dgjob{k.L[l].kc.v, k.L[l].kc.s, q, s, ko, DS_KVL, DS_KVL, D, nullptr};
                    Jv.j[l] = dgjob{k.L[l].vc.v, k.L[l].vc.s, q, s, vo, DS_KVL, DS_KVL, D, nullptr};
                    Jk.off[l + 1] = Jk.off[l] + (DS_KVL + 63) / 64;
                    Jv.off[l + 1] = Jv.off[l] + (DS_KVL + 63) / 64;
                }
                Jk.n = Jv.n = DS_NL;
                const int yt = (m + 63) / 64;
                k_pf_dgemm<false><<<dim3(Jk.off[DS_NL], yt), 256, 0, st>>>(Jk, m, m);
                k_pf_dgemm<true><<<dim3(Jv.off[DS_NL], yt), 256, 0, st>>>(Jv, m, m);
                for (int l = 0; l < DS_NL; l++)
                    k_ds_knorm_rope_store<<<dim3(DS_NKVL, m), 128, 0, st>>>(
                            k.ckf_pf + (size_t) l * m * DS_KVL, k.L[l].kcn, k.ck[l], p0, DS_EPS);
            } else {
                for (int r0 = 0; r0 < m; r0 += 8) {
                    const int n = std::min(8, m - r0);
                    for (int l = 0; l < DS_NL; l++) {
                        const unsigned char * q = k.fq + ((size_t) l * m + r0) * D;
                        const float * s = k.fs + ((size_t) l * m + r0) * (D / 32);
                        gemv(st, q, s, k.L[l].kc, k.ckf, n == 1 ? 1 : 8);
                        gemv(st, q, s, k.L[l].vc, k.cvf, n == 1 ? 1 : 8);
                        k_ds_hnorm_rope<<<dim3(DS_NKVL, n), 128, 0, st>>>(k.ckf, k.L[l].kcn, DS_NKVL, k.ckf, k.L[l].kcn, 0, p0 + r0, DS_EPS);
                        k_ds_store<<<(n * DS_KVL + 255) / 256, 256, 0, st>>>(k.ckf, k.cvf, k.ck[l], k.cv[l], p0 + r0, n);
                    }
                }
            }
            if (c == 1) ICHECK(hipEventRecord(k.ev, st));
        });
        ICHECK(hipSetDevice(qz_dev_of(0)));
    }
    // The nd drafts (1 .. 7) after anchor tok at position a, context rows 0 .. a - 1 filled.
    // The 3 DFlash layers always run DS_BS rows; only the Markov chain stops early. With sa
    // each draft is drawn from q = softmax(logit * invTq) by Gumbel-max, keyed on the
    // request's seed and this round's counter; without it the chain is the greedy argmax.
    void draft(int tok, int a, int * out, const dsamp * sa, int nd) {
        each([&](int c, card & k, hipStream_t st) {
            const ink2_card & T = E->C[c];
            k_ds_embed<<<DS_BS, 256, 0, st>>>(T.embd.v, T.embd.s, tok, T.embd_norm, E->hp.eps, k.X, k.mask);
            k_ds_quant<<<DS_BS, 256, 0, st>>>(k.X, k.L[0].an, D, k.aq, k.as, nullptr, DS_EPS);
        });
        static_assert(DS_FFL % 256 == 0, "whole chunks");
        for (int l = 0; l < DS_NL; l++) {
            const int lo = l == 1 ? 0 : std::max(0, a - WIN);      // layers 0 / 2: sliding window
            each([&](int, card & k, hipStream_t st) {
                const layer & y = k.L[l];
                gemv_batch8(st, k.aq, k.as, D, {{&y.q, k.Q}, {&y.k, k.Kb}, {&y.v, k.Vb}});
                k_ds_hnorm_rope<<<dim3(DS_NQL + DS_NKVL, DS_BS), 128, 0, st>>>(k.Q, y.qn, DS_NQL, k.Kb, y.kn, DS_NKVL, a, DS_EPS);
                // the cache keys split into nsp tile-aligned chunks, so the grid keeps
                // DS_NKVL * nsp blocks busy at every context length
                const int nc = a - lo, cks = std::max(DS_TILE, ((nc + DS_NSPL - 1) / DS_NSPL + DS_TILE - 1) / DS_TILE * DS_TILE);
                const int nsp = std::max(1, (nc + cks - 1) / cks);
                k_ds_attn<<<dim3(DS_NKVL, nsp), DS_ATT_T, 0, st>>>(k.Q, k.ck[l], k.cv[l], k.Kb, k.Vb, lo, a, nsp, cks, k.part);
                k_ds_attn_merge_quant<<<dim3(DS_NQL, DS_BS), 128, 0, st>>>(k.part, k.aq, k.as, nsp);
                gemv(st, k.aq, k.as, y.o, k.Y, 8);
            });
            int s = exchange();
            each([&](int c, card & k, hipStream_t st) {
                const layer & y = k.L[l];
                k_ds_add_quant<<<DS_BS, 256, 0, st>>>(k.X, pa(c, s), pb(c, s), y.fn, D, k.aq, k.as, DS_EPS);
                gemv_batch8(st, k.aq, k.as, D, {{&y.gate, k.G}, {&y.up, k.U}});
                k_ds_swiglu_quant<<<dim3(DS_FFL / 256, DS_BS), 256, 0, st>>>(k.G, k.U, DS_FFL, k.aq, k.as);
                gemv(st, k.aq, k.as, y.down, k.Y, 8);
            });
            s = exchange();
            // the residual add of the two partials is fused into the next quantization, the
            // next layer's attention norm. After the last layer it takes hidden rows 1 .. nd
            // instead: f32 into hid for card 0's Markov input, int8 for the base logits.
            each([&](int c, card & k, hipStream_t st) {
                if (l + 1 < DS_NL) k_ds_add_quant<<<DS_BS, 256, 0, st>>>(k.X, pa(c, s), pb(c, s), k.L[l + 1].an, D, k.aq, k.as, DS_EPS);
                else {
                    k_ds_hidden<<<nd, 256, 0, st>>>(k.X, pa(c, s), pb(c, s), k.onorm, D, hq[c], hs[c], k.hid, DS_EPS);
                    gemv(st, hq[c], hs[c], E->C[c].head, base[c], nd);
                }
            });
        }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        hipStream_t st = E->C[0].st;
        const float * hid = K[0].hid;
        ICHECK(hipMemsetAsync(state, 0, DS_R * 4, st));
        // the nd Markov steps, each chained to the peer through lat1 and the argmax push.
        // With sa the packed value is the Gumbel-max's z, so the drafts are drawn from q.
        // The handshake's sequence number only ever counts up, so a shorter round leaves
        // both cards' flags consistent.
        const int nb[2] = { (E->C[0].nvalid + 256 * DS_FE - 1) / (256 * DS_FE),
                            (E->C[1].nvalid + 256 * DS_FE - 1) / (256 * DS_FE) };
        for (int i = 1; i <= nd; i++) {
            const unsigned int sq = ++seq;
            const int r = i - 1;
            dsamp s0 = { 0.f, 0, 0, nullptr, 0.f }, s1 = { 0.f, 0, 0, nullptr, 0.f };
            if (sa) {
                s0 = { sa->invTq, sa->seed, sa->rnd, lses[0] + (size_t) r * DS_LSEB * 2, sa->cap };
                s1 = { sa->invTq, sa->seed, sa->rnd, lses[1] + (size_t) r * DS_LSEB * 2, sa->cap };
            }
            k_ds_joint<<<3 * DS_R / 8, 512, 0, st>>>(jw, jb, state, w1, i, tok, hid + (size_t) r * D, g, joint, best[0], xbest[0],
                                                     fl[0], sq - 1, errd[0], out_d);
            k_ds_markov<<<1, 256, 0, st>>>(joint, state, lat1[0] + (size_t) r * (DS_R + 1), lat1[1] + (size_t) r * (DS_R + 1),
                                           best[0], xbest[0], fl[1], sq);
            const float * L0 = lat1[0] + (size_t) r * (DS_R + 1), * L1 = lat1[1] + (size_t) r * (DS_R + 1);
            const float * B0 = base[0] + (size_t) r * E->C[0].head.N, * B1 = base[1] + (size_t) r * E->C[1].head.N;
            if (sa)
                k_ds_fold_argmax<false, true><<<nb[0], 256, 0, st>>>(B0, fold[0], nhp[0], E->C[0].nvalid, L0,
                                                                     E->C[0].id0, best[0], fl[0], sq, errd[0], s0, (unsigned int) r);
            else
                k_ds_fold_argmax<false, false><<<nb[0], 256, 0, st>>>(B0, fold[0], nhp[0], E->C[0].nvalid, L0,
                                                                      E->C[0].id0, best[0], fl[0], sq, errd[0], s0, 0u);
            ICHECK(hipSetDevice(qz_dev_of(1)));
            hipStream_t s1s = E->C[1].st;
            if (sa)
                k_ds_fold_argmax<true, true><<<nb[1], 256, 0, s1s>>>(B1, fold[1], nhp[1], E->C[1].nvalid, L1,
                                                                     E->C[1].id0, best[1], fl[1], sq, errd[1], s1, (unsigned int) r);
            else
                k_ds_fold_argmax<true, false><<<nb[1], 256, 0, s1s>>>(B1, fold[1], nhp[1], E->C[1].nvalid, L1,
                                                                      E->C[1].id0, best[1], fl[1], sq, errd[1], s1, 0u);
            k_ds_best_push<<<1, 64, 0, s1s>>>(best[1], xbest[0], fl[0], sq);
            ICHECK(hipSetDevice(qz_dev_of(0)));
        }
        if (sa) {   // each card's per-row (max, sum) of the halves, merged before the host
            for (int c = 0; c < 2; c++) {
                ICHECK(hipSetDevice(qz_dev_of(c)));
                k_ds_lse_merge<<<DS_BS - 1, 64, 0, E->C[c].st>>>(lses[c], nb[c], lsem[c]);
            }
            ICHECK(hipSetDevice(qz_dev_of(0)));
        }
        k_ds_last<<<1, 64, 0, st>>>(best[0], xbest[0], fl[0], seq, errd[0], out_d, nd - 1);
        ICHECK(hipMemcpyAsync(out_h, out_d, 8 * sizeof(int), hipMemcpyDeviceToHost, st));
        ICHECK(hipStreamSynchronize(st));
        for (int i = 0; i < nd; i++) out[i] = out_h[i];
        for (int c = 0; c < 2; c++) {                  // either card's wait may have given up
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ICHECK(hipStreamSynchronize(E->C[c].st));
            ICHECK(hipMemcpy(&out_h[7], errd[c], sizeof(int), hipMemcpyDeviceToHost));
            if (out_h[7]) { fprintf(stderr, "dspark: the Markov chain's cross-card wait timed out (step %u, card %d)\n", seq, c), exit(1); }
        }
    }
    // ---- sampled rounds, after the verify
    // The drafter's logit of the target's candidate ids: ids is the host's [DS_BS][DS_QID]
    // list (-1 padded, row r's own candidates in its slots), lg the host's logits in the
    // same layout and lse the per-row logsumexp of the whole vocab. Each card evaluates
    // the ids it owns.
    void qlog(const int * ids, float * lg, float * lse) {
        memcpy(qid_h, ids, DS_BS * DS_QID * sizeof(int));
        for (int c = 0; c < 2; c++) {
            const ink2_card & T = E->C[c];
            ICHECK(hipSetDevice(qz_dev_of(c)));
            hipStream_t st = E->C[c].st;
            ICHECK(hipMemcpyAsync(qid_d[c], qid_h, DS_BS * DS_QID * sizeof(int), hipMemcpyHostToDevice, st));
            k_ds_qids<<<dim3(DS_QID, DS_BS - 1), 64, 0, st>>>(base[c], T.head.N, fold[c], nhp[c], T.nvalid, T.id0, qid_d[c],
                                                             lat1[c], qs_d[c]);
            ICHECK(hipMemcpyAsync(qs_h[c], qs_d[c], DS_BS * DS_QID * sizeof(float), hipMemcpyDeviceToHost, st));
            ICHECK(hipMemcpyAsync(lseh[c], lsem[c], DS_BS * 2 * sizeof(float), hipMemcpyDeviceToHost, st));
        }
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamSynchronize(E->C[c].st)); }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        for (int r = 0; r < DS_BS - 1; r++)            // an id's logit sits on the card that owns it
            for (int j = 0; j < DS_QID; j++)
                lg[r * DS_QID + j] = qs_h[ids[r * DS_QID + j] < E->C[1].id0 ? 0 : 1][r * DS_QID + j];
        for (int r = 0; r < DS_BS - 1; r++) {          // logaddexp of the two cards' halves
            const float m = fmaxf(lseh[0][2 * r], lseh[1][2 * r]);
            lse[r] = m + logf(lseh[0][2 * r + 1] * expf(lseh[0][2 * r] - m) + lseh[1][2 * r + 1] * expf(lseh[1][2 * r] - m));
        }
    }
};
