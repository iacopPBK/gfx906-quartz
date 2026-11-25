// DSpine drafter: the draft model's kernels, the TG / TSV tables built at load and the
// greedy chain, with qz_ds_block_round2 running one round of DS_T rows over both cards.
// Included by qwen35_model.cpp in namespace q35.
#pragma once

#include "core/link.h"
#include "dspine.h"
#include <initializer_list>

#ifndef DS_T
#define DS_T 12                  // served block rows
#endif
#define DS_C 1024               // code width
#define DS_R 512                // bridge rank
#define DS_FF 17408             // draft FF
#define DS_V 248320             // target vocab
#define DS_TOPK 16              // candidates per row

// ------------------------------------------------------------ kernels ----
// dequantize nrows Q8 rows to f32; with ids set, output row r reads source row ids[r]
__global__ void __launch_bounds__(256) ds_deq_rows(const unsigned char * qv, const unsigned short * qs,
                                                   const int * ids, float * y, int nrows, int n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nrows * n) return;
    const int r = (int) (i / n), e = (int) (i % n);
    const int sr = ids ? ids[r] : r;
    y[i] = (float) ((int8_t) qv[(size_t) sr * n + e]) *
           __half2float(__ushort_as_half(qs[(size_t) sr * (n / 32) + e / 32]));
}

// per-head RMSNorm then rope at position p0 + r, one 128-thread block per (row, head)
__global__ void __launch_bounds__(128) ds_norm_rope(float * x, const float * nw, const float * inv,
                                                    int hcount, int nrows, int p0) {
    const int h = blockIdx.x % hcount, r = blockIdx.x / hcount;
    if (r >= nrows) return;
    __shared__ float red[4], qs[DS_HD], rinv;
    const int e = threadIdx.x;
    float * xr = x + (size_t) r * ((size_t) hcount * DS_HD) + (size_t) h * DS_HD;
    const float v = xr[e];
    float ss = v * v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down(ss, off, 32);
    if ((e & 31) == 0) red[e >> 5] = ss;
    __syncthreads();
    if (e == 0) rinv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float) DS_HD + DS_EPS);
    __syncthreads();
    qs[e] = v * rinv * nw[e];
    __syncthreads();
    const int p = p0 + r;
    if (e < DS_HD / 2) {
        const float a = (float) p * inv[e];
        const float c = cosf(a), sn = sinf(a);
        const float x0 = qs[e], x1 = qs[e + DS_HD / 2];
        xr[e]               = x0 * c - x1 * sn;
        xr[e + DS_HD / 2]   = x0 * sn + x1 * c;
    }
}

// msg_t = sigmoid(q_t + s_{t-1} + b) * v_{t-1}, for t >= 1; packed rows are [q | s | v]
__global__ void __launch_bounds__(256) ds_gate_msg(const float * packed, const float * b,
                                                   float * msg, int nrows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = i / DS_R, e = i % DS_R;
    if (r < 1 || r >= nrows) return;
    const float * pr = packed + (size_t) r * (3 * DS_R);
    const float * prev = packed + (size_t) (r - 1) * (3 * DS_R);
    const float g = 1.f / (1.f + __expf(-(pr[e] + prev[DS_R + e] + b[e])));
    msg[(size_t) r * DS_R + e] = g * prev[2 * DS_R + e];
}

// out = raw + rms * delta for rows t >= 1; the anchor row (0) is unchanged
__global__ void __launch_bounds__(256) ds_out_add(float * raw, const float * delta, const float * rms,
                                                  int nrows, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t) i >= (size_t) nrows * n) return;
    const int r = i / n;
    raw[i] = raw[i] + (r > 0 ? rms[r] : 0.f) * delta[i];
}

// silu(gate) * up elementwise, written back over gate
__global__ void __launch_bounds__(256) ds_silu_mul(float * g, const float * u, int tot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t) i >= (size_t) tot) return;
    const float x = g[i];
    g[i] = x / (1.f + __expf(-x)) * u[i];
}

// ------------------------------------------------- fast draft kernels ----
// top-16 by value, ties to the lower id. Stage A runs per chunk, stage B merges.
#define DS_TK_CH 1024
__device__ __forceinline__ bool ds_tk_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}
__device__ __forceinline__ void ds_tk_wave_best(float & v, int & i) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        const float ov = __shfl_xor(v, off, 64);
        const int oi = __shfl_xor(i, off, 64);
        if (ds_tk_better(ov, oi, v, i)) { v = ov; i = oi; }
    }
}
// NV candidates per thread held in registers; each block-argmax round retires its winner
template <int NV> __device__ __forceinline__ void ds_tk_select(float (&v)[NV], int (&id)[NV],
                                                                float * outv, int * outi) {
    __shared__ float wv[4];
    __shared__ int wi[4];
    __shared__ int win;
    const int t = threadIdx.x;
    for (int k = 0; k < DS_TOPK; k++) {
        float bv = -3e30f;
        int bi = 0x7fffffff;
        #pragma unroll
        for (int j = 0; j < NV; j++)
            if (ds_tk_better(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; }
        ds_tk_wave_best(bv, bi);
        if ((t & 63) == 0) { wv[t >> 6] = bv; wi[t >> 6] = bi; }
        __syncthreads();
        if (t == 0) {
            float mv = wv[0];
            int mi = wi[0];
            for (int w = 1; w < 4; w++) if (ds_tk_better(wv[w], wi[w], mv, mi)) { mv = wv[w]; mi = wi[w]; }
            outv[k] = mv;
            outi[k] = mi;
            win = mi;
        }
        __syncthreads();
        const int wn = win;
        #pragma unroll
        for (int j = 0; j < NV; j++) if (id[j] == wn) { v[j] = -3e30f; id[j] = 0x7fffffff; }
        __syncthreads();
    }
}
__global__ void __launch_bounds__(256) ds_top16_a(const float * lg, float * pv, int * pi, int V, int nch, int id0 = 0) {
    const int c = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const float * row = lg + (size_t) r * V;
    float v[DS_TK_CH / 256];
    int id[DS_TK_CH / 256];
    #pragma unroll
    for (int j = 0; j < DS_TK_CH / 256; j++) {
        const int i = c * DS_TK_CH + t + j * 256;
        v[j] = i < V ? row[i] : -3e30f;
        id[j] = i < V ? i + id0 : 0x7fffffff;
    }
    ds_tk_select(v, id, pv + ((size_t) r * nch + c) * DS_TOPK, pi + ((size_t) r * nch + c) * DS_TOPK);
}
template <int NV>
__global__ void __launch_bounds__(256) ds_top16_b(const float * pv, const int * pi, int * out, int nch,
                                                  float * outv = nullptr) {
    const int r = blockIdx.x, t = threadIdx.x, n = nch * DS_TOPK;
    float v[NV];
    int id[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        const int i = t + j * 256;
        v[j] = i < n ? pv[(size_t) r * n + i] : -3e30f;
        id[j] = i < n ? pi[(size_t) r * n + i] : 0x7fffffff;
    }
    __shared__ float ov[DS_TOPK];
    __shared__ int oi[DS_TOPK];
    ds_tk_select(v, id, ov, oi);
    if (t < DS_TOPK) out[(size_t) r * DS_TOPK + t] = oi[t];
    if (outv && t < DS_TOPK) outv[(size_t) r * DS_TOPK + t] = ov[t];
}

// f32 T-row GEMV with float4 loads, K % 16 == 0; lanes walk float4 groups of a K quarter
template <int T>
__global__ void __launch_bounds__(256) ds_fgemvK4(const float * x, const float * w, float * y, int N, int K) {
    __shared__ float red[4][T][4];
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    const int n0 = (int) blockIdx.x * 4;
    const int kq = K / 4, k0 = wv * kq;
    float acc[T][4];
    #pragma unroll
    for (int t = 0; t < T; t++)
        #pragma unroll
        for (int r = 0; r < 4; r++) acc[t][r] = 0.f;
    for (int i = k0 + lane * 4; i < k0 + kq; i += 256) {
        float4 ww[4];
        #pragma unroll
        for (int r = 0; r < 4; r++)
            ww[r] = n0 + r < N ? *(const float4 *) (w + (size_t) (n0 + r) * K + i) : make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float4 xv = *(const float4 *) (x + (size_t) t * K + i);
            #pragma unroll
            for (int r = 0; r < 4; r++)
                acc[t][r] += xv.x * ww[r].x + xv.y * ww[r].y + xv.z * ww[r].z + xv.w * ww[r].w;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++)
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float a = acc[t][r];
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) a += __shfl_xor(a, off, 64);
            if (lane == 0) red[wv][t][r] = a;
        }
    __syncthreads();
    if (threadIdx.x < T * 4) {
        const int t = threadIdx.x >> 2, r = threadIdx.x & 3;
        if (n0 + r < N) y[(size_t) t * N + n0 + r] = red[0][t][r] + red[1][t][r] + red[2][t][r] + red[3][t][r];
    }
}

// block-wide sum over the 256 threads (4 waves), returned to every thread
__device__ __forceinline__ float ds_bsum(float v) {
    __shared__ float r4[4];
    v = qz_wsum(v);
    __syncthreads();
    if ((threadIdx.x & 63) == 0) r4[threadIdx.x >> 6] = v;
    __syncthreads();
    return r4[0] + r4[1] + r4[2] + r4[3];
}
// one block of 256 per row (ds_rms_b / ds_bz_b / ds_code_b), the row
// reductions over the whole block (2 blocks of 4 rows -> 8 parallel rows)
__global__ void __launch_bounds__(256) ds_rms_b(const float * x, const float * w, float * y, int n, float eps) {
    const float * xr = x + (size_t) blockIdx.x * n;
    float * yr = y + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    const float inv = rsqrtf(ds_bsum(ss) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = xr[i] * inv * w[i];
}
__global__ void __launch_bounds__(256) ds_bz_b(const float * raw, float * z, float * rms, int n) {
    const int r = blockIdx.x;
    const float * xr = raw + (size_t) r * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    const float rv = sqrtf(fmaxf(ds_bsum(ss) / (float) n, 1.18e-38f));
    if (threadIdx.x == 0) rms[r] = rv;
    const float inv = 1.f / (rv + DS_EPS);
    for (int i = threadIdx.x; i < n; i += 256) z[(size_t) r * n + i] = xr[i] * inv;
}
__global__ void __launch_bounds__(256) ds_code_b(float * proj, const unsigned char * cbv, const unsigned short * cbs,
                                                 int anchor, int n) {
    const int r = blockIdx.x;
    if (r == 0) {
        for (int i = threadIdx.x; i < n; i += 256)
            proj[i] = 32.f * (float) ((int8_t) cbv[(size_t) anchor * n + i]) *
                      __half2float(__ushort_as_half(cbs[(size_t) anchor * (n / 32) + i / 32]));
        return;
    }
    float * pr = proj + (size_t) r * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss += pr[i] * pr[i];
    const float inv = 32.f / (sqrtf(ds_bsum(ss)) + DS_EPS);
    for (int i = threadIdx.x; i < n; i += 256) pr[i] *= inv;
}

// block attention split over positions: one block per (split, kv head), partials merged
// by ds_attn_comb3. Sliding window DS_W, causal.

// ds_gemvR: T-row GEMV, T <= 12, activations in LDS, up to 4 weight segments per launch.
// nkc K chunks write partials at y + kc * ystr.
struct ds_seg { const char * w; const __half * ws; int N; float * y; int b0; };
struct ds_segs { ds_seg s[4]; int n; };
#ifndef DS_RS
#define DS_RS 2
#endif
#ifndef DS_RW
#define DS_RW 8
#endif
#ifndef DS_RL
#define DS_RL 32
#endif
template <int T, int S, int NW, int LG>
__global__ void __launch_bounds__(64 * NW) ds_gemvR(const unsigned char * a_vals, const float * a_scal, int K, int ldw,
                                                    ds_segs G, int ngr, int nkc, size_t ystr) {
    constexpr int NQ = 64 / LG;
    extern __shared__ uint4 lsa[];
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6, tid = threadIdx.x;
    const int qd = lane / LG, ql = lane % LG;
    const int nb = gridDim.x / nkc, kc = blockIdx.x / nb, bl = blockIdx.x % nb;
    const int au = K / 32, lsw = ldw / 32;
    const int u0 = kc * au / nkc, n_u = (kc + 1) * au / nkc - u0;
    const int scb = (T * n_u * 2 + 15) & ~15;
    const int nst = min(n_u, (65536 - scb) / (T * 32));
    __half * lsc = (__half *) (lsa + (size_t) T * 2 * nst);
    for (int i = tid; i < T * nst * 2; i += 64 * NW) {
        const int t = i / (nst * 2), r = i % (nst * 2);
        lsa[((size_t) t * 2 + (r & 1)) * nst + (r >> 1)] = *(const uint4 *)(a_vals + (size_t) t * K + u0 * 32 + r * 16);
    }
    for (int i = tid; i < T * n_u; i += 64 * NW) {
        const int t = i / n_u, u = i % n_u;
        lsc[u * T + t] = __float2half(a_scal[(size_t) t * au + u0 + u]);
    }
    __syncthreads();
    const __amdgpu_buffer_rsrc_t ra = __builtin_amdgcn_make_buffer_rsrc((void *) a_vals, (short) 0, T * K, 0x00027000);
    const int gw = __builtin_amdgcn_readfirstlane(bl * NW + wave);
    for (int g = gw; g < ngr; g += nb * NW) {
        int i = 0;
        #pragma unroll
        for (int j = 1; j < 4; j++) if (j < G.n && g >= G.s[j].b0) i = j;
        const char * w_vals = G.s[i].w;
        const __half * w_scal = G.s[i].ws;
        const int N = G.s[i].N;
        float * y = G.s[i].y + (size_t) kc * ystr;
        const int r0 = (g - G.s[i].b0) * S * NQ + qd * S;
        float acc[T][S];
        #pragma unroll
        for (int t = 0; t < T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++) acc[t][s] = 0.f;
        const char * wr[S]; const __half * wsr[S];
        #pragma unroll
        for (int s = 0; s < S; s++) {
            const int r = min(r0 + s, N - 1);
            wr[s] = w_vals + (size_t) r * ldw + u0 * 32; wsr[s] = w_scal + (size_t) r * lsw + u0;
        }
        struct WB { qz_v4u w0[S], w1[S]; unsigned short sh[S]; };
        auto ld = [&](int u, WB & b) {
            #pragma unroll
            for (int s = 0; s < S; s++) {
                b.w0[s] = __builtin_nontemporal_load((const qz_v4u *)(wr[s] + u * 32));
                b.w1[s] = __builtin_nontemporal_load((const qz_v4u *)(wr[s] + u * 32 + 16));
                b.sh[s] = __builtin_nontemporal_load((const unsigned short *) (wsr[s] + u));
            }
        };
        auto comp = [&](int u, const WB & b) {
            float sw[S];
            #pragma unroll
            for (int s = 0; s < S; s++) sw[s] = __half2float(__ushort_as_half(b.sh[s]));
            const bool st = u < nst;
            #pragma unroll
            for (int t = 0; t < T; t++) {
                uint4 a0, a1;
                if (st) { a0 = lsa[((size_t) t * 2) * nst + u]; a1 = lsa[((size_t) t * 2 + 1) * nst + u]; }
                else {
                    a0 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(ra, t * K + (u0 + u) * 32, 0, 0));
                    a1 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(ra, t * K + (u0 + u) * 32 + 16, 0, 0));
                }
                const float sa = __half2float(lsc[u * T + t]);
                #pragma unroll
                for (int s = 0; s < S; s++) {
                    int d = 0;
                    DOT4(d, b.w0[s].x, a0.x); DOT4(d, b.w0[s].y, a0.y); DOT4(d, b.w0[s].z, a0.z); DOT4(d, b.w0[s].w, a0.w);
                    DOT4(d, b.w1[s].x, a1.x); DOT4(d, b.w1[s].y, a1.y); DOT4(d, b.w1[s].z, a1.z); DOT4(d, b.w1[s].w, a1.w);
                    acc[t][s] += (float) d * sw[s] * sa;
                }
            }
        };
        WB A, B;
        int u = ql;
        if (u < n_u) ld(u, A);
        while (u < n_u) {
            int un = u + LG;
            if (un < n_u) ld(un, B);
            comp(u, A);
            u = un;
            if (u >= n_u) break;
            un = u + LG;
            if (un < n_u) ld(un, A);
            comp(u, B);
            u = un;
        }
        #pragma unroll
        for (int t = 0; t < T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++)
                #pragma unroll
                for (int off = LG / 2; off > 0; off >>= 1) acc[t][s] += __shfl_xor(acc[t][s], off, LG);
        if (ql == 0)
            #pragma unroll
            for (int t = 0; t < T; t++)
                #pragma unroll
                for (int s = 0; s < S; s++)
                    if (r0 + s < N) y[(size_t) t * N + r0 + s] = acc[t][s];
    }
}
// launch: segments {w, ws, N, y}; K is the activation width, ldw the weight row stride
template <int T>
inline void ds_gemvR_launchT(hipStream_t st, const unsigned char * a, const float * as, int K, int ldw,
                             std::initializer_list<ds_seg> segs, int nkc, size_t ystr) {
    constexpr int RPG = DS_RS * (64 / DS_RL);
    ds_segs G = {};
    int b = 0;
    for (const ds_seg & x : segs) { G.s[G.n] = x; G.s[G.n].b0 = b; b += (x.N + RPG - 1) / RPG; G.n++; }
    static bool attr[16] = {};
    int dev = 0;
    DS_CHECK(hipGetDevice(&dev));
    if (!attr[dev]) { DS_CHECK(hipFuncSetAttribute((const void *) ds_gemvR<T, DS_RS, DS_RW, DS_RL>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536)); attr[dev] = true; }
    ds_gemvR<T, DS_RS, DS_RW, DS_RL><<<dim3(60), dim3(64 * DS_RW), 65536, st>>>(a, as, K, ldw, G, b, nkc, ystr);
}
// nt: the round's live rows, 8 or DS_T. Rows past nt are stale and never read (causal).
inline void ds_gemvR_launch(int nt, hipStream_t st, const unsigned char * a, const float * as, int K, int ldw,
                            std::initializer_list<ds_seg> segs, int nkc = 1, size_t ystr = 0) {
    if (nt <= 8) ds_gemvR_launchT<8>(st, a, as, K, ldw, segs, nkc, ystr);
    else ds_gemvR_launchT<DS_T>(st, a, as, K, ldw, segs, nkc, ystr);
}
// y[i] += y[i + n], folding the K-chunk partials before an exchange
__global__ void ds_fold2(float * y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = y[i] + y[i + n];
}

#ifndef DS_AT_PS
#define DS_AT_PS 128             // positions per split (4 tiles: a 4x shorter combine)
#endif
// ps, the positions per split, is a runtime choice and a multiple of DS_A3_TL.
#define DS_A3_TL 16
#ifndef DS_A3_W
#define DS_A3_W 3                // waves per SIMD the register budget is fitted to
#endif
__global__ void __launch_bounds__(256, DS_A3_W) ds_attn_split3(const float * q, const float * kc, const float * vc,
                                                      int n_ctx, const float * kb, const float * vb,
                                                      int S, int wbase, int ps, float * part_o, float * part_ml,
                                                      int nql = DS_NQ, int h0 = 0) {
    __shared__ float4 ks[DS_A3_TL][DS_HD / 4];
    __shared__ float4 vs[DS_A3_TL][DS_HD / 4];
    const int sp = blockIdx.x, h = blockIdx.y, th = threadIdx.x;
    const int qi = th >> 3, g = th & 7;
    const int qt_raw = blockIdx.z * 8 + (qi >> 2), qt = min(qt_raw, DS_T - 1), qj = qi & 3, hq = h * 4 + qj;   // blockIdx.z: group of 8 rows; rows past DS_T mirror the last
    const int ncl = n_ctx - wbase;
    const float4 * qrow = (const float4 *) (q + (size_t) qt * (nql * DS_HD) + (size_t) hq * DS_HD);
    float4 qr[4], acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { qr[j] = qrow[8 * j + g]; acc[j] = make_float4(0.f, 0.f, 0.f, 0.f); }
    float m = -3e30f, l = 0.f;
    const int p_lo = sp * ps, p_hi = min(S, p_lo + ps);
    const float4 * kb4 = (const float4 *) kb, * vb4 = (const float4 *) vb;
    const float4 * kc4 = (const float4 *) kc, * vc4 = (const float4 *) vc;
    float4 pk[2], pv[2];
    auto load = [&](int p0) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            const int i = th + 256 * r, a = i >> 5, e4 = i & 31, p = p0 + a;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (p < p_hi) {
                const int nb = p - ncl;
                const size_t off = (size_t) h * (DS_HD / 4) + e4;
                if (nb >= 0) {
                    kv = kb4[(size_t) nb * (nql / 4 * DS_HD / 4) + off];
                    vv = vb4[(size_t) nb * (nql / 4 * DS_HD / 4) + off];
                } else {
                    const size_t sl = (size_t) ((wbase + p) % DS_RING) * (DS_NKV * DS_HD / 4) + (size_t) h0 * (DS_HD / 4);
                    kv = kc4[sl + off];
                    vv = vc4[sl + off];
                }
            }
            pk[r] = kv;
            pv[r] = vv;
        }
    };
    for (int p0 = p_lo; p0 < p_hi; p0 += DS_A3_TL) {
        load(p0);
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            const int i = th + 256 * r;
            ks[i >> 5][i & 31] = pk[r];
            vs[i >> 5][i & 31] = pv[r];
        }
        __syncthreads();
        // positions k and 8 + k are summed together by the first reduce-scatter step,
        // with lanes g & 4 keeping 8 + k; 8 live partials remain
        float v8[8], v4[4], v2[2];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float dl = 0.f, dh = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 kl = ks[k][8 * j + g], kh = ks[8 + k][8 * j + g];
                dl += qr[j].x * kl.x + qr[j].y * kl.y + qr[j].z * kl.z + qr[j].w * kl.w;
                dh += qr[j].x * kh.x + qr[j].y * kh.y + qr[j].z * kh.z + qr[j].w * kh.w;
            }
            const bool hi = g & 4;
            v8[k] = (hi ? dh : dl) + __shfl_xor(hi ? dl : dh, 4, 8);
        }
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const bool hi = g & 2;
            const float mine = hi ? v8[4 + k] : v8[k], oth = hi ? v8[k] : v8[4 + k];
            v4[k] = mine + __shfl_xor(oth, 2, 8);
        }
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            const bool hi = g & 1;
            const float mine = hi ? v4[2 + k] : v4[k], oth = hi ? v4[k] : v4[2 + k];
            v2[k] = mine + __shfl_xor(oth, 1, 8);
        }
        float s[2], tmax = -3e30f;
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            const int p = p0 + 2 * g + k;
            bool ok = p < p_hi;
            if (ok) {
                const int nb = p - ncl;
                if (nb >= 0) ok = nb <= qt;
                else ok = wbase + p > n_ctx + qt - DS_W;
            }
            s[k] = ok ? v2[k] * 0.08838834764831845f : -3e30f;
            tmax = fmaxf(tmax, s[k]);
        }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor(tmax, off, 8));
        const float mn = fmaxf(m, tmax);
        const float corr = __expf(m - mn);
        float pw[2], ls = 0.f;
        #pragma unroll
        for (int k = 0; k < 2; k++) { pw[k] = s[k] > -1e30f ? __expf(s[k] - mn) : 0.f; ls += pw[k]; }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) ls += __shfl_xor(ls, off, 8);
        l = l * corr + ls;
        m = mn;
        #pragma unroll
        for (int j = 0; j < 4; j++) { acc[j].x *= corr; acc[j].y *= corr; acc[j].z *= corr; acc[j].w *= corr; }
        #pragma unroll 2
        for (int a = 0; a < DS_A3_TL; a++) {
            const float w = __shfl((a & 1) ? pw[1] : pw[0], a >> 1, 8);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 v = vs[a][8 * j + g];
                acc[j].x += w * v.x; acc[j].y += w * v.y; acc[j].z += w * v.z; acc[j].w += w * v.w;
            }
        }
    }
    const size_t qrw = (size_t) qt * nql + hq;
    float4 * po = (float4 *) (part_o + ((size_t) sp * (DS_T * nql) + qrw) * DS_HD);
    if (qt_raw >= DS_T) return;
    #pragma unroll
    for (int j = 0; j < 4; j++) po[8 * j + g] = acc[j];
    if (g == 0) {
        part_ml[((size_t) sp * (DS_T * nql) + qrw) * 2 + 0] = m;
        part_ml[((size_t) sp * (DS_T * nql) + qrw) * 2 + 1] = l;
    }
}
// merges ds_attn_split3's partials (a thread per split, then independent part_o loads)
__global__ void __launch_bounds__(128) ds_attn_comb3(const float * part_o, const float * part_ml, float * o, int nsp,
                                                     int nql = DS_NQ) {
    __shared__ float ws[128], red[2];
    const int qrow = blockIdx.x, e = threadIdx.x;
    const size_t st = (size_t) DS_T * nql;
    const float ms = e < nsp ? part_ml[((size_t) e * st + qrow) * 2] : -3e30f;
    const float ls = e < nsp ? part_ml[((size_t) e * st + qrow) * 2 + 1] : 0.f;
    float M = ms;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) M = fmaxf(M, __shfl_xor(M, off, 64));
    if ((e & 63) == 0) red[e >> 6] = M;
    __syncthreads();
    M = fmaxf(red[0], red[1]);
    const float w = ms > -1e30f ? __expf(ms - M) : 0.f;
    ws[e] = w;
    float L = w * ls;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) L += __shfl_xor(L, off, 64);
    __syncthreads();
    if ((e & 63) == 0) red[e >> 6] = L;
    __syncthreads();
    L = red[0] + red[1];
    float acc = 0.f;
    #pragma unroll 8
    for (int s = 0; s < nsp; s++) acc += ws[s] * part_o[((size_t) s * st + qrow) * DS_HD + e];
    o[(size_t) qrow * DS_HD + e] = L > 0.f ? acc / L : 0.f;
}
// ps for ds_attn_split3: ~3 blocks per CU over 60 CUs, at least 2 tiles, capped by at_o
static inline int ds_a3_ps(int S, int heads) {
    const int target = (180 + heads - 1) / heads;
    int ps = (S + target - 1) / target;
    ps = (ps + DS_A3_TL - 1) / DS_A3_TL * DS_A3_TL;
    return ps < 2 * DS_A3_TL ? 2 * DS_A3_TL : ps;
}

// ------------------------------------------------ the two-card round ----
// x += a + b: the two cards' partials, bit-identical on both cards
__global__ void __launch_bounds__(256) ds_add2(float * x, const float * a, const float * b, int tot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tot) x[i] = x[i] + (a[i] + b[i]);
}
// global top-16 merge of the two vocab halves, ordered by value, ties to the lower id
__global__ void __launch_bounds__(256) ds_top16_m(const float * av, const int * ai, const float * bv, const int * bi,
                                                  int * out) {
    const int r = blockIdx.x, t = threadIdx.x;
    float v[1];
    int id[1];
    v[0] = t < DS_TOPK ? av[r * DS_TOPK + t] : t < 2 * DS_TOPK ? bv[r * DS_TOPK + t - DS_TOPK] : -3e30f;
    id[0] = t < DS_TOPK ? ai[r * DS_TOPK + t] : t < 2 * DS_TOPK ? bi[r * DS_TOPK + t - DS_TOPK] : 0x7fffffff;
    __shared__ float ov[DS_TOPK];
    __shared__ int oi[DS_TOPK];
    ds_tk_select(v, id, ov, oi);
    if (t < DS_TOPK) out[r * DS_TOPK + t] = oi[t];
}

// a candidate's bas: <raw_{i+1} * onorm, deq(head row)>, one wave, double accumulation
__device__ __forceinline__ float ds_bas_dot(const float * rr, const float * onorm, const unsigned char * hv,
                                            const unsigned short * hs, int lane) {
    double acc = 0.0;
    for (int e = lane; e < DS_H; e += 64) {
        const float rw = rr[e] * onorm[e];
        const float hd = (float) ((int8_t) hv[e]) * __half2float(__ushort_as_half(hs[e / 32]));
        acc += (double) rw * hd;
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) acc += __shfl_down(acc, off, 64);
    return (float) acc;                               // the host's sdot_s returns float
}
// each card's bas for the top-16 of its own vocab half; ds_bas_pick merges them
__global__ void __launch_bounds__(256) ds_chain_bas_l(const float * raw, const float * onorm, const int * ids, int id0,
                                                      const unsigned char * hv, const unsigned short * hs, float * bas) {
    const int w = blockIdx.x * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    if (w >= (DS_T - 1) * DS_TOPK) return;
    const int i = w / DS_TOPK;
    const size_t r = (size_t) (ids[w] - id0);
    const float b = ds_bas_dot(raw + (size_t) (i + 1) * DS_H, onorm, hv + r * DS_H, hs + r * (DS_H / 32), lane);
    if (lane == 0) bas[w] = b;
}
// this card's top-16 TG / TSV rows, and the anchor's TSV row when its id is in its half
__global__ void __launch_bounds__(256) ds_rows_l(const float * TG, const float * TSV, const int * ids, int id0, int anchor,
                                                 float * tg, float * tsv, float * tsva) {
    const int w = blockIdx.x, t = threadIdx.x;
    if (w < (DS_T - 1) * DS_TOPK) {
        const size_t r = (size_t) (ids[w] - id0);
        for (int e = t; e < DS_R; e += 256) tg[(size_t) w * DS_R + e] = TG[r * DS_R + e];
        for (int e = t; e < 2 * DS_R; e += 256) tsv[(size_t) w * 2 * DS_R + e] = TSV[r * 2 * DS_R + e];
    } else if (anchor >= id0 && anchor < id0 + DS_V / QZ_NC) {
        const size_t r = (size_t) (anchor - id0);
        for (int e = t; e < 2 * DS_R; e += 256) tsva[e] = TSV[r * 2 * DS_R + e];
    }
}

// ----------------------------------------------------------------- host ----
// the block-pass state on one card; the context rings live in qz_ds_ctx
struct qz_ds_block {
    int dev = 0;
    hipStream_t st = nullptr;
    qz_loader ld;                     // the draft's block, bridge and codebook tensors
    // per-layer Q8 planes
    const unsigned char * q_v[DS_FL] = {}, * o_v[DS_FL] = {}, * g_v[DS_FL] = {}, * u_v[DS_FL] = {}, * d_v[DS_FL] = {};
    const unsigned short * q_s[DS_FL] = {}, * o_s[DS_FL] = {}, * g_s[DS_FL] = {}, * u_s[DS_FL] = {}, * d_s[DS_FL] = {};
    // f32 bridge weights (converted from BF16 at load)
    const float * anorm[DS_FL] = {}, * pnorm[DS_FL] = {}, * onorm = nullptr, * qnw[DS_FL] = {};
    const float * Rl[DS_FL] = {}, * Wl[DS_FL] = {}, * P = nullptr, * Psv = nullptr, * b_gate = nullptr;
    const unsigned char * cb_v = nullptr;
    const unsigned short * cb_s = nullptr;
    // target emb / head planes + the tables
    const unsigned char * emb_v = nullptr;
    const unsigned short * emb_s = nullptr;
    const unsigned char * hh_v = nullptr;             // this card's head rows [dev V/NC, (dev + 1) V/NC)
    const unsigned short * hh_s = nullptr;
    float * bas_l = nullptr;                          // [T-1][16] bas of this card's half top-16 (tki)
    float * pbas = nullptr;                           // card 0: the peer half's
    float * TG = nullptr, * TSV = nullptr;             // this card's rows [dev V/NC, (dev + 1) V/NC) of [V][R], [V][2R]
    float * tg_l = nullptr, * tsv_l = nullptr, * tsva_l = nullptr;   // ds_rows_l's output: this card's top-16 rows and the anchor's
    float * ptg = nullptr, * ptsv = nullptr, * ptsva = nullptr;      // card 0: the peer's
    float * tg_s = nullptr, * tsv_s = nullptr, * tsva_s = nullptr;   // card 0: ds_rows_pick's staged rows
    int ids[DS_T] = {};
    int * ids_d = nullptr;
    float * x = nullptr;                               // emb rows / resid stream [T][H]
    float * t1 = nullptr, * t2 = nullptr;  // [T][H] scratch
    unsigned char * q1 = nullptr;                     // [T][H] values
    float * s1 = nullptr;                             // [T][H/32] scales
    float * xa = nullptr;                              // [T][4096] q after gemv, roped
    float * kb = nullptr, * vb = nullptr;             // [T][NKV*HD] block k / v
    float * sc = nullptr;                             // [T][NQ][DS_W + DS_T] scores
    float * att = nullptr;                             // [T][4096] attention out
    unsigned char * qq2 = nullptr;                    // [T][4096] values
    float * qs2 = nullptr;                             // [T][4096/32]
    float * fg = nullptr, * fu = nullptr;             // [T][FF]
    unsigned char * q3 = nullptr;                     // [T][FF] values
    float * s3 = nullptr;                              // [T][FF/32]
    float * z = nullptr;                               // [T][H]
    float * proj = nullptr;                            // [T][C]
    float * packed = nullptr;                          // [T][3R]
    float * msg = nullptr;                             // [T][R]
    float * delta = nullptr;                           // [T][H]
    float * rmsb = nullptr;                            // [T]
    float * stash_raw = nullptr;                        // [T][H] the last layer's u, copied before the bridge; the chain's raw input
    float * logits = nullptr;                          // [T][DS_V]
    int * cand = nullptr;                              // [T][DS_TOPK] rows 1..T-1 used
    float * hist_hdr = nullptr;                       // [T][DS_TOPK][H] dequanted cand head rows
    float * at_o = nullptr, * at_ml = nullptr;        // ds_attn_split3 partials
    float * tk_v = nullptr;                           // ds_top16_a partials
    // the two-card round: the peer exchange inboxes and the top-16 halves
    float * xin[2] = {};                              // [T][H] peer partials (written by the peer's copy)
    float * tkv = nullptr;                            // [T-1][16] this card's half top-16 values
    int * tki = nullptr;                              // [T-1][16] ids
    float * ptkv = nullptr;                           // card 0: the peer half's values / ids
    int * ptki = nullptr;
    hipEvent_t ev = nullptr;
    int * tk_i = nullptr;
};

// read a draft tensor as f32 (BF16 converted, F32 copied); the bridge tensors and norms
inline std::vector<float> ds_read_f32(qz_loader & ld, const qz_gguf & g, const char * path,
                                      const char * name, int64_t * out_ne0 = nullptr) {
    const int64_t i = ld.find(name);
    if (i < 0) { fprintf(stderr, "draft tensor missing: %s\n", name); exit(1); }
    const qz_dev_tensor & t = ld.ts[i];
    const int64_t n = t.ne0 * t.ne1;
    std::vector<float> f(n);
    FILE * fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (t.type == GGML_TYPE_F32) {
        fseek(fp, (long) ld.raw_off[i], SEEK_SET);
        if (fread(f.data(), 4, n, fp) != (size_t) n) { fprintf(stderr, "short read %s\n", name); exit(1); }
    } else if (t.type == GGML_TYPE_BF16) {
        std::vector<uint16_t> b(n);
        fseek(fp, (long) ld.raw_off[i], SEEK_SET);
        if (fread(b.data(), 2, n, fp) != (size_t) n) { fprintf(stderr, "short read %s\n", name); exit(1); }
        for (int64_t j = 0; j < n; j++) {
            uint32_t u = (uint32_t) b[j] << 16;
            memcpy(&f[j], &u, 4);
        }
    } else {
        fprintf(stderr, "ds_read_f32: %s is not F32/BF16 (type %d)\n", name, (int) t.type);
        exit(1);
    }
    fclose(fp);
    (void) g;
    if (out_ne0) *out_ne0 = t.ne0;
    return f;
}

// ------------------------------------------------------------- tables ----
// builds TG / TSV: out = alpha * (deq(A) * ks) @ B, A Q8_0, B f32; N % 64, K % 32
__global__ void __launch_bounds__(256) ds_tab_gemm(const unsigned char * av, const unsigned short * as, int64_t lda,
                                                   const float * ks, const float * B, float * out,
                                                   int N, int K, float alpha) {
    __shared__ float At[32][64 + 4];
    __shared__ float Bt[32][64];
    const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
    const int64_t m0 = (int64_t) blockIdx.y * 64;
    const int n0 = blockIdx.x * 64;
    const int ar = t >> 2, ak = (t & 3) * 8;          // A load: row ar, k ak .. ak + 7
    const int bk = t >> 3, bn = (t & 7) * 8;          // B load: k bk, n bn .. bn + 7
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int64_t m = m0 + ar;
        const uint2 q = *(const uint2 *) (av + m * lda + k0 + ak);
        const float sc = __half2float(__ushort_as_half(as[m * (K / 32) + k0 / 32]));
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int8_t qv = (int8_t) (((e < 4 ? q.x : q.y) >> (8 * (e & 3))) & 0xff);
            const float d = (float) qv * sc;
            At[ak + e][ar] = ks ? d * ks[k0 + ak + e] : d;
        }
        const float4 * bp = (const float4 *) (B + (size_t) (k0 + bk) * N + n0 + bn);
        *(float4 *) &Bt[bk][bn] = bp[0];
        *(float4 *) &Bt[bk][bn + 4] = bp[1];
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < 32; k++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = At[k][ty * 4 + i]; b[i] = Bt[k][tx * 4 + i]; }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float4 o = make_float4(alpha * acc[i][0], alpha * acc[i][1], alpha * acc[i][2], alpha * acc[i][3]);
        *(float4 *) (out + (size_t) (m0 + ty * 4 + i) * N + n0 + tx * 4) = o;
    }
}

// --------------------------------------------------------- load / free ----
// loads the block-pass tensors from the draft GGUF, then this card's half of TG / TSV
inline int qz_ds_block_load(qz_ds_block & d, qz_ds_ctx & cx, const char * draft,
                            const qz_dev_tensor * emb, const qz_dev_tensor * head_half) {
    d.dev = cx.dev;
    DS_CHECK(hipSetDevice(d.dev));
    qz_mem_scope mem("dspine");
    qz_gguf g;
    if (!g.open(draft)) { fprintf(stderr, "cannot open draft %s\n", draft); return 1; }
    static auto need_blk = [](const char * n) {
        int l = -1;
        if (sscanf(n, "blk.%d.", &l) != 1 || l < 0 || l >= DS_FL) {
            return !strcmp(n, "output_norm.weight") || !strcmp(n, "dspine_gate.bias") ||
                   !strcmp(n, "dspine_codebook.weight");
        }
        const char * suf = strchr(n + 4, '.');
        if (!suf) return false;
        const char * s = suf + 1;
        return !strcmp(s, "attn_q.weight") || !strcmp(s, "attn_q_norm.weight") ||
               !strcmp(s, "attn_norm.weight") || !strcmp(s, "attn_output.weight") ||
               !strcmp(s, "ffn_norm.weight") || !strcmp(s, "ffn_gate.weight") ||
               !strcmp(s, "ffn_up.weight") || !strcmp(s, "ffn_down.weight") ||
               !strcmp(s, "dspine_read.weight") || !strcmp(s, "dspine_write.weight");
    };
    d.ld.skip = [](const char * n) { return !need_blk(n); };
    if (!d.ld.load(draft, g, d.dev)) return 1;
    if ((emb->type != GGML_TYPE_Q8_0 || emb->ne0 != DS_H || emb->ne1 != DS_V || (emb->vstride && emb->vstride != DS_H))) {
        fprintf(stderr, "qz_ds_block: the shared token_embd is not the dense [%d x %d] Q8_0 the drafter reads\n", DS_H, DS_V);
        return 1;
    }
    if ((head_half->type != GGML_TYPE_Q8_0 || head_half->ne0 != DS_H || head_half->ne1 != DS_V / QZ_NC ||
                      head_half->vstride != (int64_t) DS_H)) {
        fprintf(stderr, "qz_ds_block: the shared head half is not the [%d x %d] Q8_0 (row stride %d) the drafter reads\n",
                DS_H, DS_V / QZ_NC, (int) DS_H);
        return 1;
    }

    auto fd = [&](const char * n) -> const qz_dev_tensor & {
        const int64_t i = d.ld.find(n);
        if (i < 0) { fprintf(stderr, "draft tensor missing: %s\n", n); exit(1); }
        return d.ld.ts[i];
    };
    char n[80];
    auto upload_f32 = [&](const std::vector<float> & v) -> const float * {
        void * p = nullptr;
        DS_CHECK(qz_dmalloc(&p, v.size() * 4));
        DS_CHECK(hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice));
        return (const float *) p;
    };
    for (int l = 0; l < DS_FL; l++) {
        snprintf(n, sizeof(n), "blk.%d.attn_q.weight", l);
        d.q_v[l] = (const unsigned char *) fd(n).vals;   d.q_s[l] = (const unsigned short *) fd(n).scal;
        snprintf(n, sizeof(n), "blk.%d.attn_output.weight", l);
        d.o_v[l] = (const unsigned char *) fd(n).vals;   d.o_s[l] = (const unsigned short *) fd(n).scal;
        snprintf(n, sizeof(n), "blk.%d.ffn_gate.weight", l);
        d.g_v[l] = (const unsigned char *) fd(n).vals;   d.g_s[l] = (const unsigned short *) fd(n).scal;
        snprintf(n, sizeof(n), "blk.%d.ffn_up.weight", l);
        d.u_v[l] = (const unsigned char *) fd(n).vals;   d.u_s[l] = (const unsigned short *) fd(n).scal;
        snprintf(n, sizeof(n), "blk.%d.ffn_down.weight", l);
        d.d_v[l] = (const unsigned char *) fd(n).vals;   d.d_s[l] = (const unsigned short *) fd(n).scal;
        snprintf(n, sizeof(n), "blk.%d.attn_norm.weight", l);
        d.anorm[l] = upload_f32(ds_read_f32(d.ld, g, draft, n));
        snprintf(n, sizeof(n), "blk.%d.ffn_norm.weight", l);
        d.pnorm[l] = upload_f32(ds_read_f32(d.ld, g, draft, n));
        snprintf(n, sizeof(n), "blk.%d.attn_q_norm.weight", l);
        d.qnw[l] = upload_f32(ds_read_f32(d.ld, g, draft, n));
        snprintf(n, sizeof(n), "blk.%d.dspine_read.weight", l);
        d.Rl[l] = upload_f32(ds_read_f32(d.ld, g, draft, n));
        snprintf(n, sizeof(n), "blk.%d.dspine_write.weight", l);
        d.Wl[l] = upload_f32(ds_read_f32(d.ld, g, draft, n));
    }
    d.onorm = upload_f32(ds_read_f32(d.ld, g, draft, "output_norm.weight"));
    d.b_gate = upload_f32(ds_read_f32(d.ld, g, draft, "dspine_gate.bias"));
    {
        std::vector<float> P((size_t) 3 * DS_R * DS_C, 0.f);
        const char * parts[3] = { "dspine_gate.weight", "dspine_pred_gate.weight", "dspine_prev.weight" };
        for (int p = 0; p < 3; p++) {
            std::vector<float> t = ds_read_f32(d.ld, g, draft, parts[p]);
            memcpy(P.data() + (size_t) p * DS_R * DS_C, t.data(), t.size() * 4);
        }
        d.P = upload_f32(P);
        // Psv^T: [C][2R] with PsvT[c][o] = P[512 + o][c]
        std::vector<float> PsvT((size_t) DS_C * 2 * DS_R);
        for (int o = 0; o < 2 * DS_R; o++)
            for (int c = 0; c < DS_C; c++)
                PsvT[(size_t) c * 2 * DS_R + o] = P[(size_t) (DS_R + o) * DS_C + c];
        d.Psv = upload_f32(PsvT);
    }
    const qz_dev_tensor & cb = fd("dspine_codebook.weight");
    d.cb_v = (const unsigned char *) cb.vals;
    d.cb_s = (const unsigned short *) cb.scal;
    d.emb_v = (const unsigned char *) emb->vals;
    d.emb_s = (const unsigned short *) emb->scal;
    d.hh_v = (const unsigned char *) head_half->vals;
    d.hh_s = (const unsigned short *) head_half->scal;

    // TG / TSV: this card's vocab half of each table, rows [dev V/2, (dev + 1) V/2),
    // computed here from the head half, the codebook and the bridge weights
    {
        const int64_t wne0 = fd("blk.4.dspine_write.weight").ne0;
        if (wne0 != DS_R || fd("blk.4.dspine_write.weight").ne1 != DS_H || cb.ne0 != DS_C || cb.ne1 != DS_V ||
            (cb.vstride && cb.vstride != DS_C)) {
            fprintf(stderr, "qz_ds_block: dspine_write / dspine_codebook shapes differ from the drafter's\n");
            return 1;
        }
        const int V2 = DS_V / QZ_NC;
        DS_CHECK(qz_dmalloc((void **) &d.TG, (size_t) V2 * DS_R * 4, "tables"));
        DS_CHECK(qz_dmalloc((void **) &d.TSV, (size_t) V2 * 2 * DS_R * 4, "tables"));
        ds_tab_gemm<<<dim3(DS_R / 64, V2 / 64), 256>>>(d.hh_v, d.hh_s, DS_H, d.onorm, d.Wl[DS_FL - 1], d.TG,
                                                         DS_R, DS_H, 1.f);
        const size_t cro = (size_t) d.dev * V2;       // this card's codebook rows
        ds_tab_gemm<<<dim3(2 * DS_R / 64, V2 / 64), 256>>>(d.cb_v + cro * DS_C, d.cb_s + cro * (DS_C / 32), DS_C,
                                                             nullptr, d.Psv, d.TSV, 2 * DS_R, DS_C, 32.f);
        DS_CHECK(hipGetLastError());
        DS_CHECK(hipDeviceSynchronize());
    }

    auto fz = [&](void ** p, size_t b) { DS_CHECK(qz_dmalloc(p, b)); DS_CHECK(hipMemset(*p, 0, b)); };
    fz((void **) &d.ids_d, DS_T * 4);
    fz((void **) &d.x, (size_t) DS_T * DS_H * 4);
    fz((void **) &d.t1, (size_t) 2 * DS_T * DS_H * 4);       // [2][T][H], the second plane holds the down gemv's K-chunk partials
    fz((void **) &d.t2, (size_t) DS_T * DS_H * 4);
    fz((void **) &d.q1, (size_t) DS_T * DS_H);
    fz((void **) &d.s1, (size_t) DS_T * (DS_H / 32) * 4);
    fz((void **) &d.xa, (size_t) DS_T * (DS_NQ * DS_HD) * 4);
    fz((void **) &d.kb, (size_t) DS_T * (DS_NKV * DS_HD) * 4);
    fz((void **) &d.vb, (size_t) DS_T * (DS_NKV * DS_HD) * 4);
    fz((void **) &d.sc, (size_t) DS_T * DS_NQ * (DS_W + DS_T) * 4);
    fz((void **) &d.att, (size_t) DS_T * (DS_NQ * DS_HD) * 4);
    fz((void **) &d.qq2, (size_t) DS_T * (DS_NQ * DS_HD));
    fz((void **) &d.qs2, (size_t) DS_T * (DS_NQ * DS_HD / 32) * 4);
    fz((void **) &d.fg, (size_t) DS_T * DS_FF * 4);
    fz((void **) &d.fu, (size_t) DS_T * DS_FF * 4);
    fz((void **) &d.q3, (size_t) DS_T * DS_FF);
    fz((void **) &d.s3, (size_t) DS_T * (DS_FF / 32) * 4);
    fz((void **) &d.z, (size_t) DS_T * DS_H * 4);
    fz((void **) &d.proj, (size_t) DS_T * DS_C * 4);
    fz((void **) &d.packed, (size_t) DS_T * 3 * DS_R * 4);
    fz((void **) &d.msg, (size_t) DS_T * DS_R * 4);
    fz((void **) &d.delta, (size_t) DS_T * DS_H * 4);
    fz((void **) &d.rmsb, (size_t) DS_T * 4);
    fz((void **) &d.stash_raw, (size_t) DS_T * DS_H * 4);
    fz((void **) &d.logits, (size_t) DS_T * DS_V * 4);
    fz((void **) &d.cand, (size_t) (DS_T - 1) * DS_TOPK * 4);
    fz((void **) &d.hist_hdr, (size_t) (DS_T - 1) * DS_TOPK * DS_H * 4);
    {
        const int nsp = (DS_W + DS_T + DS_AT_PS - 1) / DS_AT_PS, nch = (DS_V + DS_TK_CH - 1) / DS_TK_CH;
        fz((void **) &d.at_o, (size_t) nsp * DS_T * DS_NQ * DS_HD * 4);
        fz((void **) &d.at_ml, (size_t) nsp * DS_T * DS_NQ * 2 * 4);
        fz((void **) &d.tk_v, (size_t) (DS_T - 1) * nch * DS_TOPK * 4);
        fz((void **) &d.tk_i, (size_t) (DS_T - 1) * nch * DS_TOPK * 4);
    }
    fz((void **) &d.xin[0], (size_t) DS_T * DS_H * 4);
    fz((void **) &d.xin[1], (size_t) DS_T * DS_H * 4);
    fz((void **) &d.tkv, (size_t) DS_T * DS_TOPK * 4);
    fz((void **) &d.tki, (size_t) DS_T * DS_TOPK * 4);
    fz((void **) &d.ptkv, (size_t) DS_T * DS_TOPK * 4);
    fz((void **) &d.ptki, (size_t) DS_T * DS_TOPK * 4);
    fz((void **) &d.bas_l, (size_t) DS_T * DS_TOPK * 4);
    {
        const size_t nr = (size_t) (DS_T - 1) * DS_TOPK;
        for (float ** p : { &d.tg_l, &d.ptg, &d.tg_s }) fz((void **) p, nr * DS_R * 4);
        for (float ** p : { &d.tsv_l, &d.ptsv, &d.tsv_s }) fz((void **) p, nr * 2 * DS_R * 4);
        for (float ** p : { &d.tsva_l, &d.ptsva, &d.tsva_s }) fz((void **) p, 2 * DS_R * 4);
    }
    fz((void **) &d.pbas, (size_t) DS_T * DS_TOPK * 4);
    DS_CHECK(hipEventCreateWithFlags(&d.ev, hipEventDisableTiming));
    DS_CHECK(hipStreamCreateWithFlags(&d.st, hipStreamNonBlocking));
    printf("qz_ds_block: dev %d, block tensors %.2f GB\n", d.dev, d.ld.need_bytes / 1e9);
    return 0;
}

// the two-card block round. Card c owns half the heads, FFN rows and head rows; the
// o / down partials are summed on both cards and the chain inputs end on card 0.
inline void qz_ds_block_round2(qz_ds_block * D, qz_ds_ctx * CX, int anchor, int p0, int wbase, int nt = DS_T) {
    const int S = p0 - wbase + DS_T;
    const int H = DS_H, FF2 = DS_FF / QZ_NC, QWl = DS_NQ / QZ_NC * DS_HD, KVl = DS_NKV / QZ_NC * DS_HD, V2 = DS_V / QZ_NC;
    const size_t lh = H;
    static int xk = 0;
    auto each = [&](auto && fn) {
        for (int c = 0; c < QZ_NC; c++) { DS_CHECK(hipSetDevice(D[c].dev)); fn(c, D[c], CX[c], D[c].st); }
    };
    auto exchange = [&]() {          // ship the partials in t1, then x += own + peer on both cards
        each([&](int c, qz_ds_block & d, qz_ds_ctx &, hipStream_t s) {
            // into the peer's inbox: a peer copy on this card's stream under p2p, a D2H and
            // an H2D on the peer's stream under the host link (qz_xdst, so p2p is unchanged)
            DS_CHECK(qz_xcopy(D[1 - c].xin[xk], 1 - c, qz_xdst(D[1 - c].st), d.t1, c, s, (size_t) DS_T * H * 4));
            DS_CHECK(hipEventRecord(d.ev, s));
        });
        each([&](int c, qz_ds_block & d, qz_ds_ctx &, hipStream_t s) {
            DS_CHECK(hipStreamWaitEvent(s, D[1 - c].ev, 0));
            ds_add2<<<((size_t) DS_T * H + 255) / 256, 256, 0, s>>>(d.x, c == 0 ? d.t1 : d.xin[xk], c == 0 ? d.xin[xk] : d.t1, DS_T * H);
        });
        xk ^= 1;
    };
    each([&](int, qz_ds_block & d, qz_ds_ctx &, hipStream_t s) {
        d.ids[0] = anchor;
        for (int t = 1; t < DS_T; t++) d.ids[t] = 248070;
        DS_CHECK(hipMemcpyAsync(d.ids_d, d.ids, sizeof(d.ids), hipMemcpyHostToDevice, s));
        ds_deq_rows<<<((size_t) DS_T * H + 255) / 256, 256, 0, s>>>(d.emb_v, d.emb_s, d.ids_d, d.x, DS_T, H);
        ds_rms_b<<<DS_T, 256, 0, s>>>(d.x, d.anorm[0], d.t2, H, DS_EPS);
        ds_qrows<<<(DS_T * H / 32 + 7) / 8, 256, 0, s>>>(d.t2, d.q1, d.s1, DS_T, H);
    });
    for (int l = 0; l < DS_FL; l++) {
        each([&](int c, qz_ds_block & d, qz_ds_ctx & cx, hipStream_t s) {
            ds_gemvR_launch(nt, s, d.q1, d.s1, H, H, {
                { (const char *) d.q_v[l] + (size_t) c * QWl * lh, (const __half *) d.q_s[l] + (size_t) c * QWl * (H / 32), QWl, d.xa, 0 },
                { (const char *) cx.k_v[l] + (size_t) c * KVl * lh, (const __half *) cx.k_s[l] + (size_t) c * KVl * (H / 32), KVl, d.kb, 0 },
                { (const char *) cx.v_v[l] + (size_t) c * KVl * lh, (const __half *) cx.v_s[l] + (size_t) c * KVl * (H / 32), KVl, d.vb, 0 } });
            ds_norm_rope<<<DS_T * DS_NQ / QZ_NC, 128, 0, s>>>(d.xa, d.qnw[l], cx.inv, DS_NQ / QZ_NC, DS_T, p0);
            ds_norm_rope<<<DS_T * DS_NKV / QZ_NC, 128, 0, s>>>(d.kb, cx.knw[l], cx.inv, DS_NKV / QZ_NC, DS_T, p0);
            const int aps = ds_a3_ps(S, DS_NKV / QZ_NC);
            const int nsp = (S + aps - 1) / aps;
            ds_attn_split3<<<dim3(nsp, DS_NKV / QZ_NC, (nt + 7) / 8), 256, 0, s>>>(d.xa, cx.ring_k[l], cx.ring_v[l], p0, d.kb, d.vb,
                                                                 S, wbase, aps, d.at_o, d.at_ml, DS_NQ / QZ_NC, c * DS_NKV / QZ_NC);
            ds_attn_comb3<<<DS_T * DS_NQ / QZ_NC, 128, 0, s>>>(d.at_o, d.at_ml, d.att, nsp, DS_NQ / QZ_NC);
            ds_qrows<<<(DS_T * QWl / 32 + 7) / 8, 256, 0, s>>>(d.att, d.qq2, d.qs2, DS_T, QWl);
            ds_gemvR_launch(nt, s, d.qq2, d.qs2, QWl, DS_NQ * DS_HD, {
                { (const char *) d.o_v[l] + (size_t) c * QWl, (const __half *) d.o_s[l] + (size_t) c * QWl / 32, H, d.t1, 0 } });
        });
        exchange();
        each([&](int c, qz_ds_block & d, qz_ds_ctx &, hipStream_t s) {
            ds_rms_b<<<DS_T, 256, 0, s>>>(d.x, d.pnorm[l], d.t2, H, DS_EPS);
            ds_qrows<<<(DS_T * H / 32 + 7) / 8, 256, 0, s>>>(d.t2, d.q1, d.s1, DS_T, H);
            ds_gemvR_launch(nt, s, d.q1, d.s1, H, H, {
                { (const char *) d.g_v[l] + (size_t) c * FF2 * lh, (const __half *) d.g_s[l] + (size_t) c * FF2 * (H / 32), FF2, d.fg, 0 },
                { (const char *) d.u_v[l] + (size_t) c * FF2 * lh, (const __half *) d.u_s[l] + (size_t) c * FF2 * (H / 32), FF2, d.fu, 0 } });
            ds_silu_mul<<<((size_t) DS_T * FF2 + 255) / 256, 256, 0, s>>>(d.fg, d.fu, DS_T * FF2);
            ds_qrows<<<(DS_T * FF2 / 32 + 7) / 8, 256, 0, s>>>(d.fg, d.q3, d.s3, DS_T, FF2);
            ds_gemvR_launch(nt, s, d.q3, d.s3, FF2, DS_FF, {
                { (const char *) d.d_v[l] + (size_t) c * FF2, (const __half *) d.d_s[l] + (size_t) c * FF2 / 32, H, d.t1, 0 } },
                2, (size_t) DS_T * H);
            ds_fold2<<<(DS_T * H + 255) / 256, 256, 0, s>>>(d.t1, DS_T * H);
        });
        exchange();
        each([&](int, qz_ds_block & d, qz_ds_ctx &, hipStream_t s) {
            if (l + 1 == DS_FL)
                DS_CHECK(hipMemcpyAsync(d.stash_raw, d.x, (size_t) DS_T * DS_H * 4, hipMemcpyDeviceToDevice, s));
            ds_bz_b<<<DS_T, 256, 0, s>>>(d.x, d.z, d.rmsb, H);
            if (nt <= 8) ds_fgemvK4<8><<<1024 / 4, 256, 0, s>>>(d.z, d.Rl[l], d.proj, DS_C, H);
            else ds_fgemvK4<DS_T><<<1024 / 4, 256, 0, s>>>(d.z, d.Rl[l], d.proj, DS_C, H);
            ds_code_b<<<DS_T, 256, 0, s>>>(d.proj, d.cb_v, d.cb_s, anchor, DS_C);
            if (nt <= 8) ds_fgemvK4<8><<<1536 / 4, 256, 0, s>>>(d.proj, d.P, d.packed, 3 * DS_R, DS_C);
            else ds_fgemvK4<DS_T><<<1536 / 4, 256, 0, s>>>(d.proj, d.P, d.packed, 3 * DS_R, DS_C);
            DS_CHECK(hipMemsetAsync(d.msg, 0, (size_t) DS_T * DS_R * 4, s));
            ds_gate_msg<<<(DS_T * DS_R + 255) / 256, 256, 0, s>>>(d.packed, d.b_gate, d.msg, DS_T);
            if (nt <= 8) ds_fgemvK4<8><<<5120 / 4, 256, 0, s>>>(d.msg, d.Wl[l], d.delta, H, DS_R);
            else ds_fgemvK4<DS_T><<<5120 / 4, 256, 0, s>>>(d.msg, d.Wl[l], d.delta, H, DS_R);
            ds_out_add<<<((size_t) DS_T * H + 255) / 256, 256, 0, s>>>(d.x, d.delta, d.rmsb, DS_T, H);
            const float * nx = l + 1 < DS_FL ? d.anorm[l + 1] : d.onorm;
            ds_rms_b<<<DS_T, 256, 0, s>>>(d.x, nx, d.t2, H, DS_EPS);
            ds_qrows<<<(DS_T * H / 32 + 7) / 8, 256, 0, s>>>(d.t2, d.q1, d.s1, DS_T, H);
        });
    }
    // each card's head half gives its top-16; card 0 merges the two lists
    const int nch = (V2 + DS_TK_CH - 1) / DS_TK_CH;
    each([&](int c, qz_ds_block & d, qz_ds_ctx &, hipStream_t s) {
        ds_gemvR_launch(nt, s, d.q1, d.s1, H, H, { { (const char *) d.hh_v, (const __half *) d.hh_s, V2, d.logits, 0 } });
        ds_top16_a<<<dim3(nch, DS_T - 1), 256, 0, s>>>(d.logits + V2, d.tk_v, d.tk_i, V2, nch, c * V2);
        ds_top16_b<(((DS_V / QZ_NC + DS_TK_CH - 1) / DS_TK_CH) * DS_TOPK + 255) / 256><<<DS_T - 1, 256, 0, s>>>(
            d.tk_v, d.tk_i, d.tki, nch, d.tkv);
        ds_chain_bas_l<<<((DS_T - 1) * DS_TOPK + 3) / 4, 256, 0, s>>>(d.stash_raw, d.onorm, d.tki, c * V2,
                                                                      d.hh_v, d.hh_s, d.bas_l);
        ds_rows_l<<<(DS_T - 1) * DS_TOPK + 1, 256, 0, s>>>(d.TG, d.TSV, d.tki, c * V2, anchor, d.tg_l, d.tsv_l, d.tsva_l);
    });
    // card 1's top-16 lists into card 0's staging: six copies on card 1's stream under
    // p2p. Under the host link the H2D lands on card 0's stream (qz_xdst), which is where
    // ds_top16_m reads them, so the event below and its wait stay as they are.
    DS_CHECK(hipSetDevice(D[1].dev));
    DS_CHECK(qz_xcopy(D[0].ptkv, 0, qz_xdst(D[0].st), D[1].tkv, 1, D[1].st, (size_t) (DS_T - 1) * DS_TOPK * 4));
    DS_CHECK(qz_xcopy(D[0].ptki, 0, qz_xdst(D[0].st), D[1].tki, 1, D[1].st, (size_t) (DS_T - 1) * DS_TOPK * 4));
    DS_CHECK(qz_xcopy(D[0].pbas, 0, qz_xdst(D[0].st), D[1].bas_l, 1, D[1].st, (size_t) (DS_T - 1) * DS_TOPK * 4));
    {
        const size_t nr = (size_t) (DS_T - 1) * DS_TOPK;
        DS_CHECK(qz_xcopy(D[0].ptg, 0, qz_xdst(D[0].st), D[1].tg_l, 1, D[1].st, nr * DS_R * 4));
        DS_CHECK(qz_xcopy(D[0].ptsv, 0, qz_xdst(D[0].st), D[1].tsv_l, 1, D[1].st, nr * 2 * DS_R * 4));
        if (anchor >= V2) DS_CHECK(qz_xcopy(D[0].ptsva, 0, qz_xdst(D[0].st), D[1].tsva_l, 1, D[1].st, 2 * DS_R * 4));
    }
    DS_CHECK(hipEventRecord(D[1].ev, D[1].st));
    DS_CHECK(hipSetDevice(D[0].dev));
    DS_CHECK(hipStreamWaitEvent(D[0].st, D[1].ev, 0));
    ds_top16_m<<<DS_T - 1, 256, 0, D[0].st>>>(D[0].tkv, D[0].tki, D[0].ptkv, D[0].ptki, D[0].cand);
}

// ------------------------------------------------- the chain on device ----
// the greedy draft chain on card 0: sort candidates, bas, then walk the positions
__global__ void __launch_bounds__(64) ds_cand_sort(int * cand) {
    const int i = threadIdx.x;
    if (i >= DS_T - 1) return;
    int * c = cand + i * DS_TOPK;
    for (int a = 1; a < DS_TOPK; a++) {
        const int v = c[a];
        int b = a - 1;
        while (b >= 0 && c[b] > v) { c[b + 1] = c[b]; b--; }
        c[b + 1] = v;
    }
}
// each candidate's bas from the half that holds it; card 1's list is copied to card 0
__global__ void __launch_bounds__(128) ds_bas_pick(const int * cand, const int * ai, const float * ab,
                                                   const int * bi, const float * bb, float * bas) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= (DS_T - 1) * DS_TOPK) return;
    const int r = w / DS_TOPK * DS_TOPK, id = cand[w];
    float v = 0.f;
    for (int k = 0; k < DS_TOPK; k++) {
        if (ai[r + k] == id) v = ab[r + k];
        if (bi[r + k] == id) v = bb[r + k];
    }
    bas[w] = v;
}
// TG / TSV rows in sorted-candidate order, plus the anchor's TSV row, for ds_chain_walk
__global__ void __launch_bounds__(256) ds_rows_pick(const int * cand, int anchor,
                                                    const int * ai, const float * atg, const float * atsv, const float * atsva,
                                                    const int * bi, const float * btg, const float * btsv, const float * btsva,
                                                    float * tg, float * tsv, float * tsva) {
    const int w = blockIdx.x, t = threadIdx.x;
    if (w < (DS_T - 1) * DS_TOPK) {
        const int r = w / DS_TOPK * DS_TOPK, id = cand[w];
        int k = r;
        bool own = true;
        for (int j = 0; j < DS_TOPK; j++) {
            if (ai[r + j] == id) { k = r + j; own = true; }
            if (bi[r + j] == id) { k = r + j; own = false; }
        }
        const float * sg = (own ? atg : btg) + (size_t) k * DS_R, * ss = (own ? atsv : btsv) + (size_t) k * 2 * DS_R;
        for (int e = t; e < DS_R; e += 256) tg[(size_t) w * DS_R + e] = sg[e];
        for (int e = t; e < 2 * DS_R; e += 256) tsv[(size_t) w * 2 * DS_R + e] = ss[e];
    } else {
        const float * sa = anchor < DS_V / 2 ? atsva : btsva;
        for (int e = t; e < 2 * DS_R; e += 256) tsva[e] = sa[e];
    }
}
// TG / TSV are ds_rows_pick's tables in candidate order and tsva the anchor's TSV row
__global__ void __launch_bounds__(512) ds_chain_walk(const float * bas, const int * cand, const float * rmsb,
                                                     const float * packed, const float * bg, const float * TG,
                                                     const float * TSV, int anchor, int * drafts,
                                                     const float * u = nullptr, float invT = 1.f,
                                                     float * qall = nullptr,
                                                     const float * tsva = nullptr) {
    // with u set, position i samples from softmax(sc / T) with u[i] and writes q to qall;
    // with u == nullptr it takes the greedy argmax.
    __shared__ float m[DS_R];
    __shared__ float sc[DS_TOPK];
    __shared__ int jsel;
    const int tid = threadIdx.x, g = tid >> 5, l = tid & 31;          // 16 groups of 32
    int jprev = 0;
    for (int i = 0; i < DS_T - 1; i++) {
        const float * tsv = i == 0 ? tsva : TSV + ((size_t) (i - 1) * DS_TOPK + jprev) * 2 * DS_R;
        const float * q = packed + (size_t) (i + 1) * 3 * DS_R;
        for (int r = tid; r < DS_R; r += 512)
            m[r] = 1.f / (1.f + expf(-(q[r] + tsv[r] + bg[r]))) * tsv[DS_R + r];
        __syncthreads();
        const float * tg = TG + ((size_t) i * DS_TOPK + g) * DS_R;
        double a = 0.0;
        for (int r = l; r < DS_R; r += 32) a += (double) m[r] * tg[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off, 32);
        if (l == 0) sc[g] = __fadd_rn(bas[i * DS_TOPK + g], __fmul_rn(rmsb[i + 1], (float) a));
        __syncthreads();
        if (tid == 0) {
            int bj = 0;
            float best = sc[0];
            for (int j = 1; j < DS_TOPK; j++) if (sc[j] > best) { best = sc[j]; bj = j; }
            if (u) {
                float w[DS_TOPK];
                double sum = 0.0;
                for (int j = 0; j < DS_TOPK; j++) { w[j] = expf((sc[j] - best) * invT); sum += w[j]; }
                const double tgt = sum * u[i];
                double run = 0.0;
                bj = DS_TOPK - 1;
                for (int j = 0; j < DS_TOPK; j++) { run += w[j]; if (run >= tgt) { bj = j; break; } }
                for (int j = 0; j < DS_TOPK; j++) qall[i * DS_TOPK + j] = (float) (w[j] / sum);
            }
            jsel = bj;
            drafts[i] = cand[i * DS_TOPK + bj];
        }
        __syncthreads();
        jprev = jsel;
        __syncthreads();
    }
}

// the chain walk on card 0, after ds_cand_sort and the bas: stage table rows, then walk
inline void ds_walk(qz_ds_block & b, const float * bas, int anchor, int * drafts, const float * u = nullptr,
                    float invT = 1.f, float * qall = nullptr) {
    ds_rows_pick<<<(DS_T - 1) * DS_TOPK + 1, 256, 0, b.st>>>(b.cand, anchor, b.tki, b.tg_l, b.tsv_l, b.tsva_l,
                                                             b.ptki, b.ptg, b.ptsv, b.ptsva, b.tg_s, b.tsv_s, b.tsva_s);
    ds_chain_walk<<<1, 512, 0, b.st>>>(bas, b.cand, b.rmsb, b.packed, b.b_gate, b.tg_s, b.tsv_s, anchor, drafts,
                                       u, invT, qall, b.tsva_s);
}
