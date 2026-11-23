// inkling2: batched prefill. Chunks of PF_M prompt rows through the
// 48 applications of docs/inkling2.md, application by application (exact:
// rows interact only through a slot's KV and conv history, and every chunk
// row passes one application before the next starts). The Q8 GEMMs (dense
// and grouped MoE) share one tile body on quartz's repacked Q8 planes; the norm +
// quant and the activation plane layout come from the shared pf_gemm.h; the
// depth mix, loop controllers, head post / convs along rows and the HD 128
// flash attention are arch kernels.
// Decode stays on ink2_engine::step(). Included after ink2_kernels.h / ink2_engine.h (same TU).
#pragma once

#include "kernels/pf_gemm.h"     // pf_norm_quant, pf_quant_unit, the plane types

constexpr int PF_NP = 8 * PF_M;             // max (row,expert) pairs per chunk
constexpr int GUL = 2 * FFL;                // local gate | up rows per expert
constexpr int NGRP_FF = (FFL + 127) / 128;  // groups of 128 in the down contraction: 5 (the last one half)
constexpr int FFL_PAD = NGRP_FF * 128;      // 640
constexpr int NSC_FF = NGRP_FF * 4;         // scale slots per row of the down input planes

// ------------------------------------------------------------- kernels ----

// token rows of the Q8 embedding, rmsnorm * embd_norm; one block per row
static __global__ __launch_bounds__(256) void k_pf_embed(const int * tok, const char * ev, const __half * es,
                                                         const float * w, float eps, float * x) {
    __shared__ float red[16];
    const int row = blockIdx.x, t = threadIdx.x, id = tok[row];
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = t + k * 256;
        v[k] = (float) ev[(size_t) id * D + i] * __half2float(es[(size_t) id * (D / 32) + i / 32]);
        ss[0] += v[k] * v[k];
    }
    bsum<1>(ss, red);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) x[(size_t) row * D + t + k * 256] = v[k] * r * w[t + k * 256];
}

// image rows: row r of the chunk takes image row irow[r] as is (no embedding norm), irow < 0 keeps the token's
static __global__ __launch_bounds__(256) void k_pf_img(const int * irow, const float * img, float * x) {
    const int r = irow[blockIdx.x];
    if (r < 0) return;
    for (int i = threadIdx.x; i < D; i += 256) x[(size_t) blockIdx.x * D + i] = img[(size_t) r * D + i];
}

// mix(blocks, cur, query), one row per block: the states are [PF_M][D] batch
// buffers at the same row; MQ query columns land in out[j*ostr + row]. The
// row's state values stay in registers (one read of each state)
struct pfmix { const float * p[6]; int n; const float * qy; int m; float * out; size_t ostr; };
template <int MQ>
static __global__ __launch_bounds__(256) void k_pf_mix2(pfmix A) {
    constexpr int NV = D / 256;
    __shared__ float red[4 * 6 * (MQ + 1)];
    __shared__ float wsh[6][4];
    const int row = blockIdx.x, tid = threadIdx.x, n = A.n;
    float v[6][NV];
    float ss[6];
    #pragma unroll
    for (int s = 0; s < 6; s++) {
        ss[s] = 0.f;
        #pragma unroll
        for (int k = 0; k < NV; k++) {
            v[s][k] = s < n ? A.p[s][(size_t) row * D + tid + 256 * k] : 0.f;
            ss[s] += v[s][k] * v[s][k];
        }
    }
    bsum<6>(ss, red);
    float sc[6 * MQ];
    #pragma unroll
    for (int s = 0; s < 6; s++) {
        const float rs = rsqrtf(ss[s] / D + 1e-6f);
        #pragma unroll
        for (int k = 0; k < NV; k++) v[s][k] *= rs;
        #pragma unroll
        for (int j = 0; j < MQ; j++) {
            float a = 0.f;
            #pragma unroll
            for (int k = 0; k < NV; k++) a += v[s][k] * A.qy[(size_t) j * D + tid + 256 * k];
            sc[s * MQ + j] = a;
        }
    }
    bsum<6 * MQ>(sc, red);
    if (tid < MQ) {
        const int j = tid;
        const float scale = 1.f / sqrtf((float) D);
        float mx = -INFINITY;
        for (int s = 0; s < n; s++) mx = fmaxf(mx, sc[s * MQ + j] * scale);
        float sum = 0.f;
        for (int s = 0; s < n; s++) { const float e = expf(sc[s * MQ + j] * scale - mx); wsh[s][j] = e; sum += e; }
        for (int s = 0; s < n; s++) wsh[s][j] /= sum;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < MQ; j++)
        #pragma unroll
        for (int k = 0; k < NV; k++) {
            float o = 0.f;
            #pragma unroll
            for (int s = 0; s < 6; s++) if (s < n) o += wsh[s][j] * v[s][k];
            A.out[(size_t) j * A.ostr + (size_t) row * D + tid + 256 * k] = o;
        }
}

// z[j][row][2D] = [rmsnorm(cur_j); rmsnorm(msg_j)] (no gain), one block per
// (row, j); the ctrl down GEMM reads it. grid (m, nj), job j0 + blockIdx.y
struct pfrms { const float * cur[4]; const float * msg; float * z; int j0; };
static __global__ __launch_bounds__(256) void k_pf_rms2(pfrms A) {
    __shared__ float red[4 * 2];
    const int row = blockIdx.x, j = A.j0 + blockIdx.y, tid = threadIdx.x;
    const float * c = A.cur[j] + (size_t) row * D;
    const float * g = A.msg + ((size_t) j * PF_M + row) * D;
    float ss[2] = {0.f, 0.f};
    for (int i = tid; i < D; i += 256) { ss[0] += c[i] * c[i]; ss[1] += g[i] * g[i]; }
    bsum<2>(ss, red);
    const float rc = rsqrtf(ss[0] / D + 1e-6f), rm = rsqrtf(ss[1] / D + 1e-6f);
    float * z = A.z + ((size_t) j * PF_M + row) * 2 * D;
    for (int i = tid; i < D; i += 256) { z[i] = c[i] * rc; z[D + i] = g[i] * rm; }
}

// g[j][row][64] = Z[j][row][2D] . wd[64][2D] (f32)
struct pfcdn { const float * z; const float * wd0, *wd1, *wd2, *wd3; float * g; int j0, m; };

// out = (1 - tanh(g.forget)).cur + tanh(g.gain).msg per row: both small f32 GEMMs vs g[j][row][64]
struct pfcup { const float * g, * msg; const float * cur[4]; const float * wgs[4], * wfs[4]; float * out; int j0, m; };

// controllers as 64x64-tiled f32 GEMMs (4 x 4 per thread). down: K = 2D
// split CT_S ways, partials gp[ks][j][row][GRANK]; up sums them in order
// (deterministic), then the gain / forget GEMMs against g and the update
constexpr int CT_S = 8, CT_KS = 2 * D / CT_S, CT_KC = 32;
static __global__ __launch_bounds__(256) void k_pf_ctrl_down2(pfcdn a, float * gp) {
    __shared__ float sX[64][CT_KC + 1], sW[GRANK][CT_KC + 1];
    const int t = threadIdx.x, tr = t >> 4, tc = t & 15, j = a.j0 + blockIdx.y, ks = blockIdx.z;
    const int r0 = blockIdx.x * 64;
    const float * wd = j == 0 ? a.wd0 : j == 1 ? a.wd1 : j == 2 ? a.wd2 : a.wd3;
    const float * z = a.z + (size_t) j * PF_M * 2 * D;
    float acc[4][4] = {};
    for (int k0 = ks * CT_KS; k0 < (ks + 1) * CT_KS; k0 += CT_KC) {
        __syncthreads();
        for (int i = t; i < 64 * CT_KC; i += 256) {
            const int r = i / CT_KC, k = i % CT_KC;
            sX[r][k] = r0 + r < a.m ? z[(size_t)(r0 + r) * 2 * D + k0 + k] : 0.f;
            sW[r][k] = wd[(size_t) r * 2 * D + k0 + k];
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < CT_KC; k++) {
            float xv[4], wv[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { xv[i] = sX[tr * 4 + i][k]; wv[i] = sW[tc + 16 * i][k]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[i][c] = fmaf(xv[i], wv[c], acc[i][c]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = r0 + tr * 4 + i;
        if (row >= a.m) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) gp[(((size_t) ks * 4 + j) * PF_M + row) * GRANK + tc + 16 * c] = acc[i][c];
    }
}
// g[j][row][k] = sum over the K splits, in order
static __global__ __launch_bounds__(256) void k_pf_gsum(const float * gp, float * g, int j0, int m) {
    const int i = blockIdx.x * 256 + threadIdx.x, j = j0 + blockIdx.y;
    if (i >= m * GRANK) return;
    float s = 0.f;
#pragma unroll
    for (int ks = 0; ks < CT_S; ks++) s += gp[((size_t) ks * 4 + j) * PF_M * GRANK + i];
    g[(size_t) j * PF_M * GRANK + i] = s;
}
static __global__ __launch_bounds__(256) void k_pf_ctrl_up2(pfcup a) {
    __shared__ float sG[64][GRANK + 1], sWg[GRANK][64 + 1], sWf[GRANK][64 + 1];
    const int t = threadIdx.x, tr = t >> 4, tc = t & 15, j = a.j0 + blockIdx.z;
    const int r0 = blockIdx.x * 64, c0 = blockIdx.y * 64;
    for (int i = t; i < 64 * GRANK; i += 256) {
        const int r = i / GRANK, k = i % GRANK;
        sG[r][k] = r0 + r < a.m ? a.g[((size_t) j * PF_M + r0 + r) * GRANK + k] : 0.f;
        const int col = i / GRANK;                 // weights [column][rank] -> [rank][column]
        sWg[k][col] = a.wgs[j][(size_t)(c0 + col) * GRANK + k];
        sWf[k][col] = a.wfs[j][(size_t)(c0 + col) * GRANK + k];
    }
    __syncthreads();
    float ga[4][4] = {}, fo[4][4] = {};
#pragma unroll 8
    for (int k = 0; k < GRANK; k++) {
        float gv[4], wg[4], wf[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { gv[i] = sG[tr * 4 + i][k]; wg[i] = sWg[k][tc + 16 * i]; wf[i] = sWf[k][tc + 16 * i]; }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int c = 0; c < 4; c++) { ga[i][c] = fmaf(gv[i], wg[c], ga[i][c]); fo[i][c] = fmaf(gv[i], wf[c], fo[i][c]); }
    }
    const float * cur = a.cur[j], * msg = a.msg + (size_t) j * PF_M * D;
    float * out = a.out + (size_t) j * PF_M * D;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = r0 + tr * 4 + i;
        if (row >= a.m) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const size_t o = (size_t) row * D + c0 + tc + 16 * c;
            const float x = cur[o];
            out[o] = (x - tanhf(fo[i][c]) * x) + tanhf(ga[i][c]) * msg[o];
        }
    }
}

// per local head (Q / R clip, tau; the clipped R for the attention's bias) and
// per kv head (K/V short conv along the chunk rows, clip / norm into the cache),
// row per blockIdx.y
struct pfhp {
    const float * qr, * rr, * kr, * vr, * dk, * dv;          // [M][QL/HL../KVL/16]
    const float * qclip, * rclip, * kconv, * vconv;
    float * hist, * q;                                        // slot hist, [M][QL]
    float * rvo;                                              // [M][HQL][DREL] clipped R
    __half * kc, * vc;
    int pos0, full;
    float tau_a; int tau_f;
    float qk_clip, rel_clip, eps;
};
static __global__ __launch_bounds__(128) void k_pf_head_post(pfhp a) {
    __shared__ float rv[DREL];
    __shared__ float red[2];
    const int b = blockIdx.x, row = blockIdx.y, t = threadIdx.x, pos = a.pos0 + row;
    if (b < HQL) {
        const int h = b, i = h * HD + t;
        const float tau = a.full ? 1.f + a.tau_a * logf(fmaxf(1.f, (float) (pos + 1) / (float) a.tau_f)) : 1.f;
        float x = a.qr[(size_t) row * QL + i] * a.qclip[i];
        float n2[1] = {x * x};
        bsum<1>(n2, red);
        const float nr = sqrtf(n2[0]) + 1e-6f;
        x *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        a.q[(size_t) row * QL + i] = x * tau;
        float r = t < DREL ? a.rr[(size_t) row * (HQL * DREL) + h * DREL + t] * a.rclip[h * DREL + t] : 0.f;
        float r2[1] = {r * r};
        bsum<1>(r2, red);
        const float rn = sqrtf(r2[0]) + 1e-6f;
        if (t < DREL) rv[t] = r * fminf(fmaxf(a.rel_clip / rn, 0.f), 1.f) * tau;
        if (t < DREL) a.rvo[((size_t) row * HQL + h) * DREL + t] = rv[t];
        return;
    }
    const bool isk = b < HQL + HKVL;
    const int hk = isk ? b - HQL : b - HQL - HKVL, ch = hk * HD + t;
    const float * rb = isk ? a.kr : a.vr;                     // [M][KVL]
    const float * hb = a.hist + (isk ? H_K : H_V) + ch;
    const float u = rb[(size_t) row * KVL + ch];
    const float v1 = row >= 1 ? rb[(size_t)(row - 1) * KVL + ch] : hb[(size_t)(row + 2) * CHL];
    const float v2 = row >= 2 ? rb[(size_t)(row - 2) * KVL + ch] : hb[(size_t)(row + 1) * CHL];
    const float v3 = row >= 3 ? rb[(size_t)(row - 3) * KVL + ch] : hb[(size_t) row * CHL];
    const float * w = (isk ? a.kconv : a.vconv) + (size_t) ch * NCONV;
    float o = w[0] * v3 + w[1] * v2 + w[2] * v1 + w[3] * u;
    if (a.dk) {
        const float * tp = (isk ? a.dk : a.dv) + (size_t) row * (HKVL * NCONV) + hk * NCONV;
        o += u * tp[0] + v1 * tp[1] + v2 * tp[2] + v3 * tp[3];
    }
    float y = (u + o) * (a.dk ? 1.f / sqrtf(2.f * NCONV + 1.f) : 1.f / sqrtf((float) NCONV + 1.f));
    float n2[1] = {y * y};
    bsum<1>(n2, red);
    const int rrow = a.full ? pos : pos % PF_RING;
    if (isk) {
        const float nr = sqrtf(n2[0]) + 1e-6f;
        y *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        a.kc[(size_t) rrow * KVL + ch] = __float2half(y);
    } else {
        y *= rsqrtf(n2[0] / HD + a.eps);
        a.vc[(size_t) rrow * KVL + ch] = __float2half(y);
    }
}

// attention inputs: the window (SWA: rows pos%PF_RING of the ring) or the whole
// cache (full); out: sigmoid-gated, quantized into the attn_output planes
struct pfat {
    const float * q, * gate;                  // [M][QL], [M][QL]
    const float * rv, * relb, * sk;           // [M][HQL][DREL], [REL_EXT][DREL], [M][HQL] (bias in the kernel)
    const __half * kc, * vc;
    int pos0, full;
    unsigned char * oq; float * os;
};

// tiled flash attention: one block per (64-row tile, q head), key tiles of
// 64. S = Q.K^T with packed f16 dots (4 rows x 4 keys per thread), online
// softmax per row (bias from R . relb - sink, causal / window mask, the zero sink logit),
// O += P.V in f32 (4 rows x 8 dims per thread); the 16 lanes of a row group
// own the same rows in both phases, so row statistics stay in registers
constexpr int FA_K = 64, FA_QS = HD + 8;                // key tile, Q / K / V LDS row stride (halves)
typedef __fp16 fa_h2 __attribute__((ext_vector_type(2)));
// RPT rows per thread (tile FA_R = 16 * RPT rows); QLO: Q as f16 hi + lo (f32-exact scores)
template <int RPT, bool QLO, int FA_R = 32>
static __global__ __launch_bounds__(16 * FA_R / RPT) void k_pf_fattn(pfat a, int m) {
    constexpr int NQ = QLO ? 2 : 1, NTH = 16 * FA_R / RPT;
    __shared__ __half sQ[NQ][FA_R * FA_QS];
    __shared__ __half sKV[FA_K * FA_QS];
    __shared__ float sP[FA_R][FA_K + 4];
    __shared__ float sB[FA_R + FA_K - 1][DREL + 1];          // bias basis rows of this tile's distances
    const int t = threadIdx.x, rg = t >> 4, cg = t & 15;
    const int h = blockIdx.y, kvh = h / (HQ / HKV), r0 = blockIdx.x * FA_R;
    const int rows = min(FA_R, m - r0);
    const float scale = 1.f / sqrtf((float) HD);
    for (int e = t; e < FA_R * HD / 4; e += NTH) {
        const int r = e / (HD / 4), c4 = e % (HD / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < rows) v = *(const float4 *) (a.q + (size_t)(r0 + r) * QL + h * HD + c4 * 4);
        __half2 * d = (__half2 *) (sQ[0] + r * FA_QS + c4 * 4);
        d[0] = __floats2half2_rn(v.x, v.y);
        d[1] = __floats2half2_rn(v.z, v.w);
        if (QLO) {
            __half2 * dl = (__half2 *) (sQ[NQ - 1] + r * FA_QS + c4 * 4);
            dl[0] = __floats2half2_rn(v.x - __low2float(d[0]), v.y - __high2float(d[0]));
            dl[1] = __floats2half2_rn(v.z - __low2float(d[1]), v.w - __high2float(d[1]));
        }
    }
    int rpos[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; i++) rpos[i] = a.pos0 + r0 + min(rg * RPT + i, rows - 1);
    float rvr[RPT][DREL], skr[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; i++) {
        const int r = min(rg * RPT + i, rows - 1);
        #pragma unroll
        for (int k = 0; k < DREL; k++) rvr[i][k] = a.rv[((size_t)(r0 + r) * HQL + h) * DREL + k];
        skr[i] = a.sk[(size_t)(r0 + r) * HQL + h];
    }
    const int pfirst = a.pos0 + r0;
    float mx[RPT], l[RPT], o[RPT][8];
    #pragma unroll
    for (int i = 0; i < RPT; i++) { mx[i] = -INFINITY; l[i] = 0.f; for (int d = 0; d < 8; d++) o[i][d] = 0.f; }
    const int plast = a.pos0 + r0 + rows - 1;
    const int k0 = a.full ? 0 : max(0, a.pos0 + r0 - (WIN - 1));
    for (int kb = k0; kb <= plast; kb += FA_K) {
        __syncthreads();
        for (int e = t; e < FA_K * HD / 8; e += NTH) {
            const int r = e / (HD / 8), c8 = e % (HD / 8), j = kb + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j <= plast) { const int rr = a.full ? j : j % PF_RING; v = *(const uint4 *) (a.kc + (size_t) rr * KVL + kvh * HD + c8 * 8); }
            *(uint4 *) (sKV + r * FA_QS + c8 * 8) = v;
        }
        const int dlo = pfirst - (kb + FA_K - 1);              // distance of window row 0
        for (int e = t; e < (FA_R + FA_K - 1) * DREL; e += NTH) {
            const int ii = e / DREL, k = e % DREL, d = dlo + ii;
            sB[ii][k] = d >= 0 && d < REL_EXT ? a.relb[(size_t) d * DREL + k] : 0.f;
        }
        __syncthreads();
        float sc[RPT][4] = {};
        #pragma unroll 4
        for (int d = 0; d < HD; d += 4) {
            fa_h2 k[4][2];
            #pragma unroll
            for (int c = 0; c < 4; c++) { const uint2 u = *(const uint2 *) (sKV + (cg + 16 * c) * FA_QS + d); k[c][0] = *(const fa_h2 *) &u.x; k[c][1] = *(const fa_h2 *) &u.y; }
            #pragma unroll
            for (int qi = 0; qi < NQ; qi++)
            #pragma unroll
            for (int i = 0; i < RPT; i++) {
                const uint2 u = *(const uint2 *) (sQ[qi] + (rg * RPT + i) * FA_QS + d);
                const fa_h2 q0 = *(const fa_h2 *) &u.x, q1 = *(const fa_h2 *) &u.y;
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    sc[i][c] = __builtin_amdgcn_fdot2(q0, k[c][0], sc[i][c], false);
                    sc[i][c] = __builtin_amdgcn_fdot2(q1, k[c][1], sc[i][c], false);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < RPT; i++) {
            const int pos = rpos[i], lo = a.full ? 0 : max(0, pos - (WIN - 1));
            float tm = -INFINITY;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const int j = kb + cg + 16 * c;
                float s = -INFINITY;
                if (j <= pos && j >= lo) {
                    const float * bb = sB[pos - j - dlo];
                    float bias = 0.f;
                    #pragma unroll
                    for (int k = 0; k < DREL; k++) bias += bb[k] * rvr[i][k];
                    s = sc[i][c] * scale + (bias - skr[i]);
                }
                sc[i][c] = s;
                tm = fmaxf(tm, s);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) tm = fmaxf(tm, __shfl_xor(tm, off, 16));
            const float mn = fmaxf(mx[i], tm);
            const float cr = mn == -INFINITY ? 1.f : expf(mx[i] - mn);
            float ps = 0.f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const float p = sc[i][c] == -INFINITY ? 0.f : expf(sc[i][c] - mn);
                sP[rg * RPT + i][cg + 16 * c] = p;
                ps += p;
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) ps += __shfl_xor(ps, off, 16);
            l[i] = l[i] * cr + ps;
            mx[i] = mn;
            #pragma unroll
            for (int d = 0; d < 8; d++) o[i][d] *= cr;
        }
        __syncthreads();
        for (int e = t; e < FA_K * HD / 8; e += NTH) {
            const int r = e / (HD / 8), c8 = e % (HD / 8), j = kb + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j <= plast) { const int rr = a.full ? j : j % PF_RING; v = *(const uint4 *) (a.vc + (size_t) rr * KVL + kvh * HD + c8 * 8); }
            *(uint4 *) (sKV + r * FA_QS + c8 * 8) = v;
        }
        __syncthreads();
        #pragma unroll 4
        for (int kk = 0; kk < FA_K; kk++) {
            const uint4 u = *(const uint4 *) (sKV + kk * FA_QS + cg * 8);
            const __half2 * v2 = (const __half2 *) &u;
            float vf[8];
            #pragma unroll
            for (int d = 0; d < 4; d++) { vf[2 * d] = __low2float(v2[d]); vf[2 * d + 1] = __high2float(v2[d]); }
            #pragma unroll
            for (int i = 0; i < RPT; i++) {
                const float p = sP[rg * RPT + i][kk];
                #pragma unroll
                for (int d = 0; d < 8; d++) o[i][d] += p * vf[d];
            }
        }
    }
    // the sink logit (0), normalize, gate; quantized through LDS (32 lanes = 32 dims of one row)
    __syncthreads();
    float * sO = (float *) sQ;                             // [rows of this pass][HD] f32
    constexpr int PR = NQ * FA_R * FA_QS * 2 / (HD * 4) >= FA_R ? FA_R : 32;   // rows per pass
    #pragma unroll
    for (int pass = 0; pass < FA_R / PR; pass++) {
        if (rg * RPT / PR == pass) {
            #pragma unroll
            for (int i = 0; i < RPT; i++) {
                const float M = fmaxf(mx[i], 0.f);
                const float L = l[i] * expf(mx[i] - M) + expf(-M), f = expf(mx[i] - M) / L;
                #pragma unroll
                for (int d = 0; d < 8; d++) sO[((rg * RPT + i) % PR) * HD + cg * 8 + d] = o[i][d] * f;
            }
        }
        __syncthreads();
        for (int e = t; e < PR * HD; e += NTH) {
            const int r = pass * PR + e / HD, d = e % HD;
            if (r < rows) {
                const int i = h * HD + d;
                const float y = sO[e] * sigm(a.gate[(size_t)(r0 + r) * QL + i]);
                pf_quant_unit(y, i, r0 + r, PF_M, (int8_t *) a.oq, a.os, t & 31);
            }
        }
        __syncthreads();
    }
}

// short conv along the chunk rows + x update (mode: push a fresh residual
// block, or add); row t reads rows t-1..t-3 or the slot's conv history;
// u = own + peer (the TP2 sum, peer over P2P, f32) folded into the conv.
// One thread per column walking CV_R rows: each partial element is read once,
// the 3 previous inputs stay in registers
struct pfcv { const float * own, * peer; const float * w; float * hrg; float * x, * blk; int m, push; };
constexpr int CV_R = 32;
static __global__ __launch_bounds__(256) void k_pf_conv2(pfcv a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r0 = blockIdx.y * CV_R;
    if (i >= D) return;
    const float * p0 = a.own, * p1 = a.peer;
    auto in = [&](int r) {   // input at chunk row r (r < 0: history row 3 + r)
        if (r < 0) return a.hrg[(size_t)(r + 3) * CHL + i];
        return p0[(size_t) r * D + i] + p1[(size_t) r * D + i];
    };
    float v3 = in(r0 - 3), v2 = in(r0 - 2), v1 = in(r0 - 1);
    const float * wi = a.w + (size_t) i * NCONV;
    const float w0 = wi[0], w1 = wi[1], w2 = wi[2], w3 = wi[3], sc = 1.f / sqrtf((float) NCONV + 1.f);
    const int r1 = min(r0 + CV_R, a.m);
    for (int row = r0; row < r1; row++) {
        const float u = in(row);
        const float y = (u + (w0 * v3 + w1 * v2 + w2 * v1 + w3 * u)) * sc;
        const size_t o = (size_t) row * D + i;
        if (a.push) { a.blk[o] = a.x[o]; a.x[o] = y; }
        else a.x[o] += y;
        v3 = v2; v2 = v1; v1 = u;
    }
}

// the 3 conv history rows become the chunk's last 3 inputs (rows below the
// chunk start come from the old history); peer == nullptr for the K / V convs
static __global__ __launch_bounds__(256) void k_pf_commit(const float * src, const float * peer, float * hrg, int w, int m) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= w) return;
    float v[3];
    #pragma unroll
    for (int r = 0; r < 3; r++) {
        const int idx = m - 3 + r;
        v[r] = idx < 0 ? hrg[(size_t)(idx + 3) * CHL + c]
             : src[(size_t) idx * w + c] + (peer ? peer[(size_t) idx * w + c] : 0.f);
    }
    #pragma unroll
    for (int r = 0; r < 3; r++) hrg[(size_t) r * CHL + c] = v[r];
}

// router as a tiled f32 GEMM, K split RT_S ways for occupancy: block (16
// rows, one K split) x 128 experts, 2 rows x 4 experts per thread, K chunks
// of 32 in LDS; partials [RT_S][m][NE]. k_pf_topk then sums the splits in
// order (deterministic), sigmoid, top NE_USED by p + bias with a wave argmax
// (ties: the lowest expert, as the serial first-max loop), p / sum(p) * scale
constexpr int RT_R = 16, RT_K = 32, RT_S = 4;
static __global__ __launch_bounds__(256) void k_pf_router2(const float * f, const float * wr, float * part, int m) {
    __shared__ float sF[RT_R][RT_K + 1], sW[NE][RT_K + 1];
    const int t = threadIdx.x, tr = t >> 5, te = t & 31, r0 = blockIdx.x * RT_R;
    const int kl = blockIdx.y * (D / RT_S), kh = kl + D / RT_S;
    float acc[2][4] = {};
    for (int k0 = kl; k0 < kh; k0 += RT_K) {
        __syncthreads();
        for (int e = t; e < RT_R * RT_K; e += 256) {
            const int r = e / RT_K, k = e % RT_K;
            sF[r][k] = r0 + r < m ? f[(size_t)(r0 + r) * D + k0 + k] : 0.f;
        }
        for (int e = t; e < NE * RT_K; e += 256) {
            const int x = e / RT_K, k = e % RT_K;
            sW[x][k] = wr[(size_t) x * D + k0 + k];
        }
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < RT_K; k++) {
            const float f0 = sF[tr * 2][k], f1 = sF[tr * 2 + 1][k];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const float w = sW[te + 32 * c][k];
                acc[0][c] = fmaf(f0, w, acc[0][c]);
                acc[1][c] = fmaf(f1, w, acc[1][c]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = r0 + tr * 2 + i;
        if (r < m)
            #pragma unroll
            for (int c = 0; c < 4; c++) part[((size_t) blockIdx.y * m + r) * NE + te + 32 * c] = acc[i][c];
    }
}
// one wave per row
static __global__ __launch_bounds__(256) void k_pf_topk(const float * part, const float * bias, float scale,
                                                        int * ids, float * wts, int m) {
    const int lane = threadIdx.x & 63, r = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (r >= m) return;
    float za = 0.f, zb = 0.f;
    #pragma unroll
    for (int k = 0; k < RT_S; k++) { za += part[((size_t) k * m + r) * NE + lane]; zb += part[((size_t) k * m + r) * NE + lane + 64]; }
    const float pa = sigm(za), pb = sigm(zb);
    float sa = pa + bias[lane], sb = pb + bias[lane + 64];
    float sum = 0.f, pk = 0.f;
    int idk = 0;
    for (int k = 0; k < NE_USED; k++) {
        float bv = sa; int bi = lane;
        if (sb > bv) { bv = sb; bi = lane + 64; }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 64);
            const int oi = __shfl_xor(bi, off, 64);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const float pw = __shfl(bi < 64 ? pa : pb, bi & 63, 64);
        sum += pw;
        if (lane == k) { idk = bi; pk = pw; }
        if (bi == lane) sa = -INFINITY;
        if (bi == lane + 64) sb = -INFINITY;
    }
    sum += 1e-20f;
    if (lane < NE_USED) {
        ids[(size_t) r * NE_USED + lane] = idk;
        wts[(size_t) r * NE_USED + lane] = pk / sum * scale;
    }
}

// counting sort of the 8m (row, expert) pairs by expert: start offsets,
// perm (pair -> slot), inv (pair -> slot too); within an expert the order is
// arbitrary (each pair's results are independent; the row gather is what is
// deterministic)
static __global__ __launch_bounds__(256) void k_pf_sort(const int * ids, int np, int * start, int * perm, int * inv) {
    __shared__ int cnt[NE], cur[NE], st[NE];
    const int tid = threadIdx.x;
    for (int e = tid; e < NE; e += 256) cnt[e] = 0;
    __syncthreads();
    for (int p = tid; p < np; p += 256) atomicAdd(&cnt[ids[p]], 1);
    __syncthreads();
    if (tid == 0) {
        int s = 0;
        for (int e = 0; e < NE; e++) { st[e] = s; cur[e] = s; s += cnt[e]; }
    }
    __syncthreads();
    for (int e = tid; e < NE; e += 256) start[e] = st[e];
    if (tid == 0) start[NE] = np;
    for (int p = tid; p < np; p += 256) { const int q = atomicAdd(&cur[ids[p]], 1); perm[q] = p; inv[p] = q; }
}

// ---- grouped expert GEMM ----
// One tile per (expert, BN pairs). The tile table is built on device from the
// counting sort; a GEMM block covers 64 weight rows of that expert and the
// tile's pairs, staging weights and gathered activation rows in LDS per four
// sub-blocks (128 K values), as the shared pf_gemm does.

template <int BN>
static __global__ __launch_bounds__(256) void k_pf_tiles(const int * start, int4 * tm, int * ntl) {
    __shared__ int off[NE + 1];
    if (threadIdx.x == 0) {
        int s = 0;
        for (int e = 0; e < NE; e++) { off[e] = s; s += (start[e + 1] - start[e] + BN - 1) / BN; }
        off[NE] = s;
        *ntl = s;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NE; e += 256) {
        const int a0 = start[e], a1 = start[e + 1];
        for (int j = 0; j * BN < a1 - a0; j++) {
            const int base = a0 + j * BN;
            tm[off[e] + j] = make_int4(e, base, min(BN, a1 - base), 0);
        }
    }
}

// two tile lists: BN-wide tiles (full ones, and a remainder over BN/2) and
// BN/2-wide tiles for remainders up to BN/2; counts ntl[0], ntl[1]
constexpr int TM_B = PF_NP / 16 + NE;      // list B offset in tm (list A: up to np / 16 + NE tiles)
template <int BN>
static __global__ __launch_bounds__(256) void k_pf_tiles2(const int * start, int4 * tm, int * ntl) {
    __shared__ int offa[NE], offb[NE];
    if (threadIdx.x == 0) {
        int sa = 0, sb = 0;
        for (int e = 0; e < NE; e++) {
            const int cnt = start[e + 1] - start[e], r = cnt % BN;
            offa[e] = sa; offb[e] = sb;
            sa += cnt / BN + (r > BN / 2);
            sb += (r > 0 && r <= BN / 2);
        }
        ntl[0] = sa; ntl[1] = sb;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NE; e += 256) {
        const int a0 = start[e], cnt = start[e + 1] - a0, full = cnt / BN, r = cnt % BN;
        for (int j = 0; j < full; j++) tm[offa[e] + j] = make_int4(e, a0 + j * BN, BN, 0);
        if (r > BN / 2) tm[offa[e] + full] = make_int4(e, a0 + full * BN, r, 0);
        else if (r > 0) tm[TM_B + offb[e]] = make_int4(e, a0 + full * BN, r, 0);
    }
}

// tile_meta: {expert, first pair slot, count}. GATHER: the activation rows come
// from the chunk's row planes through the sorted pair list (gate_up); else the
// planes are already slot-major (down). Blocks past the tile count exit.
// fused SITU output of the gate_up GEMM: int8 planes [NGRP_FF][np][128] + scales
struct pf_situo { unsigned char * q = nullptr; float * qs = nullptr; int np = 0; float beta = 0.f, lb = 0.f; };

template <int BN, int GATHER, int NTT, int FS = 0>
static __device__ __forceinline__ void moe_gemm_body(
        const int4 tl,
        const char * W, const __half * sw, const int * perm,
        const int8_t * Ag, const float * sag, int Mp, int K,
        float * C, int ldc, int N, const pf_situo so = pf_situo{}) {
    constexpr int NT = NTT;                    // pair lanes
    constexpr int RG = 256 / NT;               // row groups
    constexpr int RR = 64 / RG;                // rows per thread
    constexpr int NH = BN / NT;                // pair halves
    constexpr int NXF = BN * 4 >= 256 ? BN * 4 / 256 : 1;   // activation slots per thread
    __shared__ pf_u32x4 sWlo[64][4], sWhi[64][4];
    __shared__ unsigned short sWd[4][64];
    __shared__ pf_xrow sX[BN];
    __shared__ float sXd[BN][4];
    const int t = threadIdx.x, tx = t % NT, ty = t / NT;
    const int nkb = K / 32, row0 = blockIdx.x * 64, cnt = tl.z;
    W += (size_t) tl.x * N * K;
    sw += (size_t) tl.x * N * (K / 32);
    const unsigned wrow = (unsigned) min(row0 + t / 4, N - 1);
    const unsigned w_off = wrow * (unsigned) K + (t % 4) * 32u;
    const unsigned wd_off = wrow * (unsigned) nkb + t % 4;
    // this thread's staging slots; lanes past the tile's count read the last
    // pair and are dropped in the epilogue (as pf_gemm clamps its rows)
    unsigned xo[NXF], xsc[NXF];
    bool xok[NXF];
#pragma unroll
    for (int i = 0; i < NXF; i++) {
        const int e = t + 256 * i;
        xok[i] = e < BN * 4;
        const int slot = xok[i] ? e / 4 : 0;
        const int sl = min(slot, cnt - 1);
        const int row = GATHER ? (perm[tl.y + sl] >> 3) : (tl.y + sl);
        xo[i] = (unsigned) (row * 128 + (e % 4) * 32);
        xsc[i] = (unsigned) (row * 4 + e % 4);
    }
    float acc[RR][NH] = {};
    pf_u32x4 pwlo, pwhi, pxlo[NXF], pxhi[NXF];
    unsigned short pwd;
    float pxd[NXF];
    auto prefetch = [&](int sb0) {
        if (sb0 + (t % 4) < nkb) {
            const pf_u32x4 * ws = (const pf_u32x4 *) (W + w_off + (unsigned) sb0 * 32u);
            pwlo = ws[0]; pwhi = ws[1];
            pwd = ((const unsigned short *) sw)[wd_off + (unsigned) sb0];
        } else {
            pwlo = pf_u32x4{0, 0, 0, 0}; pwhi = pwlo; pwd = 0;   // past K, never used
        }
        const unsigned grp = (unsigned) (sb0 / 4) * (unsigned) Mp;
#pragma unroll
        for (int i = 0; i < NXF; i++) {
            if (!xok[i]) continue;
            const pf_u32x4 * p = (const pf_u32x4 *) (Ag + grp * 128u + xo[i]);
            pxlo[i] = p[0]; pxhi[i] = p[1];
            pxd[i] = sag[grp * 4u + xsc[i]];
        }
    };
    auto store = [&]() {
        // K slot XOR the row group: the row groups sharing a wave hit distinct banks
        const int ks = (t % 4) ^ ((t / 4 / RR) & 3);
        sWlo[t / 4][ks] = pwlo; sWhi[t / 4][ks] = pwhi; sWd[t % 4][t / 4] = pwd;
#pragma unroll
        for (int i = 0; i < NXF; i++) {
            if (!xok[i]) continue;
            const int e = t + 256 * i;
            sX[e / 4].q[e % 4][0] = pxlo[i]; sX[e / 4].q[e % 4][1] = pxhi[i];
            sXd[e / 4][e % 4] = pxd[i];
        }
    };
    prefetch(0);
    const int rb = ty * RR;
    for (int sb0 = 0; sb0 < nkb; sb0 += 4) {
        store();
        __syncthreads();
        if (sb0 + 4 < nkb) prefetch(sb0 + 4);
#pragma unroll 1
        for (int kk = 0; kk < 4; kk++) {
            if (sb0 + kk >= nkb) break;
            int xq[NH][8];
            float dx[NH];
#pragma unroll
            for (int n = 0; n < NH; n++) {
                const pf_u32x4 q0 = sX[tx + NT * n].q[kk][0], q1 = sX[tx + NT * n].q[kk][1];
                xq[n][0] = q0.x; xq[n][1] = q0.y; xq[n][2] = q0.z; xq[n][3] = q0.w;
                xq[n][4] = q1.x; xq[n][5] = q1.y; xq[n][6] = q1.z; xq[n][7] = q1.w;
                dx[n] = sXd[tx + NT * n][kk];
            }
            const int kw = kk ^ (ty & 3);
            pf_u32x4 wlo = sWlo[rb][kw], whi = sWhi[rb][kw];
#pragma unroll
            for (int r = 0; r < RR; r++) {
                const pf_u32x4 lo = wlo, hi = whi;
                if (r + 1 < RR) { wlo = sWlo[rb + r + 1][kw]; whi = sWhi[rb + r + 1][kw]; }
                const float d = __half2float(__ushort_as_half(sWd[kk][rb + r]));
#pragma unroll
                for (int n = 0; n < NH; n++) {
                    int isum = 0;
                    isum = __builtin_amdgcn_sdot4(xq[n][0], (int) lo.x, isum, false);
                    isum = __builtin_amdgcn_sdot4(xq[n][1], (int) lo.y, isum, false);
                    isum = __builtin_amdgcn_sdot4(xq[n][2], (int) lo.z, isum, false);
                    isum = __builtin_amdgcn_sdot4(xq[n][3], (int) lo.w, isum, false);
                    isum = __builtin_amdgcn_sdot4(xq[n][4], (int) hi.x, isum, false);
                    isum = __builtin_amdgcn_sdot4(xq[n][5], (int) hi.y, isum, false);
                    isum = __builtin_amdgcn_sdot4(xq[n][6], (int) hi.z, isum, false);
                    isum = __builtin_amdgcn_sdot4(xq[n][7], (int) hi.w, isum, false);
                    acc[r][n] = __builtin_fmaf((float) isum, dx[n] * d, acc[r][n]);
                }
            }
        }
        __syncthreads();
    }
    if constexpr (FS) {
        // tile rows 0..31 are gate rows 32*bx.., rows 32..63 their up rows (gu_row):
        // both halves to LDS; every thread then does SITU + the per-32 quant (one group per pair)
        __shared__ float sA[32][BN + 1];          // gate [j][slot]
        float * sU = (float *) sX;                 // up [slot][j ^ slot] (the staging buffer is free now)
#pragma unroll
        for (int n = 0; n < NH; n++) {
            const int slot = tx + NT * n;
#pragma unroll
            for (int r = 0; r < RR; r++) {
                if (rb < 32) sA[rb + r][slot] = acc[r][n];
                else sU[slot * 32 + ((rb - 32 + r) ^ (slot & 31))] = acc[r][n];
            }
        }
        __syncthreads();
        const int j0 = blockIdx.x * 32;
        for (int i = t; i < BN * 32; i += 256) {
            const int slot = i / 32, jj = i % 32;
            const float g = sA[jj][slot], u = sU[slot * 32 + (jj ^ (slot & 31))];
            const float x = so.beta * tanhf(g / so.beta) * sigm(g) * (so.lb * tanhf(u / so.lb));
            const float amax = qz_amax32(fabsf(x));
            if (slot < cnt) {
                const float d = amax / 127.f;
                const int j = j0 + jj;
                const size_t dd = (size_t) (j / 128) * so.np + tl.y + slot;
                so.q[dd * 128 + j % 128] = (unsigned char) qz_q8_code_div(x, amax, d);
                if (jj == 0) so.qs[dd * 4 + (j % 128) / 32] = qz_q8_scale(d);
            }
        }
        return;
    }
#pragma unroll
    for (int n = 0; n < NH; n++) {
        const int slot = tx + NT * n;
        if (slot >= cnt) continue;
        if (row0 + 64 > N) {                       // partial weight tile (N % 64 != 0): per-row guard
#pragma unroll
            for (int r = 0; r < RR; r++) if (row0 + rb + r < N) C[(size_t) (tl.y + slot) * ldc + row0 + rb + r] = acc[r][n];
            continue;
        }
        float * dst = C + (size_t) (tl.y + slot) * ldc + row0 + rb;
#pragma unroll
        for (int r = 0; r < RR; r += 4)
            *(float4 *) (dst + r) = make_float4(acc[r][n], acc[r + 1][n], acc[r + 2][n], acc[r + 3][n]);
    }
}

template <int BN, int GATHER, int NTT = (BN < 64 ? BN : 64), int FS = 0>
static __global__ __launch_bounds__(256, BN <= 32 ? 3 : 2) void k_pf_moe_gemm(
        const int4 * tm, const int * ntl,
        const char * W, const __half * sw, const int * perm,
        const int8_t * Ag, const float * sag, int Mp, int K,
        float * C, int ldc, int N, const pf_situo so) {
    if (blockIdx.y >= *ntl) return;
    moe_gemm_body<BN, GATHER, NTT, FS>(tm[blockIdx.y], W, sw, perm, Ag, sag, Mp, K, C, ldc, N, so);
}

// dense projections through the same tile body: one launch for up to 6 jobs
// (blockIdx.z), token tiles of 64 from blockIdx.y; a job's last weight tile may be partial
struct dgjob { const char * W; const __half * sw; const int8_t * A; const float * as; float * C; int ldc, N, K; };
struct dgjobs { dgjob j[8]; };
static __global__ __launch_bounds__(256, 2) void k_pf_dgemm(dgjobs J, int m) {
    const dgjob & g = J.j[blockIdx.z];
    if ((int) blockIdx.x * 64 >= g.N) return;
    const int a0 = blockIdx.y * 64;
    moe_gemm_body<64, 0, 16>(make_int4(0, a0, min(64, m - a0), 0), g.W, g.sw, nullptr, g.A, g.as, PF_M, g.K, g.C, g.ldc, g.N);
}

// per row, the weighted sum of its pairs' partials in the router's selection
// order (deterministic; no atomics) -> the card's partial for the exchange
static __global__ __launch_bounds__(256) void k_pf_moe_reduce(const float * y, const int * inv, const float * wts, float * out) {
    const int r = blockIdx.x;
    const int * iv = inv + (size_t) r * NE_USED;
    const float * wt = wts + (size_t) r * NE_USED;
    for (int o = threadIdx.x; o < D; o += 256) {
        float acc = 0.f;
        for (int k = 0; k < NE_USED; k++) acc = fmaf(wt[k], y[(size_t) iv[k] * D + o], acc);
        out[(size_t) r * D + o] = acc;
    }
}

// ------------------------------------------------------------ host side ----

struct pf_plane { unsigned char * q; float * s; };   // pf_gemm activation planes

struct pf_card {
    float * X, * P, * OUT[2], * BLK[8];
    float * IN;                        // [3][PF_M][D]
    float * MSG, * CT, * Z;            // [4][PF_M][D], [4][PF_M][D], [4][PF_M][2D]
    float * Gb;                        // [4][PF_M][GRANK]
    float * F, * FN, * PART[2];        // [PF_M][D]
    float * QR, * GR, * Q, * RR, * SK, * KR, * VR, * DK, * DV, * TB;
    pf_plane INP[3], FF, AT;           // input planes (K = D), ffn (K = D), attn out (K = QL)
    unsigned char * APQ; float * APS;  // per-pair situ planes [PF_NP][FFL], [PF_NP][FFL/32]
    float * GP, * YD;                  // controller K-split partials [CT_S][4][PF_M][GRANK], expert down out [PF_NP][D]
    int * IDS; float * WTS;            // [PF_M][NE_USED]
    int * PERM, * INV, * START;        // [PF_NP], [PF_NP], [NE + 1]
    int * TOK;
    int4 * TM; int * NTL;              // expert tiles: {expert, first pair, count}, count
};

// rows of out (row stride ostride) = RMSNorm(x; w)
static __global__ __launch_bounds__(256) void k_pf_rmsw(const float * x, const float * w, float * out, int ostride, float eps) {
    __shared__ float red[16];
    const float * xr = x + (size_t) blockIdx.x * D;
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) { v[k] = xr[threadIdx.x + k * 256]; ss[0] += v[k] * v[k]; }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) out[(size_t) blockIdx.x * ostride + threadIdx.x + k * 256] = v[k] * r * w[threadIdx.x + k * 256];
}

struct ink2_pf {
    ink2_engine * E = nullptr;
    std::atomic<bool> * abort = nullptr;
    pf_card B[2];
    int bnx = 0;                       // exchange parity (double-buffered partials)
    // the DSpark drafter's target features per prompt row, [row][4][D]: the residual
    // entering applications 5, 23, 41 and the head input output_norm(mix(blocks, x; out_q)).
    // dtaps: card 0, the chunk's rows from 0, then on_chunk(p0, m) (the drafter's context fill)
    float * dtaps = nullptr;
    std::function<void(int, int)> on_chunk;
    float * tapd = nullptr;            // card 0: one chunk's head mixes
    // the prompt's image rows: irow[pos] (host, per prompt position, -1: a token) indexes
    // img[c] (each card's copy, [rows][D]); null: text only
    const int * irow = nullptr;
    const float * img[2] = {};
    int * irw[2] = {};                 // the chunk's irow on each card
    bool tapping() const { return dtaps != nullptr; }
    void tap(int k, int p0, int m) {
        if (!tapping()) return;
        ICHECK(hipSetDevice(qz_dev_of(0)));
        hipStream_t st = E->C[0].st;
        ICHECK(hipMemcpy2DAsync(dtaps + (size_t) k * D, 4 * D * 4, B[0].X, D * 4, D * 4, m, hipMemcpyDeviceToDevice, st));
    }
    void tap_head(const ink2_engine::seq & S, int p0, int m) {
        if (!tapping()) return;
        ink2_engine & E = *this->E;
        ink2_card & K = E.C[0];
        ICHECK(hipSetDevice(qz_dev_of(0)));
        if (!tapd) ICHECK(hipMalloc(&tapd, (size_t) PF_M * D * 4));
        for (int r = 0; r < m; r++) {
            std::vector<const float *> bl;
            for (int ib : S.blocks) bl.push_back(B[0].BLK[ib] + (size_t) r * D);
            E.mix(0, bl, B[0].X + (size_t) r * D, K.out_q, 1, tapd + (size_t) r * D);
        }
        k_pf_rmsw<<<m, 256, 0, K.st>>>(tapd, K.out_norm, dtaps + 3 * D, 4 * D, E.hp.eps);
        if (on_chunk) on_chunk(p0, m);
    }

    bool alloc(int c) {
        ICHECK(hipSetDevice(qz_dev_of(c)));
        qz_mem_scope mem("state");
        pf_card & b = B[c];
        const size_t MD = (size_t) PF_M * D;
        size_t need = 0;
        std::vector<size_t> sz;
        auto f = [&](size_t n) { const size_t bb = (n * 4 + 255) & ~(size_t) 255; sz.push_back(bb); need += bb; };
        auto by = [&](size_t n) { const size_t bb = (n + 255) & ~(size_t) 255; sz.push_back(bb); need += bb; };
        auto plane = [&](int K) {
            by((size_t) K * PF_M);
            f((size_t) (K / 32) * PF_M);
        };
        for (int i = 0; i < 1 + 1 + 2 + 8; i++) f(MD);                       // X P OUT BLK
        f(3 * MD); f(4 * MD); f(4 * MD); f(4 * PF_M * 2 * D);                 // IN MSG CT Z
        f(4 * (size_t) PF_M * GRANK); f(MD); f(MD); f(MD); f(MD);             // Gb F FN PART x2
        f((size_t) PF_M * QL); f((size_t) PF_M * QL); f((size_t) PF_M * QL);  // QR GR Q
        f((size_t) PF_M * (HQL * DREL)); f((size_t) PF_M * HQL);              // RR SK
        f((size_t) PF_M * KVL); f((size_t) PF_M * KVL);                       // KR VR
        f((size_t) PF_M * (HKVL * NCONV)); f((size_t) PF_M * (HKVL * NCONV)); // DK DV
        f((size_t) PF_M * HQL * DREL);                                         // TB: the clipped R
        f((size_t) CT_S * 4 * PF_M * GRANK); f((size_t) PF_NP * D);           // GP YD
        f((size_t) PF_NP * NSC_FF); f((size_t) PF_M * NE_USED);               // APS WTS
        f((size_t) PF_M * NE_USED);                                           // IDS (as floats)
        f((size_t) PF_NP); f((size_t) PF_NP); f((size_t) (NE + 1)); f(PF_M);  // PERM INV START TOK
        f((size_t) (PF_NP / 16 + 2 * NE) * 4); f(1);                          // TM (lists A, B) NTL
        for (int i = 0; i < 3; i++) plane(D);                                 // INP
        plane(D); plane(QL);                                                  // FF AT
        by((size_t) PF_NP * FFL_PAD);                                         // APQ
        char * base0 = nullptr;
        if (qz_dmalloc((void **) &base0, need, "state") != hipSuccess) {
            fprintf(stderr, "inkling2: card %d: no room for %.2f GB of prefill state\n", c, need / 1e9);
            return false;
        }
        char * p = base0;
        auto nx = [&]() { char * r = p; p += sz.front(); sz.erase(sz.begin()); return r; };
        b.X = (float *) nx();
        b.P = (float *) nx();
        for (auto & q : b.OUT) q = (float *) nx();
        for (auto & q : b.BLK) q = (float *) nx();
        b.IN = (float *) nx(); b.MSG = (float *) nx(); b.CT = (float *) nx(); b.Z = (float *) nx();
        b.Gb = (float *) nx(); b.F = (float *) nx(); b.FN = (float *) nx();
        for (auto & q : b.PART) q = (float *) nx();
        b.QR = (float *) nx(); b.GR = (float *) nx(); b.Q = (float *) nx(); b.RR = (float *) nx(); b.SK = (float *) nx();
        b.KR = (float *) nx(); b.VR = (float *) nx(); b.DK = (float *) nx(); b.DV = (float *) nx(); b.TB = (float *) nx();
        b.GP = (float *) nx(); b.YD = (float *) nx(); b.APS = (float *) nx(); b.WTS = (float *) nx();
        b.IDS = (int *) nx(); b.PERM = (int *) nx(); b.INV = (int *) nx(); b.START = (int *) nx(); b.TOK = (int *) nx();
        b.TM = (int4 *) nx(); b.NTL = (int *) nx();
        for (auto & s : b.INP) { s.q = (unsigned char *) nx(); s.s = (float *) nx(); }
        b.FF.q = (unsigned char *) nx(); b.FF.s = (float *) nx();
        b.AT.q = (unsigned char *) nx(); b.AT.s = (float *) nx();
        b.APQ = (unsigned char *) nx();
        return true;
    }

    // ---- launch helpers (current device = card c)
    // ---- one application: all m chunk rows of positions pos0..pos0+m-1
    // through state slot `slot` of physical layer ip
    // ---- launch helpers (current device = card c)
    dgjob djob(const qw & W, const unsigned char * av, const float * as, float * C, int ldc) {
        return dgjob{W.v, W.s, (const int8_t *) av, as, C, ldc, (int) W.N, (int) W.K};
    }
    // several dense GEMMs in one launch (up to 8 jobs)
    void dgemm(int c, std::initializer_list<dgjob> js, int m) {
        dgjobs J{};
        int n = 0, mx = 0;
        for (const dgjob & g : js) { J.j[n++] = g; mx = std::max(mx, g.N); }
        k_pf_dgemm<<<dim3((mx + 63) / 64, (m + 63) / 64, n), 256, 0, E->C[c].st>>>(J, m);
    }
    void normq(int c, const float * x, const float * w, pf_plane & pl, int m) {
        pf_norm_quant<<<m, 256, 0, E->C[c].st>>>(x, D, w, E->hp.eps, D, (int8_t *) pl.q, pl.s, PF_M, nullptr, 0);
    }
    void commit(int c, const float * src, const float * peer, float * hrg, int w, int m) {
        hipStream_t st = E->C[c].st;
        k_pf_commit<<<(w + 255) / 256, 256, 0, st>>>(src, peer, hrg, w, m);
    }

    // ---- grouped expert GEMM: tile width from the chunk's pairs per expert
    int moe_bn(int np) {
        const int c = np / NE;
        return c < 12 ? 16 : c < 48 ? 32 : 64;
    }
    void moe_tiles(int c, int bn) {
        ICHECK(hipSetDevice(qz_dev_of(c)));
        hipStream_t st = E->C[c].st;
        if (bn == 16) k_pf_tiles<16><<<1, 256, 0, st>>>(B[c].START, B[c].TM, B[c].NTL);
        else if (bn == 32) k_pf_tiles2<32><<<1, 256, 0, st>>>(B[c].START, B[c].TM, B[c].NTL);
        else k_pf_tiles2<64><<<1, 256, 0, st>>>(B[c].START, B[c].TM, B[c].NTL);
    }
    template <int BN, int G>
    void moe_gemm(pf_card & b, hipStream_t st, const qw & W, const unsigned char * aq, const float * as,
                  int Mp, int K, float * C, int ldc, int N, int yt, const int4 * tm, const int * ntl, const pf_situo & so) {
        // 4 x 4 (rows x pairs) per thread: 16 pair lanes; the gate_up GEMM (G = 1) ends in the fused SITU
        if constexpr (G == 1)
            k_pf_moe_gemm<BN, G, 16, 1><<<dim3(N / 64, yt), 256, 0, st>>>(tm, ntl, W.v, W.s, b.PERM, (const int8_t *) aq, as, Mp, K, C, ldc, N, so);
        else
            k_pf_moe_gemm<BN, G, 16, 0><<<dim3(N / 64, yt), 256, 0, st>>>(tm, ntl, W.v, W.s, b.PERM, (const int8_t *) aq, as, Mp, K, C, ldc, N, so);
    }
    void moe_run(pf_card & b, hipStream_t st, int bn, int gather, const qw & W, const unsigned char * aq, const float * as,
                 int Mp, int K, float * C, int ldc, int N, int yt, const int4 * tm, const int * ntl, const pf_situo & so) {
        if (bn == 16) {
            if (gather) moe_gemm<16, 1>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
            else        moe_gemm<16, 0>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
        } else if (bn == 32) {
            if (gather) moe_gemm<32, 1>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
            else        moe_gemm<32, 0>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
        } else {
            if (gather) moe_gemm<64, 1>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
            else        moe_gemm<64, 0>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
        }
    }
    void moe(int c, int bn, int gather, const qw & W, const unsigned char * aq, const float * as, int Mp, int K,
             float * C, int ldc, int N, int yt, const pf_situo & so = pf_situo{}) {
        ICHECK(hipSetDevice(qz_dev_of(c)));
        pf_card & b = B[c];
        hipStream_t st = E->C[c].st;
        moe_run(b, st, bn, gather, W, aq, as, Mp, K, C, ldc, N, yt, b.TM, b.NTL, so);
        if (bn >= 32) moe_run(b, st, bn / 2, gather, W, aq, as, Mp, K, C, ldc, N, NE, b.TM + TM_B, b.NTL + 1, so);
    }

    void layer(int slot, int ip, int pos0, int m, ink2_engine::seq & S) {
        ink2_engine & E = *this->E;
        const bool full = is_full(ip), msg = S.bank >= 0;
        const int buf = bnx & 1;
        const int push = ip % ARB == 0 ? S.nblk : -1;
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ink2_card & K = E.C[c];
            pf_card & b = B[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            const float * in[3] = {b.X, b.X, b.X};
            if (L.resq) {           // attn-res mix over depth, per row
                pfmix A{};
                int nst = 0;
                for (int ib : S.blocks) A.p[nst++] = b.BLK[ib];
                A.p[nst++] = b.X;
                A.n = nst;
                A.qy = L.resq; A.m = full ? 3 : 1; A.out = b.IN; A.ostr = (size_t) PF_M * D;
                if (full) k_pf_mix2<3><<<m, 256, 0, st>>>(A); else k_pf_mix2<1><<<m, 256, 0, st>>>(A);
                for (int j = 0; j < 3; j++) in[j] = b.IN + (full ? (size_t) j : 0) * PF_M * D;
            }
            if (msg) {
                pfmix A{};
                A.p[0] = b.BLK[S.bank]; A.p[1] = b.OUT[S.bank_out]; A.n = 2;
                A.qy = L.msgq; A.m = NCTRL; A.out = b.MSG; A.ostr = (size_t) PF_M * D;
                k_pf_mix2<4><<<m, 256, 0, st>>>(A);
                pfrms R{};
                for (int j = 0; j < 3; j++) R.cur[j] = in[j];
                R.msg = b.MSG; R.z = b.Z; R.j0 = 0;
                k_pf_rms2<<<dim3(m, 3), 256, 0, st>>>(R);
                pfcdn Da{};
                Da.z = b.Z; Da.wd0 = L.cd[0]; Da.wd1 = L.cd[1]; Da.wd2 = L.cd[2]; Da.wd3 = L.cd[3];
                Da.g = b.Gb; Da.j0 = 0; Da.m = m;
                k_pf_ctrl_down2<<<dim3((m + 63) / 64, 3, CT_S), 256, 0, st>>>(Da, b.GP);
                pfcup Ua{};
                Ua.g = b.Gb; Ua.msg = b.MSG; Ua.out = b.CT; Ua.j0 = 0; Ua.m = m;
                for (int j = 0; j < 4; j++) {
                    Ua.cur[j] = j < 3 ? in[j] : b.F;
                    Ua.wgs[j] = L.cg[j];
                    Ua.wfs[j] = L.cf[j];
                }
                k_pf_gsum<<<dim3((m * GRANK + 255) / 256, 3), 256, 0, st>>>(b.GP, b.Gb, 0, m);
                k_pf_ctrl_up2<<<dim3((m + 63) / 64, D / 64, 3), 256, 0, st>>>(Ua);
                for (int j = 0; j < 3; j++) in[j] = b.CT + (size_t) j * PF_M * D;
            }
            // norm + int8 quant per stream, then the Q8 GEMMs
            const int ik = full || msg ? 1 : 0, ivv = full || msg ? 2 : 0;
            if (full) {
                normq(c, in[0], L.qn, b.INP[0], m);
                normq(c, in[1], L.kn, b.INP[1], m);
                normq(c, in[2], L.vn, b.INP[2], m);
            } else if (msg) {
                for (int j = 0; j < 3; j++) normq(c, in[j], L.an, b.INP[j], m);
            } else {
                normq(c, in[0], L.an, b.INP[0], m);
            }
            if (!full) dgemm(c, {djob(L.q, b.INP[0].q, b.INP[0].s, b.QR, QL), djob(L.gate, b.INP[0].q, b.INP[0].s, b.GR, QL),
                      djob(L.rel, b.INP[0].q, b.INP[0].s, b.RR, HQL * DREL), djob(L.k, b.INP[ik].q, b.INP[ik].s, b.KR, KVL),
                      djob(L.v, b.INP[ivv].q, b.INP[ivv].s, b.VR, KVL), djob(L.sink, b.INP[0].q, b.INP[0].s, b.SK, HQL)}, m);
            if (full) dgemm(c, {djob(L.q, b.INP[0].q, b.INP[0].s, b.QR, QL), djob(L.gate, b.INP[0].q, b.INP[0].s, b.GR, QL),
                                djob(L.rel, b.INP[0].q, b.INP[0].s, b.RR, HQL * DREL), djob(L.k, b.INP[1].q, b.INP[1].s, b.KR, KVL),
                                djob(L.v, b.INP[2].q, b.INP[2].s, b.VR, KVL), djob(L.sink, b.INP[0].q, b.INP[0].s, b.SK, HQL),
                                djob(L.kcp, b.INP[1].q, b.INP[1].s, b.DK, HKVL * NCONV), djob(L.vcp, b.INP[2].q, b.INP[2].s, b.DV, HKVL * NCONV)}, m);
            // head post: clips, tau, bias tables, K/V convs into the cache
            pfhp h{};
            h.qr = b.QR; h.rr = b.RR; h.kr = b.KR; h.vr = b.VR;
            h.dk = full ? b.DK : nullptr; h.dv = full ? b.DV : nullptr;
            h.qclip = L.qclip; h.rclip = L.rclip; h.kconv = L.kconv; h.vconv = L.vconv;
            h.hist = K.hist + (size_t) slot * 3 * CHL;
            h.q = b.Q; h.kc = K.kc[slot]; h.vc = K.vc[slot];
            h.rvo = b.TB;
            h.pos0 = pos0; h.full = full;
            h.tau_a = E.hp.tau_alpha; h.tau_f = E.hp.tau_floor;
            h.qk_clip = E.hp.qk_clip; h.rel_clip = E.hp.rel_clip; h.eps = E.hp.eps;
            k_pf_head_post<<<dim3(HQL + 2 * HKVL, m), 128, 0, st>>>(h);
            commit(c, b.KR, nullptr, h.hist + H_K, KVL, m);
            commit(c, b.VR, nullptr, h.hist + H_V, KVL, m);
            // attention, then attn_output (row-parallel: partials per card)
            pfat t{};
            t.q = b.Q; t.gate = b.GR; t.kc = K.kc[slot]; t.vc = K.vc[slot];
            t.pos0 = pos0; t.full = full; t.oq = b.AT.q; t.os = b.AT.s;
            t.rv = b.TB; t.relb = L.relb; t.sk = b.SK;
            k_pf_fattn<1, true><<<dim3((m + 31) / 32, HQL), 512, 0, st>>>(t, m);
            dgemm(c, {djob(L.out, b.AT.q, b.AT.s, b.PART[buf], D)}, m);
        }
        E.exchange();
        bnx++;
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ink2_card & K = E.C[c];
            pf_card & b = B[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            pfcv ca{};
            ca.own = b.PART[buf]; ca.peer = B[1 - c].PART[buf]; ca.w = L.pconv;
            ca.hrg = K.hist + (size_t) slot * 3 * CHL + H_A;
            ca.x = b.X; ca.blk = push >= 0 ? b.BLK[push] : nullptr; ca.m = m; ca.push = push >= 0;
            k_pf_conv2<<<dim3((D + 255) / 256, (m + CV_R - 1) / CV_R), 256, 0, st>>>(ca);
            commit(c, b.PART[buf], B[1 - c].PART[buf], ca.hrg, D, m);
        }
        if (push >= 0) { S.blocks.push_back(push); S.nblk++; }
        const int buf2 = bnx & 1;
        const int np = m * NE_USED, bn = moe_bn(np), yt = np / bn + NE;   // expert tiles
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ink2_card & K = E.C[c];
            pf_card & b = B[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            pfmix A{};
            int nst = 0;
            for (int ib : S.blocks) A.p[nst++] = b.BLK[ib];
            A.p[nst++] = b.X;
            A.n = nst;
            A.qy = L.ffq; A.m = 1; A.out = b.F; A.ostr = 0;
            k_pf_mix2<1><<<m, 256, 0, st>>>(A);
            const float * f = b.F;
            if (msg) {
                pfrms R{};
                R.cur[3] = b.F; R.msg = b.MSG; R.z = b.Z; R.j0 = 3;
                k_pf_rms2<<<dim3(m, 1), 256, 0, st>>>(R);
                pfcdn Da{};
                Da.z = b.Z; Da.wd3 = L.cd[3]; Da.g = b.Gb; Da.j0 = 3; Da.m = m;
                k_pf_ctrl_down2<<<dim3((m + 63) / 64, 1, CT_S), 256, 0, st>>>(Da, b.GP);
                pfcup Ua{};
                Ua.g = b.Gb; Ua.msg = b.MSG; Ua.out = b.CT; Ua.j0 = 3; Ua.m = m;
                Ua.cur[3] = b.F; Ua.wgs[3] = L.cg[3]; Ua.wfs[3] = L.cf[3];
                k_pf_gsum<<<dim3((m * GRANK + 255) / 256, 1), 256, 0, st>>>(b.GP, b.Gb, 3, m);
                k_pf_ctrl_up2<<<dim3((m + 63) / 64, D / 64, 1), 256, 0, st>>>(Ua);
                f = b.CT + (size_t) 3 * PF_M * D;
            }
            pf_norm_quant<<<m, 256, 0, st>>>(f, D, L.ffn, E.hp.eps, D, (int8_t *) b.FF.q, b.FF.s, PF_M, b.FN, 0);
            {
                k_pf_router2<<<dim3((m + RT_R - 1) / RT_R, RT_S), 256, 0, st>>>(b.FN, L.gin, b.YD, m);   // YD: scratch until the down GEMM
                k_pf_topk<<<(m + 3) / 4, 256, 0, st>>>(b.YD, L.pb, E.hp.route_scale, b.IDS, b.WTS, m);
            }
            k_pf_sort<<<1, 256, 0, st>>>(b.IDS, np, b.START, b.PERM, b.INV);
            moe_tiles(c, bn);
            pf_situo so;            // SITU fused into the gate_up GEMM's epilogue -> the down GEMM's planes
            so.q = b.APQ; so.qs = b.APS; so.np = np; so.beta = E.hp.situ_beta; so.lb = E.hp.situ_lin_beta;
            moe(c, bn, 1, L.gu, b.FF.q, b.FF.s, PF_M, D, nullptr, GUL, GUL, yt, so);
            moe(c, bn, 0, L.dn, b.APQ, b.APS, np, FFL, b.YD, D, D, yt);
            k_pf_moe_reduce<<<m, 256, 0, st>>>(b.YD, b.INV, b.WTS, b.PART[buf2]);
        }
        E.exchange();
        bnx++;
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ink2_card & K = E.C[c];
            pf_card & b = B[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            pfcv cf{};
            cf.own = b.PART[buf2]; cf.peer = B[1 - c].PART[buf2]; cf.w = L.fconv;
            cf.hrg = K.hist + (size_t) slot * 3 * CHL + H_F;
            cf.x = b.X; cf.blk = nullptr; cf.m = m; cf.push = 0;
            k_pf_conv2<<<dim3((D + 255) / 256, (m + CV_R - 1) / CV_R), 256, 0, st>>>(cf);
            commit(c, b.PART[buf2], B[1 - c].PART[buf2], cf.hrg, D, m);
        }
    }

    // the head for the chunk's last row only (decode kernels, row offsets)
    void head_last(const ink2_engine::seq & S, int m) {
        ink2_engine & E = *this->E;
        const size_t off = (size_t)(m - 1) * D;
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ink2_card & K = E.C[c];
            pf_card & b = B[c];
            std::vector<const float *> bl;
            for (int ib : S.blocks) bl.push_back(b.BLK[ib] + off);
            E.mix(c, bl, b.X + off, K.out_q, 1, K.F);
            njobs N{};
            N.j[0] = {K.F, K.out_norm, nullptr, K.aq[3], K.as[3]};
            k_norm_quant<<<1, 256, 0, K.st>>>(N, E.hp.eps);
            E.gemv(c, {ink2_engine::job(K.head, K.aq[3], K.as[3], K.LOG)});
            k_softcap<<<(K.nvalid + 255) / 256, 256, 0, K.st>>>(K.LOG, K.nvalid, E.hp.softcap);
            qz_topk_rows<<<dim3(QZ_TOPK_NB, 1), 256, 0, K.st>>>(K.LOG, K.nvalid, K.id0, K.tk);
            ICHECK(hipMemcpyAsync(E.tkh[c].data(), K.tk, E.tkh[c].size() * sizeof(float2), hipMemcpyDeviceToHost, K.st));
        }
        for (int c = 0; c < 2; c++) { ICHECK(hipSetDevice(qz_dev_of(c))); ICHECK(hipStreamSynchronize(E.C[c].st)); }
    }

    // ---- the batched prefill of prompt rows a .. n-1 (rows below a already in
    // the KV and conv history); logits: the head on row n-1. false if aborted
    bool prefill(const int32_t * prompt, int a, int n, bool logits) {
        ink2_engine & E = *this->E;
        for (int p0 = a; p0 < n; p0 += PF_M) {
            if (abort && abort->load()) return false;
            const int m = std::min(PF_M, n - p0);
            for (int c = 0; c < 2; c++) {
                ICHECK(hipSetDevice(qz_dev_of(c)));
                ink2_card & K = E.C[c];
                pf_card & b = B[c];
                ICHECK(hipMemcpyAsync(b.TOK, prompt + p0, (size_t) m * 4, hipMemcpyHostToDevice, K.st));
                k_pf_embed<<<m, 256, 0, K.st>>>(b.TOK, K.embd.v, K.embd.s, K.embd_norm, E.hp.eps, b.X);
                if (irow && std::any_of(irow + p0, irow + p0 + m, [](int r) { return r >= 0; })) {
                    if (!irw[c]) ICHECK(qz_dmalloc((void **) &irw[c], (size_t) PF_M * 4, "images"));
                    ICHECK(hipMemcpyAsync(irw[c], irow + p0, (size_t) m * 4, hipMemcpyHostToDevice, K.st));
                    k_pf_img<<<m, 256, 0, K.st>>>(irw[c], img[c], b.X);
                }
            }
            ink2_engine::seq S;
            for (int il = 0; il < LOOP0; il++) { if (il == 5) tap(0, p0, m); layer(il, il, p0, m, S); }
            const ink2_engine::seq pre = S;
            for (int c = 0; c < 2; c++) {
                ICHECK(hipSetDevice(qz_dev_of(c)));
                ICHECK(hipMemcpyAsync(B[c].P, B[c].X, (size_t) m * D * 4, hipMemcpyDeviceToDevice, E.C[c].st));
            }
            for (int pass = 0; pass < NPASS; pass++) {
                if (pass > 0) {
                    const int bank = S.blocks[pre.blocks.size() + 1];
                    const int ob = (pass - 1) & 1;
                    for (int c = 0; c < 2; c++) {
                        ICHECK(hipSetDevice(qz_dev_of(c)));
                        ICHECK(hipMemcpyAsync(B[c].OUT[ob], B[c].X, (size_t) m * D * 4, hipMemcpyDeviceToDevice, E.C[c].st));
                        ICHECK(hipMemcpyAsync(B[c].X, B[c].P, (size_t) m * D * 4, hipMemcpyDeviceToDevice, E.C[c].st));
                    }
                    const int nblk = S.nblk;
                    S = pre;
                    S.nblk = nblk;
                    S.bank = bank;
                    S.bank_out = ob;
                }
                for (int ip = LOOP0; ip < LOOP1; ip++) {
                    const int slot = pass == 0 ? ip : NL + (pass - 1) * (LOOP1 - LOOP0) + (ip - LOOP0);
                    if ((pass == 1 && ip == 11) || (pass == 2 && ip == 17)) tap(pass, p0, m);   // visits 23, 41
                    layer(slot, ip, p0, m, S);
                }
            }
            S.bank = -1;
            for (int il = LOOP1; il < NL; il++) layer(il, il, p0, m, S);
            tap_head(S, p0, m);
            if (logits && p0 + m == n) head_last(S, m);
        }
        return true;
    }
};
