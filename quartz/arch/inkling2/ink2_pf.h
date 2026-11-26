// inkling2 batched prefill. Prompt rows pass through the 48 applications of
// docs/inkling2.md in chunks of PF_M, one application at a time, which is exact: rows
// interact only through a slot's KV and conv history. The dense and grouped MoE Q8
// GEMMs share one tile body over the repacked Q8 planes; norm, quant and the plane
// layout come from pf_gemm.h. The depth mix, the loop controllers, the head post and
// the row convs, and the HD 128 flash attention are arch kernels. Decode stays on
// ink2_engine::step(). Same TU as ink2_kernels.h and ink2_engine.h.
#pragma once

#include "kernels/pf_gemm.h"     // pf_norm_quant, pf_quant_unit, plane types
#include "core/link.h"           // qz_xcopy: the prefill partial to the peer

constexpr int PF_NP = 8 * PF_M;             // max (row, expert) pairs per chunk
constexpr int GUL = 2 * FFL;                // local gate and up rows per expert
constexpr int NGRP_FF = (FFL + 127) / 128;  // 128-wide groups of the down contraction, 5, the last one half
constexpr int FFL_PAD = NGRP_FF * 128;
constexpr int NSC_FF = NGRP_FF * 4;         // scale slots per row of the down input planes

// ------------------------------------------------------------- kernels ----

// Token rows: the Q8 embedding row, RMSNorm, then embd_norm. One block per row.
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

// Image rows: chunk row r copies image row irow[r], no norm; irow < 0 keeps the token.
static __global__ __launch_bounds__(256) void k_pf_img(const int * irow, const float * img, float * x) {
    const int r = irow[blockIdx.x];
    if (r < 0) return;
    for (int i = threadIdx.x; i < D; i += 256) x[(size_t) blockIdx.x * D + i] = img[(size_t) r * D + i];
}

// mix(blocks, cur; query) for a whole chunk, one block per row. Each state is a
// [PF_M][D] buffer read at the same row; the MQ query columns land in out[j*ostr + row].
// Templated on the active state count NS so only live states occupy registers and reductions.
struct pfmix { const float * p[6]; int n; const float * qy; int m; float * out; size_t ostr; };
template <int MQ, int NS>
static __global__ __launch_bounds__(256) void k_pf_mix2(pfmix A) {
    constexpr int NV = D / 256;
    __shared__ float red[4 * NS * (MQ + 1)];
    __shared__ float wsh[NS][MQ];
    const int row = blockIdx.x, tid = threadIdx.x;
    float v[NS][NV];
    float ss[NS];
    #pragma unroll
    for (int s = 0; s < NS; s++) {
        ss[s] = 0.f;
        const float * ps = A.p[s] + (size_t) row * D;
        #pragma unroll
        for (int k = 0; k < NV; k++) {
            v[s][k] = ps[tid + 256 * k];
            ss[s] += v[s][k] * v[s][k];
        }
    }
    bsum<NS>(ss, red);
    float sc[NS * MQ];
    #pragma unroll
    for (int s = 0; s < NS; s++) {
        const float rs = rsqrtf(ss[s] / D + 1e-6f);
        #pragma unroll
        for (int k = 0; k < NV; k++) v[s][k] *= rs;
    }
    #pragma unroll
    for (int j = 0; j < MQ; j++) {
        const float * qj = A.qy + (size_t) j * D;
        float qv[NV];
        #pragma unroll
        for (int k = 0; k < NV; k++) qv[k] = qj[tid + 256 * k];
        #pragma unroll
        for (int s = 0; s < NS; s++) {
            float a = 0.f;
            #pragma unroll
            for (int k = 0; k < NV; k++) a += v[s][k] * qv[k];
            sc[s * MQ + j] = a;
        }
    }
    bsum<NS * MQ>(sc, red);
    if (tid < MQ) {
        const int j = tid;
        const float scale = 1.f / sqrtf((float) D);
        float mx = -INFINITY;
        #pragma unroll
        for (int s = 0; s < NS; s++) mx = fmaxf(mx, sc[s * MQ + j] * scale);
        float sum = 0.f;
        #pragma unroll
        for (int s = 0; s < NS; s++) { const float e = expf(sc[s * MQ + j] * scale - mx); wsh[s][j] = e; sum += e; }
        #pragma unroll
        for (int s = 0; s < NS; s++) wsh[s][j] /= sum;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < MQ; j++) {
        float * dst = A.out + (size_t) j * A.ostr + (size_t) row * D;
        #pragma unroll
        for (int k = 0; k < NV; k++) {
            float o = 0.f;
            #pragma unroll
            for (int s = 0; s < NS; s++) o += wsh[s][j] * v[s][k];
            dst[tid + 256 * k] = o;
        }
    }
}
template <int MQ>
static inline void launch_pf_mix(int m, hipStream_t st, const pfmix & A) {
    switch (A.n) {
        case 2: k_pf_mix2<MQ, 2><<<m, 256, 0, st>>>(A); break;
        case 3: k_pf_mix2<MQ, 3><<<m, 256, 0, st>>>(A); break;
        case 4: k_pf_mix2<MQ, 4><<<m, 256, 0, st>>>(A); break;
        case 5: k_pf_mix2<MQ, 5><<<m, 256, 0, st>>>(A); break;
        default: k_pf_mix2<MQ, 6><<<m, 256, 0, st>>>(A); break;
    }
}

// Per-row inverse RMS scales {rc, rm} for [rmsnorm(cur_j); rmsnorm(msg_j)], applied on the
// fly inside k_pf_ctrl_down2 so the 2*D concatenated buffer is never materialized.
struct pfrms { const float * cur[4]; const float * msg; float * rinv; int j0; };
static __global__ __launch_bounds__(256) void k_pf_rms_inv(pfrms A) {
    __shared__ float red[4 * 2];
    const int row = blockIdx.x, j = A.j0 + blockIdx.y, tid = threadIdx.x;
    const float * __restrict__ c = A.cur[j] + (size_t) row * D;
    const float * __restrict__ g = A.msg + ((size_t) j * PF_M + row) * D;
    float ss[2] = {0.f, 0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const float cv = c[tid + 256 * k], gv = g[tid + 256 * k];
        ss[0] = fmaf(cv, cv, ss[0]); ss[1] = fmaf(gv, gv, ss[1]);
    }
    bsum<2>(ss, red);
    if (tid == 0) {
        float * dst = A.rinv + ((size_t) j * PF_M + row) * 2;
        dst[0] = rsqrtf(ss[0] / D + 1e-6f);
        dst[1] = rsqrtf(ss[1] / D + 1e-6f);
    }
}

// The controller down GEMM: g[j][row][64] = [cur*rc; msg*rm] . wd[64][2D], in f32.
struct pfcdn { const float * cur[4]; const float * msg, * rinv; const float * wd[4]; float * g; int j0, m; };

// The state update: out = (1 - tanh(g.forget)) * cur + tanh(g.gain) * msg per row, in f32.
// wgt / wft are the transposed [GRANK][D] gain / forget weights.
struct pfcup { const float * g, * msg; const float * cur[4]; const float * wgt[4], * wft[4]; float * out; int j0, m; };

// The loop controllers as 64x64-tiled f32 GEMMs, 4 x 4 per thread with float4 loads and
// register prefetching. down splits K = 2D CT_S ways into gp[ks][j][row][GRANK] (splits
// 0..3 read cur, 4..7 read msg); up sums the parts in order (deterministic), then tiles
// GRANK in KU = 16 steps for high occupancy and applies the update.
constexpr int CT_S = 8, CT_KS = 2 * D / CT_S, CT_KC = 32;
static_assert(CT_S % 2 == 0 && CT_KS * (CT_S / 2) == D, "controller K splits align to D");
typedef float pf_f32x4 __attribute__((ext_vector_type(4)));

static __global__ __launch_bounds__(256, 3) void k_pf_ctrl_down2(pfcdn a, float * __restrict__ gp) {
    __shared__ float sX[64][CT_KC + 4], sW[GRANK][CT_KC + 4], sR[64];
    const int t = threadIdx.x, tr = t >> 4, tc = t & 15, j = a.j0 + blockIdx.y, ks = blockIdx.z;
    const int r0 = blockIdx.x * 64;
    const bool is_msg = ks >= CT_S / 2;
    const float * __restrict__ wd = a.wd[j];
    const float * __restrict__ xsrc = is_msg ? (a.msg + (size_t) j * PF_M * D) : a.cur[j];
    const float * __restrict__ rsrc = a.rinv + (size_t) j * PF_M * 2 + (is_msg ? 1 : 0);
    if (t < 64) sR[t] = r0 + t < a.m ? rsrc[(size_t)(r0 + t) * 2] : 0.f;
    __syncthreads();
    const int xk0 = (is_msg ? (ks - CT_S / 2) : ks) * CT_KS;
    const int wk0 = ks * CT_KS;
    const int lr0 = t >> 3, lr1 = (t + 256) >> 3, lc = (t & 7) << 2;
    const float rr0 = sR[lr0], rr1 = sR[lr1];
    const pf_f32x4 z4 = {0.f, 0.f, 0.f, 0.f};
    pf_f32x4 px0 = r0 + lr0 < a.m ? *(const pf_f32x4 *) (xsrc + (size_t)(r0 + lr0) * D + xk0 + lc) : z4;
    pf_f32x4 px1 = r0 + lr1 < a.m ? *(const pf_f32x4 *) (xsrc + (size_t)(r0 + lr1) * D + xk0 + lc) : z4;
    pf_f32x4 pw0 = *(const pf_f32x4 *) (wd + (size_t) lr0 * 2 * D + wk0 + lc);
    pf_f32x4 pw1 = *(const pf_f32x4 *) (wd + (size_t) lr1 * 2 * D + wk0 + lc);
    float acc[4][4] = {};
    for (int dk = 0; dk < CT_KS; dk += CT_KC) {
        *(pf_f32x4 *) &sX[lr0][lc] = px0 * rr0;
        *(pf_f32x4 *) &sX[lr1][lc] = px1 * rr1;
        *(pf_f32x4 *) &sW[lr0][lc] = pw0;
        *(pf_f32x4 *) &sW[lr1][lc] = pw1;
        __syncthreads();
        const int nk = dk + CT_KC;
        if (nk < CT_KS) {
            px0 = r0 + lr0 < a.m ? *(const pf_f32x4 *) (xsrc + (size_t)(r0 + lr0) * D + xk0 + nk + lc) : z4;
            px1 = r0 + lr1 < a.m ? *(const pf_f32x4 *) (xsrc + (size_t)(r0 + lr1) * D + xk0 + nk + lc) : z4;
            pw0 = *(const pf_f32x4 *) (wd + (size_t) lr0 * 2 * D + wk0 + nk + lc);
            pw1 = *(const pf_f32x4 *) (wd + (size_t) lr1 * 2 * D + wk0 + nk + lc);
        }
#pragma unroll 2
        for (int k = 0; k < CT_KC; k++) {
            float xv[4], wv[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { xv[i] = sX[tr * 4 + i][k]; wv[i] = sW[tc + 16 * i][k]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[i][c] = fmaf(xv[i], wv[c], acc[i][c]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = r0 + tr * 4 + i;
        if (row >= a.m) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) gp[(((size_t) ks * 4 + j) * PF_M + row) * GRANK + tc + 16 * c] = acc[i][c];
    }
}
static __global__ __launch_bounds__(256) void k_pf_gsum(const float * gp, float * g, int j0, int m) {
    const int i = blockIdx.x * 256 + threadIdx.x, j = j0 + blockIdx.y;
    if (i >= m * GRANK) return;
    float s = 0.f;
#pragma unroll
    for (int ks = 0; ks < CT_S; ks++) s += gp[((size_t) ks * 4 + j) * PF_M * GRANK + i];
    g[(size_t) j * PF_M * GRANK + i] = s;
}
constexpr int CT_KU = 16;
static __global__ __launch_bounds__(256, 3) void k_pf_ctrl_up2(pfcup a) {
    __shared__ float sG[64][CT_KU + 4], sWg[CT_KU][64 + 4], sWf[CT_KU][64 + 4];
    const int t = threadIdx.x, tr = t >> 4, tc = t & 15, j = a.j0 + blockIdx.z;
    const int r0 = blockIdx.x * 64, c0 = blockIdx.y * 64;
    const float * __restrict__ gb = a.g + (size_t) j * PF_M * GRANK;
    const float * __restrict__ wgt = a.wgt[j];
    const float * __restrict__ wft = a.wft[j];
    const int gr = t >> 2, gk = (t & 3) << 2;
    const int wk = t >> 4, wc = (t & 15) << 2;
    const pf_f32x4 z4 = {0.f, 0.f, 0.f, 0.f};
    pf_f32x4 pg = r0 + gr < a.m ? *(const pf_f32x4 *) (gb + (size_t)(r0 + gr) * GRANK + gk) : z4;
    pf_f32x4 pwg = *(const pf_f32x4 *) (wgt + (size_t) wk * D + c0 + wc);
    pf_f32x4 pwf = *(const pf_f32x4 *) (wft + (size_t) wk * D + c0 + wc);
    float ga[4][4] = {}, fo[4][4] = {};
    for (int k0 = 0; k0 < GRANK; k0 += CT_KU) {
        *(pf_f32x4 *) &sG[gr][gk] = pg;
        *(pf_f32x4 *) &sWg[wk][wc] = pwg;
        *(pf_f32x4 *) &sWf[wk][wc] = pwf;
        __syncthreads();
        const int nk = k0 + CT_KU;
        if (nk < GRANK) {
            pg = r0 + gr < a.m ? *(const pf_f32x4 *) (gb + (size_t)(r0 + gr) * GRANK + nk + gk) : z4;
            pwg = *(const pf_f32x4 *) (wgt + (size_t)(nk + wk) * D + c0 + wc);
            pwf = *(const pf_f32x4 *) (wft + (size_t)(nk + wk) * D + c0 + wc);
        }
#pragma unroll 4
        for (int k = 0; k < CT_KU; k++) {
            float gv[4];
#pragma unroll
            for (int i = 0; i < 4; i++) gv[i] = sG[tr * 4 + i][k];
            const pf_f32x4 wg = *(const pf_f32x4 *) &sWg[k][tc * 4];
            const pf_f32x4 wf = *(const pf_f32x4 *) &sWf[k][tc * 4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                ga[i][0] = fmaf(gv[i], wg.x, ga[i][0]); fo[i][0] = fmaf(gv[i], wf.x, fo[i][0]);
                ga[i][1] = fmaf(gv[i], wg.y, ga[i][1]); fo[i][1] = fmaf(gv[i], wf.y, fo[i][1]);
                ga[i][2] = fmaf(gv[i], wg.z, ga[i][2]); fo[i][2] = fmaf(gv[i], wf.z, fo[i][2]);
                ga[i][3] = fmaf(gv[i], wg.w, ga[i][3]); fo[i][3] = fmaf(gv[i], wf.w, fo[i][3]);
            }
        }
        __syncthreads();
    }
    const float * __restrict__ cur = a.cur[j], * __restrict__ msg = a.msg + (size_t) j * PF_M * D;
    float * __restrict__ out = a.out + (size_t) j * PF_M * D;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = r0 + tr * 4 + i;
        if (row >= a.m) continue;
        const size_t o = (size_t) row * D + c0 + tc * 4;
        const pf_f32x4 x = *(const pf_f32x4 *) (cur + o);
        const pf_f32x4 mg = *(const pf_f32x4 *) (msg + o);
        pf_f32x4 y;
        y.x = fmaf(tanhf(ga[i][0]), mg.x, fmaf(-tanhf(fo[i][0]), x.x, x.x));
        y.y = fmaf(tanhf(ga[i][1]), mg.y, fmaf(-tanhf(fo[i][1]), x.y, x.y));
        y.z = fmaf(tanhf(ga[i][2]), mg.z, fmaf(-tanhf(fo[i][2]), x.z, x.z));
        y.w = fmaf(tanhf(ga[i][3]), mg.w, fmaf(-tanhf(fo[i][3]), x.w, x.w));
        *(pf_f32x4 *) (out + o) = y;
    }
}

// Batched RMSNorm + int8 plane quantization (1 or 3 streams per launch). Keeps the row
// in registers so the input is read once from HBM.
struct pfnq { const float * x[3]; const float * w[3]; int8_t * q[3]; float * s[3]; float * outf; };
static __global__ __launch_bounds__(256) void k_pf_normq(pfnq A, float eps) {
    constexpr int NV = D / 256;
    __shared__ float red[4];
    const int m = blockIdx.x, j = blockIdx.y, t = threadIdx.x;
    const float * __restrict__ xr = A.x[j] + (size_t) m * D;
    const float * __restrict__ w = A.w[j];
    float v[NV], ss = 0.f;
#pragma unroll
    for (int k = 0; k < NV; k++) { v[k] = xr[t + 256 * k]; ss = fmaf(v[k], v[k], ss); }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off, 64);
    if ((t & 63) == 0) red[t >> 6] = ss;
    __syncthreads();
    const float r = 1.f / sqrtf((red[0] + red[1] + red[2] + red[3]) / D + eps);
    int8_t * __restrict__ Ag = A.q[j];
    float * __restrict__ sag = A.s[j];
    float * __restrict__ outf = j == 0 ? A.outf : nullptr;
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const int idx = t + 256 * k;
        const float y = v[k] * r * w[idx];
        if (outf) outf[(size_t) m * D + idx] = y;
        pf_quant_unit(y, idx, m, PF_M, Ag, sag, t & 31);
    }
}

// Head post, HP_R chunk rows per blockIdx.y. Local heads: the Q and R clips, tau and the
// clipped R for the attention bias. Kv heads: the K or V conv walked along rows so each
// input is read once, then clip or norm.
constexpr int HP_R = 16;
struct pfhp {
    const float * qr, * rr, * kr, * vr, * dk, * dv;          // [M][QL/HL../KVL/16]
    const float * qclip, * rclip, * kconv, * vconv;
    float * hist, * q;                                        // slot hist, [M][QL]
    float * rvo;                                              // [M][HQL][DREL] clipped R
    __half * kc, * vc;
    int pos0, m, full;
    float tau_a; int tau_f;
    float qk_clip, rel_clip, eps;
};
static __global__ __launch_bounds__(128) void k_pf_head_post(pfhp a) {
    __shared__ float red[2];
    const int b = blockIdx.x, r0 = blockIdx.y * HP_R, r1 = min(r0 + HP_R, a.m), t = threadIdx.x;
    if (b < HQL) {
        const int h = b, i = h * HD + t;
        const float qc = a.qclip[i];
        const float rc = t < DREL ? a.rclip[h * DREL + t] : 0.f;
        for (int row = r0; row < r1; row++) {
            const int pos = a.pos0 + row;
            const float tau = a.full ? 1.f + a.tau_a * logf(fmaxf(1.f, (float) (pos + 1) / (float) a.tau_f)) : 1.f;
            float x = a.qr[(size_t) row * QL + i] * qc;
            float n2[1] = {x * x};
            bsum<1>(n2, red);
            const float nr = sqrtf(n2[0]) + 1e-6f;
            x *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
            a.q[(size_t) row * QL + i] = x * tau;
            float r = t < DREL ? a.rr[(size_t) row * (HQL * DREL) + h * DREL + t] * rc : 0.f;
            float r2[1] = {r * r};
            bsum<1>(r2, red);
            const float rn = sqrtf(r2[0]) + 1e-6f;
            if (t < DREL) a.rvo[((size_t) row * HQL + h) * DREL + t] = r * fminf(fmaxf(a.rel_clip / rn, 0.f), 1.f) * tau;
        }
        return;
    }
    const bool isk = b < HQL + HKVL;
    const int hk = isk ? b - HQL : b - HQL - HKVL, ch = hk * HD + t;
    const float * __restrict__ rb = isk ? a.kr : a.vr;        // [M][KVL]
    const float * __restrict__ hb = a.hist + (isk ? H_K : H_V) + ch;
    const float4 w = *(const float4 *) ((isk ? a.kconv : a.vconv) + (size_t) ch * NCONV);
    const float sc = a.dk ? 1.f / sqrtf(2.f * NCONV + 1.f) : 1.f / sqrtf((float) NCONV + 1.f);
    auto in = [&](int r) {
        return r >= 0 ? rb[(size_t) r * KVL + ch] : hb[(size_t)(r + 3) * CHL];
    };
    float v3 = in(r0 - 3), v2 = in(r0 - 2), v1 = in(r0 - 1);
    for (int row = r0; row < r1; row++) {
        const float u = rb[(size_t) row * KVL + ch];
        float o = w.x * v3 + w.y * v2 + w.z * v1 + w.w * u;
        if (a.dk) {
            const float * tp = (isk ? a.dk : a.dv) + (size_t) row * (HKVL * NCONV) + hk * NCONV;
            o += u * tp[0] + v1 * tp[1] + v2 * tp[2] + v3 * tp[3];
        }
        float y = (u + o) * sc;
        float n2[1] = {y * y};
        bsum<1>(n2, red);
        const int pos = a.pos0 + row;
        const int rrow = a.full ? pos : pos % PF_RING;
        if (isk) {
            const float nr = sqrtf(n2[0]) + 1e-6f;
            y *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
            a.kc[(size_t) rrow * KVL + ch] = __float2half(y);
        } else {
            y *= rsqrtf(n2[0] / HD + a.eps);
            a.vc[(size_t) rrow * KVL + ch] = __float2half(y);
        }
        v3 = v2; v2 = v1; v1 = u;
    }
}

// Attention inputs. The keys and values come from the sliding window (rows pos%PF_RING
// of the ring) or from the whole cache when full is set. The output is gated and quantized.
struct pfat {
    const float * q, * gate;                  // [M][QL], [M][QL]
    const float * rv, * relb, * sk;           // [M][HQL][DREL], [REL_EXT][DREL], [M][HQL] (bias in the kernel)
    const __half * kc, * vc;
    int pos0, full;
    unsigned char * oq; float * os;
};

// Tiled flash attention, one block per (row tile, q head) and 64-wide key tiles.
// S = Q.K^T uses packed f16 dots with hi+lo Q, RPT rows x 4 keys per thread. Online
// softmax uses 16-lane DPP reductions; O += P.V in f32. Reverse blockIdx.x schedules
// long causal tiles first.
constexpr int FA_K = 64, FA_QS = HD + 8;                // key tile, Q / K / V LDS row stride in halves
typedef __fp16 fa_h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float fa_sum16(float x) {
    x += qz_dpp<0x128>(x);
    x += qz_dpp<0x124>(x);
    x += qz_dpp<0x4E>(x);
    x += qz_dpp<0xB1>(x);
    return x;
}
__device__ __forceinline__ float fa_max16(float x) {
    x = qz_dpp_fmax<0x128>(x);
    x = qz_dpp_fmax<0x124>(x);
    x = qz_dpp_fmax<0x4E>(x);
    x = qz_dpp_fmax<0xB1>(x);
    return x;
}
// RPT rows per thread, so a tile holds 16 * RPT rows. QLO splits Q into an f16 hi and lo.
template <int RPT, bool QLO, int FA_R = 32>
static __global__ __launch_bounds__(16 * FA_R / RPT) void k_pf_fattn(pfat a, int m) {
    constexpr int NQ = QLO ? 2 : 1, NTH = 16 * FA_R / RPT;
    __shared__ __half sQ[NQ][FA_R * FA_QS];
    __shared__ __half sKV[FA_K * FA_QS];
    __shared__ float sP[FA_R][FA_K + 4];
    __shared__ float sB[FA_R + FA_K - 1][DREL + 1];          // bias basis rows of this tile's distances
    const int t = threadIdx.x, rg = t >> 4, cg = t & 15;
    const int h = blockIdx.y, kvh = h / (HQ / HKV), r0 = (gridDim.x - 1 - blockIdx.x) * FA_R;
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
    // K and V rows go through registers: this tile's V and the next tile's K load while
    // the Q.K runs.
    constexpr int NLD = FA_K * HD / 8 / NTH;
    static_assert(NLD * NTH == FA_K * HD / 8, "the K / V tile splits evenly over the block");
    uint4 kr[NLD], vr[NLD];
    auto ld = [&](const __half * c, int kb, uint4 * r) {
        #pragma unroll
        for (int u = 0; u < NLD; u++) {
            const int e = t + u * NTH, j = kb + e / (HD / 8), c8 = e % (HD / 8);
            r[u] = make_uint4(0, 0, 0, 0);
            if (j <= plast) { const int rr = a.full ? j : j % PF_RING; r[u] = *(const uint4 *) (c + (size_t) rr * KVL + kvh * HD + c8 * 8); }
        }
    };
    auto st = [&](const uint4 * r) {
        #pragma unroll
        for (int u = 0; u < NLD; u++) { const int e = t + u * NTH; *(uint4 *) (sKV + (e / (HD / 8)) * FA_QS + (e % (HD / 8)) * 8) = r[u]; }
    };
    if (k0 <= plast) ld(a.kc, k0, kr);
    for (int kb = k0; kb <= plast; kb += FA_K) {
        const int dlo = pfirst - (kb + FA_K - 1);              // distance of window row 0
        // near: some distance in the tile is below REL_EXT. Otherwise every basis row is
        // zero, the bias sum is +0 and the far path adds it as 0.f - sk (same bits).
        const bool near = dlo < REL_EXT;
        // inner: every key of the tile is inside every row's causal window
        const bool inner = kb + FA_K - 1 <= pfirst && (a.full || kb >= plast - (WIN - 1));
        __syncthreads();
        st(kr);
        if (near)
            for (int e = t; e < (FA_R + FA_K - 1) * DREL; e += NTH) {
                const int ii = e / DREL, k = e % DREL, d = dlo + ii;
                sB[ii][k] = d >= 0 && d < REL_EXT ? a.relb[(size_t) d * DREL + k] : 0.f;
            }
        __syncthreads();
        // a barrier waits for every outstanding load, so the loads start right after one
        ld(a.vc, kb, vr);
        if (kb + FA_K <= plast) ld(a.kc, kb + FA_K, kr);
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
            if (near) {
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
            } else {
                const float bz = 0.f - skr[i];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int j = kb + cg + 16 * c;
                    const float s = inner || (j <= pos && j >= lo) ? sc[i][c] * scale + bz : -INFINITY;
                    sc[i][c] = s;
                    tm = fmaxf(tm, s);
                }
            }
            tm = fa_max16(tm);
            const float mn = fmaxf(mx[i], tm);
            const float cr = mn == -INFINITY ? 1.f : expf(mx[i] - mn);
            float ps = 0.f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const float p = sc[i][c] == -INFINITY ? 0.f : expf(sc[i][c] - mn);
                sP[rg * RPT + i][cg + 16 * c] = p;
                ps += p;
            }
            ps = fa_sum16(ps);
            l[i] = l[i] * cr + ps;
            mx[i] = mn;
            if (cr < 1.f) {
                #pragma unroll
                for (int d = 0; d < 8; d++) o[i][d] *= cr;
            }
        }
        __syncthreads();
        st(vr);
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
    // Add the zero sink logit, normalize, gate, then quantize through LDS (32 dims a warp).
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

// The short conv along the chunk rows, folded into the x update. push moves x into a
// fresh residual block, otherwise the conv is added to x. Row t reads rows t-1..t-3 or
// the slot's conv history, and u = own + peer is the TP2 sum, the peer's partial read from
// this card's inbox (PARTP). One thread per 4 columns (float4) walks CV_R rows.
struct pfcv { const float * own, * peer; const float * w; float * hrg; float * x, * blk; int m, push; };
constexpr int CV_R = 32;
static __global__ __launch_bounds__(256) void k_pf_conv2(pfcv a) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x, r0 = blockIdx.y * CV_R;
    if (i4 >= D / 4) return;
    const int i = i4 * 4;
    const float4 * __restrict__ p0 = (const float4 *) a.own;
    const float4 * __restrict__ p1 = (const float4 *) a.peer;
    const float4 * __restrict__ hr = (const float4 *) a.hrg;
    auto in = [&](int r) -> float4 {   // input at chunk row r (r < 0: history row 3 + r)
        if (r < 0) return hr[(size_t)(r + 3) * (CHL / 4) + i4];
        const float4 a0 = p0[(size_t) r * (D / 4) + i4], a1 = p1[(size_t) r * (D / 4) + i4];
        return make_float4(a0.x + a1.x, a0.y + a1.y, a0.z + a1.z, a0.w + a1.w);
    };
    float4 v3 = in(r0 - 3), v2 = in(r0 - 2), v1 = in(r0 - 1);
    const float4 w0 = *(const float4 *) (a.w + (size_t) (i + 0) * NCONV);
    const float4 w1 = *(const float4 *) (a.w + (size_t) (i + 1) * NCONV);
    const float4 w2 = *(const float4 *) (a.w + (size_t) (i + 2) * NCONV);
    const float4 w3 = *(const float4 *) (a.w + (size_t) (i + 3) * NCONV);
    const float sc = 1.f / sqrtf((float) NCONV + 1.f);
    float4 * __restrict__ x4 = (float4 *) a.x;
    float4 * __restrict__ b4 = (float4 *) a.blk;
    const int r1 = min(r0 + CV_R, a.m);
    for (int row = r0; row < r1; row++) {
        const float4 u = in(row);
        float4 y;
        y.x = (u.x + (w0.x * v3.x + w0.y * v2.x + w0.z * v1.x + w0.w * u.x)) * sc;
        y.y = (u.y + (w1.x * v3.y + w1.y * v2.y + w1.z * v1.y + w1.w * u.y)) * sc;
        y.z = (u.z + (w2.x * v3.z + w2.y * v2.z + w2.z * v1.z + w2.w * u.z)) * sc;
        y.w = (u.w + (w3.x * v3.w + w3.y * v2.w + w3.z * v1.w + w3.w * u.w)) * sc;
        const size_t o = (size_t) row * (D / 4) + i4;
        if (a.push) { b4[o] = x4[o]; x4[o] = y; }
        else { const float4 xv = x4[o]; x4[o] = make_float4(xv.x + y.x, xv.y + y.y, xv.z + y.z, xv.w + y.w); }
        v3 = v2; v2 = v1; v1 = u;
    }
}

// Roll the conv history forward: its 3 rows become the chunk's last inputs, and rows
// below the chunk start come from the old history. peer is null for K and V; dst may alias hrg.
static __global__ __launch_bounds__(256) void k_pf_commit(const float * src, const float * peer, const float * hrg,
                                                          float * dst, int w, int m) {
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
    for (int r = 0; r < 3; r++) dst[(size_t) r * CHL + c] = v[r];
}

// The router as a tiled f32 GEMM, with K split RT_S ways for occupancy: one block per
// (64 rows, one K split) over 128 experts, 4 rows x 8 experts per thread, K chunks of
// 16 in LDS with float4 loads and register prefetching. k_pf_topk sums the partials in
// order, which keeps the result deterministic, applies sigmoid, then picks the top NE_USED.
constexpr int RT_R = 64, RT_K = 16, RT_S = 4;
static __global__ __launch_bounds__(256, 3) void k_pf_router2(const float * __restrict__ f, const float * __restrict__ wr,
                                                              float * __restrict__ part, int m) {
    __shared__ float sF[RT_R][RT_K + 4], sW[NE][RT_K + 4];
    const int t = threadIdx.x, tr = t >> 4, te = t & 15, r0 = blockIdx.x * RT_R;
    const int kl = blockIdx.y * (D / RT_S), kh = kl + D / RT_S;
    const int fr = t >> 2, fk = (t & 3) << 2;
    const int wx0 = t >> 2, wx1 = (t + 256) >> 2;
    const pf_f32x4 z4 = {0.f, 0.f, 0.f, 0.f};
    pf_f32x4 pf = r0 + fr < m ? *(const pf_f32x4 *) (f + (size_t)(r0 + fr) * D + kl + fk) : z4;
    pf_f32x4 pw0 = *(const pf_f32x4 *) (wr + (size_t) wx0 * D + kl + fk);
    pf_f32x4 pw1 = *(const pf_f32x4 *) (wr + (size_t) wx1 * D + kl + fk);
    float acc[4][8] = {};
    for (int k0 = kl; k0 < kh; k0 += RT_K) {
        *(pf_f32x4 *) &sF[fr][fk] = pf;
        *(pf_f32x4 *) &sW[wx0][fk] = pw0;
        *(pf_f32x4 *) &sW[wx1][fk] = pw1;
        __syncthreads();
        const int nk = k0 + RT_K;
        if (nk < kh) {
            pf = r0 + fr < m ? *(const pf_f32x4 *) (f + (size_t)(r0 + fr) * D + nk + fk) : z4;
            pw0 = *(const pf_f32x4 *) (wr + (size_t) wx0 * D + nk + fk);
            pw1 = *(const pf_f32x4 *) (wr + (size_t) wx1 * D + nk + fk);
        }
        #pragma unroll 4
        for (int k = 0; k < RT_K; k++) {
            float fv[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) fv[i] = sF[tr * 4 + i][k];
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                const float w = sW[te + 16 * c][k];
                #pragma unroll
                for (int i = 0; i < 4; i++) acc[i][c] = fmaf(fv[i], w, acc[i][c]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r < m)
            #pragma unroll
            for (int c = 0; c < 8; c++) part[((size_t) blockIdx.y * m + r) * NE + te + 16 * c] = acc[i][c];
    }
}
// One 64-lane wave per row.
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

// Counting sort of the chunk's (row, expert) pairs by expert: start offsets, perm and
// inv (both pair to slot). The order inside an expert is arbitrary because each pair's
// result stands alone; only the row gather has to stay deterministic.
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
// One tile per (expert, BN pairs), with the tile table built on device from the counting
// sort. A GEMM block covers 64 weight rows of that expert and the tile's pairs, staging
// weights and gathered rows in LDS four sub-blocks (128 K values) at a time, as pf_gemm does.

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

// Three tile lists for BN = 64: A holds BN-wide tiles (full tiles + remainders > 32),
// B holds BN/2-wide tiles (17..32), and C holds BN/4-wide tiles (1..16). ntl[0..2] counts them.
constexpr int TM_B = PF_NP / 16 + NE;          // list B offset in tm
constexpr int TM_C = PF_NP / 16 + 2 * NE;      // list C offset in tm
template <int BN>
static __global__ __launch_bounds__(256) void k_pf_tiles2(const int * start, int4 * tm, int * ntl) {
    __shared__ int offa[NE], offb[NE], offc[NE];
    if (threadIdx.x == 0) {
        int sa = 0, sb = 0, sc = 0;
        for (int e = 0; e < NE; e++) {
            const int cnt = start[e + 1] - start[e], r = cnt % BN;
            offa[e] = sa; offb[e] = sb; offc[e] = sc;
            sa += cnt / BN + (r > BN / 2);
            if constexpr (BN == 64) {
                sb += (r > 16 && r <= 32);
                sc += (r > 0 && r <= 16);
            } else {
                sb += (r > 0 && r <= BN / 2);
            }
        }
        ntl[0] = sa; ntl[1] = sb; ntl[2] = sc;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NE; e += 256) {
        const int a0 = start[e], cnt = start[e + 1] - a0, full = cnt / BN, r = cnt % BN;
        for (int j = 0; j < full; j++) tm[offa[e] + j] = make_int4(e, a0 + j * BN, BN, 0);
        if (r > BN / 2) tm[offa[e] + full] = make_int4(e, a0 + full * BN, r, 0);
        else if constexpr (BN == 64) {
            if (r > 16) tm[TM_B + offb[e]] = make_int4(e, a0 + full * BN, r, 0);
            else if (r > 0) tm[TM_C + offc[e]] = make_int4(e, a0 + full * BN, r, 0);
        } else if (r > 0) {
            tm[TM_B + offb[e]] = make_int4(e, a0 + full * BN, r, 0);
        }
    }
}

// tile_meta is {expert, first pair slot, count}. With GATHER the activation rows come
// from the chunk's row planes through the sorted pair list, as gate_up needs them;
// otherwise they are already slot-major, as down has them. Blocks past the count exit.
// Fused SITU output of gate_up: int8 planes [NGRP_FF][np][128] plus scales.
struct pf_situo { unsigned char * q = nullptr; float * qs = nullptr; int np = 0; float beta = 0.f, lb = 0.f; };

template <int BN, int GATHER, int NTT, int FS = 0, bool K_DIV4 = false>
static __device__ __forceinline__ void moe_gemm_body(
        const int4 tl, int bx,
        const char * __restrict__ W, const __half * __restrict__ sw, const int * __restrict__ perm,
        const int8_t * __restrict__ Ag, const float * __restrict__ sag, int Mp, int K,
        float * __restrict__ C, int ldc, int N, const pf_situo so = pf_situo{}, float * __restrict__ C2 = nullptr) {
    constexpr int NT = NTT;                    // pair lanes
    constexpr int RG = 256 / NT;               // row groups
    constexpr int RR = 64 / RG;                // rows per thread
    constexpr int NH = BN / NT;                // pair halves
    constexpr int NXF = BN * 4 >= 256 ? BN * 4 / 256 : 1;   // activation slots per thread
    __shared__ union {
        struct {
            pf_u32x4 wlo[64][4], whi[64][4];
            unsigned short wd[4][64];
            pf_xrow x[BN];
            float xd[BN][4];
        } g;
        struct {
            float a[32][BN + 1];               // gate [j][slot]
            float u[BN * 32];                  // up [slot][j ^ slot]
        } f;
    } sm;
    const int t = threadIdx.x, tx = t % NT, ty = t / NT;
    const int nkb = K / 32, row0 = bx * 64, cnt = tl.z;
    W += (size_t) tl.x * N * K;
    sw += (size_t) tl.x * N * (K / 32);
    const unsigned wrow = (unsigned) min(row0 + t / 4, N - 1);
    const unsigned w_off = wrow * (unsigned) K + (t % 4) * 32u;
    const unsigned wd_off = wrow * (unsigned) nkb + t % 4;
    // this thread's staging slots. Lanes past the tile's count read the last pair and
    // are dropped in the epilogue, as pf_gemm clamps its rows
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
        if (K_DIV4 || sb0 + (t % 4) < nkb) {
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
        // K slot XOR the row group: row groups in one wave hit distinct banks
        const int ks = (t % 4) ^ ((t / 4 / RR) & 3);
        sm.g.wlo[t / 4][ks] = pwlo; sm.g.whi[t / 4][ks] = pwhi; sm.g.wd[t % 4][t / 4] = pwd;
#pragma unroll
        for (int i = 0; i < NXF; i++) {
            if (!xok[i]) continue;
            const int e = t + 256 * i;
            sm.g.x[e / 4].q[e % 4][0] = pxlo[i]; sm.g.x[e / 4].q[e % 4][1] = pxhi[i];
            sm.g.xd[e / 4][e % 4] = pxd[i];
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
            if (!K_DIV4 && sb0 + kk >= nkb) break;
            int xq[NH][8];
            float dx[NH];
#pragma unroll
            for (int n = 0; n < NH; n++) {
                const pf_u32x4 q0 = sm.g.x[tx + NT * n].q[kk][0], q1 = sm.g.x[tx + NT * n].q[kk][1];
                xq[n][0] = q0.x; xq[n][1] = q0.y; xq[n][2] = q0.z; xq[n][3] = q0.w;
                xq[n][4] = q1.x; xq[n][5] = q1.y; xq[n][6] = q1.z; xq[n][7] = q1.w;
                dx[n] = sm.g.xd[tx + NT * n][kk];
            }
            const int kw = kk ^ (ty & 3);
            pf_u32x4 wlo = sm.g.wlo[rb][kw], whi = sm.g.whi[rb][kw];
            const uint2 dw = *(const uint2 *) &sm.g.wd[kk][rb];
#pragma unroll
            for (int r = 0; r < RR; r++) {
                const pf_u32x4 lo = wlo, hi = whi;
                if (r + 1 < RR) { wlo = sm.g.wlo[rb + r + 1][kw]; whi = sm.g.whi[rb + r + 1][kw]; }
                const unsigned dw32 = r < 2 ? dw.x : dw.y;
                const float d = __half2float(__ushort_as_half((unsigned short) ((r & 1) ? dw32 >> 16 : dw32 & 0xffffu)));
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
        // tile rows 0..31 hold gate rows and rows 32..63 their up rows (gu_row). Stage
        // both halves in the aliased LDS union; every thread applies SITU and per-32 quant.
#pragma unroll
        for (int n = 0; n < NH; n++) {
            const int slot = tx + NT * n;
#pragma unroll
            for (int r = 0; r < RR; r++) {
                if (rb < 32) sm.f.a[rb + r][slot] = acc[r][n];
                else sm.f.u[slot * 32 + ((rb - 32 + r) ^ (slot & 31))] = acc[r][n];
            }
        }
        __syncthreads();
        const int j0 = bx * 32;
        for (int i = t; i < BN * 32; i += 256) {
            const int slot = i / 32, jj = i % 32;
            const float g = sm.f.a[jj][slot], u = sm.f.u[slot * 32 + (jj ^ (slot & 31))];
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
        if (row0 + 64 > N) {                       // partial weight tile (N % 64 != 0): guard each row
#pragma unroll
            for (int r = 0; r < RR; r++) if (row0 + rb + r < N) {
                const size_t o = (size_t) (tl.y + slot) * ldc + row0 + rb + r;
                C[o] = acc[r][n];
                if (C2) C2[o] = acc[r][n];
            }
            continue;
        }
        const size_t off = (size_t) (tl.y + slot) * ldc + row0 + rb;
        float * dst = C + off;
#pragma unroll
        for (int r = 0; r < RR; r += 4)
            *(float4 *) (dst + r) = make_float4(acc[r][n], acc[r + 1][n], acc[r + 2][n], acc[r + 3][n]);
        if (C2) {
            float * d2 = C2 + off;
#pragma unroll
            for (int r = 0; r < RR; r += 4)
                *(float4 *) (d2 + r) = make_float4(acc[r][n], acc[r + 1][n], acc[r + 2][n], acc[r + 3][n]);
        }
    }
}

template <int BN, int GATHER, int NTT = (BN < 64 ? BN : 64), int FS = 0, bool K_DIV4 = false, int GG = 4>
static __global__ __launch_bounds__(256, BN == 16 ? 3 : 2) void k_pf_moe_gemm(
        const int4 * tm, const int * ntl,
        const char * W, const __half * sw, const int * perm,
        const int8_t * Ag, const float * sag, int Mp, int K,
        float * C, int ldc, int N, const pf_situo so) {
    const int n = *ntl;
    int bx, by;
    if constexpr (GG > 1) {
        const int gx = gridDim.x, lin = blockIdx.y * gx + blockIdx.x;
        if (lin >= n * gx) return;
        const int per = GG * gx, fy = (lin / per) * GG;
        const int gs = min(n - fy, GG), rem = lin - fy * gx;
        bx = rem / gs; by = fy + rem % gs;
    } else {
        if ((int) blockIdx.y >= n) return;
        bx = blockIdx.x; by = blockIdx.y;
    }
    moe_gemm_body<BN, GATHER, NTT, FS, K_DIV4>(tm[by], bx, W, sw, perm, Ag, sag, Mp, K, C, ldc, N, so);
}

// Dense projections as packed 64x64 Q8 GEMMs (up to 8 jobs in one launch, zero empty
// blocks, 8-way L2-grouped dispatch across token tiles). C2, when set (P2P row-parallel
// partial), streams a second copy straight into the peer's inbox. All dense K are multiples of 128.
struct dgjob { const char * W; const __half * sw; const int8_t * A; const float * as; float * C; int ldc, N, K; float * C2; };
struct dgjobs { dgjob j[8]; int off[9]; int n; };
static __global__ __launch_bounds__(256, 2) void k_pf_dgemm(dgjobs J, int m) {
    constexpr int DG_GG = 8;
    int bx, by;
    {
        const int gx = gridDim.x, lin = blockIdx.y * gx + blockIdx.x, per = DG_GG * gx, fy = (lin / per) * DG_GG;
        const int gs = min((int) gridDim.y - fy, DG_GG), rem = lin - fy * gx;
        bx = rem / gs; by = fy + rem % gs;
    }
    int ji = 0;
#pragma unroll
    for (int k = 1; k < 8; k++) if (k < J.n && bx >= J.off[k]) ji = k;
    const dgjob & g = J.j[ji];
    const int a0 = by * 64;
    moe_gemm_body<64, 0, 16, 0, true>(make_int4(0, a0, min(64, m - a0), 0), bx - J.off[ji],
                                      g.W, g.sw, nullptr, g.A, g.as, PF_M, g.K, g.C, g.ldc, g.N, pf_situo{}, g.C2);
}

// Per row, the weighted sum of its pairs' partials in the router's selection order. That
// order fixes the rounding and there are no atomics, so the sum is deterministic.
// out2, when set (P2P), writes a second copy straight into the peer's inbox.
static __global__ __launch_bounds__(256) void k_pf_moe_reduce(
        const float * __restrict__ y, const int * __restrict__ inv, const float * __restrict__ wts,
        float * __restrict__ out, float * __restrict__ out2) {
    const int r = blockIdx.x, t = threadIdx.x;
    const int * iv = inv + (size_t) r * NE_USED;
    const float * wt = wts + (size_t) r * NE_USED;
    const float * yp[NE_USED];
    float w[NE_USED];
#pragma unroll
    for (int k = 0; k < NE_USED; k++) { yp[k] = y + (size_t) iv[k] * D; w[k] = wt[k]; }
    float4 * o1 = (float4 *) (out + (size_t) r * D);
    float4 * o2 = out2 ? (float4 *) (out2 + (size_t) r * D) : nullptr;
    for (int o = t; o < D / 4; o += 256) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < NE_USED; k++) {
            const float4 v = ((const float4 *) yp[k])[o];
            acc.x = fmaf(w[k], v.x, acc.x);
            acc.y = fmaf(w[k], v.y, acc.y);
            acc.z = fmaf(w[k], v.z, acc.z);
            acc.w = fmaf(w[k], v.w, acc.w);
        }
        o1[o] = acc;
        if (o2) o2[o] = acc;
    }
}

// ------------------------------------------------------------ host side ----

struct pf_plane { unsigned char * q; float * s; };   // pf_gemm activation planes

struct pf_card {
    float * X, * P, * OUT[2], * BLK[8];
    float * IN;                        // [3][PF_M][D]
    float * MSG, * CT, * RINV;         // [4][PF_M][D], [4][PF_M][D], [4][PF_M][2]
    float * Gb;                        // [4][PF_M][GRANK]
    float * F, * FN, * PART[2];        // [PF_M][D]
    float * PARTP[2];                  // the peer's PART[buf], pushed here by the peer
    float * QR, * GR, * Q, * RR, * SK, * KR, * VR, * DK, * DV, * TB;
    pf_plane INP[3], FF, AT;           // input planes (K = D), ffn (K = D), attn output (K = QL)
    unsigned char * APQ; float * APS;  // per-pair situ planes [PF_NP][FFL], [PF_NP][FFL/32]
    float * GP, * YD;                  // controller K-split partials [CT_S][4][PF_M][GRANK], expert down out [PF_NP][D]
    int * IDS; float * WTS;            // [PF_M][NE_USED]
    int * PERM, * INV, * START;        // [PF_NP], [PF_NP], [NE + 1]
    int * TOK;
    int4 * TM; int * NTL;              // expert tiles {expert, first pair, count}, and their count
};

// out rows with stride ostride = RMSNorm(x; w), one block per row
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
    int bnx = 0;                       // exchange parity, the partials are double-buffered
    // The DSpark drafter's target features per prompt row, [row][4][D]: the residual
    // entering applications 5, 23 and 41, and the head input output_norm(mix(blocks, x;
    // out_q)). Card 0 fills the rows from row 0, then runs on_chunk(p0, m).
    float * dtaps = nullptr;
    std::function<void(int, int)> on_chunk;
    float * tapd = nullptr;            // card 0: one chunk's head mixes
    // The prompt's image rows. irow[pos] on the host, one entry per prompt position, is
    // -1 for a token and otherwise indexes img[c], each card's [rows][D] copy; null is text only.
    const int * irow = nullptr;
    const float * img[2] = {};
    int * irw[2] = {};                 // the chunk's irow, one per card
    float * shist[2] = {};             // mid-chunk conv-history snapshot buffers (prefix cache)
    float * ehist[2] = {};             // full-chunk conv-history when the last prompt token rides along
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
        f(3 * MD); f(4 * MD); f(4 * MD); f(4 * (size_t) PF_M * 2);           // IN MSG CT RINV
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
        f((size_t) (PF_NP / 16 + 3 * NE) * 4); f(4);                          // TM (lists A, B, C) NTL
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
        b.IN = (float *) nx(); b.MSG = (float *) nx(); b.CT = (float *) nx(); b.RINV = (float *) nx();
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
        // Kernels never read the other card's memory: some boards (Intel client root ports)
        // forward peer writes but not peer reads, and a peer read there takes both cards
        // down. So each card keeps a copy of the peer's two alternated slots in its own
        // VRAM, filled by qz_xcopy in layer().
        for (auto & q : b.PARTP) {
            if (qz_dmalloc((void **) &q, (size_t) PF_M * D * 4, "state") != hipSuccess) {
                fprintf(stderr, "inkling2: card %d: no room for the %.2f MB prefill inbox\n", c, (double) PF_M * D * 4 / 1e6);
                return false;
            }
            ICHECK(hipMemset(q, 0, (size_t) PF_M * D * 4));
        }
        return true;
    }

    // ---- launch helpers (current device = card c), then one application: all m chunk
    // rows of positions pos0..pos0+m-1 through state slot `slot` of physical layer ip
    dgjob djob(const qw & W, const unsigned char * av, const float * as, float * C, int ldc, float * C2 = nullptr) {
        return dgjob{W.v, W.s, (const int8_t *) av, as, C, ldc, (int) W.N, (int) W.K, C2};
    }
    // several dense GEMMs in one launch, up to 8 jobs, packed with zero empty blocks
    void dgemm(int c, std::initializer_list<dgjob> js, int m) {
        dgjobs J{};
        int n = 0;
        J.off[0] = 0;
        for (const dgjob & g : js) {
            J.j[n] = g;
            J.off[n + 1] = J.off[n] + (g.N + 63) / 64;
            n++;
        }
        J.n = n;
        k_pf_dgemm<<<dim3(J.off[n], (m + 63) / 64), 256, 0, E->C[c].st>>>(J, m);
    }
    void normq1(int c, const float * x, const float * w, pf_plane & pl, int m, float * outf = nullptr) {
        pfnq A{};
        A.x[0] = x; A.w[0] = w; A.q[0] = (int8_t *) pl.q; A.s[0] = pl.s; A.outf = outf;
        k_pf_normq<<<dim3(m, 1), 256, 0, E->C[c].st>>>(A, E->hp.eps);
    }
    void normq3(int c, const float * x0, const float * x1, const float * x2,
                const float * w0, const float * w1, const float * w2, pf_plane (&pl)[3], int m) {
        pfnq A{};
        A.x[0] = x0; A.x[1] = x1; A.x[2] = x2;
        A.w[0] = w0; A.w[1] = w1; A.w[2] = w2;
        for (int j = 0; j < 3; j++) { A.q[j] = (int8_t *) pl[j].q; A.s[j] = pl[j].s; }
        k_pf_normq<<<dim3(m, 3), 256, 0, E->C[c].st>>>(A, E->hp.eps);
    }
    void commit(int c, const float * src, const float * peer, float * hrg, float * shrg, float * ehrg,
                int w, int m, int snap, int snap_end) {
        hipStream_t st = E->C[c].st;
        if (shrg && snap > 0) k_pf_commit<<<(w + 255) / 256, 256, 0, st>>>(src, peer, hrg, shrg, w, snap);
        if (ehrg && snap_end > 0) {
            k_pf_commit<<<(w + 255) / 256, 256, 0, st>>>(src, peer, hrg, ehrg, w, m);
            k_pf_commit<<<(w + 255) / 256, 256, 0, st>>>(src, peer, hrg, hrg, w, snap_end);
        } else {
            k_pf_commit<<<(w + 255) / 256, 256, 0, st>>>(src, peer, hrg, hrg, w, m);
        }
    }
    // The other card's half of a partial: this card's copy (PARTP[buf]), which push_part()
    // filled from the peer.
    const float * peer_part(int c, int buf) const { return B[c].PARTP[buf]; }
    // Hand both cards' PART[buf] to their peer's inbox, after both cards' GEMMs
    // have been queued (a separate pass on purpose: the H2D goes on the peer's stream, so
    // queuing it inside the GEMM loop would park it in front of the peer's own GEMM for the
    // same layer and the two cards would take turns instead of overlapping).
    // Ordering: p2p is one peer copy on this card's stream, after the GEMM that wrote the
    // rows, and qz_xcopy orders the peer's stream behind it. Host mode's D2H is on this
    // card's stream and its H2D on the peer's, where the peer's conv reads the slot below
    // in program order and where its previous reader of the same alternated slot was
    // queued, so the slot's WAR is the stream's own.
    void push_part(int buf, int m) {
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ICHECK(qz_xcopy(B[1 - c].PARTP[buf], 1 - c, E->C[1 - c].st, B[c].PART[buf], c, E->C[c].st, (size_t) m * D * 4));
        }
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
    template <int BN, int G, int GG>
    void moe_gemm(pf_card & b, hipStream_t st, const qw & W, const unsigned char * aq, const float * as,
                  int Mp, int K, float * C, int ldc, int N, int yt, const int4 * tm, const int * ntl, const pf_situo & so) {
        // 4 rows x 4 pairs per thread, so 16 pair lanes. The gate_up GEMM (G = 1, K = D % 128 == 0)
        // ends in the fused SITU.
        if constexpr (G == 1)
            k_pf_moe_gemm<BN, G, 16, 1, true, GG><<<dim3(N / 64, yt), 256, 0, st>>>(tm, ntl, W.v, W.s, b.PERM, (const int8_t *) aq, as, Mp, K, C, ldc, N, so);
        else
            k_pf_moe_gemm<BN, G, 16, 0, false, GG><<<dim3(N / 64, yt), 256, 0, st>>>(tm, ntl, W.v, W.s, b.PERM, (const int8_t *) aq, as, Mp, K, C, ldc, N, so);
    }
    template <int GG>
    void moe_run(pf_card & b, hipStream_t st, int bn, int gather, const qw & W, const unsigned char * aq, const float * as,
                 int Mp, int K, float * C, int ldc, int N, int yt, const int4 * tm, const int * ntl, const pf_situo & so) {
        if (bn == 16) {
            if (gather) moe_gemm<16, 1, GG>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
            else        moe_gemm<16, 0, GG>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
        } else if (bn == 32) {
            if (gather) moe_gemm<32, 1, GG>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
            else        moe_gemm<32, 0, GG>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
        } else {
            if (gather) moe_gemm<64, 1, GG>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
            else        moe_gemm<64, 0, GG>(b, st, W, aq, as, Mp, K, C, ldc, N, yt, tm, ntl, so);
        }
    }
    void moe(int c, int bn, int gather, const qw & W, const unsigned char * aq, const float * as, int Mp, int K,
             float * C, int ldc, int N, int yt, const pf_situo & so = pf_situo{}) {
        ICHECK(hipSetDevice(qz_dev_of(c)));
        pf_card & b = B[c];
        hipStream_t st = E->C[c].st;
        if (gather) moe_run<8>(b, st, bn, 1, W, aq, as, Mp, K, C, ldc, N, yt, b.TM, b.NTL, so);
        else        moe_run<1>(b, st, bn, 0, W, aq, as, Mp, K, C, ldc, N, yt, b.TM, b.NTL, so);
        if (bn >= 32) moe_run<1>(b, st, bn / 2, gather, W, aq, as, Mp, K, C, ldc, N, NE, b.TM + TM_B, b.NTL + 1, so);
        if (bn == 64) moe_run<1>(b, st, 16, gather, W, aq, as, Mp, K, C, ldc, N, NE, b.TM + TM_C, b.NTL + 2, so);
    }

    void layer(int slot, int ip, int pos0, int m, ink2_engine::seq & S, int snap, int snap_end) {
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
                if (full) launch_pf_mix<3>(m, st, A); else launch_pf_mix<1>(m, st, A);
                for (int j = 0; j < 3; j++) in[j] = b.IN + (full ? (size_t) j : 0) * PF_M * D;
            }
            if (msg) {
                pfmix A{};
                A.p[0] = b.BLK[S.bank]; A.p[1] = b.OUT[S.bank_out]; A.n = 2;
                A.qy = L.msgq; A.m = NCTRL; A.out = b.MSG; A.ostr = (size_t) PF_M * D;
                launch_pf_mix<4>(m, st, A);
                pfrms R{};
                for (int j = 0; j < 3; j++) R.cur[j] = in[j];
                R.msg = b.MSG; R.rinv = b.RINV; R.j0 = 0;
                k_pf_rms_inv<<<dim3(m, 3), 256, 0, st>>>(R);
                pfcdn Da{};
                for (int j = 0; j < 4; j++) { Da.cur[j] = j < 3 ? in[j] : b.F; Da.wd[j] = L.cd[j]; }
                Da.msg = b.MSG; Da.rinv = b.RINV; Da.g = b.Gb; Da.j0 = 0; Da.m = m;
                k_pf_ctrl_down2<<<dim3((m + 63) / 64, 3, CT_S), 256, 0, st>>>(Da, b.GP);
                pfcup Ua{};
                Ua.g = b.Gb; Ua.msg = b.MSG; Ua.out = b.CT; Ua.j0 = 0; Ua.m = m;
                for (int j = 0; j < 4; j++) {
                    Ua.cur[j] = j < 3 ? in[j] : b.F;
                    Ua.wgt[j] = L.cgt[j];
                    Ua.wft[j] = L.cft[j];
                }
                k_pf_gsum<<<dim3((m * GRANK + 255) / 256, 3), 256, 0, st>>>(b.GP, b.Gb, 0, m);
                k_pf_ctrl_up2<<<dim3((m + 63) / 64, D / 64, 3), 256, 0, st>>>(Ua);
                for (int j = 0; j < 3; j++) in[j] = b.CT + (size_t) j * PF_M * D;
            }
            // norm and int8 quant per stream, then the Q8 GEMMs
            const int ik = full || msg ? 1 : 0, ivv = full || msg ? 2 : 0;
            if (full) {
                normq3(c, in[0], in[1], in[2], L.qn, L.kn, L.vn, b.INP, m);
            } else if (msg) {
                normq3(c, in[0], in[1], in[2], L.an, L.an, L.an, b.INP, m);
            } else {
                normq1(c, in[0], L.an, b.INP[0], m);
            }
            if (!full) dgemm(c, {djob(L.q, b.INP[0].q, b.INP[0].s, b.QR, QL), djob(L.gate, b.INP[0].q, b.INP[0].s, b.GR, QL),
                      djob(L.rel, b.INP[0].q, b.INP[0].s, b.RR, HQL * DREL), djob(L.k, b.INP[ik].q, b.INP[ik].s, b.KR, KVL),
                      djob(L.v, b.INP[ivv].q, b.INP[ivv].s, b.VR, KVL), djob(L.sink, b.INP[0].q, b.INP[0].s, b.SK, HQL)}, m);
            if (full) dgemm(c, {djob(L.q, b.INP[0].q, b.INP[0].s, b.QR, QL), djob(L.gate, b.INP[0].q, b.INP[0].s, b.GR, QL),
                                djob(L.rel, b.INP[0].q, b.INP[0].s, b.RR, HQL * DREL), djob(L.k, b.INP[1].q, b.INP[1].s, b.KR, KVL),
                                djob(L.v, b.INP[2].q, b.INP[2].s, b.VR, KVL), djob(L.sink, b.INP[0].q, b.INP[0].s, b.SK, HQL),
                                djob(L.kcp, b.INP[1].q, b.INP[1].s, b.DK, HKVL * NCONV), djob(L.vcp, b.INP[2].q, b.INP[2].s, b.DV, HKVL * NCONV)}, m);
            // head post with clips, tau and bias tables, then the K/V convs into the cache
            pfhp h{};
            h.qr = b.QR; h.rr = b.RR; h.kr = b.KR; h.vr = b.VR;
            h.dk = full ? b.DK : nullptr; h.dv = full ? b.DV : nullptr;
            h.qclip = L.qclip; h.rclip = L.rclip; h.kconv = L.kconv; h.vconv = L.vconv;
            h.hist = K.hist + (size_t) slot * 3 * CHL;
            h.q = b.Q; h.kc = K.kc[slot]; h.vc = K.vc[slot];
            h.rvo = b.TB;
            h.pos0 = pos0; h.m = m; h.full = full;
            h.tau_a = E.hp.tau_alpha; h.tau_f = E.hp.tau_floor;
            h.qk_clip = E.hp.qk_clip; h.rel_clip = E.hp.rel_clip; h.eps = E.hp.eps;
            k_pf_head_post<<<dim3(HQL + 2 * HKVL, (m + HP_R - 1) / HP_R), 128, 0, st>>>(h);
            float * sh = shist[c] ? shist[c] + (size_t) slot * 3 * CHL : nullptr;
            float * eh = ehist[c] ? ehist[c] + (size_t) slot * 3 * CHL : nullptr;
            commit(c, b.KR, nullptr, h.hist + H_K, sh ? sh + H_K : nullptr, eh ? eh + H_K : nullptr, KVL, m, snap, snap_end);
            commit(c, b.VR, nullptr, h.hist + H_V, sh ? sh + H_V : nullptr, eh ? eh + H_V : nullptr, KVL, m, snap, snap_end);
            // attention, then attn_output; rows are split, so each card keeps a partial
            pfat t{};
            t.q = b.Q; t.gate = b.GR; t.kc = K.kc[slot]; t.vc = K.vc[slot];
            t.pos0 = pos0; t.full = full; t.oq = b.AT.q; t.os = b.AT.s;
            t.rv = b.TB; t.relb = L.relb; t.sk = b.SK;
            k_pf_fattn<1, true><<<dim3((m + 31) / 32, HQL), 512, 0, st>>>(t, m);
            float * peer_out = qz_link == QZ_LINK_P2P ? B[1 - c].PARTP[buf] : nullptr;
            dgemm(c, {djob(L.out, b.AT.q, b.AT.s, b.PART[buf], D, peer_out)}, m);
        }
        if (qz_link != QZ_LINK_P2P) push_part(buf, m);
        E.exchange();
        bnx++;
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ink2_card & K = E.C[c];
            pf_card & b = B[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            pfcv ca{};
            ca.own = b.PART[buf]; ca.peer = peer_part(c, buf); ca.w = L.pconv;
            ca.hrg = K.hist + (size_t) slot * 3 * CHL + H_A;
            ca.x = b.X; ca.blk = push >= 0 ? b.BLK[push] : nullptr; ca.m = m; ca.push = push >= 0;
            k_pf_conv2<<<dim3((D / 4 + 255) / 256, (m + CV_R - 1) / CV_R), 256, 0, st>>>(ca);
            float * sha = shist[c] ? shist[c] + (size_t) slot * 3 * CHL + H_A : nullptr;
            float * eha = ehist[c] ? ehist[c] + (size_t) slot * 3 * CHL + H_A : nullptr;
            commit(c, b.PART[buf], ca.peer, ca.hrg, sha, eha, D, m, snap, snap_end);
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
            launch_pf_mix<1>(m, st, A);
            const float * f = b.F;
            if (msg) {
                pfrms R{};
                R.cur[3] = b.F; R.msg = b.MSG; R.rinv = b.RINV; R.j0 = 3;
                k_pf_rms_inv<<<dim3(m, 1), 256, 0, st>>>(R);
                pfcdn Da{};
                Da.cur[3] = b.F; Da.msg = b.MSG; Da.rinv = b.RINV; Da.wd[3] = L.cd[3]; Da.g = b.Gb; Da.j0 = 3; Da.m = m;
                k_pf_ctrl_down2<<<dim3((m + 63) / 64, 1, CT_S), 256, 0, st>>>(Da, b.GP);
                pfcup Ua{};
                Ua.g = b.Gb; Ua.msg = b.MSG; Ua.out = b.CT; Ua.j0 = 3; Ua.m = m;
                Ua.cur[3] = b.F; Ua.wgt[3] = L.cgt[3]; Ua.wft[3] = L.cft[3];
                k_pf_gsum<<<dim3((m * GRANK + 255) / 256, 1), 256, 0, st>>>(b.GP, b.Gb, 3, m);
                k_pf_ctrl_up2<<<dim3((m + 63) / 64, D / 64, 1), 256, 0, st>>>(Ua);
                f = b.CT + (size_t) 3 * PF_M * D;
            }
            normq1(c, f, L.ffn, b.FF, m, b.FN);
            {
                k_pf_router2<<<dim3((m + RT_R - 1) / RT_R, RT_S), 256, 0, st>>>(b.FN, L.gin, b.YD, m);   // YD is scratch until the down GEMM
                k_pf_topk<<<(m + 3) / 4, 256, 0, st>>>(b.YD, L.pb, E.hp.route_scale, b.IDS, b.WTS, m);
            }
            k_pf_sort<<<1, 256, 0, st>>>(b.IDS, np, b.START, b.PERM, b.INV);
            moe_tiles(c, bn);
            pf_situo so;            // SITU fused into the gate_up GEMM's epilogue, feeding the down GEMM's planes
            so.q = b.APQ; so.qs = b.APS; so.np = np; so.beta = E.hp.situ_beta; so.lb = E.hp.situ_lin_beta;
            moe(c, bn, 1, L.gu, b.FF.q, b.FF.s, PF_M, D, nullptr, GUL, GUL, yt, so);
            moe(c, bn, 0, L.dn, b.APQ, b.APS, np, FFL, b.YD, D, D, yt);
            float * peer_out = qz_link == QZ_LINK_P2P ? B[1 - c].PARTP[buf2] : nullptr;
            k_pf_moe_reduce<<<m, 256, 0, st>>>(b.YD, b.INV, b.WTS, b.PART[buf2], peer_out);
        }
        if (qz_link != QZ_LINK_P2P) push_part(buf2, m);
        E.exchange();
        bnx++;
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            ink2_card & K = E.C[c];
            pf_card & b = B[c];
            const lw & L = K.L[ip];
            hipStream_t st = K.st;
            pfcv cf{};
            cf.own = b.PART[buf2]; cf.peer = peer_part(c, buf2); cf.w = L.fconv;
            cf.hrg = K.hist + (size_t) slot * 3 * CHL + H_F;
            cf.x = b.X; cf.blk = nullptr; cf.m = m; cf.push = 0;
            k_pf_conv2<<<dim3((D / 4 + 255) / 256, (m + CV_R - 1) / CV_R), 256, 0, st>>>(cf);
            float * shf = shist[c] ? shist[c] + (size_t) slot * 3 * CHL + H_F : nullptr;
            float * ehf = ehist[c] ? ehist[c] + (size_t) slot * 3 * CHL + H_F : nullptr;
            commit(c, b.PART[buf2], cf.peer, cf.hrg, shf, ehf, D, m, snap, snap_end);
        }
    }

    // The head for the chunk's last row only, using the decode kernels on row offsets
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

    // ---- the batched prefill of prompt rows a .. n-1. Rows below a are already in
    // the KV and conv history. logits asks for the head on row n-1; snap > 0 stages
    // the conv history after snap rows into shist; snap_end > 0 leaves hist at snap_end
    // rows and stages the full chunk's conv history into ehist; false if aborted
    bool prefill(const int32_t * prompt, int a, int n, bool logits, int snap = -1, int snap_end = -1) {
        ink2_engine & E = *this->E;
        for (int p0 = a; p0 < n; p0 += PF_M) {
            if (abort && abort->load()) return false;
            const int m = std::min(PF_M, n - p0);
            const int sn = (snap > 0 && p0 == a && snap < m) ? snap : -1;
            const int se = (snap_end > 0 && p0 == a && snap_end < m) ? snap_end : -1;
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
            for (int il = 0; il < LOOP0; il++) { if (il == 5) tap(0, p0, m); layer(il, il, p0, m, S, sn, se); }
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
                    if ((pass == 1 && ip == 11) || (pass == 2 && ip == 17)) tap(pass, p0, m);   // visits 23 and 41
                    layer(slot, ip, p0, m, S, sn, se);
                }
            }
            S.bank = -1;
            for (int il = LOOP1; il < NL; il++) layer(il, il, p0, m, S, sn, se);
            tap_head(S, p0, m);
            if (logits && p0 + m == n) head_last(S, m);
        }
        return true;
    }
};
