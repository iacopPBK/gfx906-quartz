// inkling2 device bodies (docs/inkling2.md). A Q8 GEMV takes int8 activations, values
// [K] plus f32 scales [K/32], as quartz/kernels/gemv.h reads them.
//
// Every d_* body runs over one "virtual block" (index b, thread t of NT, active when
// t < NT), and the persistent decode kernel (ink2_cycle.h) runs it inside 512-thread
// blocks. A body with a block reduction must be called by every thread of the real block.
#pragma once

#include "kernels/xchg.h"                     // qz_xinbox: the TP sum in d_xconv_blk

// Fused multiply-adds follow the source expression (FP_CONTRACT ON) rather than the
// backend, whose default "fast" fuses per inline context. That keeps the standalone
// kernels and the persistent kernel rounding identically, and is restored at the end.
#pragma clang fp contract(on)

constexpr int HQL = HQ / QZ_NC, HKVL = HKV / QZ_NC;    // local heads per card
constexpr int QL = HQL * HD, KVL = HKVL * HD;          // 1536, 512 at 2 cards
constexpr int FFL = FFE / QZ_NC;                       // local expert FF: 576 at 2 cards
constexpr int GUL = 2 * FFL;                            // local gate + up rows per expert: 1152 at 2 cards
// Prefill chunk rows, and the SWA ring, indexed pos % PF_RING everywhere. The ring holds
// at least a window of WIN plus the in-flight chunk, and the slack keeps a prefix-cache
// snapshot at s restorable while writes stay below s - WIN + 1 + PF_RING.
constexpr int PF_M = 2048, PF_RING = 8192;
static_assert(PF_RING >= WIN + PF_M, "the SWA ring holds a window plus a chunk");
constexpr int CHL = 2 * KVL + 2 * D;                   // local conv history channels: K | V | attn post | ffn post
constexpr int H_K = 0, H_V = KVL, H_A = 2 * KVL, H_F = 2 * KVL + D;
constexpr int TBW = REL_EXT + 1;                       // bias table width (the zero column past the extent)
// The attention works in log2 units (ink2_flash.h): head post scales the bias table by
// log2(e) and writes an f16 copy of q pre-scaled by log2(e) / sqrt(HD), so a score is one
// v_dot2 per (lane = key, head, 2 dims).
constexpr float FL_L2E = 1.4426950408889634f;
constexpr float FL_QSC = 0.12751743082459868f;

// Sum of N values over the first nw waves of the block; every thread gets the sums.
template <int N, bool DPP = true> __device__ __forceinline__ void bsum(float (&v)[N], float * red, int nw) {
    #pragma unroll
    for (int k = 0; k < N; k++)
        if constexpr (DPP) v[k] = qz_wsum_dpp(v[k]);
        else   // not qz_wsum: dev.h is outside contract(on), so x * x would fuse into the first add
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) v[k] += __shfl_xor(v[k], off, 64);
    const int w = threadIdx.x >> 6;
    __syncthreads();
    if ((threadIdx.x & 63) == 0 && w < nw)
        #pragma unroll
        for (int k = 0; k < N; k++) red[w * N + k] = v[k];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < N; k++) {
        float s = 0.f;
        for (int i = 0; i < nw; i++) s += red[i * N + k];
        v[k] = s;
    }
}
template <int N> __device__ __forceinline__ void bsum(float (&v)[N], float * red) { bsum<N, false>(v, red, (int) (blockDim.x >> 6)); }

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

// x -> int8 plus the block's f16-rounded scale, one row per 32-lane half
__device__ __forceinline__ void quant32(float x, unsigned char * q, float * qs, int i) {
    const float amax = qz_amax32(fabsf(x));
    const float d = amax / 127.f;
    q[i] = (unsigned char) qz_q8_code_div(x, amax, d);
    if ((i & 31) == 0) qs[i >> 5] = qz_q8_scale(d);
}

// ---- token row of the Q8 embedding -> rmsnorm * w (NT 256)
__device__ __forceinline__ void d_embed(const char * ev, const __half * es, int tok, const float * w, float eps, float * x,
                                        int t, bool act) {
    __shared__ float red[16];
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = t + k * 256;
        v[k] = act ? (float) ev[(size_t) tok * D + i] * __half2float(es[(size_t) tok * (D / 32) + i / 32]) : 0.f;
        ss[0] += v[k] * v[k];
    }
    bsum<1>(ss, red, 4);
    const float r = rsqrtf(ss[0] / D + eps);
    if (act)
        #pragma unroll
        for (int k = 0; k < D / 256; k++) {
            const int i = t + k * 256;
            x[i] = v[k] * r * w[i];
        }
}

// ---- mix(states, query): softmax over the rmsnormed states of Sn . q_j / sqrt(D), and
// out[j] = sum w * Sn (NT 256).
struct sptrs { const float * p[6]; int n; };
template <int MQ>
__device__ __forceinline__ void d_mix_lds(const sptrs & S, const float * qy, float * out, int tid, bool act, float * sv) {
    // A 512-thread block on both paths: in the sum pass half h takes states 3h .. 3h+2,
    // elements t + 256 k in order with one 4-wave reduction per half. The output pass
    // takes element i = tid + 512 k over all 512 threads.
    (void) act;
    constexpr int NV = D / 256;
    __shared__ float red[2 * 4 * 3 * MQ];
    __shared__ float wsh[6][4];
    const int n = S.n;
    const int h = tid >> 8, t = tid & 255, w = (tid >> 6) & 3, lane = tid & 63;
    // Every load is in flight at once: the states for LDS and the queries.
    constexpr int NS = (D + 511) / 512;
    float st[6][NS];
    #pragma unroll
    for (int s = 0; s < 6; s++)
        #pragma unroll
        for (int k = 0; k < NS; k++) {
            // branch-free: absent states read state 0, never used; the D tail is clamped
            const int i = min((int) threadIdx.x + 512 * k, D - 1);
            st[s][k] = S.p[s < n ? s : 0][i];
        }
    float qv[MQ][NV];
    #pragma unroll
    for (int j = 0; j < MQ; j++)
        #pragma unroll
        for (int k = 0; k < NV; k++) qv[j][k] = qy[(size_t) j * D + t + 256 * k];
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < 6; s++) {
        #pragma unroll
        for (int k = 0; k < D / 512; k++) sv[s * D + threadIdx.x + 512 * k] = st[s][k];
        if (D % 512 && threadIdx.x < D % 512) sv[s * D + threadIdx.x + 512 * (D / 512)] = st[s][D / 512];
    }
    __syncthreads();
    // Two passes over the LDS states, as in d_mix: the sums of squares, then the dots of
    // rms(x) with the queries. A one-pass rs * dot(x, q) is not bit-equal.
    constexpr int NQ = 3 * MQ;
    const float * sh = sv + 3 * h * D + t;                // this half's states, this thread's column
    float r[NQ];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        r[s] = 0.f;
        #pragma unroll
        for (int k = 0; k < NV; k++) { const float v = sh[s * D + 256 * k]; r[s] += v * v; }
        r[s] = qz_wsum_dpp(r[s]);
    }
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 3; s++) red[(h * 4 + w) * 3 + s] = r[s];
    __syncthreads();
    float rs[6];
    #pragma unroll
    for (int s = 0; s < 6; s++) {
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; i++) a += red[((s / 3) * 4 + i) * 3 + s % 3];
        rs[s] = rsqrtf(a / D + 1e-6f);
    }
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        const float rh = h ? rs[3 + s] : rs[s];
        #pragma unroll
        for (int j = 0; j < MQ; j++) r[s * MQ + j] = 0.f;
        #pragma unroll
        for (int k = 0; k < NV; k++) {
            const float x = sh[s * D + 256 * k] * rh;
            #pragma unroll
            for (int j = 0; j < MQ; j++) r[s * MQ + j] += x * qv[j][k];
        }
    }
    #pragma unroll
    for (int q = 0; q < NQ; q++) r[q] = qz_wsum_dpp(r[q]);
    if (lane == 0)
        #pragma unroll
        for (int q = 0; q < NQ; q++) red[(h * 4 + w) * NQ + q] = r[q];
    __syncthreads();
    // lanes 0 .. MQ-1 each run the softmax of query j = tid, on d_mix's values
    if (tid < MQ) {
        float v[6];
        #pragma unroll
        for (int s = 0; s < 6; s++) {
            float a = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; i++) a += red[((s / 3) * 4 + i) * NQ + (s % 3) * MQ + tid];
            v[s] = a;
        }
        const float scale = 1.f / sqrtf((float) D);
        float mx = -INFINITY;
        #pragma unroll
        for (int s = 0; s < 6; s++) if (s < n) mx = fmaxf(mx, v[s] * scale);
        float sum = 0.f, ev[6];
        #pragma unroll
        for (int s = 0; s < 6; s++) if (s < n) { ev[s] = expf(v[s] * scale - mx); sum += ev[s]; }
        #pragma unroll
        for (int s = 0; s < 6; s++) if (s < n) wsh[s][tid] = ev[s] / sum;
    }
    __syncthreads();
    float wr[6][MQ];
    #pragma unroll
    for (int s = 0; s < 6; s++)
        #pragma unroll
        for (int j = 0; j < MQ; j++) wr[s][j] = wsh[s][j];
    #pragma unroll
    for (int k = 0; k < NS; k++) {
        const int i = threadIdx.x + 512 * k;
        if (k < D / 512 || i < D) {
            float o[MQ];
            #pragma unroll
            for (int j = 0; j < MQ; j++) o[j] = 0.f;
            #pragma unroll
            for (int s = 0; s < 6; s++)
                if (s < n) {
                    const float x = sv[s * D + i];
                    #pragma unroll
                    for (int j = 0; j < MQ; j++) o[j] += wr[s][j] * x * rs[s];
                }
            #pragma unroll
            for (int j = 0; j < MQ; j++) out[(size_t) j * D + i] = o[j];
        }
    }
    __syncthreads();
}

constexpr int MIX_LDS = 6 * D * 4;               // d_mix_lds: the states in dynamic LDS
extern __shared__ float mix_lds[];
template <int MQ>
static __global__ __launch_bounds__(512) void k_mix(sptrs S, const float * qy, float * out) {
    d_mix_lds<MQ>(S, qy, out, threadIdx.x, true, mix_lds);
}

// ---- control(cur, msg): g = [rms(cur); rms(msg)] . down, and the result is
// (1 - tanh(g.forget)) * cur + tanh(g.gain) * msg. wgt / wft are gain / forget transposed
// to [GRANK][D], which the persistent kernel copies coalesced.
struct cjob { const float * cur, * msg, * wd, * wg, * wf; float * g, * out; const float * wgt, * wft; };
static __global__ void k_ctrl_t(const float * s, float * d) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= D) return;
    for (int k = 0; k < GRANK; k++) d[(size_t) k * D + i] = s[(size_t) i * GRANK + k];
}
struct cjobs { cjob j[3]; };

// ---- rmsnorm(x) * w -> optional f32 y and the int8 planes (NT 256)
struct njob { const float * x, * w; float * y; unsigned char * q; float * qs; };
struct njobs { njob j[3]; };
__device__ __forceinline__ void d_norm_quant(const njob & a, float eps, int tid, bool act) {
    __shared__ float red[4];
    float v[D / 256], wv[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {           // unconditional loads (no branch per load), then the select
        const float x_ = a.x[(tid & 255) + k * 256], w_ = a.w[(tid & 255) + k * 256];
        v[k] = act ? x_ : 0.f; wv[k] = act ? w_ : 0.f;
    }
    #pragma unroll
    for (int k = 0; k < D / 256; k++) ss[0] += v[k] * v[k];
    bsum<1>(ss, red, 4);
    const float r = rsqrtf(ss[0] / D + eps);
    if (act)
        #pragma unroll
        for (int k = 0; k < D / 256; k++) {
            const int i = tid + k * 256;
            const float y = v[k] * r * wv[k];
            if (a.y) a.y[i] = y;
            quant32(y, a.q, a.qs, i);
        }
}
static __global__ __launch_bounds__(256) void k_norm_quant(njobs J, float eps) { d_norm_quant(J.j[blockIdx.x], eps, threadIdx.x, true); }

// ---- several Q8 GEMVs in one launch: y[N] = W[N][K] . a, 32 rows per 512-thread block.
// yp, when given, takes a second copy of the outputs for the TP2 peer inbox.
struct gjob { const char * wv; const __half * ws; const unsigned char * av; const float * as; float * y; int N, K, g1; };
struct gjobs { gjob j[8]; int n; };
__device__ __forceinline__ void d_gemv(const gjobs & J, int b, float * yp) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    int k = 0;
    while (k + 1 < J.n && b >= J.j[k].g1) k++;
    const gjob & j = J.j[k];
    const int g = b - (k ? J.j[k - 1].g1 : 0);
    const int r0 = g * 32 + wave * 4;
    float acc[1][4];
    qz_gemv_rows_t<1, 4>(j.av, j.as, j.K, j.wv, j.ws, j.N, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 4; s++) if (r0 + s < j.N) { j.y[r0 + s] = acc[0][s]; if (yp) yp[r0 + s] = acc[0][s]; }
}
static __global__ __launch_bounds__(512) void k_gemv(gjobs J) { d_gemv(J, blockIdx.x, nullptr); }

// ---- per local head, after the projections: Q / R clip and tau, the bias table, then
// K / V conv with clip and norm into the cache (NT 128, virtual blocks HQL + 2 HKVL).
struct hpost {
    const float * qr, * rr, * sk, * kr, * vr, * dk, * dv;     // raw projections (dk / dv null on SWA layers)
    const float * qclip, * rclip, * relb, * kconv, * vconv;
    float * hist;                                            // this slot's [3][CHL]
    float * tb;                                              // out: table [HQL][TBW]
    __half * qh;                                             // out: q as f16 * log2(e)/sqrt(HD), [QL]
    __half * kc, * vc;                                       // the cache rows to write
    float tau, qk_clip, rel_clip, eps;
};
// dmax: the table entries computed (TBW, or min(TBW, pos + 1) in the persistent kernel)
__device__ __forceinline__ void d_head_post(const hpost & a, int b, int t, bool act, int dmax = TBW) {
    __shared__ float red[2 * 2];
    __shared__ float rv[DREL];
    if (b < HQL) {
        const int h = b, i = h * HD + t;
        float x = act ? a.qr[i] * a.qclip[i] : 0.f;
        float n2[1] = {x * x};
        bsum<1>(n2, red, 2);
        const float nr = sqrtf(n2[0]) + 1e-6f;
        x *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        if (act) a.qh[i] = __float2half(x * a.tau * FL_QSC);
        float r = t < DREL ? a.rr[h * DREL + t] * a.rclip[h * DREL + t] : 0.f;
        float r2[1] = {r * r};
        bsum<1>(r2, red, 2);
        const float rn = sqrtf(r2[0]) + 1e-6f;
        if (t < DREL) rv[t] = r * fminf(fmaxf(a.rel_clip / rn, 0.f), 1.f) * a.tau;
        __syncthreads();
        const float sink = a.sk[h];
        if (act)
            for (int d = t; d < dmax; d += 128) {
                float s = 0.f;
                if (d < REL_EXT)
                    #pragma unroll
                    for (int k = 0; k < DREL; k++) s += a.relb[d * DREL + k] * rv[k];
                a.tb[h * TBW + d] = s - sink;
            }
        __syncthreads();
        return;
    }
    const bool isk = b < HQL + HKVL;
    const int hk = isk ? b - HQL : b - HQL - HKVL, ch = hk * HD + t;
    const float * raw = isk ? a.kr : a.vr;
    const float * dyn = isk ? a.dk : a.dv;
    const float * w = (isk ? a.kconv : a.vconv) + (size_t) ch * NCONV;
    float * hs = a.hist + (isk ? H_K : H_V) + ch;
    float y = 0.f;
    if (act) {
        const float u = raw[ch], h0 = hs[0], h1 = hs[CHL], h2 = hs[2 * CHL];
        float o = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * u;
        if (dyn) {
            const float * tp = dyn + hk * NCONV;
            o += u * tp[0];
            o += h2 * tp[1];
            o += h1 * tp[2];
            o += h0 * tp[3];
        }
        hs[0] = h1; hs[CHL] = h2; hs[2 * CHL] = u;
        y = (u + o) * (dyn ? 1.f / sqrtf(2.f * NCONV + 1.f) : 1.f / sqrtf((float) NCONV + 1.f));
    }
    float n2[1] = {y * y};
    bsum<1>(n2, red, 2);
    if (!act) return;
    if (isk) {
        const float nr = sqrtf(n2[0]) + 1e-6f;
        y *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        a.kc[ch] = __float2half(y);
    } else {
        y *= rsqrtf(n2[0] / HD + a.eps);
        a.vc[ch] = __float2half(y);
    }
}

// ---- attention: the flash body of ink2_flash.h (lane = key position, q in LDS, K/V tiles
// in LDS, one wave per query row), shared by the plain step and the DSpark verify so that a
// verify row is bit-identical to a plain step at the same rp. The plain step splits the KV
// cache ATT_SPL ways per KV head; the verify stages the same split count (FL_NS).
constexpr int GQ = HQ / HKV, ATT_SPL = 15, ATT_PW = HD + 2;
struct attn_args {
    const float * tb, * gate;
    const __half * qh;            // q as f16 * log2(e)/sqrt(HD), pre-scaled by head post
    const __half * kc, * vc;      // cache base: row r at r * KVL
    int pos, full;
    unsigned char * oq; float * os;
    float * part;                 // [HKVL][ATT_SPL][GQ][ATT_PW]
    int * cnt;                    // [HKVL], zero between uses (the merger resets it)
};
// one 16 B read of the f16 cache as 8 floats (the drafter's attention staging)
__device__ __forceinline__ void att_h8(const uint4 & u, float * f) {
    const __half2 * h = (const __half2 *) &u;
    #pragma unroll
    for (int k = 0; k < 4; k++) { f[2 * k] = __low2float(h[k]); f[2 * k + 1] = __high2float(h[k]); }
}

// ---- the TP sum of the cards' partials, then a short conv with its history and no
// dynamic taps. blk null: x += a. Otherwise blk = x and x = a, a new residual block. One
// block covers t, t + NT, ... and issues all loads first, since it is latency bound.
template <int NT>
__device__ __forceinline__ void d_xconv_blk(const float * own, const qz_xinbox<QZ_NC> & peer, const float * w, float * hs,
                                            float * x, float * blk, int t) {
    constexpr int NE_ = (D + NT - 1) / NT;
    float u[NE_], h0[NE_], h1[NE_], h2[NE_], xv[NE_];
    float4 wi[NE_];
    #pragma unroll
    for (int k = 0; k < NE_; k++) {
        const int i = min(t + NT * k, D - 1);          // clamped: unconditional loads, the stores guarded
        u[k] = peer.add(own[i], i);
        h0[k] = hs[i]; h1[k] = hs[CHL + i]; h2[k] = hs[2 * CHL + i];
        wi[k] = *(const float4 *) (w + (size_t) i * NCONV);
        xv[k] = x[i];
    }
    #pragma unroll
    for (int k = 0; k < NE_; k++) {
        const int i = t + NT * k;
        if ((k + 1) * NT <= D || i < D) {
            const float o = wi[k].x * h0[k] + wi[k].y * h1[k] + wi[k].z * h2[k] + wi[k].w * u[k];
            hs[i] = h1[k]; hs[CHL + i] = h2[k]; hs[2 * CHL + i] = u[k];
            const float y = (u[k] + o) * (1.f / sqrtf((float) NCONV + 1.f));
            if (blk) { blk[i] = xv[k]; x[i] = y; }
            else x[i] = xv[k] + y;
        }
    }
}

// ---- router: sigmoid(f . W), the top NE_USED by p + bias, weights p / sum(p) * scale. A
// wave per expert (64-lane strided sum, xor tree). ps[e] = p, ps[NE + e] = p + bias.
static_assert(D / 64 % 12 == 0, "router chunks");
__device__ __forceinline__ void d_router(const float * f, const float * wr, const float * bias, float * ps, int e, int lane) {
    // the loads 12 + 12 at a time, the same sum order
    float s = 0.f;
    #pragma unroll 1
    for (int c0 = 0; c0 < D / 64; c0 += 12) {
        float fv[12], wv[12];
        #pragma unroll
        for (int k = 0; k < 12; k++) { fv[k] = f[lane + 64 * (c0 + k)]; wv[k] = wr[(size_t) e * D + lane + 64 * (c0 + k)]; }
        #pragma unroll
        for (int k = 0; k < 12; k++) s += fv[k] * wv[k];
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) s += __shfl_xor(s, off, 64);
    if (lane == 0) { const float p = sigm(s); ps[e] = p; ps[NE + e] = p + bias[e]; }
}
// gate_up rows are interleaved in 32s at load (k_gu_interleave): tile T of 64 rows holds
// gate rows 32T..32T+31 then the matching up rows, so gate row j has its up row 32 later.
__device__ __host__ __forceinline__ int gu_row(int j) { return (j / 32) * 64 + j % 32; }
static __global__ __launch_bounds__(256) void k_gu_interleave(const char * sv, const __half * ss, char * dv, __half * ds) {
    const int p = blockIdx.x, e = blockIdx.y, r = p % 64;
    const int src = r < 32 ? (p / 64) * 32 + r : FFL + (p / 64) * 32 + r - 32;
    const size_t so = (size_t) e * GUL + src, dof = (size_t) e * GUL + p;
    const uint4 * a = (const uint4 *) (sv + so * D);
    uint4 * b = (uint4 *) (dv + dof * D);
    for (int i = threadIdx.x; i < D / 16; i += 256) b[i] = a[i];
    for (int i = threadIdx.x; i < D / 32; i += 256) ds[dof * (D / 32) + i] = ss[so * (D / 32) + i];
}
// ---- SITU: the expert activation of a gate value g and its up value u
__device__ __forceinline__ float situ_v(float g, float u, float beta, float lb) {
    return beta * tanhf(g / beta) * sigm(g) * (lb * tanhf(u / lb));
}

static __global__ void k_softcap(float * y, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(y[i] / cap);
}

#include "ink2_flash.h"

#pragma clang fp contract(fast)
