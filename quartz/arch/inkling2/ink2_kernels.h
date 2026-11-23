// inkling2 device bodies (docs/inkling2.md). Activations into a Q8 GEMV are
// int8 planes (vals [K] + f32 scales [K/32]), as quartz/kernels/gemv.h reads them.
//
// Every body d_* runs over one "virtual block" (index b, thread t of NT,
// act = t < NT); the persistent decode kernel (ink2_cycle.h) runs them inside
// 512-thread blocks. Bodies with block reductions must be called by every
// thread of the real block (inactive threads pass act = false).
#pragma once

// fused multiply-adds decided by the source expression (FP_CONTRACT ON), not by
// the backend (HIP's default "fast" fuses per inline context): the standalone
// kernels and the persistent kernel then round identically. Restored at the end.
#pragma clang fp contract(on)

constexpr int HQL = HQ / 2, HKVL = HKV / 2;            // local heads per card
constexpr int QL = HQL * HD, KVL = HKVL * HD;          // 1536, 512
constexpr int FFL = FFE / 2;                           // local expert FF: 576
// prefill chunk rows (batched prefill) and the SWA ring size it needs: a
// window of WIN plus the in-flight chunk, indexed pos % PF_RING everywhere
constexpr int PF_M = 2048, PF_RING = WIN + PF_M;
constexpr int CHL = 2 * KVL + 2 * D;                   // local conv history channels: K | V | attn post | ffn post
constexpr int H_K = 0, H_V = KVL, H_A = 2 * KVL, H_F = 2 * KVL + D;
constexpr int TBW = REL_EXT + 1;                       // bias table width (the zero column past the extent)

// wsum_xor (the xor butterfly with DPP in-row levels): ink2_prelude.h
// sum of N values over the first nw waves of the block (every thread of the
// block calls it); every thread gets the sums
template <int N, bool DPP = true> __device__ __forceinline__ void bsum(float (&v)[N], float * red, int nw) {
    #pragma unroll
    for (int k = 0; k < N; k++)
        if (DPP) v[k] = wsum_xor(v[k]);
        else
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

// x (one row in each 32-lane half: lanes hold consecutive elements) -> int8 + the block's f16-rounded scale
__device__ __forceinline__ void quant32(float x, unsigned char * q, float * qs, int i) {
    const float amax = qz_amax32(fabsf(x));
    const float d = amax / 127.f;
    int v = amax == 0.f ? 0 : (int) roundf(x / d);
    v = v > 127 ? 127 : (v < -127 ? -127 : v);
    q[i] = (unsigned char) v;
    if ((i & 31) == 0) qs[i >> 5] = __half2float(__float2half(d));
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

// ---- mix(states, query): softmax over the rmsnormed states of Sn . q_j / sqrt(D); out[j] = sum w * Sn (NT 256)
struct sptrs { const float * p[6]; int n; };
template <int MQ>
__device__ __forceinline__ void d_mix_lds(const sptrs & S, const float * qy, float * out, int tid, bool act, float * sv) {
    // a 512-thread block (both paths): half h takes states 3h .. 3h+2 in the sum
    // pass (elements t + 256 k in order, one 4-wave reduction per half), the
    // output pass takes element i = tid + 512 k over all 512 threads
    (void) act;
    constexpr int NV = D / 256;
    __shared__ float red[2 * 4 * 3 * MQ];
    __shared__ float wsh[6][4];
    const int n = S.n;
    const int h = tid >> 8, t = tid & 255, w = (tid >> 6) & 3, lane = tid & 63;
    // every load in flight at once: the states for LDS (512 threads), the queries
    constexpr int NS = (D + 511) / 512;
    float st[6][NS];
    #pragma unroll
    for (int s = 0; s < 6; s++)
        #pragma unroll
        for (int k = 0; k < NS; k++) {
            // branch-free: absent states read state 0 (never used), the D tail is clamped
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
    // two passes over the LDS states, d_mix's order: the sums of squares, then
    // the dots of rms(x) with the queries (a one-pass rs * dot(x, q) is not
    // bit-equal and moved a long-context greedy split to a large gap)
    constexpr int NQ = 3 * MQ;
    const float * sh = sv + 3 * h * D + t;                // this half's states, this thread's column
    float r[NQ];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        r[s] = 0.f;
        #pragma unroll
        for (int k = 0; k < NV; k++) { const float v = sh[s * D + 256 * k]; r[s] += v * v; }
        r[s] = wsum_xor(r[s]);
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
    for (int q = 0; q < NQ; q++) r[q] = wsum_xor(r[q]);
    if (lane == 0)
        #pragma unroll
        for (int q = 0; q < NQ; q++) red[(h * 4 + w) * NQ + q] = r[q];
    __syncthreads();
    // lanes 0 .. MQ-1 each the softmax of query j = tid (static selects): d_mix's values
    if (tid < MQ) {
        float v[6];
        #pragma unroll
        for (int s = 0; s < 6; s++) {
            float a = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; i++) a += red[((s / 3) * 4 + i) * NQ + (s % 3) * MQ + tid];   // LDS index, j = tid
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

// ---- control(cur, msg): g = [rms(cur); rms(msg)] . down; out = (1 - tanh(g.forget)) * cur + tanh(g.gain) * msg
// wgt / wft: gain / forget transposed to [GRANK][D] (the persistent kernel's coalesced copy)
struct cjob { const float * cur, * msg, * wd, * wg, * wf; float * g, * out; const float * wgt, * wft; };
// [D][GRANK] -> [GRANK][D]
static __global__ void k_ctrl_t(const float * s, float * d) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= D) return;
    for (int k = 0; k < GRANK; k++) d[(size_t) k * D + i] = s[(size_t) i * GRANK + k];
}
struct cjobs { cjob j[3]; };
// virtual block (r, y): r < GRANK, y the job (NT 256)
__device__ __forceinline__ void d_ctrl_down(const cjob & c, int r, int tid, bool act) {
    __shared__ float red[4 * 2];
    float ss[2] = {0.f, 0.f};
    if (act) for (int i = tid; i < D; i += 256) { ss[0] += c.cur[i] * c.cur[i]; ss[1] += c.msg[i] * c.msg[i]; }
    bsum<2>(ss, red, 4);
    const float rc = rsqrtf(ss[0] / D + 1e-6f), rm = rsqrtf(ss[1] / D + 1e-6f);
    const float * w = c.wd + (size_t) r * 2 * D;
    float a[1] = {0.f};
    if (act) for (int i = tid; i < D; i += 256) a[0] += w[i] * (c.cur[i] * rc) + w[D + i] * (c.msg[i] * rm);
    bsum<1>(a, red, 4);
    if (tid == 0) c.g[r] = a[0];
    __syncthreads();
}
// virtual block (bx, y): bx < D / 256 (NT 256)
__device__ __forceinline__ void d_ctrl_up(const cjob & c, int bx, int tid, bool act) {
    __shared__ float g[GRANK];
    __syncthreads();
    if (tid < GRANK) g[tid] = c.g[tid];
    __syncthreads();
    const int i = bx * 256 + tid;
    if (act && i < D) {
        float ga = 0.f, fo = 0.f;
        // the rows as float4, 4 + 4 in flight per chunk, the same k order
        const float4 * wg4 = (const float4 *) (c.wg + (size_t) i * GRANK), * wf4 = (const float4 *) (c.wf + (size_t) i * GRANK);
        #pragma unroll 1
        for (int c0 = 0; c0 < GRANK / 4; c0 += 4) {
            float4 vg[4], vf[4];
            #pragma unroll
            for (int q = 0; q < 4; q++) { vg[q] = wg4[c0 + q]; vf[q] = wf4[c0 + q]; }
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const int k = 4 * (c0 + q);
                ga += g[k] * vg[q].x; fo += g[k] * vf[q].x;
                ga += g[k + 1] * vg[q].y; fo += g[k + 1] * vf[q].y;
                ga += g[k + 2] * vg[q].z; fo += g[k + 2] * vf[q].z;
                ga += g[k + 3] * vg[q].w; fo += g[k + 3] * vf[q].w;
            }
        }
        const float x = c.cur[i];
        c.out[i] = (x - tanhf(fo) * x) + tanhf(ga) * c.msg[i];
    }
}

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

// ---- several Q8 GEMVs in one launch: y[N] = W[N][K] . a; 32 rows per 512-thread block
// (yp: optional second copy of the outputs, the TP2 peer inbox)
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

// ---- per local head, after the projections: Q / R clip and tau, the bias table, K / V conv + clip / norm into the cache
// (NT 128, virtual blocks HQL + 2 HKVL)
struct hpost {
    const float * qr, * rr, * sk, * kr, * vr, * dk, * dv;     // raw projections (dk / dv null on SWA layers)
    const float * qclip, * rclip, * relb, * kconv, * vconv;
    float * hist;                                            // this slot's [3][CHL]
    float * q, * tb;                                         // out: q [QL], table [HQL][TBW]
    __half * kc, * vc;                                       // the cache rows to write
    float tau, qk_clip, rel_clip, eps;
};
// dmax: the table entries computed (TBW; the persistent kernel: the ones pos can read)
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
        if (act) a.q[i] = x * a.tau;
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

// ---- one query row: softmax(q.k / sqrt(HD) + table[d]) with a zero sink, times
// sigmoid(gate), into int8 planes for attn_output. A local q head's keys are
// split over ATT_SPL virtual blocks of 8 waves (512 threads): wave vw of the
// ATT_NW takes keys lo + vw, lo + vw + ATT_NW, ... (online softmax), writes
// (m, l, o[HD]) to part; the head's last block to finish (cnt) merges the
// ATT_NW partials in wave order. Both paths run this body: the same bits.
constexpr int ATT_SPL = 5, ATT_NW = 8 * ATT_SPL, ATT_PW = HD + 2;
struct attn_args {
    const float * q, * tb, * gate;
    const __half * kc, * vc;      // cache base: row r at r * KVL
    int pos, full;
    unsigned char * oq; float * os;
    float * part;                 // [HQL][ATT_NW][ATT_PW]
    int * cnt;                    // [HQL], zero between uses (the merger resets it)
};
__device__ __forceinline__ void d_attn(const attn_args & a, int h, int sp, int tid) {
    __shared__ float wm[ATT_NW], wl[ATT_NW];
    __shared__ int last;
    const int kvh = h / (HQ / HKV), wave = tid >> 6, lane = tid & 63, vw = sp * 8 + wave;
    {
        const float scale = 1.f / sqrtf((float) HD);
        const float q0 = a.q[h * HD + 2 * lane], q1 = a.q[h * HD + 2 * lane + 1];
        const int lo = a.full ? 0 : max(0, a.pos - (WIN - 1));
        float m = -INFINITY, l = 0.f, o0 = 0.f, o1 = 0.f;
        for (int j = lo + vw; j <= a.pos; j += ATT_NW) {
            const int row = a.full ? j : j % PF_RING;
            const __half2 k2 = *(const __half2 *) (a.kc + (size_t) row * KVL + kvh * HD + 2 * lane);
            const __half2 v2 = *(const __half2 *) (a.vc + (size_t) row * KVL + kvh * HD + 2 * lane);
            float s = wsum_xor(q0 * __low2float(k2) + q1 * __high2float(k2));
            const int d = min(a.pos - j, REL_EXT);
            s = s * scale + a.tb[h * TBW + d];
            const float mn = fmaxf(m, s), c = expf(m - mn), p = expf(s - mn);
            l = l * c + p;
            o0 = o0 * c + p * __low2float(v2);
            o1 = o1 * c + p * __high2float(v2);
            m = mn;
        }
        float * P = a.part + ((size_t) h * ATT_NW + vw) * ATT_PW;
        if (lane == 0) { P[0] = m; P[1] = l; }
        P[2 + 2 * lane] = o0;
        P[3 + 2 * lane] = o1;
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int old = atomicAdd(&a.cnt[h], 1);
        last = old == ATT_SPL - 1;
        if (last) { a.cnt[h] = 0; __threadfence(); }       // acquire: the other blocks' partials
    }
    __syncthreads();
    if (!last) return;                                     // block-uniform
    const float * P = a.part + (size_t) h * ATT_NW * ATT_PW;
    if (tid < ATT_NW) { wm[tid] = P[tid * ATT_PW]; wl[tid] = P[tid * ATT_PW + 1]; }
    __syncthreads();
    if (tid < HD) {
        const int d = tid;
        float ov[ATT_NW];
        #pragma unroll
        for (int w = 0; w < ATT_NW; w++) ov[w] = P[w * ATT_PW + 2 + d];
        float M = 0.f;                                  // the sink logit
        for (int w = 0; w < ATT_NW; w++) M = fmaxf(M, wm[w]);
        float L = expf(-M), O = 0.f;
        #pragma unroll
        for (int w = 0; w < ATT_NW; w++) { const float e = expf(wm[w] - M); L += wl[w] * e; O += ov[w] * e; }
        const int i = h * HD + d;
        const float y = O / L * sigm(a.gate[i]);
        quant32(y, a.oq, a.os, i);
    }
    __syncthreads();
}

// ---- the TP2 sum of the two cards' partials, then a short conv (no dynamic taps) with its history;
// blk null: x += a; else blk = x, x = a (a new residual block). Element i = bx * 256 + t.
__device__ __forceinline__ void d_xconv(const float * own, const float * peer, const float * w, float * hs,
                                        float * x, float * blk, int i) {
    if (i >= D) return;
    const float u = own[i] + __builtin_nontemporal_load(peer + i);
    const float h0 = hs[i], h1 = hs[CHL + i], h2 = hs[2 * CHL + i];
    const float * wi = w + (size_t) i * NCONV;
    const float o = wi[0] * h0 + wi[1] * h1 + wi[2] * h2 + wi[3] * u;
    hs[i] = h1; hs[CHL + i] = h2; hs[2 * CHL + i] = u;
    const float y = (u + o) * (1.f / sqrtf((float) NCONV + 1.f));
    if (blk) { blk[i] = x[i]; x[i] = y; }
    else x[i] = x[i] + y;
}
// d_xconv over elements t, t + NT, ... of one block, all loads first (latency bound)
template <int NT>
__device__ __forceinline__ void d_xconv_blk(const float * own, const float * peer, const float * w, float * hs,
                                            float * x, float * blk, int t) {
    constexpr int NE_ = (D + NT - 1) / NT;
    float u[NE_], h0[NE_], h1[NE_], h2[NE_], xv[NE_];
    float4 wi[NE_];
    #pragma unroll
    for (int k = 0; k < NE_; k++) {
        const int i = min(t + NT * k, D - 1);          // clamped: unconditional loads, the stores guarded
        u[k] = own[i] + __builtin_nontemporal_load(peer + i);
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

// ---- router: sigmoid(f . W), top NE_USED by p + bias, weights p / sum(p) * scale
// one wave per expert (the 64-lane strided sum + xor tree); ps[e] = p, ps[NE + e] = p + bias
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
// top NE_USED by p + bias (first max wins: ties to the lower id), weights
// p / sum(p) * scale with the sum in selection order; one wave, 2 experts per lane
__device__ __forceinline__ void d_router_pick(const float * ps, float scale, int * ids, float * wts, int lane) {
    const float pa = ps[lane], pb = ps[lane + 64];
    float sa = ps[NE + lane], sb = ps[NE + lane + 64];
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
    if (lane < NE_USED) { ids[lane] = idk; wts[lane] = pk / sum * scale; }
}

// gate_up rows are interleaved per 32 at load (gu_interleave): tile T of 64 rows holds
// gate rows 32T..32T+31 then the matching up rows; gate row j sits at gu_row(j), its up row 32 later
__device__ __host__ __forceinline__ int gu_row(int j) { return (j / 32) * 64 + j % 32; }
static __global__ __launch_bounds__(256) void k_gu_interleave(const char * sv, const __half * ss, char * dv, __half * ds) {
    const int p = blockIdx.x, e = blockIdx.y, r = p % 64;
    const int src = r < 32 ? (p / 64) * 32 + r : FFL + (p / 64) * 32 + r - 32;
    const size_t so = (size_t) e * FFE + src, dof = (size_t) e * FFE + p;
    const uint4 * a = (const uint4 *) (sv + so * D);
    uint4 * b = (uint4 *) (dv + dof * D);
    for (int i = threadIdx.x; i < D / 16; i += 256) b[i] = a[i];
    for (int i = threadIdx.x; i < D / 32; i += 256) ds[dof * (D / 32) + i] = ss[so * (D / 32) + i];
}
// ---- the selected experts' local gate | up rows: h[k][FFE] (interleaved per 32, see gu_row);
// 512-thread block bx of expert slot k
__device__ __forceinline__ void d_moe_up(const char * wv, const __half * ws, const int * ids,
                                         const unsigned char * av, const float * as, float * h, int bx, int k) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int e = __builtin_amdgcn_readfirstlane(ids[k]);
    const size_t base = (size_t) e * FFE;
    const int r0 = bx * 32 + wave * 4;
    float acc[1][4];
    qz_gemv_rows_t<1, 4>(av, as, D, wv + base * D, ws + base * (D / 32), FFE, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 4; s++) if (r0 + s < FFE) h[k * FFE + r0 + s] = acc[0][s];
}
// ---- SITU on the local halves -> int8 planes per expert: [k][FFL]; element j of slot k
// (32-lane halves of a wave must hold 32 consecutive j: quant32)
__device__ __forceinline__ float situ_v(float g, float u, float beta, float lb) {
    return beta * tanhf(g / beta) * sigm(g) * (lb * tanhf(u / lb));
}
__device__ __forceinline__ void d_situ(const float * h, float beta, float lb, unsigned char * q, float * qs, int k, int j) {
    const int gi = gu_row(j);
    const float g = h[k * FFE + gi], u = h[k * FFE + gi + 32];
    quant32(situ_v(g, u, beta, lb), q + k * FFL, qs + k * (FFL / 32), j);
}
// ---- sum over the selected experts of w_k * down_e . act_k, local contraction half (yp: peer inbox copy)
__device__ __forceinline__ void d_moe_down(const char * wv, const __half * ws, const int * ids, const float * wts,
                                           const unsigned char * aq, const float * as, float * y, float * yp, int bx) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int r0 = bx * 32 + wave * 4;
    float tot[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < NE_USED; k++) {
        const int e = __builtin_amdgcn_readfirstlane(ids[k]);
        const size_t base = (size_t) e * D;
        float acc[1][4];
        qz_gemv_rows_t<1, 4>(aq + k * FFL, as + k * (FFL / 32), FFL, wv + base * FFL, ws + base * (FFL / 32), D, r0, lane, acc);
        const float wk = wts[k];
        #pragma unroll
        for (int s = 0; s < 4; s++) tot[s] += wk * acc[0][s];
    }
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 4; s++) if (r0 + s < D) { y[r0 + s] = tot[s]; if (yp) yp[r0 + s] = tot[s]; }
}
static __global__ __launch_bounds__(512) void k_moe_down(const char * wv, const __half * ws, const int * ids, const float * wts,
                                                         const unsigned char * aq, const float * as, float * y) {
    d_moe_down(wv, ws, ids, wts, aq, as, y, nullptr, blockIdx.x);
}

static __global__ void k_softcap(float * y, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(y[i] / cap);
}

#pragma clang fp contract(fast)
