// inkling2 device kernels, correctness path: one token per launch
// sequence, host-orchestrated on both cards (docs/inkling2.md). Activations
// into a Q8 GEMV are int8 planes (vals [K] + f32 scales [K/32]), as
// quartz/kernels/gemv.h reads them.
#pragma once

constexpr int HQL = HQ / 2, HKVL = HKV / 2;            // local heads per card
constexpr int QL = HQL * HD, KVL = HKVL * HD;          // 1536, 512
constexpr int FFL = FFE / 2;                           // local expert FF: 576
constexpr int CHL = 2 * KVL + 2 * D;                   // local conv history channels: K | V | attn post | ffn post
constexpr int H_K = 0, H_V = KVL, H_A = 2 * KVL, H_F = 2 * KVL + D;
constexpr int TBW = REL_EXT + 1;                       // bias table width (the zero column past the extent)

// sum of N values over the block (blockDim a multiple of 64, <= 1024); every thread gets the sums
template <int N> __device__ __forceinline__ void bsum(float (&v)[N], float * red) {
    #pragma unroll
    for (int k = 0; k < N; k++)
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) v[k] += __shfl_xor(v[k], off, 64);
    const int w = threadIdx.x >> 6, nw = blockDim.x >> 6;
    __syncthreads();
    if ((threadIdx.x & 63) == 0)
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

// token row of the Q8 embedding -> rmsnorm * w
static __global__ __launch_bounds__(256) void k_embed(const char * ev, const __half * es, int tok, const float * w, float eps, float * x) {
    __shared__ float red[16];
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        v[k] = (float) ev[(size_t) tok * D + i] * __half2float(es[(size_t) tok * (D / 32) + i / 32]);
        ss[0] += v[k] * v[k];
    }
    bsum<1>(ss, red);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        x[i] = v[k] * r * w[i];
    }
}

// mix(states, query): softmax over the rmsnormed states of Sn . q_j / sqrt(D); out[j] = sum w * Sn
struct sptrs { const float * p[6]; int n; };
static __global__ __launch_bounds__(256) void k_mix(sptrs S, const float * qy, int m, float * out) {
    __shared__ float red[4 * 24];
    __shared__ float wsh[6][4];
    const int n = S.n, tid = threadIdx.x;
    float ss[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int i = tid; i < D; i += 256)
        #pragma unroll
        for (int s = 0; s < 6; s++) if (s < n) { const float v = S.p[s][i]; ss[s] += v * v; }
    bsum<6>(ss, red);
    float rs[6];
    #pragma unroll
    for (int s = 0; s < 6; s++) rs[s] = rsqrtf(ss[s] / D + 1e-6f);
    float sc[24];
    #pragma unroll
    for (int k = 0; k < 24; k++) sc[k] = 0.f;
    for (int i = tid; i < D; i += 256)
        #pragma unroll
        for (int s = 0; s < 6; s++) if (s < n) {
            const float v = S.p[s][i] * rs[s];
            #pragma unroll
            for (int j = 0; j < 4; j++) if (j < m) sc[s * 4 + j] += v * qy[(size_t) j * D + i];
        }
    bsum<24>(sc, red);
    if (tid < m) {
        const int j = tid;
        const float scale = 1.f / sqrtf((float) D);
        float mx = -INFINITY;
        for (int s = 0; s < n; s++) mx = fmaxf(mx, sc[s * 4 + j] * scale);
        float sum = 0.f;
        for (int s = 0; s < n; s++) { const float e = expf(sc[s * 4 + j] * scale - mx); wsh[s][j] = e; sum += e; }
        for (int s = 0; s < n; s++) wsh[s][j] /= sum;
    }
    __syncthreads();
    for (int i = tid; i < D; i += 256)
        for (int j = 0; j < m; j++) {
            float o = 0.f;
            #pragma unroll
            for (int s = 0; s < 6; s++) if (s < n) o += wsh[s][j] * S.p[s][i] * rs[s];
            out[(size_t) j * D + i] = o;
        }
}

// control(cur, msg): g = [rms(cur); rms(msg)] . down; out = (1 - tanh(g.forget)) * cur + tanh(g.gain) * msg
struct cjob { const float * cur, * msg, * wd, * wg, * wf; float * g, * out; };
struct cjobs { cjob j[3]; };
static __global__ __launch_bounds__(256) void k_ctrl_down(cjobs J) {
    __shared__ float red[4 * 2];
    const cjob & c = J.j[blockIdx.y];
    const int r = blockIdx.x, tid = threadIdx.x;
    float ss[2] = {0.f, 0.f};
    for (int i = tid; i < D; i += 256) { ss[0] += c.cur[i] * c.cur[i]; ss[1] += c.msg[i] * c.msg[i]; }
    bsum<2>(ss, red);
    const float rc = rsqrtf(ss[0] / D + 1e-6f), rm = rsqrtf(ss[1] / D + 1e-6f);
    const float * w = c.wd + (size_t) r * 2 * D;
    float a[1] = {0.f};
    for (int i = tid; i < D; i += 256) a[0] += w[i] * (c.cur[i] * rc) + w[D + i] * (c.msg[i] * rm);
    bsum<1>(a, red);
    if (tid == 0) c.g[r] = a[0];
}
static __global__ __launch_bounds__(256) void k_ctrl_up(cjobs J) {
    __shared__ float g[GRANK];
    const cjob & c = J.j[blockIdx.y];
    if (threadIdx.x < GRANK) g[threadIdx.x] = c.g[threadIdx.x];
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= D) return;
    float ga = 0.f, fo = 0.f;
    #pragma unroll 8
    for (int k = 0; k < GRANK; k++) { ga += g[k] * c.wg[(size_t) i * GRANK + k]; fo += g[k] * c.wf[(size_t) i * GRANK + k]; }
    const float x = c.cur[i];
    c.out[i] = (x - tanhf(fo) * x) + tanhf(ga) * c.msg[i];
}

// rmsnorm(x) * w -> optional f32 y and the int8 planes
struct njob { const float * x, * w; float * y; unsigned char * q; float * qs; };
struct njobs { njob j[3]; };
static __global__ __launch_bounds__(256) void k_norm_quant(njobs J, float eps) {
    __shared__ float red[4];
    const njob & a = J.j[blockIdx.x];
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) { v[k] = a.x[threadIdx.x + k * 256]; ss[0] += v[k] * v[k]; }
    bsum<1>(ss, red);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        const float y = v[k] * r * a.w[i];
        if (a.y) a.y[i] = y;
        quant32(y, a.q, a.qs, i);
    }
}

// several Q8 GEMVs in one launch: y[N] = W[N][K] . a; 32 rows per 512-thread block
struct gjob { const char * wv; const __half * ws; const unsigned char * av; const float * as; float * y; int N, K, g1; };
struct gjobs { gjob j[8]; int n; };
static __global__ __launch_bounds__(512) void k_gemv(gjobs J) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    int k = 0;
    while (k + 1 < J.n && (int) blockIdx.x >= J.j[k].g1) k++;
    const gjob & j = J.j[k];
    const int g = (int) blockIdx.x - (k ? J.j[k - 1].g1 : 0);
    const int r0 = g * 32 + wave * 4;
    float acc[1][4];
    qz_gemv_rows_t<1, 4>(j.av, j.as, j.K, j.wv, j.ws, j.N, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 4; s++) if (r0 + s < j.N) j.y[r0 + s] = acc[0][s];
}

// per local head, after the projections: Q / R clip and tau, the bias table, K / V conv + clip / norm into the cache
struct hpost {
    const float * qr, * rr, * sk, * kr, * vr, * dk, * dv;     // raw projections (dk / dv null on SWA layers)
    const float * qclip, * rclip, * relb, * kconv, * vconv;
    float * hist;                                            // this slot's [3][CHL]
    float * q, * tb;                                         // out: q [QL], table [HQL][TBW]
    __half * kc, * vc;                                       // the cache rows to write
    float tau, qk_clip, rel_clip, eps;
};
static __global__ __launch_bounds__(128) void k_head_post(hpost a) {
    __shared__ float red[2 * 1];
    __shared__ float rv[DREL];
    const int b = blockIdx.x, t = threadIdx.x;
    if (b < HQL) {
        const int h = b, i = h * HD + t;
        float x = a.qr[i] * a.qclip[i];
        float n2[1] = {x * x};
        bsum<1>(n2, red);
        const float nr = sqrtf(n2[0]) + 1e-6f;
        x *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        a.q[i] = x * a.tau;
        float r = t < DREL ? a.rr[h * DREL + t] * a.rclip[h * DREL + t] : 0.f;
        float r2[1] = {r * r};
        bsum<1>(r2, red);
        const float rn = sqrtf(r2[0]) + 1e-6f;
        if (t < DREL) rv[t] = r * fminf(fmaxf(a.rel_clip / rn, 0.f), 1.f) * a.tau;
        __syncthreads();
        const float sink = a.sk[h];
        for (int d = t; d < TBW; d += 128) {
            float s = 0.f;
            if (d < REL_EXT)
                #pragma unroll
                for (int k = 0; k < DREL; k++) s += a.relb[d * DREL + k] * rv[k];
            a.tb[h * TBW + d] = s - sink;
        }
        return;
    }
    const bool isk = b < HQL + HKVL;
    const int hk = isk ? b - HQL : b - HQL - HKVL, ch = hk * HD + t;
    const float * raw = isk ? a.kr : a.vr;
    const float * dyn = isk ? a.dk : a.dv;
    const float * w = (isk ? a.kconv : a.vconv) + (size_t) ch * NCONV;
    float * hs = a.hist + (isk ? H_K : H_V) + ch;
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
    float y = (u + o) * (dyn ? 1.f / sqrtf(2.f * NCONV + 1.f) : 1.f / sqrtf((float) NCONV + 1.f));
    float n2[1] = {y * y};
    bsum<1>(n2, red);
    if (isk) {
        const float nr = sqrtf(n2[0]) + 1e-6f;
        y *= fminf(fmaxf(a.qk_clip / nr, 0.f), 1.f);
        a.kc[ch] = __float2half(y);
    } else {
        y *= rsqrtf(n2[0] / HD + a.eps);
        a.vc[ch] = __float2half(y);
    }
}

// one query row, one block per local q head: softmax(q.k / sqrt(HD) + table[d]) with a zero sink,
// times sigmoid(gate), into int8 planes for attn_output
struct attn_args {
    const float * q, * tb, * gate;
    const __half * kc, * vc;      // cache base: row r at r * KVL
    int pos, full;
    unsigned char * oq; float * os;
};
static __global__ __launch_bounds__(256) void k_attn(attn_args a) {
    __shared__ float wm[4], wl[4], wo[4][HD];
    const int h = blockIdx.x, kvh = h / (HQ / HKV), wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const float scale = 1.f / sqrtf((float) HD);
    const float q0 = a.q[h * HD + 2 * lane], q1 = a.q[h * HD + 2 * lane + 1];
    const int lo = a.full ? 0 : max(0, a.pos - (WIN - 1));
    float m = -INFINITY, l = 0.f, o0 = 0.f, o1 = 0.f;
    for (int j = lo + wave; j <= a.pos; j += 4) {
        const int row = a.full ? j : j % WIN;
        const __half2 k2 = *(const __half2 *) (a.kc + (size_t) row * KVL + kvh * HD + 2 * lane);
        float s = q0 * __low2float(k2) + q1 * __high2float(k2);
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) s += __shfl_xor(s, off, 64);
        const int d = min(a.pos - j, REL_EXT);
        s = s * scale + a.tb[h * TBW + d];
        const float mn = fmaxf(m, s), c = expf(m - mn), p = expf(s - mn);
        const __half2 v2 = *(const __half2 *) (a.vc + (size_t) row * KVL + kvh * HD + 2 * lane);
        l = l * c + p;
        o0 = o0 * c + p * __low2float(v2);
        o1 = o1 * c + p * __high2float(v2);
        m = mn;
    }
    if (lane == 0) { wm[wave] = m; wl[wave] = l; }
    wo[wave][2 * lane] = o0;
    wo[wave][2 * lane + 1] = o1;
    __syncthreads();
    if (threadIdx.x < HD) {
        const int d = threadIdx.x;
        float M = 0.f;                                  // the sink logit
        for (int w = 0; w < 4; w++) M = fmaxf(M, wm[w]);
        float L = expf(-M), O = 0.f;
        for (int w = 0; w < 4; w++) { const float e = expf(wm[w] - M); L += wl[w] * e; O += wo[w][d] * e; }
        const int i = h * HD + d;
        const float y = O / L * sigm(a.gate[i]);
        quant32(y, a.oq, a.os, i);
    }
}

// the TP2 sum of the two cards' partials, then a short conv (no dynamic taps) with its history;
// mode 0: x += a; mode 1: blk = x, x = a (a new residual block)
static __global__ __launch_bounds__(256) void k_xconv(const float * own, const float * peer, const float * w, float * hs,
                                                      float * x, float * blk) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= D) return;
    const float u = own[i] + peer[i];
    const float h0 = hs[i], h1 = hs[CHL + i], h2 = hs[2 * CHL + i];
    const float * wi = w + (size_t) i * NCONV;
    const float o = wi[0] * h0 + wi[1] * h1 + wi[2] * h2 + wi[3] * u;
    hs[i] = h1; hs[CHL + i] = h2; hs[2 * CHL + i] = u;
    const float y = (u + o) * (1.f / sqrtf((float) NCONV + 1.f));
    if (blk) { blk[i] = x[i]; x[i] = y; }
    else x[i] = x[i] + y;
}

// router: sigmoid(f . W), top NE_USED by p + bias, weights p / sum(p) * scale
static __global__ __launch_bounds__(1024) void k_router(const float * f, const float * wr, const float * bias, float scale,
                                                        int * ids, float * wts) {
    __shared__ float p[NE], sel[NE];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    for (int e = wave; e < NE; e += 16) {
        float s = 0.f;
        for (int i = lane; i < D; i += 64) s += f[i] * wr[(size_t) e * D + i];
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) s += __shfl_xor(s, off, 64);
        if (lane == 0) { p[e] = sigm(s); sel[e] = p[e] + bias[e]; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        int id[NE_USED];
        for (int k = 0; k < NE_USED; k++) {
            int b = -1;
            for (int e = 0; e < NE; e++) if (sel[e] > -INFINITY && (b < 0 || sel[e] > sel[b])) b = e;
            id[k] = b;
            sel[b] = -INFINITY;
            sum += p[b];
        }
        sum += 1e-20f;
        for (int k = 0; k < NE_USED; k++) { ids[k] = id[k]; wts[k] = p[id[k]] / sum * scale; }
    }
}

// the selected experts' local gate | up rows: h[k][FFE] (gate FFL then up FFL)
static __global__ __launch_bounds__(512) void k_moe_up(const char * wv, const __half * ws, const int * ids,
                                                       const unsigned char * av, const float * as, float * h) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, k = blockIdx.y;
    const int e = __builtin_amdgcn_readfirstlane(ids[k]);
    const size_t base = (size_t) e * FFE;
    const int r0 = blockIdx.x * 32 + wave * 4;
    float acc[1][4];
    qz_gemv_rows_t<1, 4>(av, as, D, wv + base * D, ws + base * (D / 32), FFE, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 4; s++) if (r0 + s < FFE) h[k * FFE + r0 + s] = acc[0][s];
}
// SITU on the local halves -> int8 planes per expert: [k][FFL]
static __global__ __launch_bounds__(FFL) void k_situ(const float * h, float beta, float lb, unsigned char * q, float * qs) {
    const int k = blockIdx.x, j = threadIdx.x;
    const float g = h[k * FFE + j], u = h[k * FFE + FFL + j];
    const float act = beta * tanhf(g / beta) * sigm(g) * (lb * tanhf(u / lb));
    quant32(act, q + k * FFL, qs + k * (FFL / 32), j);
}
// sum over the selected experts of w_k * down_e . act_k, local contraction half
static __global__ __launch_bounds__(512) void k_moe_down(const char * wv, const __half * ws, const int * ids, const float * wts,
                                                         const unsigned char * aq, const float * as, float * y) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int r0 = blockIdx.x * 32 + wave * 4;
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
        for (int s = 0; s < 4; s++) if (r0 + s < D) y[r0 + s] = tot[s];
}

static __global__ void k_softcap(float * y, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(y[i] / cap);
}
