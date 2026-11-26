// inkling2 persistent decode. One launch per card per token runs the embedding and
// all 48 layer applications. One 512-thread block per CU, software grid barriers
// (kernels/dev.h) between the phases, TP2 sums through the peer inbox and flags of
// kernels/xchg.h. The device bodies live in ink2_kernels.h.
#pragma once

#include "kernels/xchg.h"

// fp contract(on) so these bodies round like ink2_kernels.h; contract(fast) again at the end
#pragma clang fp contract(on)

// Grid barrier. Each block has its own flag, so no atomic queue on one word. Block b
// stores k to fl[b] with release; lanes 0 .. nb-1 of wave 0 each poll one block's
// flag until it reaches k. fl holds CYC_NBMAX words, zeroed per launch; k = 1, 2, ...
constexpr int CYC_NBMAX = 64;
__device__ __forceinline__ void cyc_barrier(unsigned int * fl, int k) {
    __syncthreads();
    if (threadIdx.x < 64) {
        if (threadIdx.x == 0) {
            __threadfence();
            __hip_atomic_store(&fl[blockIdx.x], (unsigned int) k, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        if (threadIdx.x < gridDim.x)
            while (__hip_atomic_load(&fl[threadIdx.x], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned int) k) { }
        __threadfence();
    }
    __syncthreads();
}

// one layer application (built by ink2_engine::plan)
struct appk {
    int slot, ip, full, msg;
    int pre;                      // 1: P = X. 2: OUT[ob] = X, then X = P (a loop pass starts)
    float * pre_out;
    // attention half
    int has_res, mres_m;
    sptrs mres; const float * qres; float * res_out;
    sptrs mmsg; const float * qmsg; float * msg_out;
    cjobs ca;                     // the q / k / v controllers, msg layers only
    njobs na; int nna;
    gjobs ga; int nga;            // nga: blocks
    hpost hp;                     // kc / vc point at the slot's cache base; tau is 1
    attn_args at;                 // pos per token
    gjobs go; int ngo;            // attn_output -> PART[0]
    const float * pconv; float * hsa, * blk;
    // ffn half
    sptrs mff; const float * qff;
    cjobs cb;                     // the ffn controller (msg)
    njobs nb;
    const float * gin, * pb;
    const char * guv; const __half * gus; const char * dnv; const __half * dns;
    const float * fconv; float * hsf;
    float * tap;                  // DSpark: X after the application's leading xconv (null if none)
};

// static per card in device memory; the token's tok / pos / tau are kernel arguments
struct cyc_args {
    const appk * tab; int napp;
    const char * ev; const __half * es; const float * embd_norm; float eps;
    float * X, * P, * PART0, * PART1;
    float * F, * FN, * H, * WTS; int * IDS;
    float * GU;                   // the SITU values, NE_USED * FFL; also the head's logit buffer
    int * gcnt;                   // tasks done per quant group (NE_USED * FFL / 32), reset to 0
    unsigned char * aq3, * hq; float * as3, * hs;
    float route_scale, beta, lb;
    // the head launch: mix into F, norm into aq3 / as3 (block 0), then the vocab GEMV
    // into hg.y, with the softcap on the first nvalid rows
    sptrs hmix; const float * hq_out; njob hn; gjob hg; int nvalid; float softcap;
    // TP exchange (the names kernels/xchg.h expects; *_px[k - 1] is my slot in peer k)
    static constexpr int NC = QZ_NC;
    float * xin_mine, * xin_px[QZ_NC - 1];
    unsigned int * xflag_mine, * xflag_px[QZ_NC - 1], * xseq;
    int * xerr;
};

// The card's partial is in the peer inbox: pushed, fenced, grid barrier passed. The
// calling block raises its flag (block 0 only) and waits for the peer's.
template <class XA> __device__ __forceinline__ void cyc_xwait(const XA & A, unsigned int tok) {
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) qz_xflag(A, tok);
        qz_xwait(A, tok);
    }
    __syncthreads();
}

// Phase wrappers. The compiler sees the table's pointers as generic, so these pass them
// through qz_gp() (kernels/dev.h); the inlined bodies infer global loads and stores. The
// table and args are read through the constant address space, as scalar loads.
#ifdef __HIP_DEVICE_COMPILE__
#define CK const __attribute__((address_space(4)))
#else
#define CK const
#endif
constexpr int CYC_LDS = 6 * D * 4;           // dynamic LDS: the mix states, also the attention scratch
static_assert(FL_LDS1 <= CYC_LDS, "the attention fits the dynamic LDS");
extern __shared__ float cyc_lds[];
#define CYC_NI __device__ __forceinline__
template <int MQ> CYC_NI void c_mix(CK sptrs * S, const float * q, float * o, int tid) {
    asm volatile ("" : "+v"(tid));
    sptrs L;
    L.n = S->n;
    #pragma unroll
    for (int s = 0; s < 6; s++) L.p[s] = qz_gp(S->p[s]);
    d_mix_lds<MQ>(L, qz_gp(q), qz_gp(o), tid, tid < 256, cyc_lds);
}
CYC_NI void c_mix1(CK sptrs * S, const float * q, float * o, int tid) { c_mix<1>(S, q, o, tid); }
// bsum<N>(v, red, 4) for each 256-thread half of the block separately (red: 2 * 4 * N)
template <int N> CYC_NI void hsum(float (&v)[N], float * red) {
    #pragma unroll
    for (int k = 0; k < N; k++) v[k] = qz_wsum_dpp(v[k]);
    const int w = (threadIdx.x >> 6) & 3, h = threadIdx.x >> 8;
    __syncthreads();
    if ((threadIdx.x & 63) == 0)
        #pragma unroll
        for (int k = 0; k < N; k++) red[(h * 4 + w) * N + k] = v[k];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < N; k++) {
        float s = 0.f;
        for (int i = 0; i < 4; i++) s += red[(h * 4 + i) * N + k];
        v[k] = s;
    }
}
// The control-down rows, one pass per block. The blocks split into NJ groups, one job
// each, so every rms runs once. A half block takes 2 rows, with 256-thread strided sums
// and then a 4-wave reduction.
template <int NJ> CYC_NI void c_ctrl_down_all(CK cjob * cj, int tid) {
    asm volatile ("" : "+v"(tid));
    __shared__ float red[2 * 4 * 2];
    constexpr int NV = D / 256, MR = 2;
    const int bpj = gridDim.x / NJ, j = blockIdx.x / bpj, bj = blockIdx.x % bpj, h = tid >> 8, t = tid & 255;
    if (j >= NJ) return;                                   // leftover blocks, block-uniform
    const float * cur = qz_gp(cj[j].cur), * msg = qz_gp(cj[j].msg), * wd = qz_gp(cj[j].wd);
    float * g = qz_gp(cj[j].g);
    for (int r0 = bj; r0 < GRANK; r0 += 2 * MR * bpj) {
        float x[NV], y[NV], wa[MR][NV], wb[MR][NV];
        #pragma unroll
        for (int k = 0; k < NV; k++) { x[k] = cur[t + 256 * k]; y[k] = msg[t + 256 * k]; }
        #pragma unroll
        for (int m = 0; m < MR; m++) {
            const int r = min(r0 + bpj * (h + 2 * m), GRANK - 1);
            const float * w = wd + (size_t) r * 2 * D;
            #pragma unroll
            for (int k = 0; k < NV; k++) { wa[m][k] = w[t + 256 * k]; wb[m][k] = w[D + t + 256 * k]; }
        }
        float ss[2] = {0.f, 0.f};
        #pragma unroll
        for (int k = 0; k < NV; k++) { ss[0] += x[k] * x[k]; ss[1] += y[k] * y[k]; }
        hsum<2>(ss, red);
        const float rc = rsqrtf(ss[0] / D + 1e-6f), rm = rsqrtf(ss[1] / D + 1e-6f);
        float a[MR];
        #pragma unroll
        for (int m = 0; m < MR; m++) {
            a[m] = 0.f;
            #pragma unroll
            for (int k = 0; k < NV; k++) a[m] += wa[m][k] * (x[k] * rc) + wb[m][k] * (y[k] * rm);
        }
        hsum<MR>(a, red);
        if (t < MR) {
            #pragma unroll
            for (int m = 0; m < MR; m++) {
                const int r = r0 + bpj * (h + 2 * m);
                if (t == m && r < GRANK) g[r] = a[m];
            }
        }
    }
    __syncthreads();
}
// The control-up rows: NJ * D rows spread over all blocks, from the transposed weights
// (wgt / wft [GRANK][D], with lanes on consecutive rows, coalesced). Each item is a row
// and a sum, and every sum runs in sequential k order. The forget sums go through LDS.
template <int NJ> CYC_NI void c_ctrl_up_all(CK cjob * cj, int tid) {
    asm volatile ("" : "+v"(tid));
    __shared__ float g[NJ][GRANK], sf[256];
    for (int i = tid; i < NJ * GRANK; i += 512) g[i / GRANK][i % GRANK] = qz_gp(cj[i / GRANK].g)[i % GRANK];
    __syncthreads();
    constexpr int NR = NJ * D, KC = 32;
    const int r0 = blockIdx.x * NR / gridDim.x, r1 = (blockIdx.x + 1) * NR / gridDim.x;
    for (int c0 = r0; c0 < r1; c0 += 256) {
        const int n = min(256, r1 - c0), which = tid >= n, q = tid - (which ? n : 0);
        const int r = c0 + min(q, n - 1), j = r / D, i = r % D;
        float a = 0.f;
        if (tid < 2 * n) {
            const float * w = qz_gp(which ? cj[j].wft : cj[j].wgt) + i;
            #pragma unroll 1
            for (int k0 = 0; k0 < GRANK; k0 += KC) {
                float v[KC];
                #pragma unroll
                for (int k = 0; k < KC; k++) v[k] = w[(size_t) (k0 + k) * D];
                #pragma unroll
                for (int k = 0; k < KC; k++) a += g[j][k0 + k] * v[k];
            }
            if (which) sf[q] = a;
        }
        __syncthreads();
        if (tid < n) {
            const float x = qz_gp(cj[j].cur)[i], fo = sf[q], ga = a;
            qz_gp(cj[j].out)[i] = (x - tanhf(fo) * x) + tanhf(ga) * qz_gp(cj[j].msg)[i];
        }
        __syncthreads();
    }
}
CYC_NI void c_norm(CK njob * j, float eps, int tid) {
    asm volatile ("" : "+v"(tid));
    njob n;
    n.x = qz_gp(j->x); n.w = qz_gp(j->w); n.y = j->y ? qz_gp(j->y) : nullptr; n.q = qz_gp(j->q); n.qs = qz_gp(j->qs);
    d_norm_quant(n, eps, tid, tid < 256);
}
// Half a 32-row virtual block: 16 rows, S = 2 per wave, same rows and per-row code as
// d_gemv, split finer so a small job list balances over the grid.
CYC_NI void c_gemv2(CK gjobs * J, int hb, float * yp) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int b = hb >> 1, half = hb & 1;
    int k = 0;
    while (k + 1 < J->n && b >= J->j[k].g1) k++;
    const gjob * j = &J->j[k];
    const int g = b - (k ? J->j[k - 1].g1 : 0);
    const int r0 = g * 32 + half * 16 + wave * 2;
    const int N = j->N;
    float * y = qz_gp(j->y);
    float acc[1][2];
    qz_gemv_rows_t<1, 2>(qz_gp(j->av), qz_gp(j->as), j->K, j->wv, j->ws, N, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 2; s++) if (r0 + s < N) { y[r0 + s] = acc[0][s]; if (yp) qz_gp(yp)[r0 + s] = acc[0][s]; }
}
// The vocab GEMV: 32-row units spread over the grid, as d_gemv's rows. Rows < nv are
// stored with the softcap applied.
CYC_NI void c_head_gemv(CK gjob * J, int nv, float cap) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int N = J->N;
    float * y = qz_gp(J->y);
    #pragma unroll 1
    for (int g = blockIdx.x; g * 32 < N; g += gridDim.x) {
        const int r0 = g * 32 + wave * 4;
        float acc[1][4];
        qz_gemv_rows_t<1, 4>(qz_gp(J->av), qz_gp(J->as), J->K, J->wv, J->ws, N, r0, lane, acc);
        if (lane == 0)
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                const float v = acc[0][s];
                if (r0 + s < N) y[r0 + s] = r0 + s < nv ? cap * tanhf(v / cap) : v;
            }
    }
}

// ---- the chain group. CG blocks share block 0's serial work: xconv, the pass copies, a
// mix and its norms. Block w owns columns 64 w + l + 256 k, with l < 64 and k < D / 256.
// That is exactly wave w's share of every full-row sum in d_mix_lds / d_norm_quant, where
// each lane sums over k, qz_wsum_dpp reduces, and waves 0 .. 3 are added in order. Each
// block publishes its wave-w partials in slots, every block polls the group's slots and
// adds the CG partials in that order, so the bits match. The element-wise parts (output
// pass, xconv, copies) work on the owned columns with the same thread for the same column
// throughout, so they need no barrier.
constexpr int CG = 4, CQ = D / CG, CNK = D / 256;
static_assert(CQ == 64 * CNK && CNK * 256 == D, "chain group columns");
constexpr int CYC_NBX = CYC_NBMAX + 2 * CG;      // bst words: the grid barrier's, then the chain groups'
constexpr int CP_1 = 0, CP_2 = CP_1 + CG * 6, CP_N = CP_2 + CG * 6 * 4, CP_SZ = CP_N + CG * 4;   // partial slots per group
constexpr int CYC_NBS = CYC_NBX + 2 * 2 * CP_SZ;  // bst words: the flags, then the two groups' 64-bit slots
__device__ __forceinline__ int ccol(int w, int q) { return 64 * w + (q & 63) + 256 * (q >> 6); }
// A partial slot holds (epoch << 32 | value bits) in one 64-bit store, and a reader polls
// the slot itself until it carries the epoch, so a partial needs no fence, flag or reload.
// The slots sit in the launch's zeroed flag buffer, so a stale slot never matches.
struct cgrp { unsigned int * fl; unsigned long long * cp; int w; };     // w is this block's index within the group
__device__ __forceinline__ void cg_put(unsigned long long * p, float v, unsigned int e) {
    __hip_atomic_store(p, ((unsigned long long) e << 32) | __float_as_uint(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ float cg_get(unsigned long long * p, unsigned int e) {
    unsigned long long x;
    do { x = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); } while ((unsigned int) (x >> 32) != e);
    return __uint_as_float((unsigned int) x);
}
// Returns once every CG block has published epoch e, each partial fenced by its writer.
CYC_NI void cg_sync(const cgrp & g, unsigned int e) {
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_store(&g.fl[g.w], e, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    if (threadIdx.x < CG) {
        while (__hip_atomic_load(&g.fl[threadIdx.x], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < e) { }
        __threadfence();
    }
    __syncthreads();
}
// d_xconv_blk's values on the owned columns, thread q = tid and tid + 512. With tap, it
// also takes the new x: the DSpark feature, the residual entering the next application.
CYC_NI void c_cxconv(const float * own, const float * peer, const float * w, float * hs, float * x, float * blk, int cw, int tid,
                     float * tap = nullptr) {
    own = qz_gp(own); peer = qz_gp(peer); w = qz_gp(w); hs = qz_gp(hs); x = qz_gp(x); blk = blk ? qz_gp(blk) : nullptr; tap = tap ? qz_gp(tap) : nullptr;
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            const int i = ccol(cw, q);
            const float u = own[i] + __builtin_nontemporal_load(peer + i);
            const float h0 = hs[i], h1 = hs[CHL + i], h2 = hs[2 * CHL + i];
            const float4 wi = *(const float4 *) (w + (size_t) i * NCONV);
            const float xv = x[i];
            const float o = wi.x * h0 + wi.y * h1 + wi.z * h2 + wi.w * u;
            hs[i] = h1; hs[CHL + i] = h2; hs[2 * CHL + i] = u;
            const float y = (u + o) * (1.f / sqrtf((float) NCONV + 1.f));
            if (blk) { blk[i] = xv; x[i] = y; }
            else x[i] = xv + y;
            if (tap) tap[i] = blk ? y : xv + y;
        }
    }
}
// the pass copies on the owned columns
CYC_NI void c_cpre(int pre, float * P, float * X, float * po, int cw, int tid) {
    P = qz_gp(P); X = qz_gp(X);
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            const int i = ccol(cw, q);
            if (pre == 1) P[i] = X[i];
            else { qz_gp(po)[i] = X[i]; X[i] = P[i]; }
        }
    }
}
// d_norm_quant of NJ rows held in LDS v[j][CQ], owned columns. Wave j sums row j; the
// scaled and quantized output then spreads over all waves, one (row, 64 columns) item each.
constexpr int CN_IT = (3 * CNK + 7) / 8;      // output items per wave, NJ <= 3
CYC_NI void c_cnorm_tail(int NJ, const float * v, CK njob * nj, float eps, const cgrp & g, unsigned int & ce, int tid) {
    const int wave = tid >> 6, lane = tid & 63;
    float wv[CN_IT];                     // the norm weights in flight over the sync
    #pragma unroll
    for (int u = 0; u < CN_IT; u++) {
        const int it = wave + 8 * u;
        if (it < NJ * CNK) wv[u] = qz_gp(nj[it / CNK].w)[ccol(g.w, lane + 64 * (it % CNK))];
    }
    float ss = 0.f;
    if (wave < NJ)
        #pragma unroll
        for (int k = 0; k < CNK; k++) { const float x = v[wave * CQ + lane + 64 * k]; ss += x * x; }
    ss = qz_wsum_dpp(ss);
    ++ce;
    if (wave < NJ && lane == 0) cg_put(&g.cp[CP_N + g.w * 4 + wave], ss, ce);
    __shared__ float cpn[CG * 4], crn[4];
    if (tid < CG * 4 && (tid & 3) < NJ) cpn[tid] = cg_get(&g.cp[CP_N + tid], ce);
    __syncthreads();
    if (tid < NJ) {
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < CG; i++) a += cpn[i * 4 + tid];
        crn[tid] = rsqrtf(a / D + eps);
    }
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < CN_IT; u++) {
        const int it = wave + 8 * u;
        if (it < NJ * CNK) {
            const int jr = it / CNK, k = it % CNK;
            CK njob * j = nj + jr;
            float * y = j->y ? qz_gp(j->y) : nullptr;
            unsigned char * qv = qz_gp(j->q); float * qs = qz_gp(j->qs);
            const int i = ccol(g.w, lane + 64 * k);
            const float yv = v[jr * CQ + lane + 64 * k] * crn[jr] * wv[u];
            if (y) y[i] = yv;
            quant32(yv, qv, qs, i);
        }
    }
}
// NJ norms (d_norm_quant) of rows in global memory
template <int NJ> CYC_NI void c_cnorm(CK njob * nj, float eps, const cgrp & g, unsigned int & ce, int tid) {
    asm volatile ("" : "+v"(tid));
    float * v = cyc_lds;
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < NJ; j++)
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            const int q = tid + 512 * m;
            if (m == 0 || q < CQ) v[j * CQ + q] = qz_gp(nj[j].x)[ccol(g.w, q)];
        }
    __syncthreads();
    c_cnorm_tail(NJ, v, nj, eps, g, ce, tid);
    __syncthreads();
}
// d_mix_lds<MQ> in two-pass order, on the owned columns. With NRM, also the norm nj[j].
template <int MQ, bool NRM> CYC_NI void c_cmix(CK sptrs * S, const float * qy, float * out, CK njob * nj, float eps,
                                              const cgrp & g, unsigned int & ce, int tid) {
    asm volatile ("" : "+v"(tid));
    __shared__ float crs[6], cws[6][4];
    float * sv = cyc_lds;                  // [6][CQ] states
    float * sq = sv + 6 * CQ;              // [MQ][CQ] queries
    float * so = sq + MQ * CQ;             // [MQ][CQ] outputs
    const int n = S->n, wave = tid >> 6, lane = tid & 63;
    qy = qz_gp(qy); out = qz_gp(out);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            const int i = ccol(g.w, q);
            #pragma unroll
            for (int s = 0; s < 6; s++) sv[s * CQ + q] = qz_gp(S->p[s < n ? s : 0])[i];
            #pragma unroll
            for (int j = 0; j < MQ; j++) sq[j * CQ + q] = qy[(size_t) j * D + i];
        }
    }
    __syncthreads();
    // pass 1: wave s sums the squares of state s
    {
        float r = 0.f;
        if (wave < 6)
            #pragma unroll
            for (int k = 0; k < CNK; k++) { const float v = sv[wave * CQ + lane + 64 * k]; r += v * v; }
        r = qz_wsum_dpp(r);
        if (wave < 6 && lane == 0) cg_put(&g.cp[CP_1 + g.w * 6 + wave], r, ce + 1);
    }
    ++ce;
    __shared__ float cp1[CG * 6];
    if (tid < CG * 6) cp1[tid] = cg_get(&g.cp[CP_1 + tid], ce);
    __syncthreads();
    if (tid < 6) {
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < CG; i++) a += cp1[i * 6 + tid];
        crs[tid] = rsqrtf(a / D + 1e-6f);
    }
    __syncthreads();
    // pass 2: wave s takes the dots of rms(state s) with the queries
    if (wave < 6) {
        const float rh = crs[wave];
        float r[MQ];
        #pragma unroll
        for (int j = 0; j < MQ; j++) r[j] = 0.f;
        #pragma unroll
        for (int k = 0; k < CNK; k++) {
            const float x = sv[wave * CQ + lane + 64 * k] * rh;
            #pragma unroll
            for (int j = 0; j < MQ; j++) r[j] += x * sq[j * CQ + lane + 64 * k];
        }
        #pragma unroll
        for (int j = 0; j < MQ; j++) r[j] = qz_wsum_dpp(r[j]);
        if (lane == 0)
            #pragma unroll
            for (int j = 0; j < MQ; j++) cg_put(&g.cp[CP_2 + (g.w * 6 + wave) * 4 + j], r[j], ce + 1);
    }
    ++ce;
    __shared__ float cpl[CG * 6 * 4];
    if (tid < CG * 6 * 4) cpl[tid] = (tid & 3) < MQ ? cg_get(&g.cp[CP_2 + tid], ce) : 0.f;
    __syncthreads();
    if (tid < 64) {                        // d_mix_lds's softmax: lane s * 4 + j holds state s, query j
        const int ss = lane >> 2, jq = lane & 3;
        float v = 0.f;
        if (ss < 6) {
            #pragma unroll
            for (int i = 0; i < CG; i++) v += cpl[(i * 6 + ss) * 4 + jq];
        }
        const float scale = 1.f / sqrtf((float) D);
        float mx = -INFINITY;
        #pragma unroll
        for (int s = 0; s < 6; s++) { const float vs = __shfl(v, s * 4 + jq, 64); if (s < n) mx = fmaxf(mx, vs * scale); }
        const float ev = expf(v * scale - mx);
        float sum = 0.f;
        #pragma unroll
        for (int s = 0; s < 6; s++) { const float es = __shfl(ev, s * 4 + jq, 64); if (s < n) sum += es; }
        if (ss < n && jq < MQ) cws[ss][jq] = ev / sum;
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            float o[MQ];
            #pragma unroll
            for (int j = 0; j < MQ; j++) o[j] = 0.f;
            #pragma unroll
            for (int s = 0; s < 6; s++)
                if (s < n) {
                    const float x = sv[s * CQ + q];
                    const float rs = crs[s];
                    #pragma unroll
                    for (int j = 0; j < MQ; j++) o[j] += cws[s][j] * x * rs;
                }
            const int i = ccol(g.w, q);
            #pragma unroll
            for (int j = 0; j < MQ; j++) { out[(size_t) j * D + i] = o[j]; if (NRM) so[j * CQ + q] = o[j]; }
        }
    }
    if constexpr (NRM) {
        __syncthreads();
        c_cnorm_tail(MQ, so, nj, eps, g, ce, tid);
    }
    __syncthreads();
}
CYC_NI void c_head_post(CK hpost * e, int pos, float tau, int v, int tid) {
    asm volatile ("" : "+v"(tid));
    hpost h = *e;
    const int row = e->dk ? pos : pos % PF_RING;          // dk: full layers only
    h.qr = qz_gp(h.qr); h.rr = qz_gp(h.rr); h.sk = qz_gp(h.sk); h.kr = qz_gp(h.kr); h.vr = qz_gp(h.vr);
    h.dk = h.dk ? qz_gp(h.dk) : nullptr; h.dv = h.dv ? qz_gp(h.dv) : nullptr;
    h.qclip = qz_gp(h.qclip); h.rclip = qz_gp(h.rclip); h.relb = qz_gp(h.relb); h.kconv = qz_gp(h.kconv); h.vconv = qz_gp(h.vconv);
    h.hist = qz_gp(h.hist); h.qh = qz_gp(h.qh); h.tb = qz_gp(h.tb);
    h.kc = qz_gp(h.kc + (size_t) row * KVL); h.vc = qz_gp(h.vc + (size_t) row * KVL);
    h.tau = e->dk ? tau : 1.f;
    d_head_post(h, v, tid, tid < 128, min(TBW, pos + 1));     // attention at pos reads d <= pos only
}
CYC_NI void c_attn(CK attn_args * e, int pos, int v, int tid) {
    asm volatile ("" : "+v"(tid));
    attn_args t = *e;
    t.qh = qz_gp(t.qh); t.tb = qz_gp(t.tb); t.gate = qz_gp(t.gate); t.kc = qz_gp(t.kc); t.vc = qz_gp(t.vc);
    t.oq = qz_gp(t.oq); t.os = qz_gp(t.os); t.part = qz_gp(t.part); t.cnt = qz_gp(t.cnt);
    t.pos = pos;
    fl_attn_plain(t, v / ATT_SPL, v % ATT_SPL, tid, cyc_lds);   // the mix states are dead here
}
CYC_NI void c_xconv_b0(const float * own, const float * peer, const float * w, float * hs, float * x, float * blk, int tid) {
    d_xconv_blk<512>(qz_gp(own), qz_gp(peer), qz_gp(w), qz_gp(hs), qz_gp(x), blk ? qz_gp(blk) : nullptr, tid);
}
CYC_NI void c_router(const float * f, const float * wr, const float * bias, float * ps, int e, int lane) { d_router(qz_gp(f), qz_gp(wr), qz_gp(bias), qz_gp(ps), e, lane); }
// The router's top NE_USED by rank. Candidate i's place is #{j : s_j > s_i, or s_j == s_i
// and j < i}, so ties go to the lower id and the larger scores come first. The weights
// are p / sum(p) * scale, with the sum taken in that order. The whole block calls it.
CYC_NI void c_pick_rank(const float * ps, float scale, int * sids, float * swts, int tid) {
    __shared__ float lv[NE], lp[NE_USED];
    asm volatile ("" : "+v"(tid));
    ps = qz_gp(ps);
    float p = 0.f;
    if (tid < NE) { lv[tid] = ps[NE + tid]; p = ps[tid]; }
    __syncthreads();
    if (tid < NE) {
        const float v = lv[tid];
        int r = 0;
        #pragma unroll 16
        for (int j = 0; j < NE; j++) { const float o = lv[j]; r += (o > v) | ((o == v) & (j < tid)); }
        if (r < NE_USED) { sids[r] = tid; lp[r] = p; }
    }
    __syncthreads();
    if (tid < NE_USED) {
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < NE_USED; k++) sum += lp[k];
        sum += 1e-20f;
        swts[tid] = lp[tid] / sum * scale;
    }
    __syncthreads();
}
// The experts' gate | up GEMV, SITU and quant. A task is 2 activations: their 2 gate rows
// and the 2 up rows 32 later (gu_row), in one 4-row GEMV with qz_gemv_rows_t's value per
// row. Tasks go round-robin over all the grid's waves. The SITU values land in sv as f32,
// and the wave that finishes a 32-activation group (gcnt) quantizes it and resets the
// count.
constexpr int UP_A = 2, UP_TG = 32 / UP_A;       // activations per task, tasks per quant group
CYC_NI void c_up_situ_w(const char * wv, const __half * ws, const int * sids, const unsigned char * av, const float * as,
                        float beta, float lb, float * sv, int * gcnt, unsigned char * hq, float * hs, int tid) {
    asm volatile ("" : "+v"(tid));
    constexpr int NT = NE_USED * FFL / UP_A;
    av = qz_gp(av); as = qz_gp(as); sv = qz_gp(sv); gcnt = qz_gp(gcnt); hq = qz_gp(hq); hs = qz_gp(hs);
    const int wave = tid >> 6, lane = tid & 63, nw = gridDim.x * 8;
    #pragma unroll 1
    for (int task = blockIdx.x * 8 + wave; task < NT; task += nw) {
        const int a0 = task * UP_A, k = a0 / FFL, j0 = a0 % FFL;
        const int e = __builtin_amdgcn_readfirstlane(sids[k]);
        const size_t base = (size_t) e * FFE;
        float acc[1][2 * UP_A];
        qz_gemv_rows_t<1, 2 * UP_A, 32>(av, as, D, wv + base * D, ws + base * (D / 32), FFE, gu_row(j0), lane, acc);
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < UP_A; s++) sv[a0 + s] = situ_v(acc[0][s], acc[0][UP_A + s], beta, lb);
            __threadfence();
        }
        int done = 0;
        if (lane == 0) done = atomicAdd(&gcnt[a0 / 32], 1) == UP_TG - 1;
        done = __shfl(done, 0, 64);
        if (done) {                                         // wave-uniform
            __threadfence();
            const int g = a0 / 32;
            if (lane == 0) gcnt[g] = 0;
            if (lane < 32) {
                const int ai = g * 32 + lane, kk = ai / FFL, j = ai % FFL;
                const float x = __hip_atomic_load(&sv[ai], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                quant32(x, hq + kk * FFL, hs + kk * (FFL / 32), j);
            }
        }
    }
}
// The experts' down rows. The grid's row quads split evenly over the blocks, and a task
// is a quad with an expert pair: lanes 0..31 take expert 2p, lanes 32..63 expert 2p+1.
// K = FFL is 18 units, so a lane holds at most one unit and the per-row value is the unit
// products reduced by shfl_down levels 16..1 (level 32 would only add zeros), then the
// same k-order fold of wk * acc from LDS.
constexpr int MD_Q = 16;                         // quads per LDS chunk
CYC_NI void c_moe_down_q(const char * wv, const __half * ws, const int * sids, const float * swts,
                         const unsigned char * aq, const float * as, float * y, float * yp, int tid, float * lacc) {
    static_assert(FFL / 32 <= 32, "one unit per lane");
    asm volatile ("" : "+v"(tid));
    constexpr int NU = FFL / 32, NQ = D / 4;
    y = qz_gp(y); yp = yp ? qz_gp(yp) : nullptr;          // aq / as are LDS pointers
    const int wave = tid >> 6, lane = tid & 63, h = lane >> 5, l = lane & 31;
    const int q0 = blockIdx.x * NQ / gridDim.x, q1 = (blockIdx.x + 1) * NQ / gridDim.x;
    for (int c0 = q0; c0 < q1; c0 += MD_Q) {
        const int nq = min(MD_Q, q1 - c0);
        #pragma unroll 1
        for (int task = wave; task < nq * (NE_USED / 2); task += 8) {
            const int qd = task / (NE_USED / 2), k = 2 * (task % (NE_USED / 2)) + h;
            const int e = sids[k], r0 = (c0 + qd) * 4;
            const int u = min(l, NU - 1);
            const char * w = wv + ((size_t) e * D + r0) * FFL + u * 32;
            const __half * sw = ws + ((size_t) e * D + r0) * NU + u;
            const uint4 a0 = *(const uint4 *) (aq + k * FFL + u * 32), a1 = *(const uint4 *) (aq + k * FFL + u * 32 + 16);
            const float sa = as[k * NU + u];
            qz_v4u w0[4], w1[4];
            float wsc[4];
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                const qz_gu4 * wr = (const qz_gu4 *) (w + (size_t) s * FFL);
                w0[s] = wr[0]; w1[s] = wr[1];
                wsc[s] = __half2float(__ushort_as_half(((const qz_gh *) sw)[(size_t) s * NU]));
            }
            float acc[4];
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                int d = 0;
                DOT4(d, w0[s].x, a0.x); DOT4(d, w0[s].y, a0.y); DOT4(d, w0[s].z, a0.z); DOT4(d, w0[s].w, a0.w);
                DOT4(d, w1[s].x, a1.x); DOT4(d, w1[s].y, a1.y); DOT4(d, w1[s].z, a1.z); DOT4(d, w1[s].w, a1.w);
                const float v = (float) d * wsc[s] * sa;
                acc[s] = l < NU ? v : 0.f;
            }
            #pragma unroll
            for (int s = 0; s < 4; s++)
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc[s] += __shfl_down(acc[s], off, 64);
            if (l == 0)
                #pragma unroll
                for (int s = 0; s < 4; s++) lacc[(qd * NE_USED + k) * 4 + s] = acc[s];
        }
        __syncthreads();
        if (tid < nq * 4) {
            const int qd = tid >> 2, s = tid & 3, r = (c0 + qd) * 4 + s;
            float tot = 0.f;
            #pragma unroll
            for (int k = 0; k < NE_USED; k++) tot += swts[k] * lacc[(qd * NE_USED + k) * 4 + s];
            y[r] = tot;
            if (yp) yp[r] = tot;
        }
        __syncthreads();
    }
}

// The token. Per application, block 0 alone runs the serial chain: the exchange wait, the
// post conv, the next mix and norm. The grid shares the GEMVs, the head posts, attention,
// the router and the experts. Every block repeats the router's top-8 into its own LDS;
// the gate_up tiles carry their SITU value.
__global__ __launch_bounds__(512) void ink2_cycle(unsigned int * bst, const cyc_args * Ag, int tok, int pos, float tau, int head) {
    __shared__ int sids[NE_USED];
    __shared__ float swts[NE_USED];
    __shared__ float mdl[MD_Q * NE_USED * 4];
    const int tid_k = threadIdx.x, nb = gridDim.x;
    const bool b0 = blockIdx.x == 0;
    int kb = 0;
#define BAR() cyc_barrier(bst, ++kb)
    unsigned int xs = *Ag->xseq;
    if (b0) d_embed(Ag->ev, Ag->es, tok, Ag->embd_norm, Ag->eps, Ag->X, tid_k, tid_k < 256);
    const int napp = Ag->napp;
    // the chain groups: blocks 0 .. CG-1 and CG .. 2 CG-1, with flags zeroed per launch
    unsigned long long * slots = (unsigned long long *) (bst + CYC_NBX);
    const cgrp g1 = {bst + CYC_NBMAX, slots, (int) blockIdx.x};
    const cgrp g2 = {bst + CYC_NBMAX + CG, slots + CP_SZ, (int) blockIdx.x - CG};
    unsigned int ce1 = 0, ce2 = 0;
    if (blockIdx.x < CG) cg_sync(g1, ++ce1);          // block 0's embedding (X) before the group reads its columns
    for (int ai = 0; ai < napp; ai++) {
        // The args and table pointers stay opaque per application, so their fields are
        // reloaded as scalars where used instead of living across the loop.
        CK cyc_args * Ap = (CK cyc_args *) Ag;
        asm volatile ("" : "+s"(Ap));
        CK cyc_args & A = *Ap;
        CK appk * e = (CK appk *) A.tab + ai;
        // tid opaque per application: per-thread addresses are not hoisted out of the loop
        int tid = tid_k;
        asm volatile ("" : "+v"(tid));
        const int wave = tid >> 6, lane = tid & 63;
        const int pre = e->pre, msg = e->msg;

        // ---- P0, chain group 1 (blocks 0 .. CG-1): previous ffn sum and conv, the pass
        // copies, the mixes, the norms. The msg mix needs none of it, so group 2 runs it.
        if (blockIdx.x < CG) {
            if (ai > 0) {
                CK appk * ep = e - 1;
                cyc_xwait(A, xs);
                c_cxconv(A.PART1, A.xin_mine + (size_t) (xs & 1) * D, ep->fconv, ep->hsf, A.X, nullptr, g1.w, tid, e->tap);
            }
            if (pre) c_cpre(pre, A.P, A.X, e->pre_out, g1.w, tid);
            if (e->has_res && !msg) {           // the norms' inputs are the mix outputs (nna = mres_m)
                if (e->mres_m == 1) c_cmix<1, true>(&e->mres, e->qres, e->res_out, e->na.j, A.eps, g1, ce1, tid);
                else c_cmix<3, true>(&e->mres, e->qres, e->res_out, e->na.j, A.eps, g1, ce1, tid);
            } else {
                if (e->has_res) {
                    if (e->mres_m == 1) c_cmix<1, false>(&e->mres, e->qres, e->res_out, nullptr, 0.f, g1, ce1, tid);
                    else c_cmix<3, false>(&e->mres, e->qres, e->res_out, nullptr, 0.f, g1, ce1, tid);
                }
                if (msg && pre) c_cmix<4, false>(&e->mmsg, e->qmsg, e->msg_out, nullptr, 0.f, g1, ce1, tid);   // reads OUT the pass copy just wrote
                if (!msg) {
                    if (e->nna == 1) c_cnorm<1>(e->na.j, A.eps, g1, ce1, tid);
                    else c_cnorm<3>(e->na.j, A.eps, g1, ce1, tid);
                }
            }
        } else if (msg && !pre && blockIdx.x < 2 * CG) c_cmix<4, false>(&e->mmsg, e->qmsg, e->msg_out, nullptr, 0.f, g2, ce2, tid);
        BAR();
        if (msg) {
            c_ctrl_down_all<3>(e->ca.j, tid);
            BAR();
            c_ctrl_up_all<3>(e->ca.j, tid);
            BAR();
            for (int v = blockIdx.x; v < e->nna; v += nb) c_norm(&e->na.j[v], A.eps, tid);
            BAR();
        }
        // ---- attention
        for (int v = blockIdx.x; v < 2 * e->nga; v += nb) c_gemv2(&e->ga, v, nullptr);
        BAR();
        for (int v = blockIdx.x; v < HQL + 2 * HKVL; v += nb) c_head_post(&e->hp, pos, tau, v, tid);
        BAR();
        for (int v = blockIdx.x; v < HKVL * ATT_SPL; v += nb) c_attn(&e->at, pos, v, tid);
        BAR();
        ++xs;
        for (int v = blockIdx.x; v < 2 * e->ngo; v += nb) c_gemv2(&e->go, v, A.xin_px[0] + (size_t) (xs & 1) * D);
        __threadfence_system();
        BAR();
        // ---- P1, chain group 1: the attention sum + conv, the ffn mix (and norm)
        if (blockIdx.x < CG) {
            cyc_xwait(A, xs);
            c_cxconv(A.PART0, A.xin_mine + (size_t) (xs & 1) * D, e->pconv, e->hsa, A.X, e->blk, g1.w, tid);
            if (msg) c_cmix<1, false>(&e->mff, e->qff, A.F, nullptr, 0.f, g1, ce1, tid);
            else c_cmix<1, true>(&e->mff, e->qff, A.F, e->nb.j, A.eps, g1, ce1, tid);
        }
        BAR();
        if (msg) {
            c_ctrl_down_all<1>(e->cb.j, tid);
            BAR();
            c_ctrl_up_all<1>(e->cb.j, tid);
            BAR();
            if (b0) c_norm(&e->nb.j[0], A.eps, tid);
            BAR();
        }
        // ---- experts
        for (int x = blockIdx.x + nb * wave; x < NE; x += 8 * nb) c_router(A.FN, e->gin, e->pb, A.H, x, lane);   // few waves per block
        BAR();
        c_pick_rank(A.H, A.route_scale, sids, swts, tid);
        if (b0 && tid < NE_USED) { qz_gp(A.IDS)[tid] = sids[tid]; qz_gp(A.WTS)[tid] = swts[tid]; }
        c_up_situ_w(e->guv, e->gus, sids, A.aq3, A.as3, A.beta, A.lb, A.GU, A.gcnt, A.hq, A.hs, tid);
        BAR();
        ++xs;
        c_moe_down_q(e->dnv, e->dns, sids, swts, qz_gp(A.hq), qz_gp(A.hs), A.PART1, A.xin_px[0] + (size_t) (xs & 1) * D, tid, mdl);
        __threadfence_system();
        BAR();
    }
    if (b0) {        // the last ffn sum and conv
        CK cyc_args & A = *(CK cyc_args *) Ag;
        CK appk * ep = (CK appk *) A.tab + napp - 1;
        cyc_xwait(A, xs);
        c_xconv_b0(A.PART1, A.xin_mine + (size_t) (xs & 1) * D, ep->fconv, ep->hsf, A.X, nullptr, tid_k);
        if (tid_k == 0) *Ag->xseq = xs;
    }
    if (head) {
        CK cyc_args & A = *(CK cyc_args *) Ag;
        if (b0) {
            __syncthreads();
            c_mix1(&A.hmix, A.hq_out, A.F, tid_k);
            c_norm(&A.hn, A.eps, tid_k);
        }
        BAR();
        c_head_gemv(&A.hg, A.nvalid, A.softcap);
        BAR();
    }
}
#undef BAR
#pragma clang fp contract(fast)
