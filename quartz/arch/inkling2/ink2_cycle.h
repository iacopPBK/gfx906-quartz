// inkling2 persistent decode: one launch per card per token runs the embedding
// and all 48 layer applications. One 512-thread block per
// CU, software grid barriers (kernels/dev.h) between the phases, and the TP2
// sums through the peer inbox and flags of kernels/xchg.h. The device bodies
// are ink2_kernels.h's.
#pragma once

#include "kernels/xchg.h"

// the cycle-only bodies round like ink2_kernels.h (fp contract on), restored at the end
#pragma clang fp contract(on)

// the grid barrier: a flag per block (no atomic queue on one word). Block b
// stores k to fl[b] (release); lanes 0 .. nb-1 of wave 0 each poll one block's
// flag until it reaches k. fl: CYC_NBMAX words, zeroed per launch, k = 1, 2, ...
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
    int pre;                      // 1: P = X first; 2: OUT[ob] = X, X = P first (a loop pass starts)
    float * pre_out;
    // attention half
    int has_res, mres_m;
    sptrs mres; const float * qres; float * res_out;
    sptrs mmsg; const float * qmsg; float * msg_out;
    cjobs ca;                     // the q / k / v controllers (msg)
    njobs na; int nna;
    gjobs ga; int nga;            // nga: blocks
    hpost hp;                     // kc / vc: the slot's cache base; tau 1
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
};

// static per card (device memory); the token's tok / pos / tau are kernel arguments
struct cyc_args {
    const appk * tab; int napp;
    const char * ev; const __half * es; const float * embd_norm; float eps;
    float * X, * P, * PART0, * PART1;
    float * F, * FN, * H, * WTS; int * IDS;
    float * GU;                   // the SITU values (f32, NE_USED FFL; the head's logit buffer)
    int * gcnt;                   // per quant group: tasks done (NE_USED FFL / 32, zero between uses)
    unsigned char * aq3, * hq; float * as3, * hs;
    float route_scale, beta, lb;
    // the head (head launch arg): mix -> F, norm -> aq3 / as3 (block 0), then the vocab GEMV
    // into hg.y with the softcap on the first nvalid rows (d_gemv + k_softcap's values)
    sptrs hmix; const float * hq_out; njob hn; gjob hg; int nvalid; float softcap;
    float * cpart;                // the chain groups' partial sums (2 CP_SZ)
    // TP2 exchange (the names kernels/xchg.h expects)
    float * xin_mine, * xin_peer;
    unsigned int * xflag_mine, * xflag_peer, * xseq;
    int * xerr;
};

// the card's partial is in the peer inbox (pushed, fenced, grid barrier passed): the
// calling block flags (block 0 only) and waits for the peer's
template <class XA> __device__ __forceinline__ void cyc_xwait(const XA & A, unsigned int tok) {
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) qz_xflag(A, tok);
        qz_xwait(A, tok);
    }
    __syncthreads();
}

// The phase wrappers. Pointers the kernel loads from the table are generic to the
// compiler: flat loads / stores, which count in lgkmcnt too, so every LDS wait
// (reductions, barriers) also waited for the outstanding global traffic. gp()
// casts them to the global address space (inferred through the inlined bodies).
template <class T> __device__ __forceinline__ T * gp(T * p) {
    __builtin_assume(!__builtin_amdgcn_is_shared((const void *) p));
    __builtin_assume(!__builtin_amdgcn_is_private((const void *) p));
    return p;
}
// the table and args are read through the constant address space: scalar loads
#ifdef __HIP_DEVICE_COMPILE__
#define CK const __attribute__((address_space(4)))
#else
#define CK const
#endif
constexpr int CYC_LDS = 6 * D * 4;           // dynamic LDS: the mix states
extern __shared__ float cyc_lds[];
#define CYC_NI __device__ __forceinline__
template <int MQ> CYC_NI void c_mix(CK sptrs * S, const float * q, float * o, int tid) {
    asm volatile ("" : "+v"(tid));
    sptrs L;
    L.n = S->n;
    #pragma unroll
    for (int s = 0; s < 6; s++) L.p[s] = gp(S->p[s]);
    d_mix_lds<MQ>(L, gp(q), gp(o), tid, tid < 256, cyc_lds);
}
CYC_NI void c_mix1(CK sptrs * S, const float * q, float * o, int tid) { c_mix<1>(S, q, o, tid); }
CYC_NI void c_mix3(CK sptrs * S, const float * q, float * o, int tid) { c_mix<3>(S, q, o, tid); }
CYC_NI void c_mix4(CK sptrs * S, const float * q, float * o, int tid) { c_mix<4>(S, q, o, tid); }
CYC_NI cjob gcj(CK cjob * c) {
    cjob j;
    j.cur = gp(c->cur); j.msg = gp(c->msg); j.wd = gp(c->wd); j.wg = gp(c->wg); j.wf = gp(c->wf); j.g = gp(c->g); j.out = gp(c->out); j.wgt = c->wgt; j.wft = c->wft;
    return j;
}
// bsum<N>(v, red, 4) for each 256-thread half of the block on its own (red: 2 * 4 * N)
template <int N> CYC_NI void hsum(float (&v)[N], float * red) {
    #pragma unroll
    for (int k = 0; k < N; k++) v[k] = wsum_xor(v[k]);
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
// the control-down rows in one pass per block: the blocks split into NJ groups
// (one job each: its rms once), each half block takes 2 rows; d_ctrl_down's
// per-thread sums in its order and the same 4-wave reductions, so bit-identical
template <int NJ> CYC_NI void c_ctrl_down_all(CK cjob * cj, int tid) {
    asm volatile ("" : "+v"(tid));
    __shared__ float red[2 * 4 * 2];
    constexpr int NV = D / 256, MR = 2;
    const int bpj = gridDim.x / NJ, j = blockIdx.x / bpj, bj = blockIdx.x % bpj, h = tid >> 8, t = tid & 255;
    if (j >= NJ) return;                                   // the leftover blocks (block-uniform)
    const float * cur = gp(cj[j].cur), * msg = gp(cj[j].msg), * wd = gp(cj[j].wd);
    float * g = gp(cj[j].g);
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
// d_ctrl_up for the NJ jobs' D rows each, over all blocks, from the transposed
// weights (wgt / wft [GRANK][D]: lanes on consecutive rows, coalesced): item =
// (row, sum), each sum in d_ctrl_up's sequential k order; the forget sums via LDS
template <int NJ> CYC_NI void c_ctrl_up_all(CK cjob * cj, int tid) {
    asm volatile ("" : "+v"(tid));
    __shared__ float g[NJ][GRANK], sf[256];
    for (int i = tid; i < NJ * GRANK; i += 512) g[i / GRANK][i % GRANK] = gp(cj[i / GRANK].g)[i % GRANK];
    __syncthreads();
    constexpr int NR = NJ * D, KC = 32;
    const int r0 = blockIdx.x * NR / gridDim.x, r1 = (blockIdx.x + 1) * NR / gridDim.x;
    for (int c0 = r0; c0 < r1; c0 += 256) {
        const int n = min(256, r1 - c0), which = tid >= n, q = tid - (which ? n : 0);
        const int r = c0 + min(q, n - 1), j = r / D, i = r % D;
        float a = 0.f;
        if (tid < 2 * n) {
            const float * w = gp(which ? cj[j].wft : cj[j].wgt) + i;
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
            const float x = gp(cj[j].cur)[i], fo = sf[q], ga = a;
            gp(cj[j].out)[i] = (x - tanhf(fo) * x) + tanhf(ga) * gp(cj[j].msg)[i];
        }
        __syncthreads();
    }
}
CYC_NI void c_norm(CK njob * j, float eps, int tid) {
    asm volatile ("" : "+v"(tid));
    njob n;
    n.x = gp(j->x); n.w = gp(j->w); n.y = j->y ? gp(j->y) : nullptr; n.q = gp(j->q); n.qs = gp(j->qs);
    d_norm_quant(n, eps, tid, tid < 256);
}
// half a 32-row virtual block (16 rows, S = 2 per wave): the same rows with the
// same per-row code, only finer so that a small job list balances over the grid
CYC_NI void c_gemv2(CK gjobs * J, int hb, float * yp) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int b = hb >> 1, half = hb & 1;
    int k = 0;
    while (k + 1 < J->n && b >= J->j[k].g1) k++;
    const gjob * j = &J->j[k];
    const int g = b - (k ? J->j[k - 1].g1 : 0);
    const int r0 = g * 32 + half * 16 + wave * 2;
    const int N = j->N;
    float * y = gp(j->y);
    float acc[1][2];
    qz_gemv_rows_t<1, 2>(gp(j->av), gp(j->as), j->K, j->wv, j->ws, N, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int s = 0; s < 2; s++) if (r0 + s < N) { y[r0 + s] = acc[0][s]; if (yp) gp(yp)[r0 + s] = acc[0][s]; }
}
// the vocab GEMV: 32-row units over the grid (d_gemv's rows), the softcap applied
// to the stored value on rows < nv (k_softcap's expression)
CYC_NI void c_head_gemv(CK gjob * J, int nv, float cap) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int N = J->N;
    float * y = gp(J->y);
    #pragma unroll 1
    for (int g = blockIdx.x; g * 32 < N; g += gridDim.x) {
        const int r0 = g * 32 + wave * 4;
        float acc[1][4];
        qz_gemv_rows_t<1, 4>(gp(J->av), gp(J->as), J->K, J->wv, J->ws, N, r0, lane, acc);
        if (lane == 0)
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                const float v = acc[0][s];
                if (r0 + s < N) y[r0 + s] = r0 + s < nv ? cap * tanhf(v / cap) : v;
            }
    }
}

// ---- the chain group: CG blocks share block 0's serial work (xconv, the pass
// copies, a mix and its norms). Block w owns the columns 64 w + l + 256 k
// (l < 64, k < D / 256): exactly wave w's share of every full-row sum in
// d_mix_lds / d_norm_quant (per-lane sum over k, wsum_xor, waves 0 .. 3 added in
// order), so each block publishes its wave-w partials, the group syncs on flags,
// and every block adds the CG partials in that order: the same bits. The
// element-wise parts (output pass, y / quant, xconv, copies) take the owned
// columns, the same thread for the same column throughout (no barrier between).
constexpr int CG = 4, CQ = D / CG, CNK = D / 256;
static_assert(CQ == 64 * CNK && CNK * 256 == D, "chain group columns");
constexpr int CYC_NBX = CYC_NBMAX + 2 * CG;      // bst words: the grid barrier's, then the chain groups'
constexpr int CP_1 = 0, CP_2 = CP_1 + CG * 6, CP_N = CP_2 + CG * 6 * 4, CP_SZ = CP_N + CG * 4;   // partial slots per group
__device__ __forceinline__ int ccol(int w, int q) { return 64 * w + (q & 63) + 256 * (q >> 6); }
// a partial after cg_sync: plain loads (the sync's acquire fence invalidated the CU's L1), all in flight at once
__device__ __forceinline__ float cld(const float * p) { return *p; }
struct cgrp { unsigned int * fl; float * cp; int w; };     // the group's CG flags, its partial slots, this block's index
// the partials stored (each by its writer, fenced) -> all CG blocks have published epoch e
CYC_NI void cg_sync(const cgrp & g, unsigned int e) {
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_store(&g.fl[g.w], e, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    if (threadIdx.x < CG) {
        while (__hip_atomic_load(&g.fl[threadIdx.x], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < e) { }
        __threadfence();
    }
    __syncthreads();
}
__device__ __forceinline__ void cg_put(float * p, float v) { *p = v; __threadfence(); }
// d_xconv_blk's values on the owned columns (thread q = tid, tid + 512)
CYC_NI void c_cxconv(const float * own, const float * peer, const float * w, float * hs, float * x, float * blk, int cw, int tid) {
    own = gp(own); peer = gp(peer); w = gp(w); hs = gp(hs); x = gp(x); blk = blk ? gp(blk) : nullptr;
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
        }
    }
}
// the pass copies on the owned columns
CYC_NI void c_cpre(int pre, float * P, float * X, float * po, int cw, int tid) {
    P = gp(P); X = gp(X);
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            const int i = ccol(cw, q);
            if (pre == 1) P[i] = X[i];
            else { gp(po)[i] = X[i]; X[i] = P[i]; }
        }
    }
}
// d_norm_quant of NJ rows held in LDS v[j][CQ] (owned columns): wave j < NJ
CYC_NI void c_cnorm_tail(int NJ, const float * v, CK njob * nj, float eps, const cgrp & g, unsigned int & ce, int tid) {
    const int wave = tid >> 6, lane = tid & 63;
    float wv[CNK];                       // the norm weights in flight over the sync
    {
        const float * wn = gp(nj[wave < NJ ? wave : 0].w);
        #pragma unroll
        for (int k = 0; k < CNK; k++) wv[k] = wn[ccol(g.w, lane + 64 * k)];
    }
    float ss = 0.f;
    if (wave < NJ)
        #pragma unroll
        for (int k = 0; k < CNK; k++) { const float x = v[wave * CQ + lane + 64 * k]; ss += x * x; }
    ss = wsum_xor(ss);
    if (wave < NJ && lane == 0) cg_put(&g.cp[CP_N + g.w * 4 + wave], ss);
    cg_sync(g, ++ce);
    __shared__ float cpn[CG * 4];
    if (tid < CG * 4) cpn[tid] = cld(&g.cp[CP_N + tid]);
    __syncthreads();
    if (wave < NJ) {
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < CG; i++) a += cpn[i * 4 + wave];
        const float r = rsqrtf(a / D + eps);
        CK njob * j = nj + wave;
        float * y = j->y ? gp(j->y) : nullptr;
        unsigned char * qv = gp(j->q); float * qs = gp(j->qs);
        #pragma unroll
        for (int k = 0; k < CNK; k++) {
            const int i = ccol(g.w, lane + 64 * k);
            const float yv = v[wave * CQ + lane + 64 * k] * r * wv[k];
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
            if (m == 0 || q < CQ) v[j * CQ + q] = gp(nj[j].x)[ccol(g.w, q)];
        }
    __syncthreads();
    c_cnorm_tail(NJ, v, nj, eps, g, ce, tid);
    __syncthreads();
}
// d_mix_lds<MQ> (two-pass order) on the owned columns; NRM: then the norms nj[j] of output j
template <int MQ, bool NRM> CYC_NI void c_cmix(CK sptrs * S, const float * qy, float * out, CK njob * nj, float eps,
                                              const cgrp & g, unsigned int & ce, int tid) {
    asm volatile ("" : "+v"(tid));
    __shared__ float crs[6], cws[6][4];
    float * sv = cyc_lds;                  // [6][CQ] states
    float * sq = sv + 6 * CQ;              // [MQ][CQ] queries
    float * so = sq + MQ * CQ;             // [MQ][CQ] outputs
    const int n = S->n, wave = tid >> 6, lane = tid & 63;
    qy = gp(qy); out = gp(out);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int q = tid + 512 * m;
        if (m == 0 || q < CQ) {
            const int i = ccol(g.w, q);
            #pragma unroll
            for (int s = 0; s < 6; s++) sv[s * CQ + q] = gp(S->p[s < n ? s : 0])[i];
            #pragma unroll
            for (int j = 0; j < MQ; j++) sq[j * CQ + q] = qy[(size_t) j * D + i];
        }
    }
    __syncthreads();
    // pass 1: wave s the sum of squares of state s
    {
        float r = 0.f;
        if (wave < 6)
            #pragma unroll
            for (int k = 0; k < CNK; k++) { const float v = sv[wave * CQ + lane + 64 * k]; r += v * v; }
        r = wsum_xor(r);
        if (wave < 6 && lane == 0) cg_put(&g.cp[CP_1 + g.w * 6 + wave], r);
    }
    cg_sync(g, ++ce);
    __shared__ float cp1[CG * 6];
    if (tid < CG * 6) cp1[tid] = cld(&g.cp[CP_1 + tid]);
    __syncthreads();
    if (tid < 6) {
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < CG; i++) a += cp1[i * 6 + tid];
        crs[tid] = rsqrtf(a / D + 1e-6f);
    }
    __syncthreads();
    // pass 2: wave s the dots of rms(state s) with the queries
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
        for (int j = 0; j < MQ; j++) r[j] = wsum_xor(r[j]);
        if (lane == 0)
            #pragma unroll
            for (int j = 0; j < MQ; j++) cg_put(&g.cp[CP_2 + (g.w * 6 + wave) * 4 + j], r[j]);
    }
    cg_sync(g, ++ce);
    __shared__ float cpl[CG * 6 * 4];
    if (tid < CG * 6 * 4) cpl[tid] = cld(&g.cp[CP_2 + tid]);   // the partials, one load per thread
    __syncthreads();
    if (tid < MQ) {                        // d_mix_lds's softmax, query j = tid
        float v[6];
        #pragma unroll
        for (int s = 0; s < 6; s++) {
            float a = 0.f;
            #pragma unroll
            for (int i = 0; i < CG; i++) a += cpl[(i * 6 + s) * 4 + tid];
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
        for (int s = 0; s < 6; s++) if (s < n) cws[s][tid] = ev[s] / sum;
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
    h.qr = gp(h.qr); h.rr = gp(h.rr); h.sk = gp(h.sk); h.kr = gp(h.kr); h.vr = gp(h.vr);
    h.dk = h.dk ? gp(h.dk) : nullptr; h.dv = h.dv ? gp(h.dv) : nullptr;
    h.qclip = gp(h.qclip); h.rclip = gp(h.rclip); h.relb = gp(h.relb); h.kconv = gp(h.kconv); h.vconv = gp(h.vconv);
    h.hist = gp(h.hist); h.q = gp(h.q); h.tb = gp(h.tb);
    h.kc = gp(h.kc + (size_t) row * KVL); h.vc = gp(h.vc + (size_t) row * KVL);
    h.tau = e->dk ? tau : 1.f;
    d_head_post(h, v, tid, tid < 128, min(TBW, pos + 1));     // attention at pos reads d <= pos only
}
CYC_NI void c_attn(CK attn_args * e, int pos, int v, int tid) {
    asm volatile ("" : "+v"(tid));
    attn_args t = *e;
    t.q = gp(t.q); t.tb = gp(t.tb); t.gate = gp(t.gate); t.kc = gp(t.kc); t.vc = gp(t.vc); t.oq = gp(t.oq); t.os = gp(t.os);
    t.part = gp(t.part); t.cnt = gp(t.cnt);
    t.pos = pos;
    d_attn(t, v / ATT_SPL, v % ATT_SPL, tid);
}
CYC_NI void c_xconv_b0(const float * own, const float * peer, const float * w, float * hs, float * x, float * blk, int tid) {
    d_xconv_blk<512>(gp(own), gp(peer), gp(w), gp(hs), gp(x), blk ? gp(blk) : nullptr, tid);
}
CYC_NI void c_router(const float * f, const float * wr, const float * bias, float * ps, int e, int lane) { d_router(gp(f), gp(wr), gp(bias), gp(ps), e, lane); }
// d_router_pick by rank: candidate i's place = #{j : s_j > s_i or (s_j == s_i and j < i)}
// (its selection order: max first, ties to the lower id); the weights with the
// sum in that order, so the same ids and bits. The whole block calls it.
CYC_NI void c_pick_rank(const float * ps, float scale, int * sids, float * swts, int tid) {
    __shared__ float lv[NE], lp[NE_USED];
    asm volatile ("" : "+v"(tid));
    ps = gp(ps);
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
// the experts' gate | up GEMV, SITU and quant: task = 2 activations (their 2
// gate rows and the 2 up rows 32 later, gu_row) in one 4-row GEMV (each row
// d_moe_up's value), dealt round-robin over all the grid's waves; the SITU
// values go to sv (f32) and the wave finishing a 32-activation group (gcnt)
// quantizes it (d_situ's quant32) and resets the count
constexpr int UP_A = 2, UP_TG = 32 / UP_A;       // activations per task, tasks per quant group
CYC_NI void c_up_situ_w(const char * wv, const __half * ws, const int * sids, const unsigned char * av, const float * as,
                        float beta, float lb, float * sv, int * gcnt, unsigned char * hq, float * hs, int tid) {
    asm volatile ("" : "+v"(tid));
    constexpr int NT = NE_USED * FFL / UP_A;
    av = gp(av); as = gp(as); sv = gp(sv); gcnt = gp(gcnt); hq = gp(hq); hs = gp(hs);
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
// the experts' down rows, the grid's row quads split evenly over the blocks;
// task (quad, expert pair): lanes 0..31 expert 2p, 32..63 expert 2p+1. With
// K = FFL (18 units) a lane holds at most one unit, so d_moe_down's per-row
// value is this tree: the unit products, the shfl_down levels 16..1 (its
// level 32 adds zeros); then the same k-order fold of wk * acc (from LDS)
constexpr int MD_Q = 16;                         // quads per LDS chunk
CYC_NI void c_moe_down_q(const char * wv, const __half * ws, const int * sids, const float * swts,
                         const unsigned char * aq, const float * as, float * y, float * yp, int tid, float * lacc) {
    static_assert(FFL / 32 <= 32, "one unit per lane");
    asm volatile ("" : "+v"(tid));
    constexpr int NU = FFL / 32, NQ = D / 4;
    y = gp(y); yp = yp ? gp(yp) : nullptr;          // aq / as: LDS
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

// the token. Per application: block 0 alone runs the serial chain (exchange wait,
// post conv, the next mix and norm); the grid shares the GEMVs, the head posts,
// attention, the router and the experts. Each block repeats the router's top-8
// (into LDS), and gate_up tiles carry their SITU.
__global__ __launch_bounds__(512) void ink2_cycle(unsigned int * bst, const cyc_args * Ag, int tok, int pos, float tau, int head) {
    __shared__ int sids[NE_USED];
    __shared__ float swts[NE_USED];
    __shared__ float tl[64];
    __shared__ float mdl[MD_Q * NE_USED * 4];
    const int tid_k = threadIdx.x, nb = gridDim.x;
    const bool b0 = blockIdx.x == 0;
    int kb = 0;
#define BAR() cyc_barrier(bst, ++kb)
    unsigned int xs = *Ag->xseq;
    if (b0) d_embed(Ag->ev, Ag->es, tok, Ag->embd_norm, Ag->eps, Ag->X, tid_k, tid_k < 256);
    const int napp = Ag->napp;
    // the chain groups: blocks 0 .. CG-1 and CG .. 2 CG-1 (flags after the grid barrier's, zeroed per launch)
    const cgrp g1 = {bst + CYC_NBMAX, Ag->cpart, (int) blockIdx.x};
    const cgrp g2 = {bst + CYC_NBMAX + CG, Ag->cpart + CP_SZ, (int) blockIdx.x - CG};
    unsigned int ce1 = 0, ce2 = 0;
    if (blockIdx.x < CG) cg_sync(g1, ++ce1);          // block 0's embedding (X) before the group reads its columns
    for (int ai = 0; ai < napp; ai++) {
        // the args / table pointers opaque per application: their fields are
        // reloaded (scalar) where used instead of held live across the loop
        CK cyc_args * Ap = (CK cyc_args *) Ag;
        asm volatile ("" : "+s"(Ap));
        CK cyc_args & A = *Ap;
        CK appk * e = (CK appk *) A.tab + ai;
        // tid opaque per application: per-thread addresses are not hoisted out of the loop
        int tid = tid_k;
        asm volatile ("" : "+v"(tid));
        const int wave = tid >> 6, lane = tid & 63;
        const int pre = e->pre, msg = e->msg;

        // ---- P0, chain group 1 (blocks 0 .. CG-1): the previous ffn sum + conv, the pass
        // copies, the mixes, the norms; the msg mix that needs none of it on group 2
        if (blockIdx.x < CG) {
            if (ai > 0) {
                CK appk * ep = e - 1;
                cyc_xwait(A, xs);
                c_cxconv(A.PART1, A.xin_mine + (size_t) (xs & 1) * D, ep->fconv, ep->hsf, A.X, nullptr, g1.w, tid);
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
        for (int v = blockIdx.x; v < HQL * ATT_SPL; v += nb) c_attn(&e->at, pos, v, tid);
        BAR();
        ++xs;
        for (int v = blockIdx.x; v < 2 * e->ngo; v += nb) c_gemv2(&e->go, v, A.xin_peer + (size_t) (xs & 1) * D);
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
        for (int x = blockIdx.x + nb * wave; x < NE; x += 8 * nb) c_router(A.FN, e->gin, e->pb, A.H, x, lane);   // spread: few waves per block
        BAR();
        c_pick_rank(A.H, A.route_scale, sids, swts, tid);
        if (b0 && tid < NE_USED) { gp(A.IDS)[tid] = sids[tid]; gp(A.WTS)[tid] = swts[tid]; }
        c_up_situ_w(e->guv, e->gus, sids, A.aq3, A.as3, A.beta, A.lb, A.GU, A.gcnt, A.hq, A.hs, tid);
        BAR();
        ++xs;
        c_moe_down_q(e->dnv, e->dns, sids, swts, gp(A.hq), gp(A.hs), A.PART1, A.xin_peer + (size_t) (xs & 1) * D, tid, mdl);
        __threadfence_system();
        BAR();
    }
    if (b0) {        // the last ffn sum + conv
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
