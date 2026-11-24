// inkling2 flash attention, the structure of quartz/arch/qwen35/flash9.h: a tile of
// FL_TP = 64 keys by absolute index, lane = key position for the QK (one dot per lane, no
// cross-lane reduction per key), the K tile in LDS with an XOR swizzle of its 16 B chunks,
// q in LDS read as wave-uniform 16 B broadcasts, the online softmax per tile in log2 units,
// P in LDS as f16, and the PV by v_dot2 over key pairs with V stored interleaved. Key j of
// a row at rp scores (q . k_j) / sqrt(HD) + tb[h][min(rp - j, REL_EXT)], the sink (an extra
// logit 0 with value 0) joins the merge, and the output is O / L * sigmoid(gate), then
// quant32.
//
// One body, two callers: the plain step (ink2_cycle's c_attn) and the DSpark verify
// (ink2_verify's v_attn_a / v_attn_b). Split-K: split s owns the tiles t % FL_NS == s,
// walked in ascending t, and the splits merge in ascending order on top of the sink; a row
// walks the same tiles in the same order in both callers and runs the same expressions per
// (row, head), so a verify row is bit-identical to a plain step at the same position (the
// DSpark switch is timing based and depends on it).
//
// What differs between the callers is only how the work is spread over the block's waves
// and how deep the staging is pipelined:
//   plain  - one row: wave w = q head w (3 waves), the other waves stage; K is double
//            buffered so tile t+1's store runs under tile t's math.
//   verify - one row per wave (8 rows): every wave runs all 3 heads of its own row against
//            the one K/V staging, so a tile's K and V are read from the cache once for the
//            whole round; K is single buffered with the next tile's load in registers.
#pragma once

constexpr int FL_TP = 64;                    // keys per tile
constexpr int FL_NS = ATT_SPL;               // splits per kv head (tile t goes to t % FL_NS)
constexpr int FL_CH = HD / 8;                // 16 B chunks in a key row
constexpr int FL_VP = FL_TP / 2;             // key pairs per tile
constexpr int FL_KH = FL_TP * HD;            // halves in the K tile (and in the V tile)
constexpr int FL_NW = 8;                     // waves per block, and the verify's rows
constexpr float FL_FBIG = -3.4028235e38f;    // empty-partial max

typedef unsigned int fl_v4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float fl_rl(float v, int l) {
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), l));
}
// max over the wave, uniform in every lane: a 32-lane DPP tree then the two halves
__device__ __forceinline__ float fl_wmax(float v) {
    v = qz_amax32(v);
    return fmaxf(fl_rl(v, 0), fl_rl(v, 32));
}
__device__ __forceinline__ __half2 fl_h2(unsigned int u) { return __builtin_bit_cast(__half2, u); }
// the LDS writes of one wave visible to its own reads (no cross-wave barrier needed; the
// block barriers are the tile's, in fl_plain_run / fl_attn_rows)
__device__ __forceinline__ void fl_lds_wait() { asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); }

// ---- staging. The block's 512 threads each own 2 K items and 1 V item of the tile.
// A K item is 16 B: item i is key i / FL_CH, chunk i % FL_CH, so consecutive items are
// consecutive bytes of one key row and the reads stay coalesced. Chunk c of key p lands at
// (c ^ (p & 7)), so 64 lanes reading 64 different keys never bank-conflict.
constexpr int FL_ST = 512;                       // the megakernels' block size
constexpr int FL_KI = FL_TP * FL_CH / FL_ST;     // K items per thread (2)
static_assert(FL_TP * FL_CH % FL_ST == 0 && FL_VP * FL_CH == FL_ST, "staging items per thread");

__device__ __forceinline__ void fl_ld_k(const __half * kc, int tile, int lo, int rp, int full, int tid,
                                       fl_v4 & r0, fl_v4 & r1) {
    #pragma unroll
    for (int n = 0; n < FL_KI; n++) {
        const int i = tid + n * FL_ST, p = i / FL_CH, c = i - p * FL_CH;
        const int j = min(max(tile * FL_TP + p, lo), rp);       // clamped: no read past the row
        const size_t row = full ? (size_t) j : (size_t) (j & (PF_RING - 1));
        *(n ? &r1 : &r0) = *(const fl_v4 *) (kc + row * KVL + c * 8);
    }
}
__device__ __forceinline__ void fl_st_k(__half * kb, int tid, fl_v4 r0, fl_v4 r1) {
    #pragma unroll
    for (int n = 0; n < FL_KI; n++) {
        const int i = tid + n * FL_ST, p = i / FL_CH, c = i - p * FL_CH;
        *(fl_v4 *) (kb + (size_t) p * HD + ((c ^ (p & 7)) * 8)) = n ? r1 : r0;
    }
}
// The 8 dims of a pair's 2 keys to [dim][2]: dword j = (the even key's dim, the odd key's
// dim), so one v_dot2 with (p_2k, p_2k+1) gives both keys' share of one dim.
__device__ __forceinline__ void fl_pair8(fl_v4 a, fl_v4 b, __half * dst) {
    fl_v4 lo, hi;
    lo.x = (a.x & 0x0000ffffu) | (b.x << 16);   lo.y = (a.x >> 16) | (b.x & 0xffff0000u);
    lo.z = (a.y & 0x0000ffffu) | (b.y << 16);   lo.w = (a.y >> 16) | (b.y & 0xffff0000u);
    hi.x = (a.z & 0x0000ffffu) | (b.z << 16);   hi.y = (a.z >> 16) | (b.z & 0xffff0000u);
    hi.z = (a.w & 0x0000ffffu) | (b.w << 16);   hi.w = (a.w >> 16) | (b.w & 0xffff0000u);
    *(fl_v4 *) dst = lo;
    *(fl_v4 *) (dst + 8) = hi;
}
// A V item is one key pair: both keys' 8 dims at one dim octet. The keys outside the row's
// [lo, rp] are zeroed at the store: their p is 0, so they would add nothing anyway, but the
// cache row past a row's position may hold anything. The mask goes on the store and not on
// the load: a masked load would still wait for its data at the next barrier.
__device__ __forceinline__ void fl_ld_v(const __half * vc, int tile, int lo, int rp, int full, int tid,
                                       fl_v4 & a, fl_v4 & b) {
    const int i = tid, pk = i / FL_CH, c = i - pk * FL_CH;
    const int j0 = min(max(tile * FL_TP + 2 * pk, lo), rp), j1 = min(max(tile * FL_TP + 2 * pk + 1, lo), rp);
    const size_t r0 = full ? (size_t) j0 : (size_t) (j0 & (PF_RING - 1));
    const size_t r1 = full ? (size_t) j1 : (size_t) (j1 & (PF_RING - 1));
    a = *(const fl_v4 *) (vc + r0 * KVL + c * 8);
    b = *(const fl_v4 *) (vc + r1 * KVL + c * 8);
}
__device__ __forceinline__ void fl_st_v(__half * vb, int tile, int lo, int rp, int tid, fl_v4 a, fl_v4 b) {
    const int i = tid, pk = i / FL_CH, c = i - pk * FL_CH;
    const int j0 = tile * FL_TP + 2 * pk, j1 = j0 + 1;
    const unsigned m0 = j0 >= lo && j0 <= rp ? ~0u : 0u, m1 = j1 >= lo && j1 <= rp ? ~0u : 0u;
    fl_pair8(a & m0, b & m1, vb + ((size_t) pk * HD + c * 8) * 2);
}
// q, f16 and pre-scaled by log2(e)/sqrt(HD), is staged once per row into LDS: the dots then
// read it as wave-uniform 16 B broadcasts, so it costs no VGPRs and no HBM latency.
__device__ __forceinline__ void fl_st_q(__half * qb, const __half * qrow, int lane) {
    if (lane < GQ * HD / 8) *(fl_v4 *) (qb + lane * 8) = *(const fl_v4 *) (qrow + lane * 8);
}
__device__ __forceinline__ fl_v4 fl_qld(const __half * p) { return *(const fl_v4 *) p; }

// One wave's state for one (row, kv head, split): m in log2 units, l per lane until the wave
// sum at the end, o for the lane's 2 output dims. NH of the GQ slots are live.
struct fl_st { float m[GQ], l[GQ], o[GQ][2]; };
__device__ __forceinline__ void fl_init(fl_st & s) {
    #pragma unroll
    for (int h = 0; h < GQ; h++) { s.m[h] = FL_FBIG; s.l[h] = 0.f; s.o[h][0] = 0.f; s.o[h][1] = 0.f; }
}

// ---- the tile. Lane = key for the scores: the lane's own key row of K is in LDS, its 16
// chunks swizzled by lane & 7, and q comes from LDS. 16 chunks x 4 dots x NH heads.
template <int NH>
__device__ __forceinline__ void fl_qk(const __half * krow, int sw, const __half * qb, float (&s)[NH]) {
    float a0[NH], a1[NH];
    #pragma unroll
    for (int h = 0; h < NH; h++) { a0[h] = 0.f; a1[h] = 0.f; }
    // 4 chunks at a time: the LDS reads are ~30 cycles, and one wave per SIMD has nothing
    // else to hide them. Fully unrolled the scheduler hoists every read and spills.
    #pragma unroll 4
    for (int c = 0; c < FL_CH; c++) {
        const fl_v4 k = *(const fl_v4 *) (krow + (c ^ sw) * 8);
        #pragma unroll
        for (int h = 0; h < NH; h++) {
            const fl_v4 q = fl_qld(qb + h * HD + c * 8);
            float x0 = a0[h], x1 = a1[h];
            x0 = __builtin_amdgcn_fdot2(fl_h2(q.x), fl_h2(k.x), x0, false);
            x1 = __builtin_amdgcn_fdot2(fl_h2(q.y), fl_h2(k.y), x1, false);
            x0 = __builtin_amdgcn_fdot2(fl_h2(q.z), fl_h2(k.z), x0, false);
            x1 = __builtin_amdgcn_fdot2(fl_h2(q.w), fl_h2(k.w), x1, false);
            a0[h] = x0; a1[h] = x1;
        }
    }
    #pragma unroll
    for (int h = 0; h < NH; h++) s[h] = a0[h] + a1[h];
}
// The online softmax of one tile in log2 units: one wave max per (tile,
// head), p = exp2(s - m) as f16 into this wave's slot, l kept per lane, the accumulators
// rescaled once per tile. A tile whose keys are all masked leaves the state untouched.
template <int NH>
__device__ __forceinline__ void fl_softmax(fl_st & st, float (&s)[NH], __half * prow, int lane) {
    #pragma unroll
    for (int h = 0; h < NH; h++) {
        const float mt = fl_wmax(s[h]);
        const float mn = fmaxf(st.m[h], mt);
        if (mn > FL_FBIG) {
            const float al = __builtin_amdgcn_exp2f(st.m[h] - mn);      // m = FBIG: 0
            const __half ph = __float2half(s[h] > FL_FBIG ? __builtin_amdgcn_exp2f(s[h] - mn) : 0.f);
            st.l[h] = st.l[h] * al + __half2float(ph);
            prow[h * FL_TP + lane] = ph;
            if (al != 1.f) { st.o[h][0] *= al; st.o[h][1] *= al; }
            st.m[h] = mn;
        } else {
            prow[h * FL_TP + lane] = __float2half(0.f);
        }
    }
}
// PV: lane = 2 output dims, key pairs ascending. V is [pair][dim][2], so the lane's 8 B are
// dims 2 * lane and 2 * lane + 1 of both keys, and one v_dot2 per (head, dim) takes
// (p_2k, p_2k+1) as src0. All lanes read the same P word: a broadcast.
template <int NH>
__device__ __forceinline__ void fl_pv(fl_st & st, const __half * vb, const __half * prow, int lane) {
    const int d0 = lane * 2;
    #pragma unroll 8
    for (int k8 = 0; k8 < FL_TP / 8; k8++) {
        uint4 pp[NH];
        #pragma unroll
        for (int h = 0; h < NH; h++) pp[h] = *(const uint4 *) (prow + h * FL_TP + k8 * 8);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint2 vv = *(const uint2 *) (vb + ((size_t) (k8 * 4 + j) * HD + d0) * 2);
            const __half2 va = __builtin_bit_cast(__half2, vv.x), vc = __builtin_bit_cast(__half2, vv.y);
            #pragma unroll
            for (int h = 0; h < NH; h++) {
                const unsigned int pj = j == 0 ? pp[h].x : j == 1 ? pp[h].y : j == 2 ? pp[h].z : pp[h].w;
                const __half2 p2 = __builtin_bit_cast(__half2, pj);
                st.o[h][0] = __builtin_amdgcn_fdot2(p2, va, st.o[h][0], false);
                st.o[h][1] = __builtin_amdgcn_fdot2(p2, vc, st.o[h][1], false);
            }
        }
    }
}
// One tile for this wave: scores plus the table bias, softmax, P, PV. tb and qb are the
// wave's head 0. A tile entirely further back than REL_EXT takes the per-head constant
// tb[h][REL_EXT] instead of the gather.
template <int NH>
__device__ __forceinline__ void fl_tile(fl_st & st, const __half * kb, const __half * vb, __half * prow,
                                        const __half * qb, const float * tb, int rp, int lo, int tile, int lane) {
    float s[NH];
    fl_qk<NH>(kb + (size_t) lane * HD, lane & 7, qb, s);
    const int j = tile * FL_TP + lane;
    const bool ok = j <= rp && j >= lo;
    const int d = min(rp - j, REL_EXT);
    const bool far = rp - (tile * FL_TP + FL_TP - 1) > REL_EXT;
    #pragma unroll
    for (int h = 0; h < NH; h++) {
        const float b = far ? tb[h * TBW + REL_EXT] : tb[h * TBW + max(d, 0)];
        s[h] = ok ? s[h] + b * FL_L2E : FL_FBIG;
    }
    fl_softmax<NH>(st, s, prow, lane);
    fl_pv<NH>(st, vb, prow, lane);
}
// The wave's share of the partial: m, l and its 2 dims per head, ATT_PW floats per head in
// the layout both callers' merges read.
template <int NH>
__device__ __forceinline__ void fl_store_part(float * part, fl_st & st, int lane, int h0) {
    #pragma unroll
    for (int h = 0; h < NH; h++) {
        const float l = qz_wsum(st.l[h]);
        float * B = part + (size_t) (h0 + h) * ATT_PW;
        B[2 + 2 * lane] = st.o[h][0];
        B[3 + 2 * lane] = st.o[h][1];
        if (lane == 0) { B[0] = st.m[h]; B[1] = l; }
    }
}
// A partial absorbs another: both rescaled by their exponents against the joint max.
// m = m2 = FBIG leaves the state unchanged.
__device__ __forceinline__ void fl_join(float & m, float & l, float & o, float m2, float l2, float o2) {
    const float M = fmaxf(m, m2);
    const float a = m == FL_FBIG ? 0.f : __builtin_amdgcn_exp2f(m - M);
    const float b = m2 == FL_FBIG ? 0.f : __builtin_amdgcn_exp2f(m2 - M);
    l = l * a + l2 * b;
    o = o * a + o2 * b;
    m = M;
}
// Merge the FL_NS splits of one (row, kv head) in ascending split order, from the sink
// (logit 0 with value 0: m = 0, l = 1, o = 0), then the gate and the int8 planes. Thread
// hh * HD + d of the block takes (head hh, dim d), so a 32-lane half is one int8 group.
__device__ __forceinline__ void fl_merge(const float * part, const float * gate, unsigned char * oq, float * os, int h0, int hh, int d) {
    float M = 0.f, L = 1.f, O = 0.f;
    const float * B = part + (size_t) hh * ATT_PW;
    #pragma unroll 1
    for (int sp = 0; sp < FL_NS; sp++) {
        const float * S = B + (size_t) sp * GQ * ATT_PW;
        fl_join(M, L, O, S[0], S[1], S[2 + d]);
    }
    const int i = (h0 + hh) * HD + d;
    const float y = O / L * sigm(gate[i]);
    quant32(y, oq, os, i);
}

// ---- the LDS layout of the phase, laid out by the caller. The plain step needs one P and
// one q slot and two K buffers; the verify one P and one q slot per row and one K buffer.
struct fl_lds { __half * k, * k2, * v, * p, * q; };
constexpr size_t FL_LDS1 = (size_t) 3 * FL_KH * 2 + (size_t) FL_NW * GQ * FL_TP * 2 + (size_t) GQ * HD * 2;
constexpr size_t FL_LDS8 = (size_t) 2 * FL_KH * 2 + (size_t) FL_NW * (GQ * FL_TP + GQ * HD) * 2;
// 6 * D * 4 is ink2_cycle.h's CYC_LDS, the decode's dynamic LDS (asserted there)
static_assert(FL_LDS1 <= (size_t) (6 * D * 4) && FL_LDS8 <= (size_t) (6 * D * 4), "the attention fits the decode's dynamic LDS");

__device__ __forceinline__ fl_lds fl_layout1(float * lds) {
    __half * k = (__half *) lds, * k2 = k + FL_KH, * v = k2 + FL_KH;
    return fl_lds{k, k2, v, v + FL_KH, v + FL_KH + (size_t) FL_NW * GQ * FL_TP};
}
__device__ __forceinline__ fl_lds fl_layout8(float * lds) {
    __half * k = (__half *) lds, * v = k + FL_KH;
    return fl_lds{k, nullptr, v, v + FL_KH, v + FL_KH + (size_t) FL_NW * GQ * FL_TP};
}

// ---- the plain step's body: one (kv head, split) unit per block, wave w = q head w (the
// other waves only stage), K double buffered so tile t+1's store runs under tile t's math.
__device__ __forceinline__ void fl_plain_run(const attn_args & a, const fl_lds & L, int kvh, int sp, int tid) {
    const int wave = tid >> 6, lane = tid & 63;
    const bool live = wave < GQ;                                 // wave-uniform
    const int hp = live ? wave : 0;                              // this wave's q head
    const float * tb = qz_gp(a.tb) + (size_t) (kvh * GQ + hp) * TBW;
    const __half * qrow = qz_gp(a.qh) + (size_t) (kvh * GQ + hp) * HD;
    const int rp = a.pos, lo = a.full ? 0 : max(0, rp - (WIN - 1));
    const __half * kc = qz_gp(a.kc) + (size_t) kvh * HD, * vc = qz_gp(a.vc) + (size_t) kvh * HD;
    __half * prow = L.p + (size_t) wave * GQ * FL_TP, * qb = L.q + (size_t) hp * HD;
    fl_st st;
    fl_init(st);
    if (live) { fl_st_q(qb, qrow, lane); fl_lds_wait(); }
    const int t0 = lo / FL_TP, t1 = rp / FL_TP, o = (sp - t0 % FL_NS + FL_NS) % FL_NS;
    int t = t0 + o;
    // a split with no tile stores (FBIG, 0, 0), which every fl_join leaves unchanged: the
    // merge must not read a previous application's partial. The skip is block-uniform, and
    // the barriers are inside it, so every wave of the block takes the same side.
    if (t <= t1) {
        fl_v4 rk0, rk1, rva, rvb;
        fl_ld_k(kc, t, lo, rp, a.full, tid, rk0, rk1);
        fl_ld_v(vc, t, lo, rp, a.full, tid, rva, rvb);
        fl_lds_wait();                                           // the prologue's q is in LDS
        fl_st_k(L.k, tid, rk0, rk1);                             // K(t0) -> buffer 0
        for (int buf = 0;; buf ^= 1) {
            __syncthreads();                                     // V(t - 1) readers are done
            fl_st_v(L.v, t, lo, rp, tid, rva, rvb);
            __syncthreads();                                     // K(t) and V(t) are in LDS
            const int tn = min(t + FL_NS, t1);
            fl_ld_k(kc, tn, lo, rp, a.full, tid, rk0, rk1);       // in flight during the math
            fl_ld_v(vc, tn, lo, rp, a.full, tid, rva, rvb);
            __half * kb = buf ? L.k2 : L.k;                       // this tile's K buffer
            if (live) fl_tile<1>(st, kb, L.v, prow, qb, tb, rp, lo, t, lane);
            if (t + FL_NS > t1) break;
            t += FL_NS;
            fl_st_k(buf ? L.k : L.k2, tid, rk0, rk1);             // K(t + FL_NS) -> the other
        }
    }
    if (live) fl_store_part<1>(qz_gp(a.part) + (size_t) (kvh * FL_NS + sp) * GQ * ATT_PW, st, lane, hp);
}

// The plain step: one unit (kv head, split) per block. Whichever block finds the kv head's
// counter at FL_NS - 1 merges the splits.
__device__ __forceinline__ void fl_attn_plain(const attn_args & a, int kvh, int sp, int tid, float * lds) {
    __shared__ int last;
    fl_plain_run(a, fl_layout1(lds), kvh, sp, tid);
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int old = atomicAdd(&a.cnt[kvh], 1);
        last = old == FL_NS - 1;
        if (last) { a.cnt[kvh] = 0; __threadfence(); }
    }
    __syncthreads();
    if (!last) return;                                           // block-uniform
    if (tid < GQ * HD)
        fl_merge(qz_gp(a.part) + (size_t) kvh * FL_NS * GQ * ATT_PW, qz_gp(a.gate), qz_gp(a.oq), qz_gp(a.os),
                 kvh * GQ, tid >> 7, tid & (HD - 1));
}

// ---- the verify's phase A: one (kv head, split) unit per block, wave w = row w, every wave
// running all GQ heads. Rows past nrow stage but compute nothing and write nothing, so the
// barriers and the staged bytes do not depend on nrow and rows < nrow are untouched by it.
__device__ __forceinline__ void fl_attn_rows(const attn_args * at0, size_t ast, float * part, size_t pst, int pos,
                                            int kvh, int sp, int tid, int nrow, float * lds) {
    const fl_lds L = fl_layout8(lds);
    const int wave = tid >> 6, lane = tid & 63;
    // idle waves mirror the last row. row, and with it every per-row pointer, is
    // wave-uniform: readfirstlane keeps them scalar, or they cost a VGPR pair each
    const int row = (int) __builtin_amdgcn_readfirstlane((unsigned int) min(wave, nrow - 1));
    const attn_args & a0 = *at0;                                 // row 0: the cache and the layer
    const attn_args & ea = *(const attn_args *) ((const char *) at0 + (size_t) row * ast);
    const float * tb = qz_gp(ea.tb) + (size_t) kvh * GQ * TBW;
    const __half * qrow = qz_gp(ea.qh) + (size_t) kvh * GQ * HD;
    const int rp = pos + row, lo = a0.full ? 0 : max(0, rp - (WIN - 1));
    const int rp1 = pos + nrow - 1, lo0 = a0.full ? 0 : max(0, pos - (WIN - 1));
    const int t1 = rp1 / FL_TP;
    const bool live = wave < nrow;                               // wave-uniform
    const __half * kc = qz_gp(a0.kc) + (size_t) kvh * HD, * vc = qz_gp(a0.vc) + (size_t) kvh * HD;
    __half * prow = L.p + (size_t) wave * GQ * FL_TP;            // this wave's, not the row's
    __half * qb = L.q + (size_t) wave * GQ * HD;
    fl_st st;
    fl_init(st);
    if (live) fl_st_q(qb, qrow, lane);
    const int t0 = lo0 / FL_TP, o = (sp - t0 % FL_NS + FL_NS) % FL_NS;
    int t = t0 + o;
    // a unit with no tile stores (FBIG, 0, 0) for every row, which every fl_join leaves
    // unchanged: the merge must not read an earlier round's partials. The skip is
    // block-uniform, so every wave takes the same side of the barriers inside it.
    if (t <= t1) {
        // K and V of the next tile are in flight during this tile's math. The tile index is
        // clamped, never branched on, so the loads are unconditional.
        fl_v4 rk0, rk1, rva, rvb;
        fl_ld_k(kc, t, lo0, rp1, a0.full, tid, rk0, rk1);
        fl_ld_v(vc, t, lo0, rp1, a0.full, tid, rva, rvb);
        for (; t <= t1; t += FL_NS) {
            __syncthreads();                                     // the previous tile is done
            fl_st_k(L.k, tid, rk0, rk1);
            fl_st_v(L.v, t, lo0, rp1, tid, rva, rvb);
            __syncthreads();                                     // staged: every wave may read
            const int tn = min(t + FL_NS, t1);
            fl_ld_k(kc, tn, lo0, rp1, a0.full, tid, rk0, rk1);
            fl_ld_v(vc, tn, lo0, rp1, a0.full, tid, rva, rvb);
            if (live) fl_tile<GQ>(st, L.k, L.v, prow, qb, tb, rp, lo, t, lane);
        }
    }
    // part is the caller's buffer (the plain step passes attn_args.part, the verify its ws),
    // so the partials of a row live at row * pst
    if (live) fl_store_part<GQ>(qz_gp(part) + (size_t) row * pst + (size_t) (kvh * FL_NS + sp) * GQ * ATT_PW, st, lane, 0);
}
