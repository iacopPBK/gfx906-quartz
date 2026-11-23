// fl9: the verify attention as a split-K flash pass (block = kv head x split,
// wave = row, lane = position, P in fp16 for v_dot2 PV) plus a merge.
#pragma once

#ifndef FL9_NS
#define FL9_NS 30          // splits per kv head (grid = FKV * FL9_NS = 60 blocks: one per CU)
#endif
#define FL9_TP 64          // positions per tile
#define FL9_LDS (FL9_TP * QZ_HD * 2 + 8 * 6 * FL9_TP * 2)

__device__ __forceinline__ float fl9_rl(float v, int l) {
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), l));
}
__device__ __forceinline__ float fl9_wmax(float v) {      // wave max, uniform
    v = qz_amax32(v);
    return fmaxf(fl9_rl(v, 0), fl9_rl(v, 32));
}

typedef unsigned int fl9_v4 __attribute__((ext_vector_type(4)));
// rows a, b (8 dims each) -> [dim][2] pairs, 2 x 16 B at dst
__device__ __forceinline__ void fl9_pair(fl9_v4 a, fl9_v4 b, __half * dst) {
    fl9_v4 lo, hi;
    lo.x = __builtin_amdgcn_perm(b.x, a.x, 0x05040100u); lo.y = __builtin_amdgcn_perm(b.x, a.x, 0x07060302u);
    lo.z = __builtin_amdgcn_perm(b.y, a.y, 0x05040100u); lo.w = __builtin_amdgcn_perm(b.y, a.y, 0x07060302u);
    hi.x = __builtin_amdgcn_perm(b.z, a.z, 0x05040100u); hi.y = __builtin_amdgcn_perm(b.z, a.z, 0x07060302u);
    hi.z = __builtin_amdgcn_perm(b.w, a.w, 0x05040100u); hi.w = __builtin_amdgcn_perm(b.w, a.w, 0x07060302u);
    *(fl9_v4 *) dst = lo;
    *(fl9_v4 *) (dst + 8) = hi;
}

__device__ __forceinline__ qz_h2 fl9_h2(unsigned int u) { return __builtin_bit_cast(qz_h2, u); }
// a 64 B load at a wave-uniform address as s_load_dwordx16 (constant
// address space + readfirstlane'd pointer): the dots take the q words as SGPRs
typedef unsigned int fl9_v16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ fl9_v16 fl9_sload16(const __half * p) {
    const unsigned long long a = (unsigned long long) p;
    const unsigned int lo = __builtin_amdgcn_readfirstlane((unsigned int) a);
    const unsigned int hi = __builtin_amdgcn_readfirstlane((unsigned int) (a >> 32));
    return *(const __attribute__((address_space(4))) fl9_v16 *) (((unsigned long long) hi << 32) | lo);
}

template <int FQ, int FKV>
__global__ void __launch_bounds__(512) mv_fl9_k(const __half * __restrict__ kv_k, const __half * __restrict__ kv_v,
                                                const __half * __restrict__ q_f16,
                                                float * __restrict__ pm, float * __restrict__ pl,
                                                float * __restrict__ pacc, int pos, int ns, int nrows = 8) {
    const int T = min((int) blockIdx.y * 8 + 8, nrows);   // rows of this group and below
    constexpr int GQA = FQ / FKV;
    static_assert(GQA == 6, "fl9: 6 q heads per kv head");
    extern __shared__ char lds_raw[];
    __half * buf = (__half *) lds_raw;                    // K: [64][256] (16 B chunk c at c ^ (p & 7)); V: [32][256][2]
    __half * pw = buf + FL9_TP * QZ_HD;                   // [8 waves][6][64]
    const int tid = threadIdx.x, tw = blockIdx.y * 8 + (tid >> 6), t = min(tw, nrows - 1), lane = tid & 63;   // waves past nrows mirror the last row   // blockIdx.y: group of 8 rows
    const int kvh = blockIdx.x / ns, sp = blockIdx.x % ns;
    const int ntile = (pos + T + FL9_TP - 1) / FL9_TP;
    const int per = (ntile + ns - 1) / ns;
    const int tl0 = sp * per, tl1 = min(ntile, tl0 + per);
    const int np_r = pos + t + 1;                         // row t sees positions < np_r
    const int d0 = lane * 4;
    constexpr float SC = 0.0625f * 1.4426950408889634f;   // 1/sqrt(256), log2 units

    float m_run[6], l_ln[6], ac[6][4];
    #pragma unroll
    for (int h = 0; h < 6; h++) {
        m_run[h] = QZ_FBIG; l_ln[h] = 0.f;
        #pragma unroll
        for (int e = 0; e < 4; e++) ac[h][e] = 0.f;
    }
    if (tl0 < tl1) {
        // fill mappings. K: position kp = tid >> 3, chunks kc + 8 j. V: pair
        // vk = tid >> 4, chunks vc, vc + 16, both positions of the pair.
        const int kp = tid >> 3, kc = tid & 7;
        const int vk = tid >> 4, vc = tid & 15;
        const int plim = pos + T;                         // rows >= plim are not written yet
        // the prefetch registers as named values (an array here went to scratch)
        fl9_v4 rk0, rk1, rk2, rk3, rv0, rv1, rv2, rv3;
#define FL9_LOAD_K(tl_) do {                                                                  \
            const __half * src_ = kv_k + qz_kv_row(min((tl_) * FL9_TP + kp, plim - 1), FKV, kvh) + kc * 8; \
            rk0 = *(const fl9_v4 *) (src_); rk1 = *(const fl9_v4 *) (src_ + 64);                \
            rk2 = *(const fl9_v4 *) (src_ + 128); rk3 = *(const fl9_v4 *) (src_ + 192); } while (0)
#define FL9_STORE_K() do {                                                                     \
            __half * dst_ = buf + kp * QZ_HD; const int x_ = kp & 7;                              \
            *(fl9_v4 *) (dst_ + (((kc +  0) ^ x_) * 8)) = rk0; *(fl9_v4 *) (dst_ + (((kc +  8) ^ x_) * 8)) = rk1; \
            *(fl9_v4 *) (dst_ + (((kc + 16) ^ x_) * 8)) = rk2; *(fl9_v4 *) (dst_ + (((kc + 24) ^ x_) * 8)) = rk3; } while (0)
#define FL9_LOAD_V(tl_) do {                                                                  \
            const int p_ = (tl_) * FL9_TP + 2 * vk;                                               \
            const __half * s0_ = kv_v + qz_kv_row(min(p_, plim - 1), FKV, kvh) + vc * 8;          \
            const __half * s1_ = kv_v + qz_kv_row(min(p_ + 1, plim - 1), FKV, kvh) + vc * 8;      \
            rv0 = *(const fl9_v4 *) (s0_); rv1 = *(const fl9_v4 *) (s1_);                         \
            rv2 = *(const fl9_v4 *) (s0_ + 128); rv3 = *(const fl9_v4 *) (s1_ + 128); } while (0)
// the mask at the store (not the load: no wait on the data before the next barrier)
#define FL9_STORE_V(tl_) do {                                                                  \
            const int p_ = (tl_) * FL9_TP + 2 * vk;                                               \
            const unsigned int m0_ = p_ < plim ? ~0u : 0u, m1_ = p_ + 1 < plim ? ~0u : 0u;       \
            __half * dst_ = buf + ((size_t) vk * QZ_HD + vc * 8) * 2;                             \
            fl9_pair(rv0 & m0_, rv1 & m1_, dst_); fl9_pair(rv2 & m0_, rv3 & m1_, dst_ + 16 * 8 * 2); } while (0)
        // q straight from global at a wave-uniform address (scalar loads)
        const __half * qrow = q_f16 + (size_t) __builtin_amdgcn_readfirstlane(t * FQ + kvh * GQA) * QZ_HD;
        __half * prow = pw + (size_t) (tid >> 6) * 6 * FL9_TP;   // this wave's P slot (not t: rows >= 8 and mirror waves)
        // K_{i+1} loads during QK_i, V_{i+1} during PV_i
        FL9_LOAD_K(tl0);
        FL9_LOAD_V(tl0);
        for (int tl = tl0; tl < tl1; tl++) {
            __syncthreads();                              // previous tile's PV reads are done
            FL9_STORE_K();
            __syncthreads();
            // gfx906 waits for every outstanding load at a barrier: a prefetch
            // issued right after one has the whole next phase to land
            if (tl + 1 < tl1) FL9_LOAD_K(tl + 1);
            // ---- scores: lane = position p, the full dot for the row's 6 heads
            const int p = tl * FL9_TP + lane;
            float s[6];
            {
                float s0[6], s1[6];
                #pragma unroll
                for (int h = 0; h < 6; h++) { s0[h] = 0.f; s1[h] = 0.f; }
                const __half * kr = buf + lane * QZ_HD;
                const int sw = lane & 7;
                // 32 dims per step: 4 K chunks from LDS; q 3 heads at a time as
                // 64 B scalar loads (48 SGPRs), one wait per 96 dots
                #pragma unroll 1
                for (int c = 0; c < 32; c += 4) {
                    const fl9_v4 k0 = *(const fl9_v4 *) (kr + ((c ^ sw) * 8));
                    const fl9_v4 k1 = *(const fl9_v4 *) (kr + (((c + 1) ^ sw) * 8));
                    const fl9_v4 k2 = *(const fl9_v4 *) (kr + (((c + 2) ^ sw) * 8));
                    const fl9_v4 k3 = *(const fl9_v4 *) (kr + (((c + 3) ^ sw) * 8));
                    #pragma unroll
                    for (int h3 = 0; h3 < 6; h3 += 3) {
                        fl9_v16 qv[3];
                        #pragma unroll
                        for (int h = 0; h < 3; h++) qv[h] = fl9_sload16(qrow + (h3 + h) * QZ_HD + c * 8);
                        #pragma unroll
                        for (int h = 0; h < 3; h++) {
                            const fl9_v16 q = qv[h];
                            float a0 = s0[h3 + h], a1 = s1[h3 + h];
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.s0), fl9_h2(k0.x), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.s1), fl9_h2(k0.y), a1, false);
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.s2), fl9_h2(k0.z), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.s3), fl9_h2(k0.w), a1, false);
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.s4), fl9_h2(k1.x), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.s5), fl9_h2(k1.y), a1, false);
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.s6), fl9_h2(k1.z), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.s7), fl9_h2(k1.w), a1, false);
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.s8), fl9_h2(k2.x), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.s9), fl9_h2(k2.y), a1, false);
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.sa), fl9_h2(k2.z), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.sb), fl9_h2(k2.w), a1, false);
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.sc), fl9_h2(k3.x), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.sd), fl9_h2(k3.y), a1, false);
                            a0 = __builtin_amdgcn_fdot2(fl9_h2(q.se), fl9_h2(k3.z), a0, false);
                            a1 = __builtin_amdgcn_fdot2(fl9_h2(q.sf), fl9_h2(k3.w), a1, false);
                            s0[h3 + h] = a0; s1[h3 + h] = a1;
                        }
                    }
                }
                #pragma unroll
                for (int h = 0; h < 6; h++) s[h] = p < np_r ? (s0[h] + s1[h]) * SC : QZ_FBIG;
            }
            // ---- online softmax (log2 units), P to LDS as fp16
            #pragma unroll
            for (int h = 0; h < 6; h++) {
                const float mt = fl9_wmax(s[h]);
                const float mn = fmaxf(m_run[h], mt);
                if (mn > QZ_FBIG) {
                    const float al = __builtin_amdgcn_exp2f(m_run[h] - mn);   // m_run = FBIG: 0
                    const __half ph = __float2half(s[h] > QZ_FBIG ? __builtin_amdgcn_exp2f(s[h] - mn) : 0.f);
                    l_ln[h] = l_ln[h] * al + __half2float(ph);
                    prow[h * FL9_TP + lane] = ph;
                    if (al != 1.f) {
                        #pragma unroll
                        for (int e = 0; e < 4; e++) ac[h][e] *= al;
                    }
                    m_run[h] = mn;
                } else {
                    prow[h * FL9_TP + lane] = __float2half(0.f);
                }
            }
            __syncthreads();                              // every wave is done with K
            FL9_STORE_V(tl);
            __syncthreads();
            if (tl + 1 < tl1) FL9_LOAD_V(tl + 1);
            // ---- PV: lane = dims d0 .. d0 + 3, position pairs
            #pragma unroll 2
            for (int k8 = 0; k8 < FL9_TP / 8; k8++) {
                uint4 pp[6];
                #pragma unroll
                for (int h = 0; h < 6; h++) pp[h] = *(const uint4 *) (prow + h * FL9_TP + k8 * 8);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint4 vv = *(const uint4 *) (buf + ((size_t) (k8 * 4 + j) * QZ_HD + d0) * 2);
                    #pragma unroll
                    for (int h = 0; h < 6; h++) {
                        const unsigned int pj = j == 0 ? pp[h].x : j == 1 ? pp[h].y : j == 2 ? pp[h].z : pp[h].w;
                        const qz_h2 p2 = __builtin_bit_cast(qz_h2, pj);
                        ac[h][0] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.x), ac[h][0], false);
                        ac[h][1] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.y), ac[h][1], false);
                        ac[h][2] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.z), ac[h][2], false);
                        ac[h][3] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.w), ac[h][3], false);
                    }
                }
            }
        }
    }
    // one partial per (row, head): slot ((t FQ + kvh 6 + h) ns + sp)
    #pragma unroll
    for (int h = 0; h < 6; h++) {
        const float l = qz_wsum(l_ln[h]);
        const size_t slot = (size_t) (t * FQ + kvh * GQA + h) * ns + sp;
        if (tw < nrows) *(float4 *) (pacc + slot * QZ_HD + d0) = make_float4(ac[h][0], ac[h][1], ac[h][2], ac[h][3]);
        if (lane == 0 && tw < nrows) { pm[slot] = m_run[h]; pl[slot] = l; }
    }
}

#undef FL9_LOAD_K
#undef FL9_STORE_K
#undef FL9_LOAD_V
#undef FL9_STORE_V

// merge the ns partials of one (row, head) + the gate sigmoid / requant
// tail (as qz_fl9c_reduce); one block of 256 per (row, head)
template <int FQ, int AQ, int G>
__global__ void __launch_bounds__(256) mv_fl9_red(const float * __restrict__ pm, const float * __restrict__ pl,
                                                  const float * __restrict__ pacc, int ns,
                                                  const float * __restrict__ y_q, float * __restrict__ o_preg,
                                                  unsigned char * __restrict__ pl_vals, float * __restrict__ pl_scal) {
    __shared__ float ws[64];
    __shared__ float rinv;
    const int row_h = blockIdx.x, t = row_h / FQ, h = row_h % FQ, tid = threadIdx.x;
    const size_t base = (size_t) row_h * ns;
    if (tid < 64) {
        const float mi = tid < ns ? pm[base + tid] : QZ_FBIG;
        const float M = fl9_wmax(mi);
        const float w = tid < ns && mi > QZ_FBIG ? __builtin_amdgcn_exp2f(mi - M) : 0.f;
        ws[tid] = w;
        const float L = qz_wsum(tid < ns ? w * pl[base + tid] : 0.f);
        if (tid == 0) rinv = L > 0.f ? 1.f / L : 0.f;
    }
    __syncthreads();
    float os = 0.f;
    for (int s = 0; s < ns; s++) os += ws[s] * pacc[(base + s) * QZ_HD + tid];
    os *= rinv;
    o_preg[(size_t) t * FQ * QZ_HD + (size_t) h * QZ_HD + tid] = os;
    const float g = y_q[(size_t) t * AQ + QZ_HD + (size_t) h * 2 * QZ_HD + tid];
    const float x = os * (1.f / (1.f + expf(-g)));   // gate sigmoid
    float amax = fabsf(x);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    const float d = amax / 127.f;
    const float id = qz_q8_inv(d);
    int q = qz_q8_code(x, id);
    pl_vals[(size_t) t * G + (size_t) h * QZ_HD + tid] = (unsigned char) q;
    if ((tid & 31) == 0) pl_scal[(size_t) t * (G / 32) + h * 8 + (tid >> 5)] = qz_q8_scale(d);
}
