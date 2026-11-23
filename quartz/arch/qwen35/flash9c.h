// fl9c: fl9's split-K attention as a phase of the persistent T-row cycle
// (block = kvh * FL9_NS + split, one partial per query per split).
#pragma once
#include "flash9.h"

template <int T> __host__ __device__ constexpr int qz_fl9c_qw() { return (T * 6 + 7) / 8; }

template <int T, int HQ, int HKV>
__device__ __forceinline__ void qz_fl9c_flash(const __half * __restrict__ kv_k, const __half * __restrict__ kv_v,
                                              const __half * __restrict__ q_f16,
                                              float * __restrict__ pm, float * __restrict__ pl, float * __restrict__ pacc,
                                              int pos, __half * buf, __half * pw, int tid) {
    constexpr int GQA = HQ / HKV, NT = T * GQA, QW = qz_fl9c_qw<T>(), NS = FL9_NS;
    static_assert(GQA == 6, "fl9c: 6 q heads per kv head");
    if (blockIdx.x >= HKV * NS) return;                   // block-uniform (the cycle runs 60)
    // q comes through scalar loads: the grid barriers do not invalidate the
    // scalar cache
    __builtin_amdgcn_s_dcache_inv();
    const int w = tid >> 6, lane = tid & 63;
    const int kvh = blockIdx.x / NS, sp = blockIdx.x % NS;
    const int ntile = (pos + T + FL9_TP - 1) / FL9_TP;
    const int per = (ntile + NS - 1) / NS;
    const int tl0 = sp * per, tl1 = min(ntile, tl0 + per);
    const int d0 = lane * 4;
    constexpr float SC = 0.0625f * 1.4426950408889634f;   // 1/sqrt(256), log2 units
    // this wave's queries (wave-uniform)
    const int nq = w < NT ? (NT - w + 7) / 8 : 0;
    int np[QW];
    const __half * qrow[QW];
    #pragma unroll
    for (int j = 0; j < QW; j++) {
        const int qi = j < nq ? w + 8 * j : w;
        const int t = qi / GQA, hl = qi % GQA;
        np[j] = pos + t + 1;
        qrow[j] = q_f16 + (size_t) __builtin_amdgcn_readfirstlane((t * HQ + kvh * GQA + hl) * QZ_HD);
    }
    float m_run[QW], l_ln[QW], ac[QW][4];
    #pragma unroll
    for (int j = 0; j < QW; j++) {
        m_run[j] = QZ_FBIG; l_ln[j] = 0.f;
        #pragma unroll
        for (int e = 0; e < 4; e++) ac[j][e] = 0.f;
    }
    __half * prow = pw + (size_t) w * QW * FL9_TP;
    if (tl0 < tl1) {
        const int kp = tid >> 3, kc = tid & 7;
        const int vk = tid >> 4, vc = tid & 15;
        const int plim = pos + T;                         // rows >= plim are not written yet
        fl9_v4 rk0, rk1, rk2, rk3, rv0, rv1, rv2, rv3;
#define FL9C_LOAD_K(tl_) do {                                                                  \
            const __half * src_ = kv_k + qz_kv_row(min((tl_) * FL9_TP + kp, plim - 1), HKV, kvh) + kc * 8; \
            rk0 = *(const fl9_v4 *) (src_); rk1 = *(const fl9_v4 *) (src_ + 64);                \
            rk2 = *(const fl9_v4 *) (src_ + 128); rk3 = *(const fl9_v4 *) (src_ + 192); } while (0)
#define FL9C_STORE_K() do {                                                                     \
            __half * dst_ = buf + kp * QZ_HD; const int x_ = kp & 7;                              \
            *(fl9_v4 *) (dst_ + (((kc +  0) ^ x_) * 8)) = rk0; *(fl9_v4 *) (dst_ + (((kc +  8) ^ x_) * 8)) = rk1; \
            *(fl9_v4 *) (dst_ + (((kc + 16) ^ x_) * 8)) = rk2; *(fl9_v4 *) (dst_ + (((kc + 24) ^ x_) * 8)) = rk3; } while (0)
#define FL9C_LOAD_V(tl_) do {                                                                  \
            const int p_ = (tl_) * FL9_TP + 2 * vk;                                               \
            const __half * s0_ = kv_v + qz_kv_row(min(p_, plim - 1), HKV, kvh) + vc * 8;          \
            const __half * s1_ = kv_v + qz_kv_row(min(p_ + 1, plim - 1), HKV, kvh) + vc * 8;      \
            rv0 = *(const fl9_v4 *) (s0_); rv1 = *(const fl9_v4 *) (s1_);                         \
            rv2 = *(const fl9_v4 *) (s0_ + 128); rv3 = *(const fl9_v4 *) (s1_ + 128); } while (0)
#define FL9C_STORE_V(tl_) do {                                                                  \
            const int p_ = (tl_) * FL9_TP + 2 * vk;                                               \
            const unsigned int m0_ = p_ < plim ? ~0u : 0u, m1_ = p_ + 1 < plim ? ~0u : 0u;       \
            __half * dst_ = buf + ((size_t) vk * QZ_HD + vc * 8) * 2;                             \
            fl9_pair(rv0 & m0_, rv1 & m1_, dst_); fl9_pair(rv2 & m0_, rv3 & m1_, dst_ + 16 * 8 * 2); } while (0)
        FL9C_LOAD_K(tl0);
        FL9C_LOAD_V(tl0);
        for (int tl = tl0; tl < tl1; tl++) {
            __syncthreads();                              // previous tile's PV reads are done
            FL9C_STORE_K();
            __syncthreads();
            if (tl + 1 < tl1) FL9C_LOAD_K(tl + 1);
            // ---- scores: lane = position p, the full dot for each of the wave's queries
            const int p = tl * FL9_TP + lane;
            float s[QW];
            {
                float s0[QW], s1[QW];
                #pragma unroll
                for (int j = 0; j < QW; j++) { s0[j] = 0.f; s1[j] = 0.f; }
                const __half * kr = buf + lane * QZ_HD;
                const int sw = lane & 7;
                #pragma unroll 1
                for (int c = 0; c < 32; c += 4) {
                    const fl9_v4 k0 = *(const fl9_v4 *) (kr + ((c ^ sw) * 8));
                    const fl9_v4 k1 = *(const fl9_v4 *) (kr + (((c + 1) ^ sw) * 8));
                    const fl9_v4 k2 = *(const fl9_v4 *) (kr + (((c + 2) ^ sw) * 8));
                    const fl9_v4 k3 = *(const fl9_v4 *) (kr + (((c + 3) ^ sw) * 8));
                    fl9_v16 qv[QW];
                    #pragma unroll
                    for (int j = 0; j < QW; j++) qv[j] = fl9_sload16(qrow[j] + c * 8);
                    #pragma unroll
                    for (int j = 0; j < QW; j++) {
                        if (j < nq) {
                            const fl9_v16 q = qv[j];
                            float a0 = s0[j], a1 = s1[j];
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
                            s0[j] = a0; s1[j] = a1;
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < QW; j++) s[j] = p < np[j] ? (s0[j] + s1[j]) * SC : QZ_FBIG;
            }
            // ---- online softmax (log2 units), P to LDS as fp16
            #pragma unroll
            for (int j = 0; j < QW; j++) {
                if (j < nq) {
                    const float mt = fl9_wmax(s[j]);
                    const float mn = fmaxf(m_run[j], mt);
                    if (mn > QZ_FBIG) {
                        const float al = __builtin_amdgcn_exp2f(m_run[j] - mn);   // m_run = FBIG: 0
                        const __half ph = __float2half(s[j] > QZ_FBIG ? __builtin_amdgcn_exp2f(s[j] - mn) : 0.f);
                        l_ln[j] = l_ln[j] * al + __half2float(ph);
                        prow[j * FL9_TP + lane] = ph;
                        if (al != 1.f) {
                            #pragma unroll
                            for (int e = 0; e < 4; e++) ac[j][e] *= al;
                        }
                        m_run[j] = mn;
                    } else {
                        prow[j * FL9_TP + lane] = __float2half(0.f);
                    }
                }
            }
            __syncthreads();                              // every wave is done with K
            FL9C_STORE_V(tl);
            __syncthreads();
            if (tl + 1 < tl1) FL9C_LOAD_V(tl + 1);
            // ---- PV: lane = dims d0 .. d0 + 3, position pairs
            #pragma unroll 2
            for (int k8 = 0; k8 < FL9_TP / 8; k8++) {
                uint4 pp[QW];
                #pragma unroll
                for (int j = 0; j < QW; j++) pp[j] = *(const uint4 *) (prow + j * FL9_TP + k8 * 8);
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) {
                    const uint4 vv = *(const uint4 *) (buf + ((size_t) (k8 * 4 + jj) * QZ_HD + d0) * 2);
                    #pragma unroll
                    for (int j = 0; j < QW; j++) {
                        if (j < nq) {
                            const unsigned int pj = jj == 0 ? pp[j].x : jj == 1 ? pp[j].y : jj == 2 ? pp[j].z : pp[j].w;
                            const qz_h2 p2 = __builtin_bit_cast(qz_h2, pj);
                            ac[j][0] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.x), ac[j][0], false);
                            ac[j][1] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.y), ac[j][1], false);
                            ac[j][2] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.z), ac[j][2], false);
                            ac[j][3] = __builtin_amdgcn_fdot2(p2, __builtin_bit_cast(qz_h2, vv.w), ac[j][3], false);
                        }
                    }
                }
            }
        }
#undef FL9C_LOAD_K
#undef FL9C_STORE_K
#undef FL9C_LOAD_V
#undef FL9C_STORE_V
    }
    // one partial per query: slot (t HQ + kvh 6 + hl) NS + sp
    #pragma unroll
    for (int j = 0; j < QW; j++) {
        if (j < nq) {
            const int qi = w + 8 * j, t = qi / GQA, hl = qi % GQA;
            const float l = fl9_wsum(l_ln[j]);
            const size_t slot = (size_t) (t * HQ + kvh * GQA + hl) * NS + sp;
            *(float4 *) (pacc + slot * QZ_HD + d0) = make_float4(ac[j][0], ac[j][1], ac[j][2], ac[j][3]);
            if (lane == 0) { pm[slot] = m_run[j]; pl[slot] = l; }
        }
    }
    __syncthreads();                                      // buf / pw are the cycle's scratch again
}

// merge the FL9_NS partials of (row t, head h) + the gate sigmoid / requant;
// ws: 33 floats of LDS, called by every thread of the block
template <int T, int HQ>
__device__ __forceinline__ void qz_fl9c_reduce(const float * __restrict__ pm, const float * __restrict__ pl,
                                               const float * __restrict__ pacc,
                                               const float * __restrict__ gate, int gate_stride, float * __restrict__ attn_out,
                                               unsigned char * __restrict__ gv_vals, float * __restrict__ gv_scal,
                                               float * ws, int row_h, int tid) {
    constexpr int NS = FL9_NS;
    const int h = row_h % HQ;
    const size_t base = (size_t) row_h * NS;
    if (tid < 64) {
        const float mi = tid < NS ? pm[base + tid] : QZ_FBIG;
        const float M = fl9_wmax(mi);
        const float wgt = tid < NS && mi > QZ_FBIG ? __builtin_amdgcn_exp2f(mi - M) : 0.f;
        if (tid < 32) ws[tid] = wgt;
        const float L = fl9_wsum(tid < NS ? wgt * pl[base + tid] : 0.f);
        if (tid == 0) ws[32] = L > 0.f ? 1.f / L : 0.f;
    }
    __syncthreads();
    if (tid < QZ_HD) {
        float os = 0.f;
        #pragma unroll 6
        for (int s = 0; s < NS; s++) os += ws[s] * pacc[(base + s) * QZ_HD + tid];
        os *= ws[32];
        attn_out[(size_t) h * QZ_HD + tid] = os;
        const float g = gate[(size_t) h * gate_stride + tid];
        const float x = os * (1.f / (1.f + expf(-g)));   // gate sigmoid
        float amax = fabsf(x);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
        const float d = amax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        int q = (int) roundf(x * id);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        gv_vals[(size_t) h * QZ_HD + tid] = (unsigned char) q;
        if ((tid & 31) == 0) gv_scal[h * 8 + (tid >> 5)] = __half2float(__float2half(d));
    }
    __syncthreads();
}
