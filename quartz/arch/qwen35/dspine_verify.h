#include <initializer_list>

#include "devices.h"
#include <type_traits>

#ifndef MV_T
#define MV_T 8           // verify rows (the anchor + 7 drafts)
#endif
#ifndef MV_TR
#define MV_TR MV_T
#endif
#ifndef MV_LCHK
#define MV_LCHK() do { const hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) { \
    fprintf(stderr, "mv launch %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#endif

#undef D
#undef FF
#undef G
#undef CCH
#undef HV
#undef S
#undef AQ
#undef AKV
#undef HQ
#undef HKV
#undef HD
#undef BLKPOS

// ------------------------------------------------------------ kernels ----
__global__ void __launch_bounds__(512) mv_phase_a(qz_trunk_args A, const int * toks_d, int il) {
    extern __shared__ char lds_raw[];
    float * xn = (float *) lds_raw;                              // [QZ_D]
    float * red = (float *)(lds_raw + QZ_D * 4);                // [9]
    const int tid = threadIdx.x, n_u = QZ_D / 32;
    const qz_trunk_w * W = &A.w[il];
    const int parl = (il & 1), parn = ((il + 1) & 1);
    float * xprev = A.x_rows + (size_t) parl * MV_T * QZ_D;
    float * xin = A.x_rows + (size_t) parn * MV_T * QZ_D;
    const int t0 = gridDim.y > 1 ? (int) blockIdx.y : 0, t1 = gridDim.y > 1 ? t0 + 1 : MV_T;
    for (int t = t0; t < t1; t++) {
        float ss = 0.f;
        __syncthreads();          // xin writes of row t-1 visible before row t reads
        if (il > 0) {
            float xr[QZ_D / 512];
            #pragma unroll
            for (int k = 0; k < QZ_D / 512; k++) {
                const int i = tid + 512 * k;
                xr[k] = xprev[(size_t) t * QZ_D + i] + A.y_ffn[(size_t) t * QZ_D + i];
            }
            #pragma unroll
            for (int k = 0; k < QZ_D / 512; k++) {
                const int i = tid + 512 * k;
                xn[i] = xr[k];
                ss = fmaf(xr[k], xr[k], ss);          // the rolled loop's v_fmac
                if (blockIdx.x == 0) xin[(size_t) t * QZ_D + i] = xr[k];
            }
        } else {
            const int tok = toks_d[t];
            const int * em = (const int *) A.emb_vals;
            const __half * es = A.emb_scal;
            float ss2 = 0.f;
            (void) ss2;
            for (int i = tid; i < QZ_D; i += 512) {
                const unsigned int w = em[(size_t) tok * (QZ_D / 4) + (i >> 2)];
                int b = (int) ((w >> ((i & 3) * 8)) & 0xff);
                if (b > 127) b -= 256;
                const float x = (float) b * __half2float(es[(size_t) tok * n_u + (i >> 5)]);
                xn[i] = x;
                ss += x * x;
                if (blockIdx.x == 0) xin[(size_t) t * QZ_D + i] = x;
            }
        }
        qz_norm_requant_slice(xn, W->attn_norm, ss, A.eps, A.nq_vals + (size_t) t * QZ_D,
                              A.nq_scal + (size_t) t * n_u, red, tid);
    }
}

__global__ void __launch_bounds__(512) mv_phase_g(qz_trunk_args A, int il) {
    extern __shared__ char lds_raw[];
    float * xn = (float *) lds_raw;
    float * red = (float *)(lds_raw + QZ_D * 4);
    const int tid = threadIdx.x, n_u = QZ_D / 32;
    const qz_trunk_w * W = &A.w[il];
    float * xin = A.x_rows + (size_t) ((il + 1) & 1) * MV_T * QZ_D;
    const int t0 = gridDim.y > 1 ? (int) blockIdx.y : 0, t1 = gridDim.y > 1 ? t0 + 1 : MV_T;
    for (int t = t0; t < t1; t++) {
        float ss = 0.f;
        __syncthreads();
        float xr[QZ_D / 512];
        #pragma unroll
        for (int k = 0; k < QZ_D / 512; k++)
            xr[k] = xin[(size_t) t * QZ_D + tid + 512 * k] + A.y_core[(size_t) t * QZ_D + tid + 512 * k];
        #pragma unroll
        for (int k = 0; k < QZ_D / 512; k++) {
            const int i = tid + 512 * k;
            xn[i] = xr[k];
            ss = fmaf(xr[k], xr[k], ss);
        }
        qz_norm_requant_slice(xn, W->post_norm, ss, A.eps, A.nq_vals + (size_t) t * QZ_D,
                              A.nq_scal + (size_t) t * n_u, red, tid);
    }
}


__global__ void __launch_bounds__(512) mv_phase_dn(qz_trunk_args A, int il, int par, int n_rep) {
    using Gm = qz_geo<MV_T>;
    extern __shared__ char lds_raw[];
    const qz_trunk_w * W = &A.w[il];
    if (blockIdx.x >= Gm::HV) return;
    const int h = blockIdx.x;
    const int j = threadIdx.x & (QZ_S - 1), r = threadIdx.x >> 7;
    using Gr = qz_geo<MV_TR>;
    const float * rcur = W->ring + (size_t) par * Gr::RING_PAR;
    const float * rprev = W->ring + (size_t) (par ^ 1) * Gr::RING_PAR;
    qz_dn_rows<MV_T, MV_TR, MV_TR>(W->dn_state + ((size_t) h << 14) + (size_t) r * 32 * QZ_S + j, n_rep,
                              rprev, rcur, A.y_gate, W->dt_bias, W->ssm_a, W->ssm_norm_w,
                              A.pl_vals, A.pl_scal, h, A.eps, (qz_dn_rows_lds<MV_T, MV_TR> *) lds_raw, threadIdx.x);
}


// ------------------------------------------------------- verify GEMVs ----
// T-row GEMVs: mv_gemv8P (T = 8), mv_gemvR (T = 12), up to 4 planes per launch
struct mv_seg { const char * w; const __half * ws; int N; float * y; int b0; };
struct mv_segs { mv_seg s[4]; int n; };
#ifndef MV_RL
#define MV_RL 32
#endif
#ifndef MV_RS
#define MV_RS 2
#endif
#ifndef MV_RW
#define MV_RW 8
#endif
#ifndef MV_PS
#define MV_PS 2
#endif
#ifndef MV_PW
#define MV_PW 16
#endif
#define MV_PU (65536 / (MV_T * 32))              // units staged by mv_gemv8P (MV_T * 32 * 256 = 64 KB)
template <int S, int NW>
__global__ void __launch_bounds__(64 * NW) mv_gemv8P(const unsigned char * a_vals, const float * a_scal, int K,
                                                    mv_segs G, int ngr) {
    extern __shared__ uint4 lsa[];
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6, tid = threadIdx.x;
    const int n_u = K / 32, lsw = K / 32, ldw = K;
    const int nst = min(n_u, MV_PU);           // staged units; the rest (down's last 16) from global
    for (int i = tid; i < MV_T * nst * 2; i += 64 * NW) {
        const int t = i / (nst * 2), r = i % (nst * 2);
        lsa[((size_t) t * 2 + (r & 1)) * nst + (r >> 1)] = *(const uint4 *)(a_vals + (size_t) t * K + r * 16);
    }
    __syncthreads();
    const __amdgpu_buffer_rsrc_t ra = __builtin_amdgcn_make_buffer_rsrc((void *) a_vals, (short) 0, MV_T * K, 0x00027000);
    const int gw = __builtin_amdgcn_readfirstlane(blockIdx.x * NW + wave);   // wave-uniform: segment state in SGPRs
    for (int g = gw; g < ngr; g += gridDim.x * NW) {
        int i = 0;
        #pragma unroll
        for (int j = 1; j < 4; j++) if (j < G.n && g >= G.s[j].b0) i = j;
        const char * w_vals = G.s[i].w;
        const __half * w_scal = G.s[i].ws;
        const int N = G.s[i].N;
        float * y = G.s[i].y;
        const int r0 = (g - G.s[i].b0) * S;
        float acc[MV_T][S];
        #pragma unroll
        for (int t = 0; t < MV_T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++) acc[t][s] = 0.f;
        const int rb = __builtin_amdgcn_readfirstlane(r0);
        __amdgpu_buffer_rsrc_t rv[S], rs[S];
        #pragma unroll
        for (int s = 0; s < S; s++) {
            const int r = min(rb + s, N - 1);
            rv[s] = __builtin_amdgcn_make_buffer_rsrc((void *)(w_vals + (size_t) r * ldw), (short) 0, n_u * 32, 0x00027000);
            rs[s] = __builtin_amdgcn_make_buffer_rsrc((void *)(w_scal + (size_t) r * lsw), (short) 0, n_u * 2, 0x00027000);
        }
        // one unit step; GL: activations from global (the units past nst)
        auto step = [&](int u, auto gl) {
            float sag[MV_T];
            #pragma unroll
            for (int t = 0; t < MV_T; t++) sag[t] = a_scal[(size_t) t * (K / 32) + u];
            qz_v4u w0[S], w1[S];
            unsigned short sh[S];
            #pragma unroll
            for (int s = 0; s < S; s++) {
                w0[s] = __builtin_bit_cast(qz_v4u, __builtin_amdgcn_raw_buffer_load_b128(rv[s], u * 32, 0, 0));
                w1[s] = __builtin_bit_cast(qz_v4u, __builtin_amdgcn_raw_buffer_load_b128(rv[s], u * 32 + 16, 0, 0));
                sh[s] = __builtin_amdgcn_raw_buffer_load_b16(rs[s], u * 2, 0, 0);
            }
            float sw[S];
            #pragma unroll
            for (int s = 0; s < S; s++) sw[s] = __half2float(__ushort_as_half(sh[s]));
            #pragma unroll
            for (int t = 0; t < MV_T; t++) {
                uint4 a0, a1;
                if constexpr (decltype(gl)::value) {
                    a0 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(ra, t * K + u * 32, 0, 0));
                    a1 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(ra, t * K + u * 32 + 16, 0, 0));
                } else {
                    a0 = lsa[((size_t) t * 2) * nst + u];
                    a1 = lsa[((size_t) t * 2 + 1) * nst + u];
                }
                const float sa = sag[t];
                #pragma unroll
                for (int s = 0; s < S; s++) {
                    int d = 0;
                    DOT4(d, w0[s].x, a0.x);
                    DOT4(d, w0[s].y, a0.y);
                    DOT4(d, w0[s].z, a0.z);
                    DOT4(d, w0[s].w, a0.w);
                    DOT4(d, w1[s].x, a1.x);
                    DOT4(d, w1[s].y, a1.y);
                    DOT4(d, w1[s].z, a1.z);
                    DOT4(d, w1[s].w, a1.w);
                    acc[t][s] += (float) d * sw[s] * sa;
                }
            }
        };

        int u = lane;
        for (; u < nst; u += 64) step(u, std::false_type{});
        for (; u < n_u; u += 64) step(u, std::true_type{});
        #pragma unroll
        for (int t = 0; t < MV_T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++)
                #pragma unroll
                for (int off = 32; off > 0; off >>= 1) acc[t][s] += __shfl_down(acc[t][s], off, 64);
        if (lane == 0)
            #pragma unroll
            for (int t = 0; t < MV_T; t++)
                #pragma unroll
                for (int s = 0; s < S; s++)
                    if (r0 + s < N) y[(size_t) t * N + r0 + s] = acc[t][s];
    }
}

template <int S, int NW, int LG>
__global__ void __launch_bounds__(64 * NW) mv_gemvR(const unsigned char * a_vals, const float * a_scal, int K,
                                                 mv_segs G, int ngr, int nkc = 1, size_t ystr = 0) {
    constexpr int NQ = 64 / LG;
    extern __shared__ uint4 lsa[];
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6, tid = threadIdx.x;
    const int qd = lane / LG, ql = lane % LG;
    const int lsw = K / 32, ldw = K;
    // K chunks: blocks [c nb, (c+1) nb) take units [u0, u0 + n_u); chunk c's partial goes to y + c ystr
    const int nb = gridDim.x / nkc, kc = blockIdx.x / nb, bl = blockIdx.x % nb;
    const int u0 = kc * (K / 32) / nkc, n_u = (kc + 1) * (K / 32) / nkc - u0;
    const int scb = (MV_T * n_u * 2 + 15) & ~15;
    const int nst = min(n_u, (65536 - scb) / (MV_T * 32));
    __half * lsc = (__half *) (lsa + (size_t) MV_T * 2 * nst);
    for (int i = tid; i < MV_T * nst * 2; i += 64 * NW) {
        const int t = i / (nst * 2), r = i % (nst * 2);
        lsa[((size_t) t * 2 + (r & 1)) * nst + (r >> 1)] = *(const uint4 *)(a_vals + (size_t) t * K + u0 * 32 + r * 16);
    }
    for (int i = tid; i < MV_T * n_u; i += 64 * NW) {
        const int t = i / n_u, u = i % n_u;
        lsc[u * MV_T + t] = __float2half(a_scal[(size_t) t * lsw + u0 + u]);
    }
    __syncthreads();
    const __amdgpu_buffer_rsrc_t ra = __builtin_amdgcn_make_buffer_rsrc((void *) a_vals, (short) 0, MV_T * K, 0x00027000);
    const int gw = __builtin_amdgcn_readfirstlane(bl * NW + wave);
    for (int g = gw; g < ngr; g += nb * NW) {
        int i = 0;
        #pragma unroll
        for (int j = 1; j < 4; j++) if (j < G.n && g >= G.s[j].b0) i = j;
        const char * w_vals = G.s[i].w;
        const __half * w_scal = G.s[i].ws;
        const int N = G.s[i].N;
        float * y = G.s[i].y + (size_t) kc * ystr;
        const int r0 = (g - G.s[i].b0) * S * NQ + qd * S;
        float acc[MV_T][S];
        #pragma unroll
        for (int t = 0; t < MV_T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++) acc[t][s] = 0.f;
        const char * wr[S]; const __half * wsr[S];
        #pragma unroll
        for (int s = 0; s < S; s++) {
            const int r = min(r0 + s, N - 1);
            wr[s] = w_vals + (size_t) r * ldw + u0 * 32; wsr[s] = w_scal + (size_t) r * lsw + u0;
        }
        struct WB { qz_v4u w0[S], w1[S]; unsigned short sh[S]; };
        auto ld = [&](int u, WB & b) {
            #pragma unroll
            for (int s = 0; s < S; s++) {
                b.w0[s] = __builtin_nontemporal_load((const qz_v4u *)(wr[s] + u * 32));
                b.w1[s] = __builtin_nontemporal_load((const qz_v4u *)(wr[s] + u * 32 + 16));
                b.sh[s] = __builtin_nontemporal_load((const unsigned short *) (wsr[s] + u));
            }
        };
        auto comp = [&](int u, const WB & b) {
            float sw[S];
            #pragma unroll
            for (int s = 0; s < S; s++) sw[s] = __half2float(__ushort_as_half(b.sh[s]));
            const bool st = u < nst;
            #pragma unroll
            for (int t = 0; t < MV_T; t++) {
                uint4 a0, a1;
                if (st) { a0 = lsa[((size_t) t * 2) * nst + u]; a1 = lsa[((size_t) t * 2 + 1) * nst + u]; }
                else {
                    a0 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(ra, t * K + (u0 + u) * 32, 0, 0));
                    a1 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(ra, t * K + (u0 + u) * 32 + 16, 0, 0));
                }
                const float sa = __half2float(lsc[u * MV_T + t]);
                #pragma unroll
                for (int s = 0; s < S; s++) {
                    int d = 0;
                    DOT4(d, b.w0[s].x, a0.x); DOT4(d, b.w0[s].y, a0.y); DOT4(d, b.w0[s].z, a0.z); DOT4(d, b.w0[s].w, a0.w);
                    DOT4(d, b.w1[s].x, a1.x); DOT4(d, b.w1[s].y, a1.y); DOT4(d, b.w1[s].z, a1.z); DOT4(d, b.w1[s].w, a1.w);
                    acc[t][s] += (float) d * sw[s] * sa;
                }
            }
        };
        WB A, B;
        int u = ql;
        if (u < n_u) ld(u, A);
        while (u < n_u) {
            int un = u + LG;
            if (un < n_u) ld(un, B);
            comp(u, A);
            u = un;
            if (u >= n_u) break;
            un = u + LG;
            if (un < n_u) ld(un, A);
            comp(u, B);
            u = un;
        }
        #pragma unroll
        for (int t = 0; t < MV_T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++)
                #pragma unroll
                for (int off = LG / 2; off > 0; off >>= 1) acc[t][s] += __shfl_xor(acc[t][s], off, LG);
        if (ql == 0)
            #pragma unroll
            for (int t = 0; t < MV_T; t++)
                #pragma unroll
                for (int s = 0; s < S; s++)
                    if (r0 + s < N) y[(size_t) t * N + r0 + s] = acc[t][s];
    }
}

// mv_gemv8P stages MV_PU units of the rows in LDS and reads the rest (at most
// 64 per lane) from global memory: every K the verify uses must fit that
static_assert(MV_T > 8 || (QZ_D / 32 <= MV_PU + 64 && QZ_G / 64 <= MV_PU + 64 && QZ_FF / 64 <= MV_PU + 64),
              "verify GEMV: a K too wide for mv_gemv8P");
inline void mv_gemv8P_launch(hipStream_t st, const unsigned char * a, const float * as, int K,
                             std::initializer_list<mv_seg> segs) {
    mv_segs G = {};
    int b = 0;
    for (const mv_seg & x : segs) { G.s[G.n] = x; G.s[G.n].b0 = b; b += (x.N + MV_PS - 1) / MV_PS; G.n++; }
    if constexpr (MV_T > 8 && MV_T <= 12) {
        constexpr int RPG = MV_RS * (64 / MV_RL);
        G = {}; b = 0;
        for (const mv_seg & x : segs) { G.s[G.n] = x; G.s[G.n].b0 = b; b += (x.N + RPG - 1) / RPG; G.n++; }
        mv_gemvR<MV_RS, MV_RW, MV_RL><<<dim3(60), dim3(64 * MV_RW), 65536, st>>>(a, as, K, G, b);
    } else
        mv_gemv8P<MV_PS, MV_PW><<<dim3(60), dim3(64 * MV_PW), (size_t) MV_T * 32 * std::min(K / 32, MV_PU), st>>>(a, as, K, G, b);
}
// down at > 8 rows: T x 8704 activations do not fit 64 KB, so the grid splits
// K in MV_DKC chunks; chunk c's partial lands at y + c T N (y_part sized for it)
#ifndef MV_DKC
#define MV_DKC 2
#endif
inline void mv_gemv_down_launch(hipStream_t st, const unsigned char * a, const float * as, int K,
                                const char * w, const __half * ws, int N, float * y) {
    constexpr int RPG = MV_RS * (64 / MV_RL);
    mv_segs G = {};
    G.s[0] = { w, ws, N, y, 0 }; G.n = 1;
    mv_gemvR<MV_RS, MV_RW, MV_RL><<<dim3(60), dim3(64 * MV_RW), 65536, st>>>(a, as, K, G, (N + RPG - 1) / RPG, MV_DKC,
                                                                          (size_t) MV_T * N);
}
inline void mv_gemv8_launch(hipStream_t st, const unsigned char * a, const float * as, int K,
                            const char * w, const __half * ws, int N, float * y) {
    mv_gemv8P_launch(st, a, as, K, { { w, ws, N, y, 0 } });
}

__global__ void __launch_bounds__(256) mv_qkv_conv(const float * y_qkv, int N, float * conv_state,
                                                   const float * conv1d, float * conv_out,
                                                   const float * prev_qkv, int n_rep) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= N) return;
    const float * w = conv1d + c * 4;
    float c0_ = conv_state[c], c1_ = conv_state[N + c], c2_ = conv_state[2 * N + c];
    for (int t = 0; t < n_rep; t++) {
        c0_ = c1_; c1_ = c2_; c2_ = prev_qkv[(size_t) t * N + c];
    }
    if (n_rep > 0) {
        conv_state[c] = c0_;
        conv_state[N + c] = c1_;
        conv_state[2 * N + c] = c2_;
    }
    #pragma unroll
    for (int t = 0; t < MV_T; t++) {
        const float cur = y_qkv[(size_t) t * N + c];
        const float o = c0_ * w[0] + c1_ * w[1] + c2_ * w[2] + cur * w[3];
        conv_out[(size_t) t * N + c] = o / (1.f + expf(-o));
        c0_ = c1_; c1_ = c2_; c2_ = cur;
    }
}

__global__ void __launch_bounds__(512) mv_xfin(qz_trunk_args A, unsigned int tok, int down) {
    {
        const int par = tok & 1;
        float * dst = A.xin_peer + (size_t) par * MV_T * QZ_D;
        for (int g = blockIdx.x; g * 32 < QZ_D; g += gridDim.x)
            for (int i = threadIdx.x; i < 8 * MV_T; i += blockDim.x) {
                const int t = i >> 3, q = i & 7;
                const size_t o = (size_t) t * QZ_D + g * 32 + q * 4;
                float4 v = *(const float4 *)(A.y_part + o);
                if (MV_T > 8 && MV_T <= 12 && down) {
                    #pragma unroll
                    for (int c = 1; c < MV_DKC; c++) {
                        const float4 w = *(const float4 *)(A.y_part + (size_t) c * MV_T * QZ_D + o);
                        v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
                    }
                    *(float4 *)(A.y_part + o) = v;
                }
                *(float4 *)(dst + o) = v;
            }
        __threadfence_system();
    }
    int kb = 0;
    qz_xfinish_t<MV_T, MV_T, false>(nullptr, kb, A, tok, down ? A.y_ffn : A.y_core);
}

__global__ void __launch_bounds__(256) mv_ffn_act(qz_trunk_args A, int il, const float * yg, const float * yu, int N) {
    unsigned char * ffn_q = A.ffn_q;
    float * ffn_sc = A.ffn_sc;
    {   // the phase's residual promote xin += y_core, spread over the grid
        float * xin = A.x_rows + (size_t) ((il + 1) & 1) * MV_T * QZ_D;
        for (int i = blockIdx.x * 256 + threadIdx.x; i < MV_T * QZ_D; i += gridDim.x * 256)
            xin[i] = xin[i] + A.y_core[i];
    }
    const int wv = blockIdx.x * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    const int ng = N / 32;
    if (wv >= MV_T * ng) return;
    const int t = wv / ng, g = wv % ng;
    float xu = 0.f, ax = 0.f;
    if (lane < 32) {
        const float gg = yg[(size_t) t * N + g * 32 + lane];
        xu = (gg / (1.f + expf(-gg))) * yu[(size_t) t * N + g * 32 + lane];
        ax = fabsf(xu);
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ax = fmaxf(ax, __shfl_xor(ax, off, 64));
    const float d = ax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    if (lane < 32) {
        int q = (int) roundf(xu * id);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        ffn_q[(size_t) t * N + g * 32 + lane] = (unsigned char) q;
    }
    if (lane == 0) ffn_sc[(size_t) t * (N / 32) + g] = __half2float(__float2half(d));
}


__global__ void __launch_bounds__(512) mv_phase_rope(qz_trunk_args A, int il, int pos) {
    using Gm = qz_geo<MV_T>;
    extern __shared__ char lds_raw[];
    const qz_trunk_w * W = &A.w[il];
    float * red = (float *) lds_raw;                             // [9]
    __half * aq = (__half *)(red + 16);                         // [QZ_HD]
    const int tid = threadIdx.x;
    for (int it = blockIdx.x; it < MV_T * Gm::HQ; it += gridDim.x) {
        const int t = it / Gm::HQ, h = it % Gm::HQ;
        qz_head_norm_rope(A.y_q + (size_t) t * Gm::AQ + (size_t) h * 2 * QZ_HD, W->q_norm_w,
                          pos + t + A.rope_shift, A.eps, aq, red, tid);
        if (tid < QZ_HD)
            A.q_f16[(size_t) t * Gm::HQ * QZ_HD + (size_t) h * QZ_HD + tid] = aq[tid];
        __syncthreads();
    }
    for (int it = blockIdx.x; it < MV_T * Gm::HKV; it += gridDim.x) {
        const int t = it / Gm::HKV, kh = it % Gm::HKV;
        const int p = pos + t;
        const size_t row = qz_kv_row(p, Gm::HKV, kh);
        qz_head_norm_rope(A.y_k + (size_t) t * Gm::AKV + (size_t) kh * QZ_HD, W->k_norm_w,
                          p + A.rope_shift, A.eps, aq, red, tid);
        if (tid < QZ_HD) W->kv_k[row + tid] = aq[tid];
        if (tid < QZ_HD) W->kv_v[row + tid] = __float2half(A.y_v[(size_t) t * Gm::AKV + (size_t) kh * QZ_HD + tid]);
        __syncthreads();
    }
}

#include "flash9.h"
// the verify attention: flash9.h's split-K kernel + its reduce
inline void mv_fl9_init() {
    using Gm = qz_geo<MV_T>;
    CHECK(hipFuncSetAttribute((const void *) mv_fl9_k<Gm::HQ, Gm::HKV>, hipFuncAttributeMaxDynamicSharedMemorySize, FL9_LDS));
}
inline void mv_fl9_flash(const qz_trunk_args & A, const qz_trunk_w & W, int pos, hipStream_t s) {
    using Gm = qz_geo<MV_T>;
    mv_fl9_k<Gm::HQ, Gm::HKV><<<dim3(Gm::HKV * FL9_NS, (MV_T + 7) / 8), dim3(512), FL9_LDS, s>>>(W.kv_k, W.kv_v, A.q_f16, A.pm, A.pl, A.pacc, pos, FL9_NS, MV_T);
}
inline void mv_fl9_reduce(const qz_trunk_args & A, hipStream_t s) {
    using Gm = qz_geo<MV_T>;
    mv_fl9_red<Gm::HQ, Gm::AQ, Gm::G><<<dim3(MV_T * Gm::HQ), dim3(256), 0, s>>>(A.pm, A.pl, A.pacc, FL9_NS, A.y_q, A.o_preg,
                                                                              A.pl_vals, A.pl_scal);
}
__global__ void __launch_bounds__(512) mv_head_norm(qz_trunk_args A) {
    extern __shared__ char lds_raw[];
    float * xn = (float *) lds_raw;
    float * red = (float *)(lds_raw + QZ_D * 4);
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= MV_T) return;
    float nwr[QZ_D / 512];
    #pragma unroll
    for (int k = 0; k < QZ_D / 512; k++) nwr[k] = A.out_norm[tid + 512 * k];
    float * xlast = A.x_rows + (size_t) (A.n_layer & 1) * MV_T * QZ_D;
    float ss = qz_res_row(xlast + (size_t) t * QZ_D, A.y_ffn + (size_t) t * QZ_D, xn, nullptr, tid);
    qz_norm_requant_row(xn, nwr, ss, A.eps, A.nq_vals + (size_t) t * QZ_D,
                        A.nq_scal + (size_t) t * (QZ_D / 32), red, tid);
    // the normed row: the MTP entry input when decode switches to MTP
    for (int i = tid; i < QZ_D; i += 512) A.h_nextn[(size_t) t * QZ_D + i] = xn[i];
}


__global__ void __launch_bounds__(512) mv_argmax(qz_trunk_args A, unsigned int htok) {
    extern __shared__ char lds_raw[];
    float * part = (float *) lds_raw;                  // [512] + int [512]
    const int tid = threadIdx.x;
    if (blockIdx.x >= MV_T) return;
    const float * yh = A.y_head + (size_t) blockIdx.x * A.hrows;
    float bv = -3.4028235e38f;
    int bi = A.hrows - 1;
    for (int r = tid; r < A.hrows; r += 512) {
        const float v = yh[r];
        if (v > bv) { bv = v; bi = r; }
    }
    part[tid] = bv;
    ((int *)(part + 512))[tid] = bi;
    __syncthreads();
    if (tid == 0) {
        float b0 = part[0];
        int i0 = ((int *)(part + 512))[0];
        for (int t2 = 1; t2 < 512; t2++) {
            const float v = part[t2];
            const int ix = ((int *)(part + 512))[t2];
            if (v > b0 || (v == b0 && ix < i0)) { b0 = v; i0 = ix; }
        }
        const int gid = i0 + A.tp_card * A.hrows;
        A.y_part[2 * blockIdx.x] = b0;
        A.y_part[2 * blockIdx.x + 1] = __int_as_float(gid);
        float * ph = A.xhm_peer + (size_t) (htok & 1) * MV_T * 2;
        *(float2 *)(ph + 2 * blockIdx.x) = make_float2(b0, __int_as_float(gid));
        __threadfence_system();
        qz_xflag_b(A, htok, 64 + blockIdx.x);
    }
}

__global__ void __launch_bounds__(64) mv_verify(qz_trunk_args A, const int * toks_d,
                                               unsigned int htok, unsigned int vtok, int verify) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void) vtok;
    for (int b = 64; b < 64 + MV_T; b++) qz_xwait_b(A, htok, b);
    const float * mh = A.xhm_mine + (size_t) (htok & 1) * MV_T * 2;
    for (int t = 0; t < MV_T; t++)
        A.out_tok[t] = qz_xpick(__builtin_nontemporal_load(A.y_part + 2 * t),
                                __float_as_int(__builtin_nontemporal_load(A.y_part + 2 * t + 1)),
                                __builtin_nontemporal_load(mh + 2 * t),
                                __float_as_int(__builtin_nontemporal_load(mh + 2 * t + 1)));
    int m = 0;
    if (verify) {
        while (m < MV_T - 1 && A.out_tok[m] == toks_d[m + 1]) m++;
    } else {
        m = MV_T - 1;
    }
    A.ctl[0] = m + 1;
    A.ctl[1] = A.out_tok[m];
    A.ctl[2] = A.pos + m + 1;
}

struct mv_ctx {
    int dev = 0;
    hipStream_t st = nullptr;
    qz_trunk_args Ah = {};                // host staging, passed by value per launch
    bool part_shared = false;              // pm / pl / pacc carved from the engine's scratch (not freed here)
    float * ring8 = nullptr;               // [n_recr][2 slots][RING_PAR(TR = 8)]
    float * xhm = nullptr;                 // head-merge inbox [2 slots][MV_T*2]
    float * g8a = nullptr, * g8b = nullptr; // ffn gate / up outputs [MV_T][FF per card]
    unsigned int * xflag = nullptr;        // [64] exchange flags
    int * ctl_d = nullptr;                // [8] ctl (both cards compute it)
    int * toks_d = nullptr;                // [MV_T] the round's row tokens
    long long * st_d = nullptr;            // the flash clock-counter dummy
    std::vector<qz_trunk_w> wh;            // host copy of the engine's per-layer
    qz_trunk_w * w_d = nullptr;             // ring pointers patched, device copy
};

inline void mv_b_recr_g8(mv_ctx & m, int il, int par, int n_rep) {
    using Gm = qz_geo<MV_T>;
    const qz_trunk_w & W = m.wh[il];
    const qz_trunk_args & A = m.Ah;
    using Gr = qz_geo<MV_TR>;
    float * rcur = W.ring + (size_t) par * Gr::RING_PAR;
    const float * rprev = W.ring + (size_t) (par ^ 1) * Gr::RING_PAR;
    mv_gemv8P_launch(m.st, A.nq_vals, A.nq_scal, QZ_D,
                     { { W.qkv_vals, W.qkv_scal, Gm::CCH, rcur + Gr::RING_QKV, 0 },
                       { W.gat_vals, W.gat_scal, Gm::G, A.y_gate, 0 },
                       { W.bet_vals, W.bet_scal, Gm::HV, rcur + Gr::RING_BETA, 0 },
                       { W.alp_vals, W.alp_scal, Gm::HV, rcur + Gr::RING_ALPH, 0 } });
    mv_qkv_conv<<<dim3((Gm::CCH + 255) / 256), dim3(256), 0, m.st>>>(rcur + Gr::RING_QKV, Gm::CCH, W.conv_state,
                                                                      W.conv1d, rcur + Gr::RING_CONV,
                                                                      rprev + Gr::RING_QKV, n_rep);
}
inline void mv_b_attn_g8(mv_ctx & m, int il) {
    using Gm = qz_geo<MV_T>;
    const qz_trunk_w & W = m.wh[il];
    const qz_trunk_args & A = m.Ah;
    mv_gemv8P_launch(m.st, A.nq_vals, A.nq_scal, QZ_D,
                     { { W.q_vals, W.q_scal, Gm::AQ, A.y_q, 0 },
                       { W.k_vals, W.k_scal, Gm::AKV, A.y_k, 0 },
                       { W.v_vals, W.v_scal, Gm::AKV, A.y_v, 0 } });
}
inline void mv_xrow_g8(mv_ctx & m, int il, int down, unsigned int tok) {
    using Gm = qz_geo<MV_T>;
    const qz_trunk_w & W = m.wh[il];
    const qz_trunk_args & A = m.Ah;
    const int par = tok & 1;
    float * dst = A.xin_peer + (size_t) par * MV_T * QZ_D;
    (void) dst;
    if (down && MV_T > 8 && MV_T <= 12) mv_gemv_down_launch(m.st, A.ffn_q, A.ffn_sc, Gm::FF, W.fd_vals, W.fd_scal, QZ_D, A.y_part);
    else if (down) mv_gemv8_launch(m.st, A.ffn_q, A.ffn_sc, Gm::FF, W.fd_vals, W.fd_scal, QZ_D, A.y_part);
    else mv_gemv8_launch(m.st, A.pl_vals, A.pl_scal, Gm::G, W.out_vals, W.out_scal, QZ_D, A.y_part);
    mv_xfin<<<dim3(60), dim3(512), 0, m.st>>>(A, tok, down);
}

// the fl9 attention partials: FL9_NS slots per (row, head)
inline size_t mv_partial_bytes() {
    const size_t pmb = (size_t) MV_T * qz_geo<MV_T>::HQ * FL9_NS * 4;
    return qz_scr_size({ pmb, pmb, pmb * QZ_HD });
}

inline int mv_init(mv_ctx & m, qz_engine & eng, int c, float * ring_share = nullptr) {
    using Gm = qz_geo<MV_T>;
    m.dev = qz_dev_of(c);
    qz_mem_scope mem("verify");
    qz_card & K = eng.C[c];
    CHECK(hipSetDevice(qz_dev_of(c)));
    m.st = K.st;
    m.Ah = K.A;                                        // weights / state / emb / head / arenas / tp card fields
    m.wh = K.wh;
    auto fz = [&](auto * p, size_t b) { CHECK(qz_dmalloc((void **) p, b)); CHECK(hipMemset(*p, 0, b)); };
    auto fu = [&](auto * p, size_t b) { CHECK(qz_dmalloc((void **) p, b)); };
    size_t n_recr = 0;
    for (const auto & W : m.wh) n_recr += W.recr != 0;
    if (ring_share) m.ring8 = ring_share;
    else fz((void **) &m.ring8, n_recr * 2 * (size_t) qz_geo<MV_TR>::RING_PAR * 4);
    CHECK(qz_dmalloc_uncached((void **) &m.xhm, (size_t) 2 * MV_T * 2 * 4));
    CHECK(hipMemset(m.xhm, 0, (size_t) 2 * MV_T * 2 * 4));
    CHECK(qz_dmalloc_uncached((void **) &m.xflag, 512));
    CHECK(hipMemset(m.xflag, 0, 512));
    fu((void **) &m.ctl_d, 8 * 4);
    fz((void **) &m.g8a, (size_t) MV_T * Gm::FF * 4);
    fz((void **) &m.g8b, (size_t) MV_T * Gm::FF * 4);
    fu((void **) &m.toks_d, (size_t) MV_T * 4);
    fu((void **) &m.st_d, 8 * sizeof(long long));
    qz_trunk_args & A = m.Ah;
    fz(&A.x_rows, (size_t) MV_T * 2 * QZ_D * 4);
    fz(&A.y_core, (size_t) MV_T * QZ_D * 4);
    fz(&A.y_ffn, (size_t) MV_T * QZ_D * 4);
    fz(&A.y_gate, (size_t) MV_T * Gm::G * 4);
    fz((void **) &A.pl_vals, (size_t) MV_T * Gm::G);
    fz(&A.pl_scal, (size_t) MV_T * (Gm::G / 32) * 4);
    fz((void **) &A.ffn_q, (size_t) MV_T * Gm::FF);
    fz(&A.ffn_sc, (size_t) MV_T * (Gm::FF / 32) * 4);
    fz((void **) &A.nq_vals, (size_t) MV_T * QZ_D);
    fz((void **) &A.nq_scal, (size_t) MV_T * (QZ_D / 32) * 4);
    fz((void **) &A.q_f16, (size_t) MV_T * Gm::HQ * QZ_HD * 2);
    fz(&A.y_q, (size_t) MV_T * Gm::AQ * 4);
    fz(&A.y_k, (size_t) MV_T * Gm::AKV * 4);
    fz(&A.y_v, (size_t) MV_T * Gm::AKV * 4);
    {
        const size_t pmb = (size_t) MV_T * Gm::HQ * FL9_NS * 4, pab = pmb * QZ_HD;
        m.part_shared = K.scr_carve({ { (void **) &A.pm, pmb }, { (void **) &A.pl, pmb }, { (void **) &A.pacc, pab } });
        if (!m.part_shared) {
            qz_mem_scope mem_part("verify partials");
            fz(&A.pm, pmb);
            fz(&A.pl, pmb);
            fz(&A.pacc, pab);
        }
    }
    fz(&A.o_preg, (size_t) MV_T * Gm::HQ * QZ_HD * 4);
    fz(&A.h_nextn, (size_t) MV_T * QZ_D * 4);          // normed rows: MTP entries on a switch
    fz(&A.y_head, (size_t) MV_T * eng.hrows * 4);
    fz(&A.out_tok, (size_t) MV_T * 4);
    fz(&A.y_part, (size_t) MV_DKC * MV_T * QZ_D * 4);
    CHECK(qz_dmalloc_uncached((void **) &A.xin_mine,
                             ((size_t) 2 * MV_T * QZ_D + (size_t) 2 * MV_T * (QZ_D / 32)) * 4));
    CHECK(hipMemset(A.xin_mine, 0, ((size_t) 2 * MV_T * QZ_D + (size_t) 2 * MV_T * (QZ_D / 32)) * 4));
    A.xhm_mine = m.xhm;
    A.xflag_mine = m.xflag;
    A.ctl = m.ctl_d;
    A.mtp = nullptr;
    A.hrows = eng.hrows;
    size_t nr = 0;
    for (auto & W : m.wh)
        if (W.recr) { W.ring = m.ring8 + nr * 2 * (size_t) qz_geo<MV_TR>::RING_PAR; nr++; }
    fu((void **) &m.w_d, m.wh.size() * sizeof(qz_trunk_w));
    CHECK(hipMemcpy(m.w_d, m.wh.data(), m.wh.size() * sizeof(qz_trunk_w), hipMemcpyHostToDevice));
    A.w = m.w_d;
    const int ls[] = { (int) (QZ_D * 4 + 12 * 4 + 64),                         // phase a
                       (int) sizeof(qz_dn_rows_lds<MV_T, MV_TR>),               // dn scan
                       (int) (512 * 4 * 2),                                    // argmax fold
                       (int) (QZ_D * 4 + 12 * 4 + 64),                         // phase g
                       (int) (QZ_D * 4 + 12 * 4 + 64) };                       // head norm
    const void * ks[] = { (const void *) mv_phase_a, (const void *) mv_phase_dn,
                         (const void *) mv_argmax,
                         (const void *) mv_phase_g, (const void *) mv_head_norm };
    static_assert(sizeof(ks) / sizeof(ks[0]) == sizeof(ls) / sizeof(ls[0]), "one LDS size per kernel");
    for (int i = 0; i < (int) (sizeof(ks) / sizeof(ks[0])); i++) {
        const hipError_t e_ = hipFuncSetAttribute(ks[i], hipFuncAttributeMaxDynamicSharedMemorySize, ls[i]);
        if (e_ != hipSuccess) { fprintf(stderr, "mv attr %d (size %d): %s\n", i, ls[i], hipGetErrorString(e_)); return 1; }
    }
    CHECK(hipFuncSetAttribute((const void *) mv_gemv8P<MV_PS, MV_PW>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536));
    if (MV_T > 8 && MV_T <= 12) CHECK(hipFuncSetAttribute((const void *) mv_gemvR<MV_RS, MV_RW, MV_RL>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536));
    mv_fl9_init();
    return 0;
}

inline void mv_wire_peers(mv_ctx * c) {
    for (int i = 0; i < 2; i++) {
        c[i].Ah.xin_peer = c[1 - i].Ah.xin_mine;
        c[i].Ah.xhm_peer = c[1 - i].Ah.xhm_mine;
        c[i].Ah.xflag_peer = c[1 - i].xflag;
    }
}

static const int mv_ldnk[5] = { 1, 16, 31, 46, 61 };   // mirrors PF_FEAT_LAYERS

static float * mv_cap_rows = nullptr;   // [MV_T][5][QZ_D] f32, device 0
inline float * mv_feat_cap(qz_engine & eng) {
    if (!mv_cap_rows) {
        CHECK(hipSetDevice(qz_dev_of(0)));
        CHECK(qz_dmalloc(&mv_cap_rows, (size_t) MV_T * 5 * QZ_D * 4));
    }
    return mv_cap_rows;
}
static float * mv_cap_rows1 = nullptr;  // [MV_T][5][QZ_D] f32, device 1
inline float * mv_feat_cap1() {
    if (!mv_cap_rows1) {
        CHECK(hipSetDevice(qz_dev_of(1)));
        CHECK(qz_dmalloc(&mv_cap_rows1, (size_t) MV_T * 5 * QZ_D * 4));
        CHECK(hipSetDevice(qz_dev_of(0)));
    }
    return mv_cap_rows1;
}


inline void mv_round(mv_ctx * c, int pos, int cycle, int n_rep, int verify,
                     const int32_t * toks_h, unsigned int & xtok) {
    const int par = cycle & 1;
    for (int i = 0; i < 2; i++) {
        mv_ctx & m = c[i];
        CHECK(hipSetDevice(m.dev));
        m.Ah.pos = pos;
        m.Ah.cycle = cycle;
        m.Ah.verify = verify;
        CHECK(hipMemcpyAsync(m.toks_d, toks_h, (size_t) MV_T * 4, hipMemcpyHostToDevice, m.st));
    }
    const int lds_a = QZ_D * 4 + 12 * 4 + 64;
    const int lds_dn = (int) sizeof(qz_dn_rows_lds<MV_T, MV_TR>);
    const int lds_am = 512 * 4 * 2;
    unsigned int htok = ++xtok;
    for (int il = 0; il < (int) c[0].wh.size(); il++) {
        const bool recr = c[0].wh[il].recr != 0;
        for (int i = 0; i < 2; i++) {
            mv_ctx & m = c[i];
            CHECK(hipSetDevice(m.dev));
            mv_phase_a<<<dim3(10, MV_T), dim3(512), lds_a, m.st>>>(m.Ah, m.toks_d, il);
            MV_LCHK();
            MV_LCHK();
        }
        if (mv_cap_rows) {
            int fq = -1;
            for (int q2 = 0; q2 < 5; q2++) if (mv_ldnk[q2] == il - 1) fq = q2;
            if (fq >= 0) {
                const int parn = ((il + 1) & 1);
                CHECK(hipSetDevice(qz_dev_of(0)));
                CHECK(hipMemcpy2DAsync(mv_cap_rows + (size_t) fq * QZ_D, (size_t) 5 * QZ_D * 4,
                                       c[0].Ah.x_rows + (size_t) parn * MV_T * QZ_D, (size_t) QZ_D * 4,
                                       (size_t) QZ_D * 4, MV_T, hipMemcpyDeviceToDevice, c[0].st));
                if (mv_cap_rows1) {
                    CHECK(hipSetDevice(qz_dev_of(1)));
                    CHECK(hipMemcpy2DAsync(mv_cap_rows1 + (size_t) fq * QZ_D, (size_t) 5 * QZ_D * 4,
                                           c[1].Ah.x_rows + (size_t) parn * MV_T * QZ_D, (size_t) QZ_D * 4,
                                           (size_t) QZ_D * 4, MV_T, hipMemcpyDeviceToDevice, c[1].st));
                    CHECK(hipSetDevice(qz_dev_of(0)));
                }
            }
        }
        if (recr) {
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_b_recr_g8(m, il, par, n_rep);
            MV_LCHK();
            }
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_phase_dn<<<dim3(qz_geo<MV_T>::HV), dim3(512), lds_dn, m.st>>>(m.Ah, il, par, n_rep);
            MV_LCHK();
            }
            ++xtok;
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_xrow_g8(m, il, 0, xtok);
            MV_LCHK();
            }
        } else {
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_b_attn_g8(m, il);
            MV_LCHK();
            }
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_phase_rope<<<dim3(60), dim3(512), 640, m.st>>>(m.Ah, il, pos);
            MV_LCHK();
            }
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_fl9_flash(m.Ah, m.wh[il], pos, m.st);
            MV_LCHK();
            }
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_fl9_reduce(m.Ah, m.st);
            MV_LCHK();
            }
            ++xtok;
            for (int i = 0; i < 2; i++) {
                mv_ctx & m = c[i];
                CHECK(hipSetDevice(m.dev));
                mv_xrow_g8(m, il, 0, xtok);
            MV_LCHK();
            }
        }
        for (int i = 0; i < 2; i++) {
            mv_ctx & m = c[i];
            CHECK(hipSetDevice(m.dev));
            mv_phase_g<<<dim3(10, MV_T), dim3(512), lds_a, m.st>>>(m.Ah, il);
            MV_LCHK();
        }
        for (int i = 0; i < 2; i++) {
            mv_ctx & m = c[i];
            CHECK(hipSetDevice(m.dev));
            const qz_trunk_w & Wl = m.wh[il];
            const int N = qz_geo<>::FF;
            mv_gemv8P_launch(m.st, m.Ah.nq_vals, m.Ah.nq_scal, QZ_D,
                             { { Wl.fg_vals, Wl.fg_scal, N, m.g8a, 0 }, { Wl.fu_vals, Wl.fu_scal, N, m.g8b, 0 } });
            mv_ffn_act<<<dim3((MV_T * (N / 32) + 3) / 4), dim3(256), 0, m.st>>>(m.Ah, il, m.g8a, m.g8b, N);
            MV_LCHK();
        }
        ++xtok;
        for (int i = 0; i < 2; i++) {
            mv_ctx & m = c[i];
            CHECK(hipSetDevice(m.dev));
            mv_xrow_g8(m, il, 1, xtok);
            MV_LCHK();
        }
    }
    for (int i = 0; i < 2; i++) {
        mv_ctx & m = c[i];
        CHECK(hipSetDevice(m.dev));
        mv_head_norm<<<dim3(MV_T), dim3(512), lds_a, m.st>>>(m.Ah);
            MV_LCHK();
    }
    for (int i = 0; i < 2; i++) {
        mv_ctx & m = c[i];
        CHECK(hipSetDevice(m.dev));
        mv_gemv8_launch(m.st, m.Ah.nq_vals, m.Ah.nq_scal, QZ_D, m.Ah.hd_vals, m.Ah.hd_scal, m.Ah.hrows, m.Ah.y_head);
            MV_LCHK();
    }
    for (int i = 0; i < 2; i++) {
        mv_ctx & m = c[i];
        CHECK(hipSetDevice(m.dev));
        mv_argmax<<<dim3(MV_T), dim3(512), lds_am, m.st>>>(m.Ah, htok);
            MV_LCHK();
    }
    const unsigned int vtok = ++xtok;
    for (int i = 0; i < 2; i++) {
        mv_ctx & m = c[i];
        CHECK(hipSetDevice(m.dev));
        mv_verify<<<dim3(1), dim3(64), 0, m.st>>>(m.Ah, m.toks_d, htok, vtok, verify);
            MV_LCHK();
    }
    CHECK(hipSetDevice(qz_dev_of(0)));
}
