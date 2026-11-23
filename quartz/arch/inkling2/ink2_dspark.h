// inkling2 DSpark drafter (GGUF arch "dflash" with the DSpark recurrent Markov head):
// 3 DFlash layers over a block of BS = 8 rows (the anchor token's normed target
// embedding, then 7 mask rows) that attend, non-causally among themselves, to a
// per-layer context K / V cache built from 4 target features per committed
// position: the residual entering applications 5, 23, 41 and the head's normed
// input, mixed per layer by softmax(dspark_fuse) + RMSNorm (dflare_norm; the
// Markov context g: dspark_norm). The Markov head chains the 7 drafts: joint =
// joint_proj [state | markov_w1[prev] | hidden_i | g(anchor - 1)] -> gate, cand,
// latent; the draft logits are the target head on hidden_i + hidden_out(latent),
// computed as head . hidden_i (one 7-row GEMV) + fold . [latent; 1] with
// fold = head . [hidden_out | bias] built at load (per card, its vocab half).
// The layers run on card 0; the head GEMVs / folds on both cards.
// The 7 Markov steps run as one device-side chain (no host round trip): card 0's
// joint kernel waits for the peer's argmax through peer memory flags, card 1's
// fold waits for card 0's lat1 the same way; only the 7 tokens come back.
// Included after ink2_engine.h / ink2_pf.h (same TU).

constexpr int DS_BS = 8, DS_NL = 3, DS_R = 256, DS_FF = 9216, DS_Q = 3072, DS_KV = 1024, DS_NQ = 24, DS_NKV = 8;
constexpr int DS_SPL = 16;                     // attention key splits (x 4 waves)
constexpr int DS_NT = 256;                     // the drafter GEMV's block (4 waves, S = 4 rows per wave)
constexpr float DS_EPS = 1e-6f, DS_THETA = 1e6f;
constexpr long long DS_SPIN = 2000000000ll;    // clock64 ticks a cross-card wait may spin

// ---- the Markov chain's cross-card handshake (peer memory + system-scope flags)
__device__ __forceinline__ void ds_flag(unsigned int * p, unsigned int seq) {
    __threadfence_system();
    __hip_atomic_store(p, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
// wait for the peer's flag >= seq (bounded: past DS_SPIN it sets err and goes on)
__device__ __forceinline__ void ds_wait(unsigned int * p, unsigned int seq, int * err) {
    const long long t0 = clock64();
    for (;;) {
        const unsigned int v = __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if ((int) (v - seq) >= 0) return;
        if (err && __hip_atomic_load(err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) return;
        if (clock64() - t0 > DS_SPIN) { if (err) atomicCAS(err, 0, 1); return; }
    }
}

// ---- kernels
// rows of out = RMSNorm(sum_k a_k tap_k; w): taps [row][4][D] (row stride tstride floats)
static __global__ __launch_bounds__(256) void k_ds_mixnorm(const float * taps, size_t tstride, float4 a, const float * w,
                                                           float * out, int ostride, float eps) {
    __shared__ float red[16];
    const float * t = taps + (size_t) blockIdx.x * tstride;
    float v[D / 256], ss[1] = {0.f};
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        v[k] = a.x * t[i] + a.y * t[D + i] + a.z * t[2 * D + i] + a.w * t[3 * D + i];
        ss[0] += v[k] * v[k];
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / D + eps);
    #pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int i = threadIdx.x + k * 256;
        out[(size_t) blockIdx.x * ostride + i] = v[k] * r * w[i];
    }
}
// rows (stride K) -> int8 + per-32 scales; w: RMSNorm with weight first (null: quantize as is);
// y: also the normed f32 rows
static __global__ __launch_bounds__(256) void k_ds_quant(const float * x, const float * w, int K, unsigned char * q, float * qs,
                                                         float * y, float eps) {
    __shared__ float red[16];
    const float * xr = x + (size_t) blockIdx.x * K;
    float r = 1.f;
    if (w) {
        float ss[1] = {0.f};
        for (int i = threadIdx.x; i < K; i += 256) ss[0] += xr[i] * xr[i];
        bsum<1, false>(ss, red, 4);
        r = rsqrtf(ss[0] / K + eps);
    }
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = w ? xr[i] * r * w[i] : xr[i];
        if (y) y[(size_t) blockIdx.x * K + i] = v;
        quant32(v, q + (size_t) blockIdx.x * K, qs + (size_t) blockIdx.x * (K / 32), i);
    }
}
// x += y, then the row's int8 planes from RMSNorm(x) * w (one kernel: the residual and its
// quantization, as k_ds_add + k_ds_quant did)
static __global__ __launch_bounds__(256) void k_ds_add_quant(float * x, const float * y, const float * w, int K,
                                                             unsigned char * q, float * qs, float eps) {
    __shared__ float red[16];
    float * xr = x + (size_t) blockIdx.x * K;
    const float * yr = y + (size_t) blockIdx.x * K;
    float ss[1] = {0.f};
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        ss[0] += v * v;
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / K + eps);
    for (int i = threadIdx.x; i < K; i += 256) quant32(xr[i] * r * w[i], q + (size_t) blockIdx.x * K, qs + (size_t) blockIdx.x * (K / 32), i);
}
// g = silu(g) * u, then its int8 planes
// (grid (K / 256, rows): a 256-element chunk per block)
static __global__ __launch_bounds__(256) void k_ds_swiglu_quant(float * g, const float * u, int K, unsigned char * q, float * qs) {
    float * gr = g + (size_t) blockIdx.y * K;
    const float * ur = u + (size_t) blockIdx.y * K;
    {
        const int i = blockIdx.x * 256 + threadIdx.x;
        const float x = gr[i];
        const float v = x / (1.f + expf(-x)) * ur[i];
        gr[i] = v;
        quant32(v, q + (size_t) blockIdx.y * K, qs + (size_t) blockIdx.y * (K / 32), i);
    }
}
// T-row Q8_0 GEMV, y = [T][N], with the activation planes staged in LDS: 4 rows per
// wave, qz_gemv_slice_t's per-row values (ink2_verify's v_core_acc: the same per-lane
// walk, the units past 64 spread over the wave) and wave sums (v_tsum: the butterfly's
// pairs for all T x 4 values at once, T padded to 8), so the results are bit-identical
template <int T, int NT = DS_NT>
static __global__ __launch_bounds__(NT) void k_ds_gemv(const unsigned char * aq, const float * as, int K,
                                                       const char * wv, const __half * ws, int N, float * y) {
    extern __shared__ char lds[];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tid = threadIdx.x;
    const int n_u = K / 32;
    unsigned char * la = (unsigned char *) lds;                 // [T][K]
    float * ls = (float *) (lds + (size_t) T * K);              // [T][n_u]
    for (int i = tid; i < T * K / 16; i += NT) ((uint4 *) la)[i] = ((const uint4 *) aq)[i];
    for (int i = tid; i < T * n_u; i += NT) ls[i] = as[i];
    __syncthreads();
    const int S = 4;
    constexpr int TP = T == 1 ? 1 : 8, NV = TP * 4;
    static_assert(T <= TP, "T <= 8");
    for (int g = blockIdx.x; g * (NT / 64 * S) < N; g += gridDim.x) {
        const int r0 = g * (NT / 64 * S) + wave * S;
        float acc[T][4];
        v_core_acc<T, 4>(la, ls, K, n_u, n_u, wv, ws, K, n_u, N, r0, lane, acc);
        float fv[NV];
        #pragma unroll
        for (int t = 0; t < TP; t++)
            #pragma unroll
            for (int q = 0; q < 4; q++) fv[t * 4 + q] = t < T ? acc[t < T ? t : 0][q] : 0.f;   // (the index kept in range)
        const float val = v_tsum<NV>(fv, lane);
        const int kv = v_tsum_k<NV>(lane), vt = kv / 4, vs = kv % 4;
        if ((lane & (64 / NV - 1)) == 0 && vt < T && r0 + vs < N) y[(size_t) vt * N + r0 + vs] = val;
    }
}
// the same walk for a K too wide to stage (the FFN down projection, T = 8, K = 9216):
// activations read from HBM / L2, 4 waves x 2 rows per block (N / 8 blocks, enough to
// fill the 60 CUs; the per-row value does not depend on the rows per wave)
template <int T>
static __global__ __launch_bounds__(256) void k_ds_gemv_wide(const unsigned char * aq, const float * as, int K,
                                                            const char * wv, const __half * ws, int N, float * y) {
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    constexpr int S = 2;
    const int r0 = blockIdx.x * 4 * S + wave * S;
    float acc[T][S];
    qz_gemv_rows_t<T, S>(aq, as, K, wv, ws, N, r0, lane, acc);
    if (lane == 0)
        #pragma unroll
        for (int t = 0; t < T; t++)
            #pragma unroll
            for (int s = 0; s < S; s++)
                if (r0 + s < N) y[(size_t) t * N + r0 + s] = acc[t][s];
}
// per (row, head) RMSNorm(w) over 128 + NeoX rope at pos0 + row; x [rows][nh][128];
// heads < nq use q, the rest k (blockIdx.x counted over both)
static __global__ __launch_bounds__(128) void k_ds_hnorm_rope(float * xq, const float * wq, int nq, float * xk, const float * wk,
                                                              int nk, int pos0, float eps) {
    __shared__ float red[4], s[HD];
    const bool isq = (int) blockIdx.x < nq;
    float * x = isq ? xq : xk;
    const float * w = isq ? wq : wk;
    const int nh = isq ? nq : nk;
    float * xr = x + ((size_t) blockIdx.y * nh + (isq ? blockIdx.x : blockIdx.x - nq)) * HD;
    const int e = threadIdx.x;
    const float v = xr[e];
    float ss = v * v;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 64);
    if ((e & 63) == 0) red[e >> 6] = ss;
    __syncthreads();
    const float r = rsqrtf((red[0] + red[1]) / HD + eps);
    s[e] = v * r * w[e];
    __syncthreads();
    if (e < HD / 2) {
        const float inv = powf(DS_THETA, -2.f * e / HD);
        const float ang = (float) (pos0 + (int) blockIdx.y) * inv;
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float x0 = s[e], x1 = s[e + HD / 2];
        xr[e] = x0 * cs - x1 * sn;
        xr[e + HD / 2] = x0 * sn + x1 * cs;
    }
}
// f32 rows [m][DS_KV] -> the f16 cache at rows pos0 ..
static __global__ void k_ds_store(const float * k, const float * v, __half * ck, __half * cv, int pos0, int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * DS_KV) return;
    ck[(size_t) pos0 * DS_KV + i] = __float2half(k[i]);
    cv[(size_t) pos0 * DS_KV + i] = __float2half(v[i]);
}
// attention of the block's BS query rows (head blockIdx.x) over context keys [lo, a)
// (f16 cache) and the BS block keys (f32), no mask; split sp of DS_SPL x 4 waves ->
// partials (m, l, o[128]) per (head, split-wave, row)
constexpr int DS_PW = HD + 2, DS_NP = DS_SPL * 4;
static __global__ __launch_bounds__(256) void k_ds_attn(const float * q, const __half * ck, const __half * cv,
                                                        const float * kb, const float * vb, int lo, int a, float * part) {
    const int h = blockIdx.x, kh = h / (DS_NQ / DS_NKV), wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int vw = blockIdx.y * 4 + wave;
    const float scale = 1.f / sqrtf((float) HD);
    float q0[DS_BS], q1[DS_BS], m[DS_BS], l[DS_BS], o0[DS_BS], o1[DS_BS];
    #pragma unroll
    for (int r = 0; r < DS_BS; r++) {
        q0[r] = q[((size_t) r * DS_NQ + h) * HD + 2 * lane] * scale;
        q1[r] = q[((size_t) r * DS_NQ + h) * HD + 2 * lane + 1] * scale;
        m[r] = -INFINITY; l[r] = 0.f; o0[r] = o1[r] = 0.f;
    }
    const int nk = a - lo + DS_BS;
    for (int j = vw; j < nk; j += DS_NP) {
        float k0, k1, v0, v1;
        if (j < a - lo) {
            const size_t o = (size_t) (lo + j) * DS_KV + kh * HD + 2 * lane;
            const __half2 kk = *(const __half2 *) (ck + o), vv = *(const __half2 *) (cv + o);
            k0 = __low2float(kk); k1 = __high2float(kk); v0 = __low2float(vv); v1 = __high2float(vv);
        } else {
            const size_t o = (size_t) (j - (a - lo)) * DS_KV + kh * HD + 2 * lane;
            k0 = kb[o]; k1 = kb[o + 1]; v0 = vb[o]; v1 = vb[o + 1];
        }
        #pragma unroll
        for (int r = 0; r < DS_BS; r++) {
            float s = q0[r] * k0 + q1[r] * k1;
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) s += __shfl_xor(s, off, 64);
            const float mn = fmaxf(m[r], s), c = expf(m[r] - mn), p = expf(s - mn);
            l[r] = l[r] * c + p; o0[r] = o0[r] * c + p * v0; o1[r] = o1[r] * c + p * v1; m[r] = mn;
        }
    }
    #pragma unroll
    for (int r = 0; r < DS_BS; r++) {
        float * P = part + (((size_t) h * DS_NP + vw) * DS_BS + r) * DS_PW;
        if (lane == 0) { P[0] = m[r]; P[1] = l[r]; }
        P[2 + 2 * lane] = o0[r]; P[3 + 2 * lane] = o1[r];
    }
}
// merge the DS_NP partials of (head blockIdx.x, row blockIdx.y) -> o [row][DS_Q]
static __global__ __launch_bounds__(128) void k_ds_attn_merge(const float * part, float * o) {
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    float M = -INFINITY;
    for (int w = 0; w < DS_NP; w++) M = fmaxf(M, part[(((size_t) h * DS_NP + w) * DS_BS + r) * DS_PW]);
    float L = 0.f, O = 0.f;
    #pragma unroll 1
    for (int w0 = 0; w0 < DS_NP; w0 += 16) {               // 16 partials' loads in flight, folded in order
        float pm[16], pl[16], po[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            const float * P = part + (((size_t) h * DS_NP + w0 + k) * DS_BS + r) * DS_PW;
            pm[k] = P[0]; pl[k] = P[1]; po[k] = P[2 + d];
        }
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            if (pm[k] == -INFINITY) continue;
            const float e = expf(pm[k] - M);
            L += pl[k] * e; O += po[k] * e;
        }
    }
    o[(size_t) r * DS_Q + h * HD + d] = O / L;
}
static __global__ __launch_bounds__(256) void k_ds_embed(const char * ev, const __half * es, int tok, const float * w, float eps,
                                                         float * x, const float * mask) {
    if (blockIdx.x == 0) d_embed(ev, es, tok, w, eps, x, threadIdx.x, true);
    else for (int i = threadIdx.x; i < D; i += 256) x[(size_t) blockIdx.x * D + i] = mask[i];
}
__device__ __forceinline__ float ds_bf16(uint16_t b) { return __uint_as_float((unsigned int) b << 16); }
// joint = W [state | w1[prev] | hid | g] + b, one wave per output row (768). prev is the
// previous step's token: the argmax card 0 and card 1 agreed on (step > 1), read out of
// best / xbest once the peer's flag arrives; every block computes it (no extra launch)
static __global__ __launch_bounds__(512) void k_ds_joint(const uint16_t * W, const float * bias, const float * state,
                                                         const uint16_t * w1, int step, int anc, const float * hid, const float * g,
                                                         float * joint, const unsigned long long * best,
                                                         const unsigned long long * xbest, unsigned int * fl, unsigned int seq,
                                                         int * err, int * out) {
    __shared__ int sprev;
    if (threadIdx.x == 0) {
        int prev = anc;
        if (step > 1) {
            ds_wait(fl, seq, err);
            const unsigned long long b0 = ((volatile unsigned long long *) best)[0];
            const unsigned long long b1 = ((volatile unsigned long long *) xbest)[0];
            prev = (int) ~(unsigned int) ((b0 > b1 ? b0 : b1) & 0xffffffffu);
        }
        if (blockIdx.x == 0 && step > 1) out[step - 2] = prev;
        sprev = prev;
    }
    __syncthreads();
    const int prev = sprev;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
    if (r >= 3 * DS_R) return;
    const uint16_t * wr = W + (size_t) r * (2 * DS_R + 2 * D);
    const uint16_t * e = w1 + (size_t) prev * DS_R;
    float acc = 0.f;
    for (int k = lane; k < DS_R; k += 64) acc += ds_bf16(wr[k]) * state[k] + ds_bf16(wr[DS_R + k]) * ds_bf16(e[k]);
    for (int k = lane; k < D; k += 64) acc += ds_bf16(wr[2 * DS_R + k]) * hid[k] + ds_bf16(wr[2 * DS_R + D + k]) * g[k];
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 64);
    if (lane == 0) joint[r] = acc + bias[r];
}
// gate / cand / latent: state = cand + gate (state - cand); lat1 = [tanh latent | 1];
// card 0 pushes lat1 into the peer's buffer and flags it (lat1_peer != null)
static __global__ __launch_bounds__(256) void k_ds_markov(const float * joint, float * state, float * lat1,
                                                         float * lat1_peer, unsigned long long * best, unsigned long long * xbest,
                                                         unsigned int * fl, unsigned int seq) {
    const int k = threadIdx.x;
    if (k == 0) { ((volatile unsigned long long *) best)[0] = 0; ((volatile unsigned long long *) xbest)[0] = 0; }
    const float gate = 1.f / (1.f + expf(-joint[k])), cand = tanhf(joint[DS_R + k]);
    state[k] = cand + gate * (state[k] - cand);
    const float l = tanhf(joint[2 * DS_R + k]);
    lat1[k] = l;
    if (k == 0) lat1[DS_R] = 1.f;
    if (lat1_peer) {
        volatile float * p = (volatile float *) lat1_peer;
        p[k] = l;
        if (k == 0) p[DS_R] = 1.f;
        __syncthreads();
        if (k == 0) ds_flag(fl, seq);
    }
}
// draft row logits over this card's head rows: base + fold . lat1; the argmax as a packed
// (orderable value, ~id) max into best (ties: the lower id). WT: wait for lat1 first
// (card 1). fold rows are nhp halves apart (128-byte aligned), walked in j order;
// DS_FE consecutive entries per thread (8-byte loads)
constexpr int DS_FE = 4;
template <bool WT>
static __global__ __launch_bounds__(256) void k_ds_fold_argmax(const float * base, const __half * fold, int nhp, int nvalid,
                                                               const float * lat1, int id0, unsigned long long * best,
                                                               unsigned int * fl, unsigned int seq, int * err) {
    __shared__ float sl[DS_R + 1];
    if (WT) {
        if (threadIdx.x == 0) ds_wait(fl, seq, err);
        __syncthreads();
    }
    for (int j = threadIdx.x; j <= DS_R; j += 256) sl[j] = lat1[j];
    __syncthreads();
    // each entry's sum in j order
    const int v0 = (blockIdx.x * 256 + threadIdx.x) * DS_FE;
    unsigned long long key = 0;
    if (v0 < nvalid) {
        const __half * p = fold + v0;
        float s[DS_FE];
        #pragma unroll
        for (int e = 0; e < DS_FE; e++) s[e] = v0 + e < nvalid ? base[v0 + e] : 0.f;
        #pragma unroll 8
        for (int j = 0; j <= DS_R; j++) {
            const uint2 hv = *(const uint2 *) p;
            const __half2 h01 = *(const __half2 *) &hv.x, h23 = *(const __half2 *) &hv.y;
            const float lj = sl[j];
            s[0] += lj * __low2float(h01);
            s[1] += lj * __high2float(h01);
            s[2] += lj * __low2float(h23);
            s[3] += lj * __high2float(h23);
            p += nhp;
        }
        #pragma unroll
        for (int e = 0; e < DS_FE; e++)
            if (v0 + e < nvalid) {
                unsigned int u = __float_as_uint(s[e]);
                u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                const unsigned long long k = ((unsigned long long) u << 32) | (unsigned int) ~(unsigned int) (id0 + v0 + e);
                key = k > key ? k : key;
            }
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) { const unsigned long long o = __shfl_xor(key, off, 64); key = o > key ? o : key; }
    __shared__ unsigned long long sb[4];
    if ((threadIdx.x & 63) == 0) sb[threadIdx.x >> 6] = key;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long b = sb[0];
        for (int i = 1; i < 4; i++) b = sb[i] > b ? sb[i] : b;
        atomicMax(best, b);
    }
}
// card 1: hand this step's argmax to card 0 (peer store + flag), then free the slot
static __global__ __launch_bounds__(64) void k_ds_best_push(unsigned long long * best, unsigned long long * xbest_peer,
                                                            unsigned int * fl_peer, unsigned int seq) {
    if (threadIdx.x != 0) return;
    *(volatile unsigned long long *) xbest_peer = *best;
    ds_flag(fl_peer, seq);
    *best = 0;
}
// the chain's last token: the argmax of step BS - 1, once the peer's push has landed
static __global__ __launch_bounds__(64) void k_ds_last(const unsigned long long * best, const unsigned long long * xbest,
                                                       unsigned int * fl, unsigned int seq, int * err, int * out) {
    if (threadIdx.x != 0) return;
    ds_wait(fl, seq, err);
    const unsigned long long b0 = ((volatile const unsigned long long *) best)[0];
    const unsigned long long b1 = ((volatile const unsigned long long *) xbest)[0];
    out[DS_BS - 2] = (int) ~(unsigned int) ((b0 > b1 ? b0 : b1) & 0xffffffffu);
}
static __global__ void k_ds_tohalf(const float * y, __half * o, int n, int stride) {
    const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) o[(size_t) r * stride + c] = __float2half(y[(size_t) r * n + c]);
}
// the block's hidden rows 1 .. 7: X += Y, then the Markov input (f32, hid) and its int8 planes
static __global__ __launch_bounds__(256) void k_ds_hidden(float * x, const float * y, const float * w, int K,
                                                          unsigned char * q, float * qs, float * hf, float eps) {
    __shared__ float red[16];
    float * xr = x + (size_t) (blockIdx.x + 1) * K;
    const float * yr = y + (size_t) (blockIdx.x + 1) * K;
    float ss[1] = {0.f};
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        ss[0] += v * v;
    }
    bsum<1, false>(ss, red, 4);
    const float r = rsqrtf(ss[0] / K + eps);
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = xr[i] * r * w[i];
        hf[(size_t) blockIdx.x * K + i] = v;
        quant32(v, q + (size_t) blockIdx.x * K, qs + (size_t) blockIdx.x * (K / 32), i);
    }
}
// columns of hidden_out (bf16 [D][R]) and its bias as T = 8 f32 rows [8][D] from column j0
static __global__ void k_ds_fold_rows(const uint16_t * how, const float * hob, int j0, float * rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 8 * D) return;
    const int t = i / D, d = i % D, j = j0 + t;
    rows[i] = j < DS_R ? ds_bf16(how[(size_t) d * DS_R + j]) : (j == DS_R ? hob[d] : 0.f);
}

struct ink2_dspark {
    struct layer { qw q, k, v, kc, vc, o, gate, up, down; const float * an, * fn, * qn, * kn, * kcn, * dn; };
    ink2_engine * E = nullptr;
    qz_loader ld;
    layer L[DS_NL];
    float4 alpha[DS_NL + 1];
    const float * mask = nullptr, * dsn = nullptr, * onorm = nullptr, * jb = nullptr, * hob = nullptr;
    const uint16_t * jw = nullptr, * w1 = nullptr, * how = nullptr;
    int ctx = 0;
    // card 0: context cache, block buffers
    __half * ck[DS_NL] = {}, * cv[DS_NL] = {};           // [ctx][DS_KV] (+ the nhp padding? no: plain)
    float * g = nullptr;                       // the Markov context of the last committed row
    float * tap = nullptr;                     // decode features [4][D] (the cycle writes them)
    float * vtap = nullptr;                    // verify features [VR][4][D] (the verify kernel writes them)
    float * cx = nullptr, * ckf = nullptr, * cvf = nullptr;   // context fill: mixed rows [8][D], k / v [8][DS_KV]
    float * X = nullptr, * Y = nullptr, * Q = nullptr, * Kb = nullptr, * Vb = nullptr, * O = nullptr, * G = nullptr, * U = nullptr;
    float * hid = nullptr, * part = nullptr, * joint = nullptr, * state = nullptr;
    unsigned char * aq = nullptr; float * as = nullptr;
    // per card: the draft rows' quantized hidden, base logits [8][nvalid], fold [(R+1)][nhp], lat1
    unsigned char * hq[2] = {}; float * hs[2] = {}, * base[2] = {}, * lat1[2] = {};
    __half * fold[2] = {};
    int nhp[2] = {};
    unsigned long long * best[2] = {};         // this card's fold target (local, zeroed per step)
    unsigned long long * xbest[2] = {};        // the peer's argmax pushed here (uncached)
    unsigned int * fl[2] = {};                 // the peer's handshake flag (uncached)
    int * out_d = nullptr, * out_h = nullptr;  // the 7 tokens + the chain's error word
    unsigned int seq = 0;                      // the chain's step counter (monotone)
    hipEvent_t ev = nullptr;
    // stats
    long rounds = 0, accepted = 0;

    static qw Q8(const qz_loader & l, const char * n) {
        const qz_dev_tensor * t = l.get(n);
        if (!t || t->type != GGML_TYPE_Q8_0) { fprintf(stderr, "dspark: missing Q8_0 %s\n", n); exit(1); }
        qw r; r.v = (const char *) t->vals; r.s = (const __half *) t->scal; r.K = (int) t->ne0; r.N = (int) t->ne1;
        return r;
    }
    static const void * Raw(const qz_loader & l, const char * n, ggml_type ty) {
        const qz_dev_tensor * t = l.get(n);
        if (!t || t->type != ty) { fprintf(stderr, "dspark: missing %s (type %d)\n", n, (int) ty); exit(1); }
        return t->vals;
    }
    template <class T> static void al(T ** p, size_t n) { ICHECK(qz_dmalloc((void **) p, n * sizeof(T), "dspark")); }

    bool load(const char * path, int ctx_) {
        ctx = ctx_;
        qz_gguf g_;
        if (!g_.open(path) || g_.str("general.architecture") != std::string("dflash")) {
            fprintf(stderr, "dspark: %s is not a dflash (DSpark) GGUF\n", path); return false;
        }
        if (g_.u32("dflash.block_count") != DS_NL || g_.u32("dflash.block_size") != DS_BS || g_.u32("dflash.embedding_length") != D) {
            fprintf(stderr, "dspark: unsupported drafter shape\n"); return false;
        }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        {
            qz_mem_scope mem("dspark weights");
            if (!ld.load(path, g_, 0)) { fprintf(stderr, "dspark: load failed\n"); return false; }
        }
        char n[96];
        for (int l = 0; l < DS_NL; l++) {
            auto nm = [&](const char * s) { snprintf(n, sizeof(n), "blk.%d.%s.weight", l, s); return n; };
            layer & y = L[l];
            y.q = Q8(ld, nm("attn_q")); y.k = Q8(ld, nm("attn_k")); y.v = Q8(ld, nm("attn_v"));
            y.kc = Q8(ld, nm("attn_k_ctx")); y.vc = Q8(ld, nm("attn_v_ctx")); y.o = Q8(ld, nm("attn_output"));
            y.gate = Q8(ld, nm("ffn_gate")); y.up = Q8(ld, nm("ffn_up")); y.down = Q8(ld, nm("ffn_down"));
            y.an = (const float *) Raw(ld, nm("attn_norm"), GGML_TYPE_F32); y.fn = (const float *) Raw(ld, nm("ffn_norm"), GGML_TYPE_F32);
            y.qn = (const float *) Raw(ld, nm("attn_q_norm"), GGML_TYPE_F32); y.kn = (const float *) Raw(ld, nm("attn_k_norm"), GGML_TYPE_F32);
            y.kcn = (const float *) Raw(ld, nm("attn_k_ctx_norm"), GGML_TYPE_F32); y.dn = (const float *) Raw(ld, nm("dflare_norm"), GGML_TYPE_F32);
        }
        mask = (const float *) Raw(ld, "mask_embd.weight", GGML_TYPE_F32);
        dsn = (const float *) Raw(ld, "dspark_norm.weight", GGML_TYPE_F32);
        onorm = (const float *) Raw(ld, "output_norm.weight", GGML_TYPE_F32);
        jb = (const float *) Raw(ld, "joint_proj.bias", GGML_TYPE_F32);
        hob = (const float *) Raw(ld, "dspark_hidden_out.bias", GGML_TYPE_F32);
        jw = (const uint16_t *) Raw(ld, "joint_proj.weight", GGML_TYPE_BF16);
        w1 = (const uint16_t *) Raw(ld, "markov_w1.weight", GGML_TYPE_BF16);
        how = (const uint16_t *) Raw(ld, "dspark_hidden_out.weight", GGML_TYPE_BF16);
        {   // the 4 feature mixes: softmax of each dspark_fuse row
            float f[16];
            ICHECK(hipMemcpy(f, Raw(ld, "dspark_fuse.weight", GGML_TYPE_F32), sizeof(f), hipMemcpyDeviceToHost));
            for (int r = 0; r <= DS_NL; r++) {
                float mx = f[4 * r], e[4], s = 0;
                for (int k = 1; k < 4; k++) mx = fmaxf(mx, f[4 * r + k]);
                for (int k = 0; k < 4; k++) { e[k] = expf(f[4 * r + k] - mx); s += e[k]; }
                alpha[r] = make_float4(e[0] / s, e[1] / s, e[2] / s, e[3] / s);
            }
        }
        qz_mem_scope mem("dspark state");
        for (int l = 0; l < DS_NL; l++) { al(&ck[l], (size_t) ctx * DS_KV); al(&cv[l], (size_t) ctx * DS_KV); }
        al(&g, D); al(&tap, 4 * D); al(&vtap, (size_t) VR * 4 * D); al(&cx, 8 * D); al(&ckf, 8 * DS_KV); al(&cvf, 8 * DS_KV);
        al(&X, DS_BS * D); al(&Y, DS_BS * D); al(&Q, DS_BS * DS_Q); al(&Kb, DS_BS * DS_KV); al(&Vb, DS_BS * DS_KV);
        al(&O, DS_BS * DS_Q); al(&G, DS_BS * DS_FF); al(&U, DS_BS * DS_FF); al(&hid, DS_BS * D);
        al(&part, (size_t) DS_NQ * DS_NP * DS_BS * DS_PW); al(&joint, 3 * DS_R); al(&state, DS_R);
        al(&aq, DS_BS * DS_FF); al(&as, DS_BS * DS_FF / 32);
        ICHECK(qz_dmalloc((void **) &out_d, 8 * sizeof(int), "dspark"));
        ICHECK(hipHostMalloc((void **) &out_h, 8 * sizeof(int), 0));
        ICHECK(hipMemset(out_d, 0, 8 * sizeof(int)));
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            const qw & H = E->C[c].head;
            nhp[c] = (H.N + 63) & ~63;             // 128-byte aligned fold rows
            al(&hq[c], DS_BS * D); al(&hs[c], DS_BS * D / 32); al(&base[c], (size_t) DS_BS * H.N);
            // lat1 is pushed over P2P by the peer's Markov step: uncached, so the reader never
            // sees a stale line
            ICHECK(qz_dmalloc_uncached((void **) &lat1[c], (DS_R + 1) * sizeof(float), "dspark"));
            al(&fold[c], (size_t) (DS_R + 1) * nhp[c]);
            ICHECK(qz_dmalloc((void **) &best[c], 8, "dspark"));
            ICHECK(hipMemset(best[c], 0, 8));
            ICHECK(qz_dmalloc_uncached((void **) &xbest[c], 8, "dspark"));
            ICHECK(qz_dmalloc_uncached((void **) &fl[c], 4, "dspark"));
            ICHECK(hipMemset(xbest[c], 0, 8));
            ICHECK(hipMemset(fl[c], 0, 4));
            // fold rows: head . hidden_out column j (and . bias), 8 columns per GEMV
            float * rows = nullptr, * y = nullptr;
            ICHECK(hipMalloc(&rows, 8 * D * 4)); ICHECK(hipMalloc(&y, (size_t) 8 * H.N * 4));
            const uint16_t * how_c = how; const float * hob_c = hob;
            if (c == 1) {   // the drafter lives on card 0: its two small tensors over P2P once
                uint16_t * h1 = nullptr; float * b1 = nullptr;
                ICHECK(hipMalloc(&h1, (size_t) D * DS_R * 2)); ICHECK(hipMalloc(&b1, D * 4));
                ICHECK(hipMemcpyPeer(h1, qz_dev_of(1), how, qz_dev_of(0), (size_t) D * DS_R * 2)); ICHECK(hipMemcpyPeer(b1, qz_dev_of(1), hob, qz_dev_of(0), D * 4));
                how_c = h1; hob_c = b1;
            }
            for (int j0 = 0; j0 <= DS_R; j0 += 8) {
                k_ds_fold_rows<<<(8 * D + 255) / 256, 256>>>(how_c, hob_c, j0, rows);
                k_ds_quant<<<8, 256>>>(rows, nullptr, D, hq[c], hs[c], nullptr, DS_EPS);
                k_ds_gemv<8><<<240, DS_NT, 8 * D + 8 * (D / 32) * 4>>>(hq[c], hs[c], D, H.v, H.s, H.N, y);
                const int nj = std::min(8, DS_R + 1 - j0);
                k_ds_tohalf<<<dim3((H.N + 255) / 256, nj), 256>>>(y, fold[c] + (size_t) j0 * nhp[c], H.N, nhp[c]);
            }
            ICHECK(hipDeviceSynchronize());
            (void) hipFree(rows); (void) hipFree(y);
            if (c == 1) { (void) hipFree((void *) how_c); (void) hipFree((void *) hob_c); }
        }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        ICHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
        E->dtap = tap; E->dtap_head = tap + 3 * D;
        E->vtap = vtap;
        return true;
    }

    // the T-row GEMV: the activation planes staged in LDS while they fit, else the wide walk
    void gemv(hipStream_t st, const unsigned char * q, const float * s, const qw & W, float * y, int T) {
        const int lds = T * W.K + T * (W.K / 32) * 4;
        if (lds <= 32768) {
            const int nb = std::min((W.N + 15) / 16, 240);
            if (T == 8) k_ds_gemv<8><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 7) k_ds_gemv<7><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else k_ds_gemv<1><<<nb, DS_NT, lds, st>>>(q, s, W.K, W.v, W.s, W.N, y);
        } else {
            const int nb = (W.N + 7) / 8;
            if (T == 8) k_ds_gemv_wide<8><<<nb, 256, 0, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else if (T == 7) k_ds_gemv_wide<7><<<nb, 256, 0, st>>>(q, s, W.K, W.v, W.s, W.N, y);
            else k_ds_gemv_wide<1><<<nb, 256, 0, st>>>(q, s, W.K, W.v, W.s, W.N, y);
        }
    }
    // context rows p0 .. p0 + m - 1 from their features (row stride tstride floats); g = the last row's
    void fill(const float * taps, size_t tstride, int p0, int m) {
        ICHECK(hipSetDevice(qz_dev_of(0)));
        hipStream_t st = E->C[0].st;
        for (int r0 = 0; r0 < m; r0 += 8) {
            const int n = std::min(8, m - r0);
            for (int l = 0; l < DS_NL; l++) {
                k_ds_mixnorm<<<n, 256, 0, st>>>(taps + (size_t) r0 * tstride, tstride, alpha[l], L[l].dn, cx, D, DS_EPS);
                k_ds_quant<<<n, 256, 0, st>>>(cx, nullptr, D, aq, as, nullptr, DS_EPS);
                gemv(st, aq, as, L[l].kc, ckf, n == 1 ? 1 : 8);
                gemv(st, aq, as, L[l].vc, cvf, n == 1 ? 1 : 8);
                k_ds_hnorm_rope<<<dim3(DS_NKV, n), 128, 0, st>>>(ckf, L[l].kcn, DS_NKV, ckf, L[l].kcn, 0, p0 + r0, DS_EPS);
                k_ds_store<<<(n * DS_KV + 255) / 256, 256, 0, st>>>(ckf, cvf, ck[l], cv[l], p0 + r0, n);
            }
        }
        k_ds_mixnorm<<<1, 256, 0, st>>>(taps + (size_t) (m - 1) * tstride, tstride, alpha[DS_NL], dsn, g, D, DS_EPS);
    }
    // the 7 greedy drafts after anchor tok at position a (context rows 0 .. a - 1 filled)
    void draft(int tok, int a, int * out) {
        ICHECK(hipSetDevice(qz_dev_of(0)));
        hipStream_t st = E->C[0].st;
        const ink2_card & K0 = E->C[0];
        k_ds_embed<<<DS_BS, 256, 0, st>>>(K0.embd.v, K0.embd.s, tok, K0.embd_norm, E->hp.eps, X, mask);
        k_ds_quant<<<DS_BS, 256, 0, st>>>(X, L[0].an, D, aq, as, nullptr, DS_EPS);
        for (int l = 0; l < DS_NL; l++) {
            const layer & y = L[l];
            gemv(st, aq, as, y.q, Q, 8); gemv(st, aq, as, y.k, Kb, 8); gemv(st, aq, as, y.v, Vb, 8);
            k_ds_hnorm_rope<<<dim3(DS_NQ + DS_NKV, DS_BS), 128, 0, st>>>(Q, y.qn, DS_NQ, Kb, y.kn, DS_NKV, a, DS_EPS);
            const int lo = l == 1 ? 0 : std::max(0, a - WIN);      // layers 0 / 2: sliding window
            k_ds_attn<<<dim3(DS_NQ, DS_SPL), 256, 0, st>>>(Q, ck[l], cv[l], Kb, Vb, lo, a, part);
            k_ds_attn_merge<<<dim3(DS_NQ, DS_BS), 128, 0, st>>>(part, O);
            k_ds_quant<<<DS_BS, 256, 0, st>>>(O, nullptr, DS_Q, aq, as, nullptr, DS_EPS);
            gemv(st, aq, as, y.o, Y, 8);
            k_ds_add_quant<<<DS_BS, 256, 0, st>>>(X, Y, y.fn, D, aq, as, DS_EPS);
            gemv(st, aq, as, y.gate, G, 8); gemv(st, aq, as, y.up, U, 8);
            static_assert(DS_FF % 256 == 0, "whole chunks");
            k_ds_swiglu_quant<<<dim3(DS_FF / 256, DS_BS), 256, 0, st>>>(G, U, DS_FF, aq, as);
            gemv(st, aq, as, y.down, Y, 8);
            // the residual add of X + Y is fused into the next quantization (the FFN norm here,
            // the next layer's attention norm there; the last layer's hidden rows)
            if (l + 1 < DS_NL) k_ds_add_quant<<<DS_BS, 256, 0, st>>>(X, Y, L[l + 1].an, D, aq, as, DS_EPS);
            else k_ds_hidden<<<DS_BS - 1, 256, 0, st>>>(X, Y, onorm, D, hq[0], hs[0], hid, DS_EPS);
        }
        // hidden rows 1 .. 7: f32 (the Markov input) + quantized (the base logits) on both cards
        ICHECK(hipMemcpyPeerAsync(hq[1], qz_dev_of(1), hq[0], qz_dev_of(0), (DS_BS - 1) * D, st));
        ICHECK(hipMemcpyPeerAsync(hs[1], qz_dev_of(1), hs[0], qz_dev_of(0), (DS_BS - 1) * D / 32 * 4, st));
        ICHECK(hipEventRecord(ev, st));
        for (int c = 0; c < 2; c++) {
            ICHECK(hipSetDevice(qz_dev_of(c)));
            hipStream_t s = E->C[c].st;
            if (c == 1) ICHECK(hipStreamWaitEvent(s, ev, 0));
            gemv(s, hq[c], hs[c], E->C[c].head, base[c], 7);
        }
        ICHECK(hipSetDevice(qz_dev_of(0)));
        ICHECK(hipMemsetAsync(state, 0, DS_R * 4, st));
        // the 7 Markov steps: card 0 joint -> markov (push lat1) -> fold; card 1 fold (wait) -> push
        for (int i = 1; i < DS_BS; i++) {
            const unsigned int sq = ++seq;
            k_ds_joint<<<3 * DS_R / 8, 512, 0, st>>>(jw, jb, state, w1, i, tok, hid + (size_t) (i - 1) * D, g, joint, best[0], xbest[0],
                                                     fl[0], sq - 1, out_d + 7, out_d);
            k_ds_markov<<<1, 256, 0, st>>>(joint, state, lat1[0], lat1[1], best[0], xbest[0], fl[1], sq);
            k_ds_fold_argmax<false><<<(E->C[0].nvalid + 256 * DS_FE - 1) / (256 * DS_FE), 256, 0, st>>>(base[0] + (size_t) (i - 1) * E->C[0].head.N,
                                                                                 fold[0], nhp[0], E->C[0].nvalid, lat1[0],
                                                                                 E->C[0].id0, best[0], fl[0], sq, out_d + 7);
            ICHECK(hipSetDevice(qz_dev_of(1)));
            hipStream_t s1 = E->C[1].st;
            k_ds_fold_argmax<true><<<(E->C[1].nvalid + 256 * DS_FE - 1) / (256 * DS_FE), 256, 0, s1>>>(base[1] + (size_t) (i - 1) * E->C[1].head.N,
                                                                                fold[1], nhp[1], E->C[1].nvalid, lat1[1],
                                                                                E->C[1].id0, best[1], fl[1], sq, out_d + 7);
            k_ds_best_push<<<1, 64, 0, s1>>>(best[1], xbest[0], fl[0], sq);
            ICHECK(hipSetDevice(qz_dev_of(0)));
        }
        k_ds_last<<<1, 64, 0, st>>>(best[0], xbest[0], fl[0], seq, out_d + 7, out_d);
        ICHECK(hipMemcpyAsync(out_h, out_d, 8 * sizeof(int), hipMemcpyDeviceToHost, st));
        ICHECK(hipStreamSynchronize(st));
        for (int i = 0; i < DS_BS - 1; i++) out[i] = out_h[i];
        if (out_h[7]) { fprintf(stderr, "dspark: the Markov chain's cross-card wait timed out (step %u)\n", seq), exit(1); }
    }
};
