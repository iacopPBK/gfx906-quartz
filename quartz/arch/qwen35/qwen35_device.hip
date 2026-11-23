// Device helpers, model geometry, per-layer weight / argument structs; the
// decode cycle is qwen35_cycle.hip (included at the end).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

#include "kernels/dev.h"

// recurrent (gated delta net) layer geometry
#define QZ_D    5120       // d_model
#define QZ_HV   48         // v heads
#define QZ_HK   16         // k heads
#define QZ_S    128        // ssm state dim
#define QZ_CCH  10240      // conv channels (q | k | v)
#define QZ_G    6144       // gate rows, also ssm_out K

// dn scratch lives in its own LDS region: q, k, 4-band partials, delta,
// output row, norm inverses
struct qz_dn_lds {
    float q[QZ_S];
    float k[QZ_S];
    float part[4][QZ_S];
    float d[QZ_S];
    float o[QZ_S];
    float inv[3];
};

// attention layer geometry: 24 q / 4 kv heads x 256, kq scale 1/16,
// rope base 1e7 on 64 dims (pairs (j, j+32))
#define QZ_AQ   12288      // attn_q rows
#define QZ_AKV  1024       // attn_k / attn_v rows (4 kv heads x 256)
#define QZ_HQ   24
#define QZ_HKV  4
#define QZ_HD   256

#define QZ_FBIG (-3.4028235e38f)   // empty-partial max (flash9.h / flash9c.h)
typedef _Float16 qz_h2 __attribute__((ext_vector_type(2)));

#define QZ_FF   17408      // ffn intermediate rows

struct qz_trunk_w {
    // present in every layer
    const float * attn_norm;                     // [QZ_D]
    const float * post_norm;                     // [QZ_D]
    const char * out_vals;  const __half * out_scal;   // ssm_out | attn_output, [QZ_D x QZ_G]
    const char * fg_vals;   const __half * fg_scal;   // ffn_gate [QZ_FF x QZ_D]
    const char * fu_vals;   const __half * fu_scal;   // ffn_up   [QZ_FF x QZ_D]
    const char * fd_vals;   const __half * fd_scal;   // ffn_down [QZ_D x QZ_FF]
    // recurrent layers
    const char * qkv_vals;  const __half * qkv_scal;  // attn_qkv [QZ_CCH x QZ_D]
    const char * gat_vals;  const __half * gat_scal;  // attn_gate [QZ_G x QZ_D]
    const char * bet_vals;  const __half * bet_scal;  // ssm_beta [QZ_HV x QZ_D]
    const char * alp_vals;  const __half * alp_scal;  // ssm_alpha [QZ_HV x QZ_D]
    const float * conv1d;                       // [QZ_CCH][4]
    const float * dt_bias;                       // [QZ_HV]
    const float * ssm_a;                        // [QZ_HV]
    const float * ssm_norm_w;                   // [QZ_S]
    float * dn_state;                            // [QZ_HV][QZ_S][QZ_S], persistent
    float * conv_state;                          // [3][QZ_CCH], persistent
    float * ring;   // qz_cycle replay rows: [2 parity][qz_geo::RING_PAR floats], see qwen35_cycle.hip
    // attention layers
    const char * q_vals;    const __half * q_scal;   // attn_q [QZ_AQ x QZ_D]
    const char * k_vals;    const __half * k_scal;   // attn_k [QZ_AKV x QZ_D]
    const char * v_vals;    const __half * v_scal;   // attn_v [QZ_AKV x QZ_D]
    const float * q_norm_w;                     // [QZ_HD]
    const float * k_norm_w;                     // [QZ_HD]
    __half * kv_k;                              // paged arenas, persistent
    __half * kv_v;
    int recr;
};

#define QZ_TMAX 4                 // max rows per cycle (qz_cycle<T>)

// MTP draft block (blk.64): one trunk attn layer plus the nextn extras;
// lm head and token embedding are shared with the trunk.
struct qz_mtp_w {
    qz_trunk_w L;
    const char * eh_vals;  const __half * eh_scal;   // nextn.eh_proj [QZ_D x 2*QZ_D]
    const float * enorm;                    // nextn.enorm [QZ_D]
    const float * hnorm;                    // nextn.hnorm [QZ_D]
    const float * shnorm;                   // nextn.shared_head_norm [QZ_D]
};

struct qz_trunk_args {
    static constexpr int DM = QZ_D;   // row width (kernels/xchg.h)
    const int * emb_vals;         // [vocab][QZ_D/4]
    const __half * emb_scal;      // [vocab][QZ_D/32]
    const float * out_norm;       // [QZ_D]
    const char * hd_vals;  const __half * hd_scal;    // lm_head [n_vocab x QZ_D]
    qz_trunk_w * w;               // [n_layer]
    // scratch rows, reused layer to layer
    float * x_rows;               // [2][QZ_D] ping-pong residual rows
    float * y_core;               // [QZ_D] ssm_out | attn_output gemv row
    float * y_g;                  // [QZ_FF]
    float * y_u;                  // [QZ_FF]
    float * y_ffn;               // [QZ_D] ffn down gemv row
    unsigned char * pl_vals;      // [QZ_G] core-out gemv planes
    float * pl_scal;              // [QZ_G/32]
    unsigned char * ffn_q;       // [QZ_FF] ffn down planes
    float * ffn_sc;               // [QZ_FF/32]
    // recr scratch
    float * y_gate;               // [QZ_G]
    // attn scratch
    float * y_q;                   // [QZ_AQ]
    float * y_k;                   // [QZ_AKV]
    float * y_v;                   // [QZ_AKV]
    __half * q_f16;               // [QZ_HQ][QZ_HD]
    float * pm;                   // [rows][QZ_HQ][FL9_NS] attention partials (fl9c)
    float * pl;                   // [rows][QZ_HQ][FL9_NS]
    float * pacc;                 // [rows][QZ_HQ][FL9_NS][QZ_HD]
    float * o_preg;               // [rows][QZ_HQ][QZ_HD] attention rows before the gate
    // head
    float * y_head;               // [n_vocab]
    int * out_tok;                // [1] argmax result
    int toks[QZ_TMAX];             // qz_cycle<T>: row tokens
    int cycle;                     // qz_cycle: cycle index (ring parity = cycle & 1)
    int verify;                    // qz_cycle: 1 = greedy verify, 0 = accept all rows
    int * ctl;                     // qz_cycle: [3] n_acc, next_tok, pos_next of the last cycle; ctl[0] on entry = rows to replay
    int pos;
    int n_layer;
    float eps;
    // MTP draft block, qz_cycle only; null keeps the trunk-only path
    const qz_mtp_w * mtp;
    float * h_nextn;              // trunk head-norm rows [QZ_TMAX][QZ_D]
    float * mtp_h;                // MTP shared-head-norm rows [QZ_TMAX][QZ_D]
    unsigned char * cat_vals;     // eh_proj input planes [QZ_TMAX][2*QZ_D]
    float * cat_scal;             //                   scales [QZ_TMAX][2*QZ_D/32]
    float * mtp_yhead;            // MTP head rows [QZ_TMAX][n_vocab]
    // TP2: this card's exchange state, see qwen35_cycle.hip
    int tp_card;                  // 0 or 1
    int hrows;                    // local lm_head rows (n_vocab / 2)
    float * y_part;               // [QZ_TMAX][QZ_D] own row-parallel partial
    float * xin_mine;             // uncached inbox [2][QZ_TMAX][QZ_D] + checksums [2][QZ_TMAX][QZ_D/32]
    float * xin_peer;             // the peer's inbox (P2P)
    float * xhm_mine;             // uncached head-merge inbox [2][QZ_TMAX][2] {value, id bits}
    float * xhm_peer;
    unsigned int * xflag_mine;    // uncached, written by the peer only
    unsigned int * xflag_peer;
    unsigned int * xseq;          // [1] exchange counter, carried across launches
    int * xerr;                   // [8] chk mismatches, abort, code, tok, seen, first bad tok, slab, row
    unsigned char * nq_vals;      // qz_cycle norm planes [QZ_TMAX][QZ_D]: each block requants its
    float * nq_scal;              // units, all copy the rows to LDS after a grid barrier [QZ_TMAX][QZ_D/32]
    int mtp_t;                    // qz_mtp<T>: rows of the next cycle (0 = T); -(Tc + 8 slack) =
                                  // adaptive: T if a Tc-row cycle accepted >= Tc - slack, else T-1 (ctl[7])
    float * fcap;                 // qz_cycle (card 0, null: off): DSpine feature rows [5][fcap_cap][QZ_D],
    int fcap_cap;                 // row = position % fcap_cap
    int rope_shift;               // rope position = arena position + rope_shift (images in the
                                  // prompt: each advances the rope by max(nx, ny), not nx * ny)
};

#include "qwen35_cycle.hip"
