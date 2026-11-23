// GGUF v3 metadata reader that mirrors ggml's gguf.h names and signatures. Tensor
// bytes are left to the loader. Header-only.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define GGUF_MAGIC "GGUF"
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32

#define GGML_MAX_DIMS 4

// numeric ids match ggml's ggml.h
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_Q1_0    = 41,
    GGML_TYPE_Q2_0    = 42,
    GGML_TYPE_COUNT   = 43,
};

// gguf v3 kv value types
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,
};

struct ggml_context;   // opaque, unused: kept for gguf_init_params compatibility

struct gguf_init_params {
    bool no_alloc;
    struct ggml_context * ctx;
};

// block size in elements and block bytes, as in ggml's type traits table
static inline int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_I8:  case GGML_TYPE_I16: case GGML_TYPE_I32:
        case GGML_TYPE_I64: case GGML_TYPE_F64:
            return 1;
        case GGML_TYPE_Q4_0:    case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:    case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:    case GGML_TYPE_Q8_1:
        case GGML_TYPE_IQ4_NL:  case GGML_TYPE_MXFP4:
            return 32;
        case GGML_TYPE_Q1_0: return 128;
        case GGML_TYPE_Q2_0: case GGML_TYPE_NVFP4: return 64;
        case GGML_TYPE_TQ1_0: case GGML_TYPE_TQ2_0: return 256;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ1_M:
            return 256;
        default: return 0;   // unassigned id
    }
}

static inline size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_I8:  return 1;
        case GGML_TYPE_I16: return 2;
        case GGML_TYPE_I32: return 4;
        case GGML_TYPE_I64: case GGML_TYPE_F64: return 8;
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q8_K: return 292;
        case GGML_TYPE_IQ2_XXS: return 66;
        case GGML_TYPE_IQ2_XS:  return 74;
        case GGML_TYPE_IQ3_XXS: return 98;
        case GGML_TYPE_IQ1_S:   return 50;
        case GGML_TYPE_IQ4_NL:  return 18;
        case GGML_TYPE_IQ3_S:   return 110;
        case GGML_TYPE_IQ2_S:   return 82;
        case GGML_TYPE_IQ4_XS: return 136;
        case GGML_TYPE_IQ1_M:   return 56;
        case GGML_TYPE_TQ1_0: return 54;
        case GGML_TYPE_TQ2_0: return 66;
        case GGML_TYPE_MXFP4: return 17;
        case GGML_TYPE_NVFP4: return 36;
        case GGML_TYPE_Q1_0: return 18;
        case GGML_TYPE_Q2_0: return 18;
        default: return 0;
    }
}

// -- parsed metadata --
struct gguf_kv {
    std::string key;
    enum gguf_type type = GGUF_TYPE_UINT32;
    std::vector<uint8_t> data;      // scalar bytes, or a non-string array's elements
    std::vector<std::string> strs;   // a STRING value, or a string array's elements
    enum gguf_type arr_type = GGUF_TYPE_UINT32;   // element type, when type is ARRAY
};

struct gguf_tensor_info {
    std::string name;
    uint32_t n_dims = 0;
    int64_t ne[GGML_MAX_DIMS] = {0, 0, 0, 0};   // dims at index >= n_dims are 1
    enum ggml_type type = GGML_TYPE_F32;
    uint64_t offset = 0;   // byte offset from data_offset_
};

struct gguf_context {
    std::vector<gguf_kv> kvs;
    std::vector<gguf_tensor_info> tinfo;
    uint32_t version_ = 0;
    size_t alignment_ = GGUF_DEFAULT_ALIGNMENT;
    size_t data_offset_ = 0;
};

// -- reader: little-endian host assumed --

static bool gguf_rd(FILE * f, void * p, size_t n) {
    return fread(p, 1, n, f) == n;
}

static bool gguf_rd_string(FILE * f, std::string & s) {
    uint64_t len;
    if (!gguf_rd(f, &len, sizeof(len))) return false;
    if (len > (1ull << 30)) return false;
    s.resize((size_t) len);
    return len == 0 || gguf_rd(f, &s[0], (size_t) len);
}

static size_t gguf_scalar_size(enum gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static bool gguf_rd_value(FILE * f, enum gguf_type t, gguf_kv & kv) {
    if (t == GGUF_TYPE_STRING) {
        std::string s;
        if (!gguf_rd_string(f, s)) return false;
        kv.strs.push_back(std::move(s));
        return true;
    }
    if (t == GGUF_TYPE_ARRAY) {
        uint32_t et;
        uint64_t n;
        if (!gguf_rd(f, &et, sizeof(et)) || !gguf_rd(f, &n, sizeof(n))) return false;
        if (et >= GGUF_TYPE_COUNT || et == GGUF_TYPE_ARRAY) return false;
        kv.arr_type = (enum gguf_type) et;
        if (et == GGUF_TYPE_STRING) {
            kv.strs.reserve((size_t) n);
            for (uint64_t i = 0; i < n; i++)
                if (!gguf_rd_value(f, GGUF_TYPE_STRING, kv)) return false;
            return true;
        }
        const size_t sz = gguf_scalar_size((enum gguf_type) et);
        if (sz == 0) return false;
        kv.data.resize((size_t) n * sz);
        return n == 0 || gguf_rd(f, kv.data.data(), kv.data.size());
    }
    const size_t sz = gguf_scalar_size(t);
    if (sz == 0) return false;
    const size_t off = kv.data.size();
    kv.data.resize(off + sz);
    return gguf_rd(f, kv.data.data() + off, sz);
}

static inline struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    (void) params;   // metadata only, this never reads tensor bytes
    FILE * f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "gguf_init_from_file: cannot open %s\n", fname); return nullptr; }
    char magic[4];
    uint32_t version, n_dims, ttype;
    uint64_t n_tensors, n_kv;
    if (!gguf_rd(f, magic, sizeof(magic)) || memcmp(magic, GGUF_MAGIC, 4) != 0) {
        fprintf(stderr, "gguf_init_from_file: %s is not a GGUF file\n", fname); fclose(f); return nullptr;
    }
    if (!gguf_rd(f, &version, sizeof(version))) { fclose(f); return nullptr; }
    if (version != GGUF_VERSION) {
        fprintf(stderr, "gguf_init_from_file: unsupported GGUF version %u\n", version); fclose(f); return nullptr;
    }
    if (!gguf_rd(f, &n_tensors, sizeof(n_tensors)) || !gguf_rd(f, &n_kv, sizeof(n_kv))) { fclose(f); return nullptr; }
    if (n_kv > (1ull << 27) || n_tensors > (1ull << 27)) {
        fprintf(stderr, "gguf_init_from_file: %s has implausible counts\n", fname); fclose(f); return nullptr;
    }

    gguf_context * ctx = new gguf_context();
    ctx->version_ = version;
    ctx->kvs.resize((size_t) n_kv);
    for (uint64_t i = 0; i < n_kv; i++) {
        gguf_kv & kv = ctx->kvs[(size_t) i];
        uint32_t t;
        if (!gguf_rd_string(f, kv.key) || !gguf_rd(f, &t, sizeof(t)) || t >= GGUF_TYPE_COUNT ||
            !gguf_rd_value(f, (enum gguf_type) t, kv)) {
            fprintf(stderr, "gguf_init_from_file: bad kv %llu\n", (unsigned long long) i);
            delete ctx; fclose(f); return nullptr;
        }
        kv.type = (enum gguf_type) t;
    }

    ctx->tinfo.resize((size_t) n_tensors);
    const gguf_kv * align = nullptr;
    for (auto & kv : ctx->kvs)
        if (kv.key == "general.alignment") { align = &kv; break; }
    if (align) {
        if (align->type != GGUF_TYPE_UINT32 || align->data.size() != 4) {
            fprintf(stderr, "gguf_init_from_file: general.alignment must be u32\n");
            delete ctx; fclose(f); return nullptr;
        }
        uint32_t a;
        memcpy(&a, align->data.data(), 4);
        if (a == 0 || (a & (a - 1)) != 0) {
            fprintf(stderr, "gguf_init_from_file: alignment %u is not a power of 2\n", a);
            delete ctx; fclose(f); return nullptr;
        }
        ctx->alignment_ = a;
    }

    ctx->tinfo.resize((size_t) n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info & ti = ctx->tinfo[(size_t) i];
        if (!gguf_rd_string(f, ti.name) || !gguf_rd(f, &n_dims, sizeof(n_dims)) ||
            n_dims == 0 || n_dims > GGML_MAX_DIMS) {
            fprintf(stderr, "gguf_init_from_file: bad tensor %llu\n", (unsigned long long) i);
            delete ctx; fclose(f); return nullptr;
        }
        ti.n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t ne;
            if (!gguf_rd(f, &ne, sizeof(ne))) {
                fprintf(stderr, "gguf_init_from_file: bad tensor %llu\n", (unsigned long long) i);
                delete ctx; fclose(f); return nullptr;
            }
            ti.ne[d] = (int64_t) ne;
        }
        for (uint32_t d = n_dims; d < GGML_MAX_DIMS; d++) ti.ne[d] = 1;
        if (!gguf_rd(f, &ttype, sizeof(ttype)) ||
            ttype >= GGML_TYPE_COUNT || ggml_blck_size((enum ggml_type) ttype) == 0) {
            fprintf(stderr, "gguf_init_from_file: unsupported tensor type %u\n", ttype);
            delete ctx; fclose(f); return nullptr;
        }
        ti.type = (enum ggml_type) ttype;
        if (!gguf_rd(f, &ti.offset, sizeof(ti.offset))) { fclose(f); return nullptr; }
    }

    // tensor data starts after the infos, aligned up
    size_t pos = (size_t) ftell(f);
    if (n_tensors > 0)
        pos = (pos + ctx->alignment_ - 1) / ctx->alignment_ * ctx->alignment_;
    ctx->data_offset_ = pos;

    fclose(f);
    return ctx;
}

static inline void gguf_free(struct gguf_context * ctx) { delete ctx; }

// -- kv getters, for keys already found with gguf_find_key --

static inline size_t   gguf_get_data_offset(const struct gguf_context * ctx) { return ctx->data_offset_; }

static inline int64_t gguf_find_key(const struct gguf_context * ctx, const char * key) {
    for (size_t i = 0; i < ctx->kvs.size(); i++)
        if (ctx->kvs[i].key == key) return (int64_t) i;
    return -1;
}

static inline enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int64_t key_id) {
    return key_id >= 0 && (size_t) key_id < ctx->kvs.size() ? ctx->kvs[(size_t) key_id].type : GGUF_TYPE_COUNT;
}

// any integer kv that fits in u32, including u64 and i64 in range
static inline uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int64_t key_id) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size()) return 0;
    const gguf_kv & kv = ctx->kvs[(size_t) key_id];
    uint64_t v = 0;
    switch (kv.type) {
        case GGUF_TYPE_UINT8:  { uint8_t  x; memcpy(&x, kv.data.data(), 1); v = x; break; }
        case GGUF_TYPE_INT8:   { int8_t   x; memcpy(&x, kv.data.data(), 1); v = (uint64_t) (int64_t) x; break; }
        case GGUF_TYPE_UINT16: { uint16_t x; memcpy(&x, kv.data.data(), 2); v = x; break; }
        case GGUF_TYPE_INT16:  { int16_t  x; memcpy(&x, kv.data.data(), 2); v = (uint64_t) (int64_t) x; break; }
        case GGUF_TYPE_UINT32: { uint32_t x; memcpy(&x, kv.data.data(), 4); v = x; break; }
        case GGUF_TYPE_INT32:  { int32_t  x; memcpy(&x, kv.data.data(), 4); v = (uint64_t) (int64_t) x; break; }
        case GGUF_TYPE_UINT64: { memcpy(&v, kv.data.data(), 8); break; }
        case GGUF_TYPE_INT64:  { int64_t x; memcpy(&x, kv.data.data(), 8); v = (uint64_t) x; break; }
        default:
            fprintf(stderr, "gguf_get_val_u32: %s is not an integer\n", kv.key.c_str());
            return 0;
    }
    if (v > 0xffffffffull)
        fprintf(stderr, "gguf_get_val_u32: %s does not fit\n", kv.key.c_str());
    return (uint32_t) v;
}

static inline float gguf_get_val_f32(const struct gguf_context * ctx, int64_t key_id) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size()) return 0.0f;
    const gguf_kv & kv = ctx->kvs[(size_t) key_id];
    if (kv.type == GGUF_TYPE_FLOAT32) {
        float x; memcpy(&x, kv.data.data(), sizeof(x)); return x;
    }
    if (kv.type == GGUF_TYPE_FLOAT64) {
        double x; memcpy(&x, kv.data.data(), sizeof(x)); return (float) x;
    }
    fprintf(stderr, "gguf_get_val_f32: %s is not a float\n", kv.key.c_str());
    return 0.0f;
}

static inline const char * gguf_get_val_str(const struct gguf_context * ctx, int64_t key_id) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size() || ctx->kvs[(size_t) key_id].type != GGUF_TYPE_STRING) {
        fprintf(stderr, "gguf_get_val_str: bad key id %lld\n", (long long) key_id);
        return "";
    }
    return ctx->kvs[(size_t) key_id].strs[0].c_str();
}

static inline bool gguf_get_val_bool(const struct gguf_context * ctx, int64_t key_id) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size() || ctx->kvs[(size_t) key_id].type != GGUF_TYPE_BOOL) {
        fprintf(stderr, "gguf_get_val_bool: bad key id %lld\n", (long long) key_id);
        return false;
    }
    return ctx->kvs[(size_t) key_id].data[0] != 0;
}

static inline size_t gguf_get_arr_n(const struct gguf_context * ctx, int64_t key_id) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size() || ctx->kvs[(size_t) key_id].type != GGUF_TYPE_ARRAY) return 0;
    const gguf_kv & kv = ctx->kvs[(size_t) key_id];
    return kv.arr_type == GGUF_TYPE_STRING ? kv.strs.size() : kv.data.size() / (gguf_scalar_size(kv.arr_type) ? gguf_scalar_size(kv.arr_type) : 1);
}

static inline enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size() || ctx->kvs[(size_t) key_id].type != GGUF_TYPE_ARRAY) return GGUF_TYPE_COUNT;
    return ctx->kvs[(size_t) key_id].arr_type;
}

static inline const void * gguf_get_arr_data(const struct gguf_context * ctx, int64_t key_id) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size() || ctx->kvs[(size_t) key_id].type != GGUF_TYPE_ARRAY ||
        ctx->kvs[(size_t) key_id].arr_type == GGUF_TYPE_STRING) {
        fprintf(stderr, "gguf_get_arr_data: bad key id %lld\n", (long long) key_id);
        return nullptr;
    }
    return ctx->kvs[(size_t) key_id].data.data();
}

static inline const char * gguf_get_arr_str(const struct gguf_context * ctx, int64_t key_id, size_t i) {
    if (key_id < 0 || (size_t) key_id >= ctx->kvs.size() || ctx->kvs[(size_t) key_id].type != GGUF_TYPE_ARRAY ||
        i >= ctx->kvs[(size_t) key_id].strs.size()) {
        fprintf(stderr, "gguf_get_arr_str: bad key id %lld or index %zu\n", (long long) key_id, i);
        return "";
    }
    return ctx->kvs[(size_t) key_id].strs[i].c_str();
}

// -- tensor info getters --

static inline int64_t gguf_get_n_tensors(const struct gguf_context * ctx) { return (int64_t) ctx->tinfo.size(); }

static inline const char * gguf_get_tensor_name(const struct gguf_context * ctx, int64_t tensor_id) {
    return tensor_id >= 0 && (size_t) tensor_id < ctx->tinfo.size() ? ctx->tinfo[(size_t) tensor_id].name.c_str() : "";
}

static inline size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id) {
    return tensor_id >= 0 && (size_t) tensor_id < ctx->tinfo.size() ? (size_t) ctx->tinfo[(size_t) tensor_id].offset : 0;
}

static inline enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int64_t tensor_id) {
    return tensor_id >= 0 && (size_t) tensor_id < ctx->tinfo.size() ? ctx->tinfo[(size_t) tensor_id].type : GGML_TYPE_F32;
}

static inline const int64_t * gguf_get_tensor_ne(const struct gguf_context * ctx, int64_t tensor_id) {
    static const int64_t zeros[GGML_MAX_DIMS] = {0, 0, 0, 0};
    if (tensor_id < 0 || (size_t) tensor_id >= ctx->tinfo.size()) return zeros;
    return ctx->tinfo[(size_t) tensor_id].ne;
}
