// Tokenizer: byte-level BPE + the GGUF's pre-tokenizer (tokenizer.ggml.pre),
// mirroring llama.cpp's BPE path; C entry points for ctypes.
#include "gguf_lite.h"
#include "unicode.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// llama.cpp token attribute flags (subset used by BPE models)
enum {
    ATTR_UNKNOWN     = 1 << 0,
    ATTR_UNUSED      = 1 << 1,
    ATTR_NORMAL      = 1 << 2,
    ATTR_CONTROL     = 1 << 3,
    ATTR_USER_DEFINED = 1 << 4,
    ATTR_BYTE        = 1 << 5,
};

struct tok_data {
    std::string text;
    uint32_t attr = 0;
};

// pre-tokenizers by tokenizer.ggml.pre: llama.cpp's regex lists for those pre
// types, exact strings (qwen35 dispatches to unicode.cpp's custom splitter;
// gpt-4o is the lookahead form llama.cpp runs on std::regex)
const char * const QWEN35_RE =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const char * const GPT4O_RE =
    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

struct tok_pre {
    const char * name;                       // tokenizer.ggml.pre
    std::initializer_list<const char *> re;  // applied in order
    bool no_bos;                             // llama.cpp clears the bos id
};
const tok_pre TOK_PRES[] = {
    { "qwen35",   { QWEN35_RE }, false },
    { "inkling2", { GPT4O_RE }, true },
    { "gpt-4o",   { GPT4O_RE }, false },
};

struct tok_vocab {
    std::vector<tok_data> toks;                    // by id
    std::map<std::string, int32_t> text_to_id;     // like llama's token_to_id
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks;
    bool add_bos = false;
    bool add_eos = false;
    int32_t bos_id = -1, eos_id = -1;
    std::vector<int32_t> special;   // CONTROL|USER_DEFINED|UNKNOWN, text len desc
    std::vector<int32_t> eog;       // ascending
    std::string chat_template;      // tokenizer.chat_template, empty when absent
    std::vector<std::string> pre_re;   // the pre-tokenizer's regexes
    bool no_bos = false;

    int32_t find_token(const std::string & s) const {
        auto it = text_to_id.find(s);
        return it == text_to_id.end() ? -1 : it->second;
    }
    int32_t find_bpe_rank(const std::string & l, const std::string & r) const {
        auto it = bpe_ranks.find(std::make_pair(l, r));
        return it == bpe_ranks.end() ? -1 : it->second;
    }
    const std::string & text(int32_t id) const { return toks[id].text; }
    uint32_t attr(int32_t id) const { return toks[id].attr; }
};

// text-sorted id order, like iterating llama's token_to_id map
static std::vector<int32_t> sorted_by_text(const tok_vocab & v) {
    std::vector<int32_t> order(v.toks.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int32_t) i;
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return v.text(a) < v.text(b);
    });
    return order;
}

// first id (in text-sorted order) whose text is in `names`
static int32_t find_by_texts(const std::vector<int32_t> & order, const tok_vocab & v,
                             std::initializer_list<const char *> names) {
    for (const char * n : names)
        for (int32_t id : order)
            if (v.text(id) == n) return id;
    return -1;
}

// llama.cpp EOG text list entry "<｜end▁of▁sentence｜>" (deepseek-ocr), built
// from codepoints so this file stays pure ASCII
static const std::string & deepseek_eog_text() {
    static const std::string s = "<" + unicode_cpt_to_utf8(0xFF5C) + "end" +
        unicode_cpt_to_utf8(0x2581) + "of" + unicode_cpt_to_utf8(0x2581) +
        "sentence" + unicode_cpt_to_utf8(0xFF5C) + ">";
    return s;
}

// tokens that end generation: fim pad/repo/file_sep, the eog text list, then
// eos / eot / eom
static void build_eog(tok_vocab & v, const std::vector<int32_t> & order) {
    const int32_t eot = find_by_texts(order, v, {"<|eot_id|>", "<|im_end|>", "<|end|>",
        "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]"});
    (void) eot;
    const int32_t eom = find_by_texts(order, v, {"<|eom_id|>"});
    (void) eom;
    int32_t fim_pad = find_by_texts(order, v, {"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"});
    int32_t fim_rep = find_by_texts(order, v, {"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"});

    std::set<int32_t> ids;
    if (fim_pad >= 0) ids.insert(fim_pad);
    if (fim_rep >= 0) ids.insert(fim_rep);
    int32_t fim_sep = find_by_texts(order, v, {"<|file_sep|>", "<|plamo:file_separator|>"});
    if (fim_sep >= 0) ids.insert(fim_sep);
    for (int32_t id : order) {
        const std::string & t = v.text(id);
        if (t == "<|eot_id|>" || t == "<|im_end|>" || t == "<|end|>" || t == "<|return|>" ||
            t == "<|call|>" || t == "<|flush|>" || t == "<|calls|>" || t == "<end_of_turn>" ||
            t == "<|endoftext|>" || t == "</s>" || t == "<|eom_id|>" || t == "<EOT>" || t == "_<EOT>" ||
            t == "[EOT]" || t == "[EOS]" || t == "<|end_of_text|>" || t == "<end_of_utterance>" ||
            t == "<eos>" || t == "<turn|>" || t == "<|tool_response>" ||
            t == deepseek_eog_text() ||
            t == "[e~[" || t == "<|ifm|im_end|>") {
            ids.insert(id);
        }
    }
    if (v.eos_id >= 0) ids.insert(v.eos_id);
    if (eot >= 0) ids.insert(eot);
    if (eom >= 0) ids.insert(eom);
    v.eog.assign(ids.begin(), ids.end());
}

// CONTROL or UNKNOWN tokens with "unused" in the text are UNUSED as well,
// so they drop out of the special-token list
static void mark_unused_by_name(tok_vocab & v) {
    for (auto & t : v.toks)
        if ((t.attr & ATTR_CONTROL) && !(t.attr & ATTR_UNUSED) &&
            t.text.find("unused") != std::string::npos)
            t.attr |= ATTR_UNUSED;
}

static bool vocab_load(tok_vocab & v, const gguf_context * g) {
    const int64_t model_id = gguf_find_key(g, "tokenizer.ggml.model");
    if (model_id < 0 || gguf_get_kv_type(g, model_id) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(g, model_id)) != "gpt2") {
        fprintf(stderr, "qz_tok: unsupported tokenizer.ggml.model\n");
        return false;
    }
    {   // absent key: qwen35, the only pre-tokenizer before the table
        const int64_t id = gguf_find_key(g, "tokenizer.ggml.pre");
        const std::string pre = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "qwen35";
        const tok_pre * tp = nullptr;
        for (const auto & e : TOK_PRES) if (pre == e.name) tp = &e;
        if (!tp) {
            fprintf(stderr, "qz_tok: unsupported tokenizer.ggml.pre '%s'; supported:", pre.c_str());
            for (const auto & e : TOK_PRES) fprintf(stderr, " %s", e.name);
            fprintf(stderr, "\n");
            return false;
        }
        v.pre_re.assign(tp->re.begin(), tp->re.end());
        v.no_bos = tp->no_bos;
    }
    const int64_t tok_id = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tok_id < 0 || gguf_get_kv_type(g, tok_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, tok_id) != GGUF_TYPE_STRING) {
        fprintf(stderr, "qz_tok: no tokenizer.ggml.tokens string array\n");
        return false;
    }
    const size_t n = gguf_get_arr_n(g, tok_id);

    const int32_t * types = nullptr;
    const int64_t type_id = gguf_find_key(g, "tokenizer.ggml.token_type");
    if (type_id >= 0) {
        if (gguf_get_kv_type(g, type_id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g, type_id) != GGUF_TYPE_INT32 ||
            gguf_get_arr_n(g, type_id) < n) {
            fprintf(stderr, "qz_tok: bad tokenizer.ggml.token_type array\n");
            return false;
        }
        types = (const int32_t *) gguf_get_arr_data(g, type_id);
    }

    v.toks.resize(n);
    for (size_t i = 0; i < n; i++) {
        std::string word = gguf_get_arr_str(g, tok_id, i);
        if (word.empty()) word = "[EMPTY_" + std::to_string(i) + "]";
        uint32_t attr = ATTR_NORMAL;
        if (types) {
            switch (types[i]) {
                case 2: attr = ATTR_UNKNOWN; break;
                case 3: attr = ATTR_CONTROL; break;
                case 4: attr = ATTR_USER_DEFINED; break;
                case 5: attr = ATTR_UNUSED; break;
                case 6: attr = ATTR_BYTE; break;
                case 1: attr = ATTR_NORMAL; break;
                default: attr = 0; break;   // UNDEFINED
            }
        }
        v.toks[i].text = std::move(word);
        v.toks[i].attr = attr;
        v.text_to_id[v.toks[i].text] = (int32_t) i;
    }

    const int64_t merge_id = gguf_find_key(g, "tokenizer.ggml.merges");
    if (merge_id < 0 || gguf_get_kv_type(g, merge_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, merge_id) != GGUF_TYPE_STRING) {
        fprintf(stderr, "qz_tok: no tokenizer.ggml.merges string array\n");
        return false;
    }
    const size_t nm = gguf_get_arr_n(g, merge_id);
    for (size_t i = 0; i < nm; i++) {
        const std::string word = gguf_get_arr_str(g, merge_id, i);
        std::string first, second;   // empty pair if there is no space
        const size_t pos = word.find(' ', 1);
        if (pos != std::string::npos) {
            first = word.substr(0, pos);
            second = word.substr(pos + 1);
        }
        v.bpe_ranks.emplace(std::make_pair(first, second), (int32_t) i);
    }

    {   // add_bos / add_eos: absent key means the BPE default (false here)
        const int64_t id = gguf_find_key(g, "tokenizer.ggml.add_bos_token");
        if (id >= 0) v.add_bos = gguf_get_val_bool(g, id);
    }
    {
        const int64_t id = gguf_find_key(g, "tokenizer.ggml.add_eos_token");
        if (id >= 0) v.add_eos = gguf_get_val_bool(g, id);
    }
    // gpt2 defaults (llama.cpp) when the GGUF does not carry the ids
    v.bos_id = 11;
    v.eos_id = 11;
    {
        const int64_t id = gguf_find_key(g, "tokenizer.ggml.bos_token_id");
        if (id >= 0) v.bos_id = (int32_t) gguf_get_val_u32(g, id);
    }
    if (v.no_bos) v.bos_id = -1;
    {
        const int64_t id = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
        if (id >= 0) v.eos_id = (int32_t) gguf_get_val_u32(g, id);
    }

    mark_unused_by_name(v);
    const std::vector<int32_t> order = sorted_by_text(v);
    build_eog(v, order);

    // special tokens: CONTROL | USER_DEFINED | UNKNOWN, longest text first
    v.special.clear();
    for (size_t i = 0; i < v.toks.size(); i++)
        if (v.toks[i].attr & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN))
            v.special.push_back((int32_t) i);
    std::sort(v.special.begin(), v.special.end(), [&](int32_t a, int32_t b) {
        return v.text(a).size() > v.text(b).size();
    });
    return true;
}

// raw-text / token fragments of the input, after the special-token split
struct frag {
    bool is_tok = false;
    int32_t tok = -1;
    int64_t off = 0, len = 0;   // into the raw text when !is_tok
};

// llama.cpp tokenizer_st_partition: split raw fragments on each special token
// (CONTROL / UNKNOWN only with parse_special)
static void st_partition(std::vector<frag> & frags, const std::string & text,
                         const tok_vocab & v, bool parse_special) {
    for (int32_t sp : v.special) {
        if (!parse_special && (v.attr(sp) & (ATTR_CONTROL | ATTR_UNKNOWN))) continue;
        const std::string & s = v.text(sp);
        if (s.empty()) continue;
        std::vector<frag> out;
        out.reserve(frags.size());
        for (const frag & f : frags) {
            if (f.is_tok) { out.push_back(f); continue; }
            int64_t off = f.off, len = f.len;
            while (len > 0) {
                const std::string_view view(text.data() + off, (size_t) len);
                const size_t m = view.find(s);
                if (m == std::string_view::npos || m + s.size() > (size_t) len) break;
                if (m > 0) out.push_back(frag{false, -1, off, (int64_t) m});
                out.push_back(frag{true, sp, 0, 0});
                off += (int64_t) (m + s.size());
                len -= (int64_t) (m + s.size());
            }
            if (len > 0) out.push_back(frag{false, -1, off, len});
        }
        frags = std::move(out);
    }
}

// one BPE symbol: a byte run inside the current word
struct sym {
    int prev = -1, next = -1;
    int32_t off = 0;   // offset into the word
    size_t n = 0;
};

struct bigram {
    int left = -1, right = -1;
    std::string text;
    int32_t rank = 0;
};

struct bigram_cmp {
    bool operator()(const bigram & l, const bigram & r) const {
        return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }
};

// llm_tokenizer_bpe_session::tokenize for one raw fragment (byte_encode BPE)
static void bpe_fragment(const std::string & text, const tok_vocab & v, std::vector<int32_t> & output) {
    const std::vector<std::string> words = unicode_regex_split(text, v.pre_re, true);
    for (const std::string & word : words) {
        std::vector<sym> symbols;
        size_t off = 0;
        while (off < word.size()) {
            const size_t cl = std::min(word.size() - off, unicode_len_utf8(word[off]));
            sym s;
            s.off = (int32_t) off;
            s.n = cl;
            s.prev = (int32_t) symbols.size() - 1;
            s.next = off + cl == word.size() ? -1 : (int32_t) symbols.size() + 1;
            symbols.push_back(s);
            off += cl;
        }
        std::priority_queue<bigram, std::vector<bigram>, bigram_cmp> queue;
        auto add = [&](int left, int right) {
            if (left < 0 || right < 0 || (size_t) right >= symbols.size()) return;
            const std::string lt = word.substr((size_t) symbols[left].off, symbols[left].n);
            const std::string rt = word.substr((size_t) symbols[right].off, symbols[right].n);
            const int32_t rank = v.find_bpe_rank(lt, rt);
            if (rank < 0) return;
            queue.push(bigram{left, right, lt + rt, rank});
        };
        for (size_t i = 1; i < symbols.size(); i++) add((int32_t) i - 1, (int32_t) i);

        while (!queue.empty()) {
            const bigram b = queue.top();
            queue.pop();
            sym & ls = symbols[b.left];
            sym & rs = symbols[b.right];
            if (ls.n == 0 || rs.n == 0) continue;
            const std::string lt = word.substr((size_t) ls.off, ls.n);
            const std::string rt = word.substr((size_t) rs.off, rs.n);
            if (lt + rt != b.text) continue;   // stale
            // merge the right symbol into the left one
            ls.n += rs.n;
            rs.n = 0;
            ls.next = rs.next;
            if (rs.next >= 0) symbols[rs.next].prev = b.left;
            add(ls.prev, b.left);
            add(b.left, ls.next);
        }

        for (const sym & s : symbols) {
            if (s.n == 0) continue;
            const std::string str = word.substr((size_t) s.off, s.n);
            int32_t tok = v.find_token(str);
            if (tok < 0) {
                for (size_t j = 0; j < s.n; j++) {   // one lookup per raw byte
                    const std::string bs = word.substr((size_t) s.off + j, 1);
                    const int32_t tb = v.find_token(bs);
                    if (tb >= 0) output.push_back(tb);
                }
            } else {
                output.push_back(tok);
            }
        }
    }
}

// llama.cpp llama_decode_text: byte-level decode of a NORMAL token
static std::string decode_text(const std::string & text) {
    std::string out;
    for (uint32_t cpt : unicode_cpts_from_utf8(text)) {
        const std::string utf8 = unicode_cpt_to_utf8(cpt);
        try {
            out += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range &) {
            char buf[32];
            snprintf(buf, sizeof(buf), "[UNK_BYTE_0x");
            out += buf;
            for (char c : utf8) { snprintf(buf, sizeof(buf), "%02x", (uint8_t) c); out += buf; }
            out += text + "]";
        }
    }
    return out;
}

} // namespace

extern "C" {

void * qz_tok_open(const char * model) {
    gguf_init_params p = {};
    p.no_alloc = true;
    p.ctx = nullptr;
    gguf_context * g = gguf_init_from_file(model, p);
    if (!g) return nullptr;
    tok_vocab * v = new tok_vocab();
    const bool ok = vocab_load(*v, g);
    const int64_t ct = gguf_find_key(g, "tokenizer.chat_template");
    if (ok && ct >= 0 && gguf_get_kv_type(g, ct) == GGUF_TYPE_STRING) v->chat_template = gguf_get_val_str(g, ct);
    gguf_free(g);
    if (!ok) { delete v; return nullptr; }
    return v;
}

// returns the token count, or -(needed) when out is too small
int qz_tok_encode(void * m, const char * text, int add_special, int parse_special, int32_t * out, int max) {
    if (!m) return -1;
    const tok_vocab & v = *(tok_vocab *) m;
    const std::string raw(text ? text : "");
    std::vector<frag> frags;
    if (!raw.empty()) {
        frags.push_back(frag{false, -1, 0, (int64_t) raw.size()});
        st_partition(frags, raw, v, parse_special != 0);
    }
    std::vector<int32_t> output;
    if (add_special && v.add_bos && v.bos_id >= 0) output.push_back(v.bos_id);
    for (const frag & f : frags) {
        if (f.is_tok) { output.push_back(f.tok); continue; }
        bpe_fragment(raw.substr((size_t) f.off, (size_t) f.len), v, output);
    }
    if (add_special && v.add_eos && v.eos_id >= 0) output.push_back(v.eos_id);
    if (!out || (int) output.size() > max) return -(int) output.size();
    memcpy(out, output.data(), output.size() * sizeof(int32_t));
    return (int) output.size();
}

// bytes of one token's piece (special tokens rendered when special != 0);
// negative: -(needed)
int qz_tok_piece(void * m, int32_t tok, char * buf, int len, int special) {
    if (!m) return 0;
    const tok_vocab & v = *(tok_vocab *) m;
    if (tok < 0 || (size_t) tok >= v.toks.size()) return 0;
    const uint32_t attr = v.attr(tok);
    if (!special && (attr & (ATTR_UNKNOWN | ATTR_CONTROL))) return 0;
    std::string piece;
    if (attr & (ATTR_UNKNOWN | ATTR_CONTROL | ATTR_USER_DEFINED)) {
        piece = v.text(tok);
    } else if (attr & ATTR_NORMAL) {
        piece = decode_text(v.text(tok));
    } else if (attr & ATTR_BYTE) {
        piece = (char) (uint8_t) strtol(v.text(tok).substr(3, 2).c_str(), nullptr, 16);
    } else {
        return 0;   // UNUSED and UNDEFINED render empty
    }
    if (!buf || len < (int) piece.size()) return -(int) piece.size();
    memcpy(buf, piece.data(), piece.size());
    return (int) piece.size();
}

// all end-of-generation ids, ascending, up to max
int qz_tok_eog(void * m, int32_t * out, int max) {
    if (!m) return 0;
    const tok_vocab & v = *(tok_vocab *) m;
    if (!out || max < (int) v.eog.size()) return -(int) v.eog.size();
    memcpy(out, v.eog.data(), v.eog.size() * sizeof(int32_t));
    return (int) v.eog.size();
}

// the GGUF's tokenizer.chat_template (UTF-8), or NULL when it has none
const char * qz_tok_chat_template(void * m) {
    if (!m) return nullptr;
    const tok_vocab & v = *(tok_vocab *) m;
    return v.chat_template.empty() ? nullptr : v.chat_template.c_str();
}

}
