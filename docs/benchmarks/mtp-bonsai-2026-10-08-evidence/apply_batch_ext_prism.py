#!/usr/bin/env python3
"""Prism fork (adfffbe): add upstream llama.cpp's extended-batch API (llama_batch_ext_* + llama_process) as a
thin layer over the fork's llama_batch, so a client written against upstream (merlin-endpoint's MTP path)
can drive this runtime. Semantics follow upstream 889edf43 (src/llama-batch.cpp); the batch is materialized
into the old llama_batch the way the fork's own draft-mtp does (tokens AND per-token embeddings, one
position per token broadcast across rope sections by the allocator)."""
import sys
from pathlib import Path
root = Path(sys.argv[1])

def edit(rel, before, after):
    p = root / rel; t = p.read_text()
    assert t.count(before) == 1, (rel, before[:60])
    p.write_text(t.replace(before, after)); print("edited", rel)

edit("include/llama.h",
"    LLAMA_API void llama_batch_free(struct llama_batch batch);\n",
"""    LLAMA_API void llama_batch_free(struct llama_batch batch);

    //
    // Extended batch API (upstream llama.cpp, ported: a layer over llama_batch)
    //

    enum llama_process_type {
        LLAMA_PROCESS_TYPE_ENCODE,
        LLAMA_PROCESS_TYPE_DECODE,
    };

    struct llama_batch_ext;

    struct llama_embd {
        const float * data;
        size_t n_rows; // number of embedding rows in data
        size_t n_embd; // size of one row
    };

    LLAMA_API struct llama_batch_ext * llama_batch_ext_init (struct llama_context * ctx);
    LLAMA_API void                     llama_batch_ext_free (struct llama_batch_ext * batch);
    LLAMA_API void                     llama_batch_ext_clear(struct llama_batch_ext * batch);

    // Add an input token: id = LLAMA_TOKEN_NULL, no embedding, position unset (set it with llama_batch_ext_set_pos).
    // Returns the batch index (>= 0); -1 batch full, -2 invalid token, -3 invalid sequence id.
    LLAMA_API int32_t llama_batch_ext_add      (struct llama_batch_ext * batch, llama_seq_id seq_id);
    LLAMA_API int32_t llama_batch_ext_add_token(struct llama_batch_ext * batch, llama_seq_id seq_id, llama_token id);

    LLAMA_API bool llama_batch_ext_add_seq(struct llama_batch_ext * batch, int32_t idx, llama_seq_id seq_id);

    // Set the embedding row for the entry at idx; after add_token the entry carries both an id and an embedding
    // (the MTP draft head reads the token id and the target's hidden state)
    LLAMA_API bool llama_batch_ext_set_embd_token(struct llama_batch_ext * batch, int32_t idx, struct llama_embd embd);

    LLAMA_API bool llama_batch_ext_set_output_embd  (struct llama_batch_ext * batch, int32_t idx, bool value);
    LLAMA_API bool llama_batch_ext_set_output_logits(struct llama_batch_ext * batch, int32_t idx, bool value);

    // One position for a token entry; n_pos_per_embd positions for an embedding-only entry (M-RoPE)
    LLAMA_API bool llama_batch_ext_set_pos(struct llama_batch_ext * batch, int32_t idx, const llama_pos * pos);

    // Return values are the same as llama_decode()
    LLAMA_API int32_t llama_process(struct llama_context * ctx, enum llama_process_type type, struct llama_batch_ext * batch);
""")

p = root / "src/llama-batch.h"; t = p.read_text()
assert "struct llama_batch_ext" not in t
t += """
// upstream llama.cpp's extended batch, ported as a layer over llama_batch (include/llama.h "Extended batch API")
struct llama_batch_ext {
    const size_t n_tokens_max;
    const size_t n_embd_inp;      // row width an embedding entry must carry (MTP context: the target's hidden width)
    const llama_seq_id n_seq_max;
    const llama_token n_vocab;
    const size_t n_pos_per_embd;

    struct token {
        llama_token id = LLAMA_TOKEN_NULL;
        bool        has_embd = false;
        size_t      embd_off = 0;
        bool        output = false;
        std::vector<llama_seq_id> seq_ids;
        std::array<llama_pos, 4> pos = {0, 0, 0, 0};
    };
    std::vector<token> tokens;
    std::vector<float> embd;

    llama_batch_ext(size_t n_tokens_max, size_t n_embd_inp, llama_seq_id n_seq_max, llama_token n_vocab, size_t n_pos_per_embd);

    void    clear();
    int32_t add_token(llama_seq_id seq_id);
    bool    add_seq(int32_t idx, llama_seq_id seq_id);
    bool    set_token_id(int32_t idx, llama_token id);
    bool    set_token_embd(int32_t idx, llama_embd embd_in);
    bool    set_token_pos(int32_t idx, const llama_pos * pos_in);
    bool    set_output(int32_t idx, bool value);

    // materialize into a llama_batch (storage owned by this object, valid until the next call or clear())
    // returns false when the entries cannot form one batch (mixed with/without embeddings or ids)
    bool to_batch(llama_batch & out);

private:
    std::vector<llama_token>    m_token;
    std::vector<float>          m_embd;
    std::vector<llama_pos>      m_pos;
    std::vector<int32_t>        m_n_seq_id;
    std::vector<llama_seq_id *> m_seq_id;
    std::vector<llama_seq_id>   m_seq_storage;
    std::vector<int8_t>         m_logits;
};
"""
p.write_text(t); print("edited src/llama-batch.h")

edit("src/llama-batch.cpp",
"#include \"llama-batch.h\"\n\n#include \"llama-impl.h\"\n#include \"llama-vocab.h\"\n#include \"llama-memory.h\"\n",
"#include \"llama-batch.h\"\n\n#include \"llama-impl.h\"\n#include \"llama-vocab.h\"\n#include \"llama-memory.h\"\n#include \"llama-context.h\"\n#include \"llama-model.h\"\n")

p = root / "src/llama-batch.cpp"; t = p.read_text()
t += r"""
//
// llama_batch_ext (upstream extended batch API as a layer over llama_batch)
//

llama_batch_ext::llama_batch_ext(size_t n_tokens_max, size_t n_embd_inp, llama_seq_id n_seq_max, llama_token n_vocab, size_t n_pos_per_embd) :
        n_tokens_max(n_tokens_max), n_embd_inp(n_embd_inp), n_seq_max(n_seq_max), n_vocab(n_vocab), n_pos_per_embd(n_pos_per_embd) {
    clear();
}

void llama_batch_ext::clear() {
    tokens.clear();
    embd.clear();
}

int32_t llama_batch_ext::add_token(llama_seq_id seq_id) {
    if (tokens.size() >= n_tokens_max) {
        return -1;
    }
    if (seq_id < 0 || seq_id >= n_seq_max) {
        return -3;
    }
    token t;
    t.seq_ids.push_back(seq_id);
    tokens.push_back(std::move(t));
    return (int32_t) (tokens.size() - 1);
}

bool llama_batch_ext::add_seq(int32_t idx, llama_seq_id seq_id) {
    if (idx < 0 || idx >= (int32_t) tokens.size() || seq_id < 0 || seq_id >= n_seq_max) {
        return false;
    }
    auto & ids = tokens[idx].seq_ids;
    if (std::find(ids.begin(), ids.end(), seq_id) == ids.end()) {
        ids.push_back(seq_id);
    }
    return true;
}

bool llama_batch_ext::set_token_id(int32_t idx, llama_token id) {
    if (idx < 0 || idx >= (int32_t) tokens.size() || id < 0 || id >= n_vocab) {
        return false;
    }
    tokens[idx].id = id;
    return true;
}

bool llama_batch_ext::set_token_embd(int32_t idx, llama_embd embd_in) {
    if (idx < 0 || idx >= (int32_t) tokens.size() || !embd_in.data) {
        return false;
    }
    const size_t n_total = embd_in.n_rows * embd_in.n_embd;
    if (n_total != n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embedding size mismatch, got %zu rows x %zu = %zu, expected %zu\n",
                __func__, embd_in.n_rows, embd_in.n_embd, n_total, n_embd_inp);
        return false;
    }
    token & t = tokens[idx];
    if (t.has_embd) {
        LLAMA_LOG_ERROR("%s: embedding for token %d is already set\n", __func__, idx);
        return false;
    }
    t.has_embd = true;
    t.embd_off = embd.size();
    embd.insert(embd.end(), embd_in.data, embd_in.data + n_total);
    return true;
}

bool llama_batch_ext::set_token_pos(int32_t idx, const llama_pos * pos_in) {
    if (idx < 0 || idx >= (int32_t) tokens.size() || !pos_in) {
        return false;
    }
    token & t = tokens[idx];
    const size_t n_pos = t.id != LLAMA_TOKEN_NULL ? 1 : std::min<size_t>(n_pos_per_embd, t.pos.size());
    for (size_t i = 0; i < n_pos; ++i) {
        t.pos[i] = pos_in[i];
    }
    return true;
}

bool llama_batch_ext::set_output(int32_t idx, bool value) {
    if (idx < 0 || idx >= (int32_t) tokens.size()) {
        return false;
    }
    tokens[idx].output = value;
    return true;
}

bool llama_batch_ext::to_batch(llama_batch & out) {
    const size_t n = tokens.size();
    out = { /*n_tokens=*/ 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    if (n == 0) {
        LLAMA_LOG_ERROR("%s: empty batch\n", __func__);
        return false;
    }
    size_t n_ids = 0, n_embd_set = 0, n_seq_total = 0;
    for (const auto & t : tokens) {
        n_ids       += t.id != LLAMA_TOKEN_NULL;
        n_embd_set  += t.has_embd;
        n_seq_total += t.seq_ids.size();
    }
    if (n_embd_set != 0 && n_embd_set != n) {
        LLAMA_LOG_ERROR("%s: %zu of %zu entries carry an embedding; a batch carries embeddings for all entries or none\n", __func__, n_embd_set, n);
        return false;
    }
    if (n_ids != 0 && n_ids != n) {
        LLAMA_LOG_ERROR("%s: %zu of %zu entries carry a token id; a batch carries ids for all entries or none\n", __func__, n_ids, n);
        return false;
    }
    if (n_ids == 0 && n_embd_set == 0) {
        LLAMA_LOG_ERROR("%s: entries carry neither token ids nor embeddings\n", __func__);
        return false;
    }
    const bool with_ids = n_ids == n;
    // a token batch carries one position per entry (the allocator broadcasts it across rope sections);
    // an embedding-only batch carries n_pos_per_embd positions laid out [section * n_tokens + i]
    const size_t n_pos = with_ids ? 1 : n_pos_per_embd;

    m_token.assign(n, LLAMA_TOKEN_NULL);
    m_pos.assign(n * n_pos, 0);
    m_n_seq_id.assign(n, 0);
    m_seq_id.assign(n + 1, nullptr);
    m_seq_storage.assign(n_seq_total, 0);
    m_logits.assign(n, 0);
    if (n_embd_set) {
        m_embd.assign(n * n_embd_inp, 0.0f);
    } else {
        m_embd.clear();
    }

    size_t seq_off = 0;
    for (size_t i = 0; i < n; ++i) {
        const token & t = tokens[i];
        m_token[i] = t.id;
        for (size_t j = 0; j < n_pos; ++j) {
            m_pos[j * n + i] = t.pos[j];
        }
        m_n_seq_id[i] = (int32_t) t.seq_ids.size();
        m_seq_id[i] = m_seq_storage.data() + seq_off;
        for (size_t s = 0; s < t.seq_ids.size(); ++s) {
            m_seq_storage[seq_off + s] = t.seq_ids[s];
        }
        seq_off += t.seq_ids.size();
        m_logits[i] = t.output ? 1 : 0;
        if (n_embd_set) {
            std::memcpy(m_embd.data() + i * n_embd_inp, embd.data() + t.embd_off, n_embd_inp * sizeof(float));
        }
    }

    out.n_tokens = (int32_t) n;
    out.token    = with_ids ? m_token.data() : nullptr;
    out.embd     = n_embd_set ? m_embd.data() : nullptr;
    out.pos      = m_pos.data();
    out.n_seq_id = m_n_seq_id.data();
    out.seq_id   = m_seq_id.data();
    out.logits   = m_logits.data();
    return true;
}

//
// extended batch C API
//

struct llama_batch_ext * llama_batch_ext_init(struct llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);
    const auto & hparams = model->hparams;
    // an MTP draft context is fed the target's hidden state rows, not token embeddings
    const size_t n_embd_inp = ctx->get_cparams().ctx_type == LLAMA_CONTEXT_TYPE_MTP ? hparams.n_embd_out() : hparams.n_embd_inp();
    return new llama_batch_ext(
            llama_n_batch(ctx),
            n_embd_inp,
            llama_n_seq_max(ctx),
            llama_vocab_n_tokens(llama_model_get_vocab(model)),
            hparams.n_pos_per_embd());
}

void llama_batch_ext_free(struct llama_batch_ext * batch) {
    delete batch;
}

void llama_batch_ext_clear(struct llama_batch_ext * batch) {
    batch->clear();
}

int32_t llama_batch_ext_add(struct llama_batch_ext * batch, llama_seq_id seq_id) {
    return batch->add_token(seq_id);
}

int32_t llama_batch_ext_add_token(struct llama_batch_ext * batch, llama_seq_id seq_id, llama_token id) {
    const int32_t idx = batch->add_token(seq_id);
    if (idx < 0) {
        return idx;
    }
    if (!batch->set_token_id(idx, id)) {
        batch->tokens.pop_back();
        return -2;
    }
    return idx;
}

bool llama_batch_ext_add_seq(struct llama_batch_ext * batch, int32_t idx, llama_seq_id seq_id) {
    return batch->add_seq(idx, seq_id);
}

bool llama_batch_ext_set_embd_token(struct llama_batch_ext * batch, int32_t idx, struct llama_embd embd) {
    return batch->set_token_embd(idx, embd);
}

bool llama_batch_ext_set_output_embd(struct llama_batch_ext * batch, int32_t idx, bool value) {
    return batch->set_output(idx, value);
}

bool llama_batch_ext_set_output_logits(struct llama_batch_ext * batch, int32_t idx, bool value) {
    return batch->set_output(idx, value);
}

bool llama_batch_ext_set_pos(struct llama_batch_ext * batch, int32_t idx, const llama_pos * pos) {
    return batch->set_token_pos(idx, pos);
}

int32_t llama_process(struct llama_context * ctx, enum llama_process_type type, struct llama_batch_ext * batch) {
    llama_batch b;
    if (!batch->to_batch(b)) {
        return -1;
    }
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return llama_encode(ctx, b);
        case LLAMA_PROCESS_TYPE_DECODE: return llama_decode(ctx, b);
    }
    return -1;
}
"""
p.write_text(t); print("edited src/llama-batch.cpp")
print("all edits applied")
