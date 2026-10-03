#pragma once

#include "llama-batch.h"
#include "merlin-workload.h"

#include <array>

inline ggml_merlin_workload merlin_workload_from_ubatch(const llama_ubatch & batch, bool encoder) {
    const ggml_merlin_workload unknown{0, 0, GGML_MERLIN_PHASE_UNKNOWN, GGML_MERLIN_SOURCE_UNKNOWN};
    if (!batch.n_tokens || !batch.n_seqs_unq || batch.n_seqs_unq > LLAMA_MAX_SEQ ||
            !batch.n_seq_id || !batch.seq_id) {
        return unknown;
    }
    std::array<uint32_t, LLAMA_MAX_SEQ> counts{};
    std::array<uint32_t, LLAMA_MAX_SEQ> last_token{};
    uint32_t distinct = 0;
    for (uint32_t i = 0; i < batch.n_tokens; ++i) {
        if (!batch.seq_id[i] || batch.n_seq_id[i] <= 0 || batch.n_seq_id[i] > LLAMA_MAX_SEQ) {
            return unknown;
        }
        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            const llama_seq_id sequence = batch.seq_id[i][j];
            if (sequence < 0 || sequence >= LLAMA_MAX_SEQ || last_token[sequence] == i + 1) {
                return unknown;
            }
            last_token[sequence] = i + 1;
            if (counts[sequence]++ == 0) { ++distinct; }
        }
    }
    if (distinct != batch.n_seqs_unq) { return unknown; }
    bool single = false, multiple = false;
    for (uint32_t count : counts) {
        single |= count == 1;
        multiple |= count > 1;
    }
    const int32_t phase = encoder ? GGML_MERLIN_PHASE_ENCODER :
        single && multiple ? GGML_MERLIN_PHASE_MIXED :
        multiple ? GGML_MERLIN_PHASE_MULTI_TOKEN : GGML_MERLIN_PHASE_SINGLE_TOKEN;
    return {batch.n_seqs_unq, batch.n_tokens, phase, GGML_MERLIN_SOURCE_UBATCH};
}
