#pragma once

#include "ggml.h"
#include <stdint.h>

enum ggml_merlin_phase {
    GGML_MERLIN_PHASE_UNKNOWN = -1,
    GGML_MERLIN_PHASE_SINGLE_TOKEN = 0,
    GGML_MERLIN_PHASE_MULTI_TOKEN = 1,
    GGML_MERLIN_PHASE_MIXED = 2,
    GGML_MERLIN_PHASE_ENCODER = 3,
};

enum ggml_merlin_workload_source {
    GGML_MERLIN_SOURCE_UNKNOWN = 0,
    GGML_MERLIN_SOURCE_UBATCH = 1,
    GGML_MERLIN_SOURCE_CALIBRATION = 2,
};

struct ggml_merlin_workload {
    int64_t sequence_batch;
    int64_t tokens_in_flight;
    int32_t phase;
    int32_t source;
};

#ifdef __cplusplus
extern "C" {
#endif
GGML_API struct ggml_merlin_workload ggml_merlin_workload_get(void);
GGML_API struct ggml_merlin_workload ggml_merlin_workload_set(struct ggml_merlin_workload workload);
#ifdef __cplusplus
}

inline bool merlin_workload_valid(const ggml_merlin_workload & value) {
    return value.sequence_batch > 0 && value.tokens_in_flight > 0 &&
        value.phase >= GGML_MERLIN_PHASE_SINGLE_TOKEN && value.phase <= GGML_MERLIN_PHASE_ENCODER &&
        (value.source == GGML_MERLIN_SOURCE_UBATCH || value.source == GGML_MERLIN_SOURCE_CALIBRATION);
}

inline const char * merlin_workload_phase_name(int phase) {
    switch (phase) {
        case GGML_MERLIN_PHASE_SINGLE_TOKEN: return "single_token_per_sequence";
        case GGML_MERLIN_PHASE_MULTI_TOKEN: return "multi_token_per_sequence";
        case GGML_MERLIN_PHASE_MIXED: return "mixed_token_multiplicity";
        case GGML_MERLIN_PHASE_ENCODER: return "encoder";
        default: return "unknown";
    }
}

class merlin_workload_scope {
    ggml_merlin_workload previous;
public:
    explicit merlin_workload_scope(ggml_merlin_workload value) : previous(ggml_merlin_workload_set(value)) {}
    ~merlin_workload_scope() { ggml_merlin_workload_set(previous); }
    merlin_workload_scope(const merlin_workload_scope &) = delete;
    merlin_workload_scope & operator=(const merlin_workload_scope &) = delete;
};
#endif
