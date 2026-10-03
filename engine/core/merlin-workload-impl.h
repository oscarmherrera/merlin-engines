#pragma once

#include "merlin-workload.h"

// Included in ggml-backend.cpp exactly once: llama and CUDA call this shared-library storage.
static thread_local ggml_merlin_workload merlin_current_workload{0, 0, GGML_MERLIN_PHASE_UNKNOWN, 0};

ggml_merlin_workload ggml_merlin_workload_get(void) {
    return merlin_current_workload;
}

ggml_merlin_workload ggml_merlin_workload_set(ggml_merlin_workload value) {
    const ggml_merlin_workload previous = merlin_current_workload;
    merlin_current_workload = merlin_workload_valid(value) ? value :
        ggml_merlin_workload{0, 0, GGML_MERLIN_PHASE_UNKNOWN, GGML_MERLIN_SOURCE_UNKNOWN};
    return previous;
}
