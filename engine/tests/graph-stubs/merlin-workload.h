#pragma once
#include <cstdint>
struct ggml_merlin_workload { int64_t sequence_batch, tokens_in_flight; int32_t phase, source; };
inline ggml_merlin_workload workload{1, 1, 0, 1};
inline ggml_merlin_workload ggml_merlin_workload_get() { return workload; }
