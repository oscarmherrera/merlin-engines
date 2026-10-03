#pragma once

#include "common.cuh"

// Stable addresses until backend destruction; one slab per actually used stream.
constexpr size_t merlin_prefill_slab_bytes = size_t(128) * 1024 * 1024;
void merlin_prefill_initialize(ggml_backend_cuda_context & ctx, bool enabled);
void merlin_prefill_prepare_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * graph);
void merlin_prefill_release(ggml_backend_cuda_context & ctx);
char * merlin_prefill_scratch(ggml_backend_cuda_context & ctx);
size_t merlin_prefill_reserved_bytes(ggml_backend_cuda_context & ctx);
