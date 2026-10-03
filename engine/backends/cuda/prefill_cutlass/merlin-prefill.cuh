#pragma once

#include "common.cuh"

size_t merlin_prefill_workspace_bytes(const ggml_tensor * input);
size_t merlin_prefill_shared_bytes(bool wide);
bool merlin_prefill_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                             const ggml_tensor * input, ggml_tensor * output);
void merlin_prefill_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                          const ggml_tensor * input, ggml_tensor * output);
void merlin_prefill_launch_wide(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                               const ggml_tensor * input, ggml_tensor * output);
