#pragma once

#include "common.cuh"
#include "merlin-prefill-workspace.cuh"

size_t merlin_prefill_workspace_bytes(const ggml_tensor * input);
size_t merlin_prefill_shared_bytes(bool wide, bool single = false);
size_t merlin_prefill_shared_bytes_rect();
bool merlin_prefill_eligible(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                            const ggml_tensor * input, ggml_tensor * output);
bool merlin_prefill_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                             const ggml_tensor * input, ggml_tensor * output);
void merlin_prefill_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                          const ggml_tensor * input, ggml_tensor * output);
void merlin_prefill_launch_wide(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                               const ggml_tensor * input, ggml_tensor * output);

void merlin_prefill_launch_single(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                 const ggml_tensor * input, ggml_tensor * output);
void merlin_prefill_launch_wide_single(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                      const ggml_tensor * input, ggml_tensor * output);
void merlin_prefill_launch_rect_single(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                      const ggml_tensor * input, ggml_tensor * output);
