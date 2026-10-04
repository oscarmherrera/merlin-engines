#include "merlin-prefill.cuh"

#include "common.cuh"
#include "ggml-cuda.h"
#include "mmq.cuh"
#include "quantize.cuh"
#include <cstdint>
#include "merlin-prefill-workspace-impl.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "merlin-prefill-tile.cuh"

namespace merlin_prefill {

static bool device_span(const ggml_tensor * tensor, ggml_backend_buffer_type_t device_type) {
    if (!tensor->data || !tensor->buffer || ggml_backend_buffer_get_type(tensor->buffer) != device_type) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    const size_t alignment = tensor->type == GGML_TYPE_F32 ? alignof(float) : alignof(block_pq2_0);
    return base && data >= base && data - base <= size && data % alignment == 0 &&
           ggml_nbytes(tensor) <= size - (data - base);
}

static bool spans_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b->data);
    return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
}

} // namespace merlin_prefill
#endif

size_t merlin_prefill_workspace_bytes(const ggml_tensor * input) {
    const int64_t k = input->ne[0], n = input->ne[1];
    if (k <= 0 || n <= 0 || k > INT32_MAX - MATRIX_ROW_PADDING) {
        return 0;
    }
    const size_t row = size_t(GGML_PAD(k, MATRIX_ROW_PADDING)/QK8_1_MMQ)*sizeof(block_q8_1_mmq);
    if (size_t(n) > (128u*1024u*1024u)/row) {
        return 0;
    }
    return size_t(n)*row;
}

size_t merlin_prefill_shared_bytes(bool wide, bool single) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (single) { return wide ? merlin_prefill::shared_bytes_wide_single : merlin_prefill::shared_bytes_single; }
    return wide ? merlin_prefill::shared_bytes_wide : merlin_prefill::shared_bytes;
#else
    GGML_UNUSED(wide); GGML_UNUSED(single);
    return 0;
#endif
}

size_t merlin_prefill_shared_bytes_rect() {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return merlin_prefill::shared_bytes_rect_single;
#else
    return 0;
#endif
}

bool merlin_prefill_eligible(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                              const ggml_tensor * input, ggml_tensor * output) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_TURING || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_TURING ||
        weights->type != GGML_TYPE_PQ2_0 || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(weights) || !ggml_is_contiguous(input) || !ggml_is_contiguous(output)) {
        return false;
    }
    for (int dim = 2; dim < GGML_MAX_DIMS; ++dim) {
        if (weights->ne[dim] != 1 || input->ne[dim] != 1 || output->ne[dim] != 1) {
            return false;
        }
    }
    const auto device_type = ggml_backend_cuda_buffer_type(ctx.device);
    if (!merlin_prefill::device_span(weights, device_type) ||
        !merlin_prefill::device_span(input, device_type) ||
        !merlin_prefill::device_span(output, device_type) ||
        merlin_prefill::spans_overlap(output, weights) ||
        merlin_prefill::spans_overlap(output, input)) {
        return false;
    }
    const int64_t m = weights->ne[1], n = input->ne[1], k = weights->ne[0];
    if (m <= 0 || n < 1 || k <= 0 || k % QK_PQ2_0 != 0 || input->ne[0] != k ||
        output->ne[0] != m || output->ne[1] != n ||
        m > int64_t(2147483647)*32 || n > int64_t(65535)*32) {
        return false;
    }
    if (reinterpret_cast<uintptr_t>(input->data) % 16 != 0 || k > INT32_MAX - MATRIX_ROW_PADDING ||
        merlin_prefill_workspace_bytes(input) == 0) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    return false;
#endif
}

bool merlin_prefill_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                              const ggml_tensor * input, ggml_tensor * output) {
    return merlin_prefill_eligible(ctx, weights, input, output) && merlin_prefill_scratch(ctx);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template<int Rows, int Stages, int Cols = Rows>
static void merlin_prefill_launch_tile(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                       const ggml_tensor * input, ggml_tensor * output) {
    const int64_t m = weights->ne[1], n = input->ne[1], k = weights->ne[0];
    char * q8 = merlin_prefill_scratch(ctx);
    GGML_ASSERT(q8 && merlin_prefill_workspace_bytes(input) <= merlin_prefill_slab_bytes);
    quantize_mmq_q8_1_cuda(static_cast<const float *>(input->data), nullptr, q8, GGML_TYPE_PQ2_0,
                          k, input->nb[1]/sizeof(float), input->nb[2]/sizeof(float), input->nb[3]/sizeof(float),
                          GGML_PAD(k, MATRIX_ROW_PADDING), n, 1, 1, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    const dim3 grid((m + Rows - 1)/Rows, (n + Cols - 1)/Cols);
    if constexpr (Rows == Cols) {
        merlin_prefill::pq2_cutlass<Rows, Stages><<<grid, 128, 0, ctx.stream()>>>(
            static_cast<const block_pq2_0 *>(weights->data), reinterpret_cast<const block_q8_1_mmq *>(q8),
            static_cast<float *>(output->data), m, n, k);
    } else {
        merlin_prefill::pq2_cutlass_rect<Rows, Cols, Stages><<<grid, 256, 0, ctx.stream()>>>(
            static_cast<const block_pq2_0 *>(weights->data), reinterpret_cast<const block_q8_1_mmq *>(q8),
            static_cast<float *>(output->data), m, n, k);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif

void merlin_prefill_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                  const ggml_tensor * input, ggml_tensor * output) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    merlin_prefill_launch_tile<32, 2>(ctx, weights, input, output);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_ABORT("CUTLASS prefill requires NVIDIA SM75");
#endif
}

void merlin_prefill_launch_wide(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                       const ggml_tensor * input, ggml_tensor * output) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    merlin_prefill_launch_tile<64, 2>(ctx, weights, input, output);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_ABORT("CUTLASS prefill requires NVIDIA SM75");
#endif
}

void merlin_prefill_launch_single(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                  const ggml_tensor * input, ggml_tensor * output) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    merlin_prefill_launch_tile<32, 1>(ctx, weights, input, output);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_ABORT("CUTLASS prefill requires NVIDIA SM75");
#endif
}

void merlin_prefill_launch_wide_single(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                  const ggml_tensor * input, ggml_tensor * output) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    merlin_prefill_launch_tile<64, 1>(ctx, weights, input, output);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_ABORT("CUTLASS prefill requires NVIDIA SM75");
#endif
}

void merlin_prefill_launch_rect_single(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                                      const ggml_tensor * input, ggml_tensor * output) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    merlin_prefill_launch_tile<128, 1, 64>(ctx, weights, input, output);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_ABORT("CUTLASS prefill requires NVIDIA SM75");
#endif
}
