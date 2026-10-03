#pragma once

#include "common.cuh"
#include "ggml-cuda.h"
#include "merlin-kernel-log.cuh"
#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>

namespace merlin_prefill {
constexpr int tile_m = 32;
constexpr int tile_n = 32;
constexpr int tile_k = QK_PQ2_0;
constexpr size_t shared_bytes = (tile_m + tile_n)*tile_k*sizeof(half) + tile_m*tile_n*sizeof(float);

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

// Only CTA-local FP16 tiles exist; packed model weights are never expanded in VRAM.
static __global__ void pq2_wmma(const block_pq2_0 * weights, const float * input, float * output,
                               int64_t m, int64_t n, int64_t k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    namespace wmma = nvcuda::wmma;
    __shared__ __align__(32) half a[tile_m*tile_k];
    __shared__ __align__(32) half b[tile_n*tile_k];
    __shared__ __align__(32) float c[tile_m*tile_n];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = (warp / 2)*16;
    const int warp_n = (warp % 2)*16;
    const int64_t row0 = int64_t(blockIdx.x)*tile_m;
    const int64_t col0 = int64_t(blockIdx.y)*tile_n;
    const int64_t blocks_per_row = k / tile_k;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int64_t kb = 0; kb < blocks_per_row; ++kb) {
        // One lane reads one packed byte and emits four consecutive weights.
        for (int index = tid; index < tile_m*(tile_k/4); index += 128) {
            const int row = index / (tile_k/4);
            const int byte = index % (tile_k/4);
            unsigned packed = 0x55; // encoded zeros for the M tail
            float scale = 0.0f;
            if (row0 + row < m) {
                const block_pq2_0 & block = weights[(row0 + row)*blocks_per_row + kb];
                packed = block.qs[byte];
                scale = __half2float(block.d);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int q = int((packed >> (2*j)) & 3) - 1;
                a[row*tile_k + byte*4 + j] = __float2half_rn(float(q)*scale);
            }
        }
        for (int index = tid; index < tile_n*tile_k; index += 128) {
            const int col = index / tile_k;
            const int inner = index % tile_k;
            b[index] = col0 + col < n ? __float2half_rn(input[(col0 + col)*k + kb*tile_k + inner])
                                      : __float2half_rn(0.0f);
        }
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < tile_k; inner += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a + warp_m*tile_k + inner, tile_k);
            wmma::load_matrix_sync(bf, b + warp_n*tile_k + inner, tile_k);
            wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c + warp_m*tile_n + warp_n, acc, tile_n, wmma::mem_row_major);
    __syncthreads();
    for (int index = tid; index < tile_m*tile_n; index += 128) {
        const int row = index / tile_n;
        const int col = index % tile_n;
        if (row0 + row < m && col0 + col < n) {
            output[(col0 + col)*m + row0 + row] = c[index];
        }
    }
#else
    GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_UNUSED(m); GGML_UNUSED(n); GGML_UNUSED(k);
#endif
}
} // namespace merlin_prefill
#endif

static bool try_merlin_prefill(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
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
    if (m <= 0 || n < 16 || k <= 0 || k % QK_PQ2_0 != 0 || input->ne[0] != k ||
        output->ne[0] != m || output->ne[1] != n ||
        m > int64_t(2147483647)*merlin_prefill::tile_m || n > int64_t(65535)*merlin_prefill::tile_n) {
        return false;
    }
    const dim3 grid((m + merlin_prefill::tile_m - 1)/merlin_prefill::tile_m,
                    (n + merlin_prefill::tile_n - 1)/merlin_prefill::tile_n);
    merlin_prefill::pq2_wmma<<<grid, 128, 0, ctx.stream()>>>(
        static_cast<const block_pq2_0 *>(weights->data), static_cast<const float *>(input->data),
        static_cast<float *>(output->data), m, n, k);
    CUDA_CHECK(cudaGetLastError());
    merlin_kernel_log("prefill", "pq2_wmma_32x32x128", ctx.device, m, n, k,
                      merlin_prefill::shared_bytes, false);
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    return false;
#endif
}
