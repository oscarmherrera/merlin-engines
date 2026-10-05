#pragma once

#include "common.cuh"
#include "unary.cuh"
#include "merlin-dispatch.cuh"
#include "merlin-prefill.cuh"

#include <cstdint>

static __device__ __forceinline__ void merlin_pq2_dp4a_pair(int q, int & x, int & y) {
    const int even = __byte_perm(0x020100FF, 0x020100FF, q);
    const int odd = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
    x = __byte_perm(even, odd, 0x5140);
    y = __byte_perm(even, odd, 0x7362);
}

// A warp owns one output row; each lane accumulates complete 32-value integer dot groups.
template<int batch, bool gated>
static __global__ void merlin_pq2_decode(
        const block_pq2_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ activations,
        float * __restrict__ output,
        int64_t m, int64_t k, int64_t activation_stride, ggml_cuda_mm_fusion_args_device fusion) {
    const int lane = threadIdx.x;
    const int64_t row = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= m) {
        return;
    }
    const int64_t blocks = k / QK_PQ2_0;
    const block_pq2_0 * w = weights + row * blocks;
    const block_pq2_0 * gate = static_cast<const block_pq2_0 *>(fusion.gate);
    if constexpr (gated) {
        gate += row * blocks;
    }
    float sums[batch] = {};
    float gates[batch] = {};
    // Match Prism's 32-value Q8 dot/scaling boundary inside each 128-weight PQ2 block.
    for (int64_t chunk = lane; chunk < k / QK8_1; chunk += 32) {
        const int64_t b = chunk / 4;
        const int packed_offset = (chunk % 4) * 8;
        const float scale = __half2float(w[b].d);
        float gate_scale = 0;
        if constexpr (gated) { gate_scale = __half2float(gate[b].d); }
#pragma unroll
        for (int n = 0; n < batch; ++n) {
            const block_q8_1 & activation = activations[int64_t(n) * activation_stride + chunk];
            int dot = 0, gate_dot = 0;
            const int32_t * a4 = reinterpret_cast<const int32_t *>(activation.qs);
            const uint16_t * w2 = reinterpret_cast<const uint16_t *>(w[b].qs + packed_offset);
            const uint16_t * gate2 = nullptr;
            if constexpr (gated) { gate2 = reinterpret_cast<const uint16_t *>(gate[b].qs + packed_offset); }
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int packed = static_cast<int16_t>(w2[pair]);
                const int left = a4[2 * pair];
                const int right = a4[2 * pair + 1];
                int x, y;
                merlin_pq2_dp4a_pair(packed, x, y);
                dot = __dp4a(left, x, dot);
                dot = __dp4a(right, y, dot);
                if constexpr (gated) {
                    const int packed_gate = static_cast<int16_t>(gate2[pair]);
                    int gx, gy;
                    merlin_pq2_dp4a_pair(packed_gate, gx, gy);
                    gate_dot = __dp4a(left, gx, gate_dot);
                    gate_dot = __dp4a(right, gy, gate_dot);
                }
            }
            const float activation_scale = __low2float(activation.ds);
            sums[n] += scale * activation_scale * dot;
            if constexpr (gated) { gates[n] += gate_scale * activation_scale * gate_dot; }
        }
    }
#pragma unroll
    for (int n = 0; n < batch; ++n) {
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            sums[n] += __shfl_down_sync(0xffffffff, sums[n], delta);
            if constexpr (gated) {
                gates[n] += __shfl_down_sync(0xffffffff, gates[n], delta);
            }
        }
        if (lane == 0) {
            float result = sums[n];
            if (fusion.x_bias) {
                result += static_cast<const float *>(fusion.x_bias)[row];
            }
            if constexpr (gated) {
                float g = gates[n];
                if (fusion.gate_bias) {
                    g += static_cast<const float *>(fusion.gate_bias)[row];
                }
                switch (fusion.glu_op) {
                    case GGML_GLU_OP_SWIGLU: result *= ggml_cuda_op_silu_single(g); break;
                    case GGML_GLU_OP_GEGLU: result *= ggml_cuda_op_gelu_single(g); break;
                    case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(g, result); break;
                    default: result *= g; break;
                }
            }
            output[int64_t(n) * m + row] = result;
        }
    }
}

// Output rows share each quantized activation load at small batch widths.
template<int batch, int rows>
static __global__ void merlin_pq2_decode_batched(
        const block_pq2_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ activations,
        float * __restrict__ output,
        int64_t m, int64_t k, int64_t activation_stride) {
    const int lane = threadIdx.x;
    const int64_t first_row = (int64_t(blockIdx.x) * blockDim.y + threadIdx.y) * rows;
    if (first_row >= m) { return; }
    const int64_t blocks = k / QK_PQ2_0;
    float sums[batch][rows] = {};

    for (int64_t chunk = lane; chunk < k / QK8_1; chunk += 32) {
        const int64_t b = chunk / 4;
        const int packed_offset = (chunk % 4) * 8;
        int unpacked[rows][8] = {};
        float weight_scale[rows] = {};
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            if (first_row + r >= m) { continue; }
            const block_pq2_0 & weight = weights[(first_row + r) * blocks + b];
            weight_scale[r] = __half2float(weight.d);
            const uint16_t * packed_words = reinterpret_cast<const uint16_t *>(weight.qs + packed_offset);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int q = static_cast<int16_t>(packed_words[pair]);
                const int even = __byte_perm(0x020100FF, 0x020100FF, q);
                const int odd = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                unpacked[r][2 * pair] = __byte_perm(even, odd, 0x5140);
                unpacked[r][2 * pair + 1] = __byte_perm(even, odd, 0x7362);
            }
        }

#pragma unroll
        for (int n = 0; n < batch; ++n) {
            const block_q8_1 & activation = activations[int64_t(n) * activation_stride + chunk];
            const int32_t * q8 = reinterpret_cast<const int32_t *>(activation.qs);
            int dots[rows] = {};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int left = q8[2 * pair];
                const int right = q8[2 * pair + 1];
#pragma unroll
                for (int r = 0; r < rows; ++r) {
                    if (first_row + r >= m) { continue; }
                    dots[r] = __dp4a(left, unpacked[r][2 * pair], dots[r]);
                    dots[r] = __dp4a(right, unpacked[r][2 * pair + 1], dots[r]);
                }
            }
            const float activation_scale = __low2float(activation.ds);
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                if (first_row + r >= m) { continue; }
                sums[n][r] += weight_scale[r] * activation_scale * dots[r];
            }
        }
    }

#pragma unroll
    for (int n = 0; n < batch; ++n) {
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            float sum = sums[n][r];
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, delta);
            }
            if (lane == 0 && first_row + r < m) {
                output[int64_t(n) * m + first_row + r] = sum;
            }
        }
    }
}

static bool merlin_decode_matrix(const ggml_tensor * t, int device) {
    if (!t) {
        return false;
    }
    const auto buffer = t->view_src ? t->view_src->buffer : t->buffer;
    return t->data && buffer && t->ne[0] > 0 && t->ne[1] > 0 &&
        t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t) &&
        ggml_backend_buffer_get_type(buffer) == ggml_backend_cuda_buffer_type(device);
}

static bool merlin_decode_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b->data);
    return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
}

static bool merlin_decode_bias(const ggml_tensor * bias, const ggml_tensor * dst, int device) {
    return !bias || (merlin_decode_matrix(bias, device) && bias->type == GGML_TYPE_F32 &&
        bias->ne[0] == dst->ne[0] && bias->ne[1] == 1 && !merlin_decode_overlaps(bias, dst));
}

template<int batch>
static void merlin_launch_decode(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * /* src1 */, ggml_tensor * dst,
        ggml_cuda_mm_fusion_args_device fusion, const block_q8_1 * activations, int64_t activation_stride) {
    const dim3 threads(32, 4);
    const auto * weights = static_cast<const block_pq2_0 *>(src0->data);
    auto * output = static_cast<float *>(dst->data);
    if constexpr (batch == 1) {
        if (fusion.gate || fusion.x_bias || fusion.gate_bias) {
            const dim3 blocks((src0->ne[1] + 3) / 4);
            if (fusion.gate) {
                merlin_pq2_decode<batch, true><<<blocks, threads, 0, ctx.stream()>>>(
                    weights, activations, output, src0->ne[1], src0->ne[0], activation_stride, fusion);
            } else {
                merlin_pq2_decode<batch, false><<<blocks, threads, 0, ctx.stream()>>>(
                    weights, activations, output, src0->ne[1], src0->ne[0], activation_stride, fusion);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    if constexpr (batch == 1 || batch == 4) {
        if (ggml_cuda_info().devices[ctx.device].cc == 860) {
            const dim3 blocks((src0->ne[1] + 7) / 8);
            merlin_pq2_decode_batched<batch, 2><<<blocks, threads, 0, ctx.stream()>>>(
                weights, activations, output, src0->ne[1], src0->ne[0], activation_stride);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const dim3 blocks((src0->ne[1] + 15) / 16);
    merlin_pq2_decode_batched<batch, 4><<<blocks, threads, 0, ctx.stream()>>>(
        weights, activations, output, src0->ne[1], src0->ne[0], activation_stride);
    CUDA_CHECK(cudaGetLastError());
}

static bool merlin_decode_supported(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if ((cc != GGML_CUDA_CC_TURING && cc != 860) || ggml_cuda_highest_compiled_arch(cc) < cc || ids ||
            src0->type != GGML_TYPE_PQ2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !merlin_decode_matrix(src0, ctx.device) || !merlin_decode_matrix(src1, ctx.device) ||
            !merlin_decode_matrix(dst, ctx.device)) {
        return false;
    }
    const int64_t n = src1->ne[1];
    // On SM86, batch 2 lost 18-39% to Prism across the three measured decode shapes.
    if (cc == 860 && n == 2) { return false; }
    if ((n != 1 && n != 2 && n != 4 && n != 8) || src0->ne[0] % QK_PQ2_0 ||
            src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != n ||
            src0->ne[1] > INT32_MAX || merlin_decode_overlaps(src0, dst) || merlin_decode_overlaps(src1, dst)) {
        return false;
    }
    if (fusion) {
        if (n != 1 || fusion->x_scale || fusion->gate_scale ||
                !merlin_decode_bias(fusion->x_bias, dst, ctx.device) ||
                !merlin_decode_bias(fusion->gate_bias, dst, ctx.device)) {
            return false;
        }
        if (fusion->gate && (!merlin_decode_matrix(fusion->gate, ctx.device) ||
                fusion->gate->type != src0->type || !ggml_are_same_shape(fusion->gate, src0) ||
                merlin_decode_overlaps(fusion->gate, dst))) {
            return false;
        }
        if (fusion->gate && fusion->glu_op != GGML_GLU_OP_SWIGLU &&
                fusion->glu_op != GGML_GLU_OP_GEGLU && fusion->glu_op != GGML_GLU_OP_SWIGLU_OAI) {
            return false;
        }
    }
    return true;
}

template<typename Quantize, typename Reference>
static bool merlin_cuda_decode(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion,
        ggml_cuda_mm_fusion_args_device device_fusion, const block_q8_1 * activations,
        int64_t activation_stride, Quantize prepare_q8, Reference baseline) {
    if (merlin_dispatch_reference_active()) { return false; }
    const auto workload = ggml_merlin_workload_get();
    if (!merlin_workload_valid(workload)) {
        merlin_dispatch_bypass(ctx, src0, src1, dst, "unknown_workload", fusion != nullptr);
        return false;
    }
    const bool native = merlin_decode_supported(ctx, src0, src1, ids, dst, fusion);
    const bool matrix = !ids && !fusion && merlin_prefill_supported(ctx, src0, src1, dst);
    if (!native && !matrix) {
        if (src0->type == GGML_TYPE_PQ2_0) {
            merlin_dispatch_bypass(ctx, src0, src1, dst, "unsupported_decode_shape_or_fusion", fusion != nullptr);
        }
        return false;
    }
    struct arguments {
        ggml_backend_cuda_context & ctx;
        const ggml_tensor * weights, * input, * ids;
        ggml_tensor * output;
        const ggml_cuda_mm_fusion_args_host * fusion;
        ggml_cuda_mm_fusion_args_device device_fusion;
        const block_q8_1 * activations;
        int64_t activation_stride;
        Quantize & prepare_q8;
        Reference & baseline;
    } args{ctx, src0, src1, ids, dst, fusion, device_fusion, activations, activation_stride, prepare_q8, baseline};
    const auto reference = [](void * opaque) {
        auto & a = *static_cast<arguments *>(opaque);
        a.baseline();
    };
    merlin_dispatch_candidate candidates[5]{};
    size_t count = 0;
    if (native) {
        candidates[count++] = {"pq2_q8_1_dp4a_warp", size_t(src1->ne[1] * activation_stride) * sizeof(block_q8_1), [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            a.prepare_q8();
            switch (a.input->ne[1]) {
                case 1: merlin_launch_decode<1>(a.ctx, a.weights, a.input, a.output, a.device_fusion, a.activations, a.activation_stride); break;
                case 2: merlin_launch_decode<2>(a.ctx, a.weights, a.input, a.output, a.device_fusion, a.activations, a.activation_stride); break;
                case 4: merlin_launch_decode<4>(a.ctx, a.weights, a.input, a.output, a.device_fusion, a.activations, a.activation_stride); break;
                case 8: merlin_launch_decode<8>(a.ctx, a.weights, a.input, a.output, a.device_fusion, a.activations, a.activation_stride); break;
            }
        }, &args};
    }
    if (matrix) {
        const size_t workspace = merlin_prefill_workspace_bytes(src1);
        candidates[count++] = {merlin_dispatch::cutlass_narrow, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(false)};
        candidates[count++] = {merlin_dispatch::cutlass_wide, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_wide(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(true)};
        candidates[count++] = {merlin_dispatch::cutlass_narrow_single, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_single(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(false, true)};
        candidates[count++] = {merlin_dispatch::cutlass_wide_single, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_wide_single(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(true, true)};
    }
    int64_t fusion_key = 0;
    if (fusion) {
        fusion_key = (fusion->gate ? 1 : 0) | (fusion->x_bias ? 2 : 0) | (fusion->gate_bias ? 4 : 0);
        if (fusion->gate) { fusion_key |= (int64_t(fusion->glu_op) + 1) << 8; }
    }
    const merlin_dispatch::key shape{ggml_cuda_info().devices[ctx.device].cc, GGML_OP_MUL_MAT,
        src0->ne[1], src1->ne[1], src0->ne[0], workload.sequence_batch, workload.tokens_in_flight,
        src0->type, workload.phase, fusion_key};
    merlin_dispatch_run(ctx, shape, static_cast<float *>(dst->data), ggml_nelements(dst), reference, &args,
        candidates, count, merlin_dispatch_logical_bytes(src0, src1, dst, fusion));
    return true;
}
