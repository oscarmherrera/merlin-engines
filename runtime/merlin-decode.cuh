#pragma once

#include "common.cuh"
#include "unary.cuh"
#include "merlin-kernel-log.cuh"

#include <cstdint>

// A warp owns one output row. Each packed byte is reused across the entire batch.
template<int batch, bool gated>
static __global__ void merlin_pq2_decode(
        const block_pq2_0 * __restrict__ weights,
        const float * __restrict__ activations,
        float * __restrict__ output,
        int64_t m, int64_t k, ggml_cuda_mm_fusion_args_device fusion) {
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
    for (int64_t b = 0; b < blocks; ++b) {
        const unsigned packed = w[b].qs[lane];
        const float scale = __half2float(w[b].d);
        unsigned packed_gate = 0;
        float gate_scale = 0;
        if constexpr (gated) {
            packed_gate = gate[b].qs[lane];
            gate_scale = __half2float(gate[b].d);
        }
        const int64_t offset = b * QK_PQ2_0 + lane * 4;
#pragma unroll
        for (int n = 0; n < batch; ++n) {
            float dot = 0;
            float gate_dot = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float a = activations[int64_t(n) * k + offset + i];
                // PQ2's fourth code is +2, even for a nominally ternary model.
                const float q = float(int((packed >> (2 * i)) & 3) - 1);
                dot = fmaf(q, a, dot);
                if constexpr (gated) {
                    const float g = float(int((packed_gate >> (2 * i)) & 3) - 1);
                    gate_dot = fmaf(g, a, gate_dot);
                }
            }
            sums[n] = fmaf(scale, dot, sums[n]);
            if constexpr (gated) {
                gates[n] = fmaf(gate_scale, gate_dot, gates[n]);
            }
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
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        ggml_cuda_mm_fusion_args_device fusion) {
    const dim3 threads(32, 4);
    const dim3 blocks((src0->ne[1] + 3) / 4);
    const auto * weights = static_cast<const block_pq2_0 *>(src0->data);
    const auto * activations = static_cast<const float *>(src1->data);
    auto * output = static_cast<float *>(dst->data);
    if (fusion.gate) {
        merlin_pq2_decode<batch, true><<<blocks, threads, 0, ctx.stream()>>>(
            weights, activations, output, src0->ne[1], src0->ne[0], fusion);
    } else {
        merlin_pq2_decode<batch, false><<<blocks, threads, 0, ctx.stream()>>>(
            weights, activations, output, src0->ne[1], src0->ne[0], fusion);
    }
    CUDA_CHECK(cudaGetLastError());
    merlin_kernel_log("decode", "pq2_f32_warp", ctx.device, src0->ne[1], batch, src0->ne[0], 0,
        fusion.gate || fusion.x_bias || fusion.gate_bias);
}

static bool merlin_cuda_decode(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion,
        ggml_cuda_mm_fusion_args_device device_fusion) {
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_TURING || ids ||
            src0->type != GGML_TYPE_PQ2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !merlin_decode_matrix(src0, ctx.device) || !merlin_decode_matrix(src1, ctx.device) ||
            !merlin_decode_matrix(dst, ctx.device)) {
        return false;
    }
    const int64_t n = src1->ne[1];
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
    switch (n) {
        case 1: merlin_launch_decode<1>(ctx, src0, src1, dst, device_fusion); break;
        case 2: merlin_launch_decode<2>(ctx, src0, src1, dst, device_fusion); break;
        case 4: merlin_launch_decode<4>(ctx, src0, src1, dst, device_fusion); break;
        case 8: merlin_launch_decode<8>(ctx, src0, src1, dst, device_fusion); break;
    }
    return true;
}
