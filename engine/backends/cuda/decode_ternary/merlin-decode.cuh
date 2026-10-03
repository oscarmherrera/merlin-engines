#pragma once

#include "common.cuh"
#include "unary.cuh"
#include "merlin-dispatch.cuh"
#include "merlin-prefill.cuh"

#include <cstdint>

static __device__ __forceinline__ float merlin_pq2_add(float sum, float value, unsigned code) {
    // PQ2 codes are -1, 0, +1, +2. Explicit adds keep symbol accumulation multiply-free.
    const float magnitude = code == 3 ? __fadd_rn(value, value) : value;
    const float term = code == 0 ? -magnitude : code == 1 ? 0.0f : magnitude;
    return __fadd_rn(sum, term);
}

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
                dot = merlin_pq2_add(dot, a, (packed >> (2 * i)) & 3);
                if constexpr (gated) {
                    gate_dot = merlin_pq2_add(gate_dot, a, (packed_gate >> (2 * i)) & 3);
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
}

static bool merlin_decode_supported(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
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
    return true;
}

static bool merlin_cuda_decode(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion,
        ggml_cuda_mm_fusion_args_device device_fusion) {
    if (merlin_dispatch_reference_active()) { return false; }
    const auto workload = ggml_merlin_workload_get();
    if (!merlin_workload_valid(workload)) { return false; }
    const bool native = merlin_decode_supported(ctx, src0, src1, ids, dst, fusion);
    const bool matrix = !ids && !fusion && merlin_prefill_supported(ctx, src0, src1, dst);
    if (!native && !matrix) { return false; }
    struct arguments {
        ggml_backend_cuda_context & ctx;
        const ggml_tensor * weights, * input, * ids;
        ggml_tensor * output;
        const ggml_cuda_mm_fusion_args_host * fusion;
        ggml_cuda_mm_fusion_args_device device_fusion;
    } args{ctx, src0, src1, ids, dst, fusion, device_fusion};
    const auto reference = [](void * opaque) {
        auto & a = *static_cast<arguments *>(opaque);
        merlin_dispatch_reference_scope scope;
        ggml_cuda_mul_mat_vec_q(a.ctx, a.weights, a.input, a.ids, a.output, a.fusion);
    };
    merlin_dispatch_candidate candidates[5]{};
    size_t count = 0;
    if (native) {
        candidates[count++] = {"pq2_f32_warp", 0, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            switch (a.input->ne[1]) {
                case 1: merlin_launch_decode<1>(a.ctx, a.weights, a.input, a.output, a.device_fusion); break;
                case 2: merlin_launch_decode<2>(a.ctx, a.weights, a.input, a.output, a.device_fusion); break;
                case 4: merlin_launch_decode<4>(a.ctx, a.weights, a.input, a.output, a.device_fusion); break;
                case 8: merlin_launch_decode<8>(a.ctx, a.weights, a.input, a.output, a.device_fusion); break;
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
