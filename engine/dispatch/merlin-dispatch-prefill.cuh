#pragma once

#include "merlin-dispatch.cuh"
#include "merlin-prefill.cuh"

static bool merlin_dispatch_prefill(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
        const ggml_tensor * input, ggml_tensor * output,
        void (*reference)(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *)) {
    // Small batches reach central MMVQ, where the native and matrix candidates compete together.
    if (merlin_dispatch_reference_active() || input->ne[1] <= 8 ||
            !merlin_prefill_supported(ctx, weights, input, output)) {
        return false;
    }
    struct arguments {
        ggml_backend_cuda_context & ctx;
        const ggml_tensor * weights, * input;
        ggml_tensor * output;
        decltype(reference) baseline;
    } args{ctx, weights, input, output, reference};
    const auto baseline = [](void * opaque) {
        auto & a = *static_cast<arguments *>(opaque);
        merlin_dispatch_reference_scope scope;
        a.baseline(a.ctx, a.weights, a.input, a.output);
    };
    const size_t workspace = merlin_prefill_workspace_bytes(input);
    const merlin_dispatch_candidate candidates[] = {
        {"pq2_cutlass_32x32x64", workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(false)},
        {"pq2_cutlass_64x64x64", workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_wide(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(true)},
    };
    const merlin_dispatch::key shape{ggml_cuda_info().devices[ctx.device].cc, GGML_OP_MUL_MAT,
        weights->ne[1], input->ne[1], weights->ne[0], input->ne[1], input->ne[1], weights->type, 1, 0};
    merlin_dispatch_run(ctx, shape, static_cast<float *>(output->data), ggml_nelements(output), baseline, &args,
        candidates, sizeof(candidates) / sizeof(candidates[0]));
    return true;
}
