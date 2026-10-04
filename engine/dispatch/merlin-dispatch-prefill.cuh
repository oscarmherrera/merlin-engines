#pragma once

#include "merlin-dispatch.cuh"
#include "merlin-prefill.cuh"

// Startup reserves for calibration, forced diagnosis, or a measured CUTLASS winner.
static bool merlin_prefill_profile_eligible(int device) {
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_TURING) { return false; }
    if (merlin_dispatch_calibrating() || merlin_dispatch_force_custom()) { return true; }
    auto & storage = merlin_dispatch::state(device);
    std::lock_guard<std::mutex> lock(storage.mutex);
    for (const auto & row : storage.costs.entries) {
        if (storage.costs.choose(row.first, merlin_dispatch::cutlass_candidates, 5) >= 0) { return true; }
    }
    return false;
}

static bool merlin_dispatch_prefill(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
        const ggml_tensor * input, ggml_tensor * output,
        void (*reference)(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *)) {
    // Small batches reach central MMVQ, where the native and matrix candidates compete together.
    if (merlin_dispatch_reference_active() || input->ne[1] <= 8) { return false; }
    if (!merlin_prefill_eligible(ctx, weights, input, output)) {
        if (weights->type == GGML_TYPE_PQ2_0) {
            merlin_dispatch_bypass(ctx, weights, input, output, "unsupported_prefill_shape_or_layout");
        }
        return false;
    }
    if (!merlin_prefill_scratch(ctx)) {
        merlin_dispatch_bypass(ctx, weights, input, output, "no_reserved_workspace");
        return false;
    }
    const auto workload = ggml_merlin_workload_get();
    if (!merlin_workload_valid(workload)) {
        merlin_dispatch_bypass(ctx, weights, input, output, "unknown_workload");
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
        {merlin_dispatch::cutlass_narrow, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(false)},
        {merlin_dispatch::cutlass_wide, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_wide(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(true)},
        {merlin_dispatch::cutlass_narrow_single, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_single(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(false, true)},
        {merlin_dispatch::cutlass_wide_single, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_wide_single(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes(true, true)},
        {merlin_dispatch::cutlass_rect_single, workspace, [](void * opaque) {
            auto & a = *static_cast<arguments *>(opaque);
            merlin_prefill_launch_rect_single(a.ctx, a.weights, a.input, a.output);
        }, &args, merlin_prefill_shared_bytes_rect()},
    };
    const merlin_dispatch::key shape{ggml_cuda_info().devices[ctx.device].cc, GGML_OP_MUL_MAT,
        weights->ne[1], input->ne[1], weights->ne[0], workload.sequence_batch, workload.tokens_in_flight,
        weights->type, workload.phase, 0};
    merlin_dispatch_run(ctx, shape, static_cast<float *>(output->data), ggml_nelements(output), baseline, &args,
        candidates, sizeof(candidates) / sizeof(candidates[0]),
        merlin_dispatch_logical_bytes(weights, input, output));
    return true;
}
