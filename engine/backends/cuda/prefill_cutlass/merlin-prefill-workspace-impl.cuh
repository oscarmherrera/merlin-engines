#include "merlin-engine-log.h"
#pragma once

#include "merlin-prefill-workspace.cuh"
#include <array>
#include <map>
#include <mutex>

namespace merlin_prefill {
struct scratch_owner {
    std::array<char *, GGML_CUDA_MAX_STREAMS> streams{};
};
static std::mutex scratch_mutex;
static std::map<ggml_backend_cuda_context *, scratch_owner> scratch_contexts;

static void reserve_stream(ggml_backend_cuda_context & ctx, scratch_owner & owner, int stream) {
    GGML_ASSERT(stream >= 0 && stream < GGML_CUDA_MAX_STREAMS);
    if (!owner.streams[stream]) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&owner.streams[stream]), merlin_prefill_slab_bytes));
        merlin_engine_diagnostic(
            "merlin-engine: scratch_reserved device=%d stream=%d reserved_bytes=%zu lifetime=backend\n",
            ctx.device, stream, merlin_prefill_slab_bytes);
    }
}
} // namespace merlin_prefill

void merlin_prefill_initialize(ggml_backend_cuda_context & ctx, bool enabled) {
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_TURING) { return; }
    std::lock_guard<std::mutex> lock(merlin_prefill::scratch_mutex);
    auto & owner = merlin_prefill::scratch_contexts[&ctx];
    merlin_prefill::reserve_stream(ctx, owner, 0);
}

void merlin_prefill_prepare_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * graph) {
    std::lock_guard<std::mutex> lock(merlin_prefill::scratch_mutex);
    const auto found = merlin_prefill::scratch_contexts.find(&ctx);
    if (found == merlin_prefill::scratch_contexts.end()) { return; }
    auto & owner = found->second;
    if (ctx.stream_context().concurrent_events.empty()) { return; }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1] ||
                !merlin_prefill_eligible(ctx, node->src[0], node->src[1], node)) { continue; }
        for (const auto & entry : ctx.stream_context().concurrent_events) {
            const auto mapping = entry.second.stream_mapping.find(node);
            if (mapping != entry.second.stream_mapping.end()) {
                merlin_prefill::reserve_stream(ctx, owner, mapping->second);
            }
        }
    }
}

char * merlin_prefill_scratch(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(merlin_prefill::scratch_mutex);
    const auto found = merlin_prefill::scratch_contexts.find(&ctx);
    if (found == merlin_prefill::scratch_contexts.end()) { return nullptr; }
    GGML_ASSERT(ctx.curr_stream_no >= 0 && ctx.curr_stream_no < GGML_CUDA_MAX_STREAMS);
    return found->second.streams[ctx.curr_stream_no];
}

size_t merlin_prefill_reserved_bytes(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(merlin_prefill::scratch_mutex);
    const auto found = merlin_prefill::scratch_contexts.find(&ctx);
    if (found == merlin_prefill::scratch_contexts.end()) { return 0; }
    size_t result = 0;
    for (const auto pointer : found->second.streams) {
        if (pointer) { result += merlin_prefill_slab_bytes; }
    }
    return result;
}

void merlin_prefill_release(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(merlin_prefill::scratch_mutex);
    const auto found = merlin_prefill::scratch_contexts.find(&ctx);
    if (found == merlin_prefill::scratch_contexts.end()) { return; }
    ggml_cuda_set_device(ctx.device);
    // Teardown only: finish all users before freeing addresses retained by graphs.
    for (int stream = 0; stream < GGML_CUDA_MAX_STREAMS; ++stream) {
        if (found->second.streams[stream] && ctx.streams[ctx.device][stream]) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.streams[ctx.device][stream]));
        }
    }
    for (auto pointer : found->second.streams) {
        if (pointer) { CUDA_CHECK(cudaFree(pointer)); }
    }
    merlin_prefill::scratch_contexts.erase(found);
}
