#pragma once
#include "common.cuh"
#include "merlin-dispatch.cuh"
#include "merlin-graph-policy.h"
#include "merlin-workload.h"
#include <unordered_map>
#include <atomic>
#include <chrono>

#ifdef USE_CUDA_GRAPH
namespace merlin_graph {
inline uint64_t next_monitor_id() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}
struct selection { merlin_dispatch::key shape; std::string kernel; };
struct monitor {
    struct sample {
        cudaEvent_t start = nullptr, stop = nullptr;
        bool pending = false;
        uint64_t monitor_id = 0, generation = 0, submission = 0, signature = 0, epoch = 0;
        int64_t batch = 0, tokens = 0;
        int phase = -1;
    };
    const uint64_t id = next_monitor_id();
    ggml_cuda_graph * owner;
    uint64_t epoch = 0, generation = 0, submissions = 0, signature = 0;
    int64_t batch = 0, tokens = 0;
    int phase = -1;
    merlin_graph_cost_window window;
    std::array<sample, 2> samples{};
    std::vector<selection> selected;
    merlin_graph_lifetime lifetime;
    explicit monitor(ggml_cuda_graph * graph) : owner(graph) {
        selected.reserve(256);
        for (auto & item : samples) {
            CUDA_CHECK(cudaEventCreate(&item.start));
            CUDA_CHECK(cudaEventCreate(&item.stop));
        }
    }
    ~monitor() {
        for (auto & item : samples) {
            CUDA_CHECK(cudaEventDestroy(item.start));
            CUDA_CHECK(cudaEventDestroy(item.stop));
        }
    }
    static bool record(void * user, const merlin_dispatch::key & shape, const char * kernel) {
        auto & self = *static_cast<monitor *>(user);
        for (const auto & item : self.selected) {
            if (item.shape.fields() == shape.fields() && item.kernel == kernel) { return true; }
        }
        if (self.selected.size() == 256) { return false; }
        self.selected.push_back({shape, kernel});
        return true;
    }
    void reset() { ++generation; window = {}; selected.clear(); }
};

using graphs = std::unordered_map<const void *, std::unique_ptr<monitor>>;
struct registry {
    std::mutex mutex;
    std::unordered_map<ggml_backend_cuda_context *, graphs> contexts;
};
inline registry & all() { static registry result; return result; }
inline monitor * & current() { static thread_local monitor * result = nullptr; return result; }

// Shape/stride/op-parameter identity is checked even when upstream reuses cgraph->uid.
// In particular, changing attention/KV extents cannot train a latency-drift baseline.
inline uint64_t signature(const ggml_cgraph * graph) {
    uint64_t result = 1469598103934665603ULL;
    const auto add = [&](const void * pointer, size_t length) {
        const auto * bytes = static_cast<const unsigned char *>(pointer);
        for (size_t i = 0; i < length; ++i) { result = (result ^ bytes[i]) * 1099511628211ULL; }
    };
    add(&graph->n_nodes, sizeof(graph->n_nodes));
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        add(&node->op, sizeof(node->op));
        add(&node->type, sizeof(node->type));
        add(node->ne, sizeof(node->ne));
        add(node->nb, sizeof(node->nb));
        add(node->op_params, sizeof(node->op_params));
        for (const auto * source : node->src) {
            const bool present = source != nullptr;
            add(&present, sizeof(present));
            if (source) {
                add(&source->type, sizeof(source->type));
                add(source->ne, sizeof(source->ne));
                add(source->nb, sizeof(source->nb));
            }
        }
    }
    return result;
}

inline void log_completion(const monitor::sample & item, int device,
        double elapsed, bool used_for_drift) {
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr,
        "{\"schema\":1,\"event\":\"merlin_graph_completion\",\"device\":%d,"
        "\"monitor_id\":%llu,\"unix_ms\":%lld,\"graph_signature\":%llu,\"generation\":%llu,\"submission\":%llu,"
        "\"sequence_batch\":%lld,\"tokens_in_flight\":%lld,\"phase\":%d,\"epoch\":%llu,"
        "\"elapsed_ms\":%.6f,\"gpu_completion\":true,\"counts_graph_replays\":true,"
        "\"sampled\":true,\"used_for_drift\":%s,\"latency_scope\":\"whole_graph\"}\n",
        device, (unsigned long long) item.monitor_id, (long long) timestamp, (unsigned long long) item.signature, (unsigned long long) item.generation,
        (unsigned long long) item.submission, (long long) item.batch, (long long) item.tokens, item.phase,
        (unsigned long long) item.epoch, elapsed, used_for_drift ? "true" : "false");
}

inline void poll(monitor & graph, merlin_dispatch::device_state & storage, int device) {
    bool drift = false;
    for (auto & item : graph.samples) {
        if (!item.pending) { continue; }
        const auto ready = cudaEventQuery(item.stop);
        if (ready == cudaErrorNotReady) { continue; }
        CUDA_CHECK(ready);
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, item.start, item.stop));
        const bool same = item.generation == graph.generation && graph.epoch == storage.epoch;
        log_completion(item, device, elapsed, same);
        if (same) { drift = graph.window.observe(elapsed) || drift; }
        item.pending = false;
    }
    if (!drift || graph.selected.empty()) { return; }
    size_t retired = 0;
    for (const auto & item : graph.selected) {
        auto row = storage.costs.entries.find(item.shape);
        if (row == storage.costs.entries.end()) { continue; }
        for (auto & cost : row->second) {
            if (cost.kernel == item.kernel && cost.correct) { cost.correct = false; ++retired; }
        }
    }
    if (!retired) { return; }
    ++storage.epoch;
    const bool saved = storage.costs.save(storage.path, storage.fingerprint);
    std::fprintf(stderr,
        "merlin-engine: profile_invalidated reason=graph_cohort_latency_drift "
        "device=%d epoch=%llu records=%zu baseline_ms=%.6f observed_median_ms=%.6f "
        "per_kernel_cause_known=false profile_saved=%d\n", device,
        (unsigned long long) storage.epoch, retired, graph.window.baseline_ms,
        merlin_dispatch::median(graph.window.samples), int(saved));
}

inline void drain_ready(monitor & graph, ggml_backend_cuda_context & ctx) {
    auto & storage = merlin_dispatch::state(ctx.device);
    std::lock_guard<std::mutex> lock(storage.mutex);
    poll(graph, storage, ctx.device);
    size_t pending = 0;
    for (const auto & item : graph.samples) { pending += item.pending; }
    if (pending) {
        std::fprintf(stderr, "merlin-engine: graph_monitor_release incomplete_samples=%zu device=%d no_wait=true\n",
            pending, ctx.device);
    }
}

inline monitor * acquire(ggml_backend_cuda_context & ctx, const void * key, ggml_cuda_graph * graph) {
    auto & registry = all();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto context = registry.contexts.find(&ctx);
    if (context == registry.contexts.end()) {
        if (registry.contexts.size() == 64) { return nullptr; }
        context = registry.contexts.emplace(&ctx, graphs{}).first;
    }
    auto & entries = context->second;
    for (auto it = entries.begin(); it != entries.end();) {
        const auto live = ctx.cuda_graphs.find(it->first);
        if (live == ctx.cuda_graphs.end() || live->second.get() != it->second->owner) {
            drain_ready(*it->second, ctx);
            it = entries.erase(it);
        } else { ++it; }
    }
    auto found = entries.find(key);
    if (found == entries.end()) {
        if (entries.size() == 64) { return nullptr; }
        found = entries.emplace(key, std::unique_ptr<monitor>(new monitor(graph))).first;
    }
    return found->second.get();
}

inline bool prepare(monitor & graph, ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    const auto workload = ggml_merlin_workload_get();
    const uint64_t identity = signature(cgraph);
    auto & storage = merlin_dispatch::state(ctx.device);
    std::lock_guard<std::mutex> lock(storage.mutex);
    storage.prepare_drift();
    storage.poll_drift();
    const bool lifetime_changed = graph.lifetime.observe(graph.owner->instance != nullptr);
    const bool changed = lifetime_changed || graph.signature != identity ||
        graph.batch != workload.sequence_batch || graph.tokens != workload.tokens_in_flight ||
        graph.phase != workload.phase || graph.epoch != storage.epoch;
    if (changed) { graph.reset(); }
    graph.signature = identity;
    graph.batch = workload.sequence_batch;
    graph.tokens = workload.tokens_in_flight;
    graph.phase = workload.phase;
    poll(graph, storage, ctx.device);
    const bool invalidated = graph.epoch != storage.epoch;
    if (invalidated && !changed) { graph.reset(); }
    graph.epoch = storage.epoch;
    return changed || invalidated;
}

struct capture_scope {
    monitor * previous = current();
    merlin_dispatch::capture_observer observer = merlin_dispatch::capture_observer_current();
    capture_scope(monitor * graph, bool recapture) {
        current() = graph;
        if (graph && recapture) { graph->reset(); }
        merlin_dispatch::capture_observer_current() = graph ?
            merlin_dispatch::capture_observer{monitor::record, graph} : merlin_dispatch::capture_observer{};
    }
    ~capture_scope() {
        current() = previous;
        merlin_dispatch::capture_observer_current() = observer;
    }
};

inline void launch(ggml_backend_cuda_context & ctx, cudaGraphExec_t instance) {
    auto * graph = current();
    if (graph) { graph->lifetime.observe(true); }
    monitor::sample * sample = nullptr;
    if (graph && graph->window.should_sample(++graph->submissions)) {
        for (auto & item : graph->samples) {
            if (!item.pending) { sample = &item; break; }
        }
    }
    if (sample) { CUDA_CHECK(cudaEventRecord(sample->start, ctx.stream())); }
    CUDA_CHECK(cudaGraphLaunch(instance, ctx.stream()));
    if (sample) {
        CUDA_CHECK(cudaEventRecord(sample->stop, ctx.stream()));
        sample->monitor_id = graph->id;
        sample->generation = graph->generation;
        sample->submission = graph->submissions;
        sample->signature = graph->signature;
        sample->batch = graph->batch;
        sample->tokens = graph->tokens;
        sample->phase = graph->phase;
        sample->epoch = graph->epoch;
        sample->pending = true;
    }
}

inline void release(ggml_backend_cuda_context & ctx) {
    auto & registry = all();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.contexts.find(&ctx);
    if (found == registry.contexts.end()) { return; }
    ggml_cuda_set_device(ctx.device);
    for (auto & item : found->second) { drain_ready(*item.second, ctx); }
    registry.contexts.erase(found);
}
} // namespace merlin_graph
#endif
