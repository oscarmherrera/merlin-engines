#include "merlin-graph-monitor.cuh"

int main(int argc, char ** argv) {
    assert(argc == 2);
    ggml_backend_cuda_context ctx;
    ggml_tensor tensor;
    ggml_tensor * nodes[]{&tensor};
    ggml_cgraph cgraph{1, nodes};
    auto & owner = ctx.cuda_graphs[&tensor];
    owner.reset(new ggml_cuda_graph);
    auto * monitor = merlin_graph::acquire(ctx, &tensor, owner.get());
    assert(monitor);
    {
        ggml_backend_cuda_context other_ctx;
        auto & other_owner = other_ctx.cuda_graphs[&tensor];
        other_owner.reset(new ggml_cuda_graph);
        auto * other = merlin_graph::acquire(other_ctx, &tensor, other_owner.get());
        assert(other && other->id > monitor->id);
        merlin_graph::release(other_ctx);
    }
    assert(merlin_graph::prepare(*monitor, ctx, &cgraph));
    assert(!merlin_graph::prepare(*monitor, ctx, &cgraph)); // Stable second call can capture.
    merlin_dispatch::key key{750, 29, 5120, 1, 5120, 1, 1, 142, 0, 0};
    auto & state = merlin_dispatch::state(0);
    state.path = argv[1];
    state.costs.entries[key] = {{"prism", 2, 0, true}, {"custom", 1, 0, true}};
    owner->instance = &ctx;
    {
        merlin_graph::capture_scope capture(monitor, true);
        assert(merlin_dispatch::capture_observer_current().record(monitor, key, "custom"));
        merlin_graph::launch(ctx, owner->instance);
    }
    assert(monitor->samples[0].monitor_id == monitor->id);
    assert(!merlin_graph::current());
    assert(!merlin_dispatch::capture_observer_current().record);
    // Observe 5 completed launches at baseline10; subsequent real replays remain stable.
    for (int i = 0; i < 10; ++i) {
        assert(!merlin_graph::prepare(*monitor, ctx, &cgraph));
        merlin_graph::capture_scope capture(monitor, false);
        merlin_graph::launch(ctx, owner->instance);
    }
    assert(monitor->window.baseline_ms == 10);
    // Five sampled graph completions at20 retire this graph's custom record, persist,
    // bump the shared device epoch and request recapture exactly once.
    mock_elapsed_ms = 20;
    bool recapture = false;
    for (int i = 0; i < 700 && !recapture; ++i) {
        recapture = merlin_graph::prepare(*monitor, ctx, &cgraph);
        if (!recapture) {
            merlin_graph::capture_scope capture(monitor, false);
            merlin_graph::launch(ctx, owner->instance);
        }
    }
    assert(recapture);
    assert(state.epoch == 1);
    assert(!state.costs.entries[key][1].correct);
    assert(state.costs.entries[key][0].correct);
    merlin_dispatch::profile persisted;
    assert(persisted.load(state.path, state.fingerprint));
    assert(!persisted.entries[key][1].correct);
    assert(!merlin_graph::prepare(*monitor, ctx, &cgraph));
    assert(monitor->window.baseline_ms == 0);
    // Same cgraph address with new KV extent or workload cannot share the baseline.
    tensor.ne[1] = 2048;
    assert(merlin_graph::prepare(*monitor, ctx, &cgraph));
    assert(!merlin_graph::prepare(*monitor, ctx, &cgraph));
    workload.tokens_in_flight = 2;
    assert(merlin_graph::prepare(*monitor, ctx, &cgraph));
    assert(!merlin_graph::prepare(*monitor, ctx, &cgraph));
    owner->instance = nullptr;
    assert(merlin_graph::prepare(*monitor, ctx, &cgraph));
    assert(!merlin_graph::prepare(*monitor, ctx, &cgraph));
    // Captures above the bounded selection capacity must reject the extra custom route.
    {
        merlin_graph::capture_scope capture(monitor, true);
        for (int i = 0; i < 257; ++i) {
            key.m = i + 1;
            assert(merlin_graph::monitor::record(monitor, key, "custom") == (i < 256));
        }
    }
    merlin_graph::release(ctx);
    assert(merlin_graph::all().contexts.empty());
}
