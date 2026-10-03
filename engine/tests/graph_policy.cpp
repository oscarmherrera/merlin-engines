#include "merlin-graph-policy.h"
#include <cassert>
#include <limits>

int main() {
    merlin_graph_lifetime lifecycle;
    bool warmup_complete = false, has_instance = false;
    for (int call = 0; call < 3; ++call) {
        const bool changed = lifecycle.observe(has_instance);
        bool capture = false, replay = false;
        if (!warmup_complete) {
            if (!changed) { warmup_complete = true; capture = true; }
        } else if (changed) {
            warmup_complete = false;
        } else { replay = has_instance; }
        assert(capture == (call == 1));
        assert(replay == (call == 2));
        if (capture) { has_instance = true; lifecycle.observe(true); }
    }
    assert(lifecycle.observe(false)); // Eviction/address reuse resets exactly once.
    assert(!lifecycle.observe(false));

    merlin_graph_cost_window graph;
    for (int i = 0; i < 5; ++i) { assert(!graph.observe(10)); }
    assert(graph.baseline_ms == 10);
    assert(!graph.should_sample(127));
    assert(graph.should_sample(128));
    assert(!graph.observe(std::numeric_limits<double>::quiet_NaN()));
    assert(!graph.observe(0));
    assert(!graph.observe(-2));
    // One unrelated latency spike cannot invalidate a healthy graph.
    assert(!graph.observe(100));
    for (int i = 0; i < 4; ++i) { assert(!graph.observe(10)); }
    for (int i = 0; i < 32; ++i) { assert(!graph.observe(14.9)); }
    // A sustained >1.5x cost shift must fire; equality alone must not.
    for (int i = 0; i < 5; ++i) { assert(!graph.observe(15)); }
    assert(!graph.observe(16));
    assert(!graph.observe(16));
    assert(graph.observe(16));
    // A new context/shape/workload establishes its own baseline.
    graph = {};
    for (int i = 0; i < 10; ++i) { assert(!graph.observe(25)); }
    assert(graph.baseline_ms == 25);
}
