#pragma once
#include "merlin-dispatch-profile.h"

// Compare graph completion costs only within one unchanged workload/graph generation.
struct merlin_graph_cost_window {
    std::array<double, 5> samples{};
    unsigned observed = 0;
    double baseline_ms = 0;
    bool observe(double elapsed) {
        if (!std::isfinite(elapsed) || elapsed <= 0) { return false; }
        samples[observed++ % samples.size()] = elapsed;
        if (observed < samples.size()) { return false; }
        if (baseline_ms == 0) {
            baseline_ms = merlin_dispatch::median(samples);
            return false;
        }
        return observed >= 10 && merlin_dispatch::material_drift(samples, baseline_ms);
    }
    bool should_sample(uint64_t submission) const { return observed < 5 || submission % 128 == 0; }
};

// A missing executable is expected throughout upstream's two-call warmup.
struct merlin_graph_lifetime {
    bool initialized = false;
    bool had_instance = false;
    bool observe(bool has_instance) {
        const bool reset = !initialized || (had_instance && !has_instance);
        initialized = true;
        had_instance = has_instance;
        return reset;
    }
};
