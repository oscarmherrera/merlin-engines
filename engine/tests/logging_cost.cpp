#include "merlin-kernel-log.cuh"
#include <chrono>
#include <cstdio>

int main() {
    constexpr int operations = 100000;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < operations; ++i) {
        const int n = 1 << (i % 5);
        merlin_kernel_log("single_token_per_sequence", "custom", 0, 5120, n, 5120, 0, false);
        if (i % 128 == 0) {
            merlin_kernel_completion_log(0, i + 1, 1000, "custom", 5120, n, 5120, 1.0);
        }
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("{\"operations\":%d,\"logging_enabled\":%s,\"host_wall_seconds\":%.9f}\n",
        operations, merlin_engine_log_enabled() ? "true" : "false", elapsed);
}
