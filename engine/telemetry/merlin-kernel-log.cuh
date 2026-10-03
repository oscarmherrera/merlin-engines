#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

// Host submissions, not CUDA graph replays or completed GPU work.
inline void merlin_kernel_log(const char * phase, const char * kernel, int device,
        int64_t m, int64_t n, int64_t k, size_t workspace_bytes, bool fused,
        size_t shared_bytes_per_block = 0) {
    static std::atomic<unsigned long long> submissions[5]{};
    const unsigned bucket = std::strcmp(phase, "prefill") == 0 ? 4 :
        n == 1 ? 0 : n == 2 ? 1 : n == 4 ? 2 : 3;
    const auto sequence = submissions[bucket].fetch_add(1, std::memory_order_relaxed) + 1;
    static FILE * file = [] {
        const char * path = std::getenv("MERLIN_KERNEL_LOG");
        if (!path || !path[0]) {
            return static_cast<FILE *>(nullptr);
        }
        FILE * result = std::fopen(path, "wx");
        if (!result) {
            std::fputs("merlin-engine: cannot create MERLIN_KERNEL_LOG\n", stderr);
            std::abort();
        }
        return result;
    }();
    // Per phase/batch: first 256 submissions, then every 1024; no device sync.
    if (sequence > 256 && sequence % 1024 != 0) {
        return;
    }
    if (!file && sequence > 16) {
        return;
    }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    FILE * output = file ? file : stderr;
    char workspace[32];
    if (workspace_bytes == SIZE_MAX) {
        std::strcpy(workspace, "null");
    } else {
        std::snprintf(workspace, sizeof(workspace), "%zu", workspace_bytes);
    }
    std::fprintf(output,
        "{\"schema\":1,\"event\":\"merlin_kernel_dispatch\","
        "\"unix_ms\":%lld,\"phase_batch_submission\":%llu,\"phase\":\"%s\","
        "\"kernel\":\"%s\",\"device\":%d,\"m\":%lld,\"n\":%lld,"
        "\"k\":%lld,\"workspace_bytes\":%s,\"shared_bytes_per_block\":%zu,\"fused\":%s,"
        "\"gpu_completion\":false,\"counts_graph_replays\":false}\n",
        static_cast<long long>(timestamp), sequence, phase, kernel, device,
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k),
        workspace, shared_bytes_per_block, fused ? "true" : "false");
    if (std::fflush(output) != 0 || std::ferror(output)) {
        std::fputs("merlin-engine: kernel dispatch log write failed\n", stderr);
        std::abort();
    }
}
