#pragma once
#include "merlin-engine-log.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

struct merlin_logical_bytes {
    size_t read_bytes = SIZE_MAX;
    size_t write_bytes = SIZE_MAX;
};

inline void merlin_json_bytes(char (&text)[32], size_t bytes) {
    if (bytes == SIZE_MAX) { std::strcpy(text, "null"); }
    else { std::snprintf(text, sizeof(text), "%zu", bytes); }
}

// Host submissions, not CUDA graph replays or completed GPU work.
inline void merlin_kernel_log(const char * phase, const char * kernel, int device,
        int64_t m, int64_t n, int64_t k, size_t workspace_bytes, bool fused,
        size_t shared_bytes_per_block = 0, merlin_logical_bytes logical = {},
        size_t workspace_reserved_bytes = SIZE_MAX, int64_t sequence_batch = 0, int64_t tokens_in_flight = 0, const char * phase_source = "unknown", const char * reason = "measured_selection") {
    if (!merlin_engine_log_enabled()) { return; }
    static std::atomic<unsigned long long> submissions[5]{};
    const unsigned bucket = n > 8 ? 4 :
        n == 1 ? 0 : n == 2 ? 1 : n == 4 ? 2 : 3;
    const auto sequence = submissions[bucket].fetch_add(1, std::memory_order_relaxed) + 1;
    // Per matrix-column sampling bucket (not workload phase or sequence batch): first 256 submissions, then every 1024; no device sync.
    if (sequence > 256 && sequence % 1024 != 0) {
        return;
    }
    const auto timestamp = merlin_log_unix_ms();
    char workspace[32], reserved[32], reads[32], writes[32], shared[32];
    merlin_json_bytes(workspace, workspace_bytes);
    merlin_json_bytes(shared, shared_bytes_per_block);
    merlin_json_bytes(reserved, workspace_reserved_bytes);
    merlin_json_bytes(reads, logical.read_bytes);
    merlin_json_bytes(writes, logical.write_bytes);
    merlin_engine_log(
        "{\"schema\":2,\"event\":\"merlin_kernel_dispatch\","
        "\"unix_ms\":%lld,\"matrix_column_bucket_submission\":%llu,\"phase\":\"%s\","
        "\"kernel\":\"%s\",\"device\":%d,\"m\":%lld,\"n\":%lld,"
        "\"k\":%lld,\"workspace_bytes\":%s,\"shared_bytes_per_block\":%s,\"fused\":%s,"
        "\"workspace_reserved_bytes\":%s,\"logical_tensor_read_bytes_estimate\":%s,"
        "\"logical_tensor_write_bytes_estimate\":%s,\"physical_dram_traffic_measured\":false,"
        "\"logical_bytes_scope\":\"operator_operand_footprints_excluding_scratch\","
        "\"sequence_batch\":%lld,\"tokens_in_flight\":%lld,\"phase_source\":\"%s\","
        "\"reason\":\"%s\",\"mode\":\"%s\",\"pid\":%ld,"
        "\"gpu_completion\":false,\"counts_graph_replays\":false}\n",
        static_cast<long long>(timestamp), sequence, phase, kernel, device,
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k),
        workspace, shared, fused ? "true" : "false", reserved, reads, writes,
        static_cast<long long>(sequence_batch), static_cast<long long>(tokens_in_flight), phase_source, reason, merlin_log_mode(), (long)getpid());
}

inline void merlin_kernel_completion_log(int device, uint64_t completion_id, long long submitted,
        const char * kernel, int64_t m, int64_t n, int64_t k, double elapsed) {
    if (!merlin_engine_log_enabled()) { return; }
    merlin_engine_log("{\"schema\":1,\"event\":\"merlin_kernel_completion\",\"mode\":\"%s\","
        "\"pid\":%ld,\"device\":%d,\"completion_id\":%llu,\"submitted_unix_ms\":%lld,\"unix_ms\":%lld,"
        "\"kernel\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,\"elapsed_ms\":%.6f,"
        "\"gpu_completion\":true,\"sampled\":true,\"completed_custom_operations\":1}\n",
        merlin_log_mode(), (long)getpid(), device, (unsigned long long) completion_id, submitted,
        merlin_log_unix_ms(), kernel, (long long)m, (long long)n, (long long)k, elapsed);
}
