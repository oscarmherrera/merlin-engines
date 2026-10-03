#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// Diagnostic timings include launch gaps. They are not calibration samples.
static bool merlin_cuda_profile_enabled() {
    static const bool enabled = [] {
        const char * path = std::getenv("MERLIN_CUDA_PROFILE");
        return path && path[0];
    }();
    return enabled;
}

class merlin_cuda_profile {
    struct entry {
        int first, last, stream;
        cudaEvent_t begin, end;
        float ms = 0;
    };
    ggml_cgraph * graph;
    int device;
    unsigned long long id = 0;
    bool active = false;
    std::vector<entry> entries;

    static FILE * output() {
        static FILE * file = [] {
            FILE * f = std::fopen(std::getenv("MERLIN_CUDA_PROFILE"), "wx");
            if (!f) {
                GGML_ABORT("Cannot create a new Merlin CUDA profile file");
            }
            return f;
        }();
        return file;
    }

    static void shape(FILE * file, const ggml_tensor * t) {
        if (!t) {
            std::fputs("null", file);
            return;
        }
        std::fprintf(file, "{\"type\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld]}",
            ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3]);
    }

public:
    merlin_cuda_profile(ggml_cgraph * graph, int device) : graph(graph), device(device) {
        if (!merlin_cuda_profile_enabled()) {
            return;
        }
        static std::atomic<unsigned long long> next{0};
        id = next.fetch_add(1);
        // Bound disk use and event allocation independently of request length.
        if (id >= 128) {
            return;
        }
        if (graph->n_nodes > 16384) {
            GGML_ABORT("Merlin profile graph exceeds 16384 nodes");
        }
        output();
        entries.reserve(graph->n_nodes);
        active = true;
    }

    class scope {
        merlin_cuda_profile & owner;
        int & last;
        cudaStream_t stream;
        entry value{};
    public:
        scope(merlin_cuda_profile & owner, int & index, int stream_no, cudaStream_t stream)
            : owner(owner), last(index), stream(stream) {
            if (!owner.active) {
                return;
            }
            value.first = index;
            value.stream = stream_no;
            CUDA_CHECK(cudaEventCreate(&value.begin));
            CUDA_CHECK(cudaEventCreate(&value.end));
            CUDA_CHECK(cudaEventRecord(value.begin, stream));
        }
        ~scope() {
            if (!owner.active) {
                return;
            }
            value.last = last;
            CUDA_CHECK(cudaEventRecord(value.end, stream));
            owner.entries.push_back(value);
        }
        scope(const scope &) = delete;
        scope & operator=(const scope &) = delete;
    };

    ~merlin_cuda_profile() {
        if (!active) {
            return;
        }
        for (auto & e : entries) {
            CUDA_CHECK(cudaEventSynchronize(e.end));
            CUDA_CHECK(cudaEventElapsedTime(&e.ms, e.begin, e.end));
            CUDA_CHECK(cudaEventDestroy(e.begin));
            CUDA_CHECK(cudaEventDestroy(e.end));
        }
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        FILE * f = output();
        std::fprintf(f, "{\"schema\":1,\"graph\":%llu,\"device\":%d,"
            "\"nodes\":%d,\"cuda_graphs\":false,\"operations\":[", id, device, graph->n_nodes);
        bool first = true;
        for (const auto & e : entries) {
            const auto * node = graph->nodes[e.first];
            std::fprintf(f, "%s{\"first\":%d,\"last\":%d,\"stream\":%d,\"ms\":%.6f,\"ops\":[",
                first ? "" : ",", e.first, e.last, e.stream, e.ms);
            first = false;
            for (int i = e.first; i <= e.last; ++i) {
                std::fprintf(f, "%s\"%s\"", i == e.first ? "" : ",", ggml_op_name(graph->nodes[i]->op));
            }
            std::fputs("],\"src0\":", f); shape(f, node->src[0]);
            std::fputs(",\"src1\":", f); shape(f, node->src[1]);
            std::fputs(",\"dst\":", f); shape(f, graph->nodes[e.last]);
            std::fputc('}', f);
        }
        std::fputs("]}\n", f);
        if (std::fflush(f) != 0 || std::ferror(f)) {
            GGML_ABORT("Cannot write Merlin CUDA profile");
        }
    }
    merlin_cuda_profile(const merlin_cuda_profile &) = delete;
    merlin_cuda_profile & operator=(const merlin_cuda_profile &) = delete;
};
