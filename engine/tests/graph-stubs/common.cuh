#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <unordered_map>
#define USE_CUDA_GRAPH
#define CUDA_CHECK(call) assert((call) == 0)
using cudaStream_t = int;
using cudaGraphExec_t = void *;
struct event { bool ready = false; };
using cudaEvent_t = event *;
constexpr int cudaErrorNotReady = 1;
inline float mock_elapsed_ms = 10;
inline int cudaEventCreate(cudaEvent_t * value) { *value = new event; return 0; }
inline int cudaEventDestroy(cudaEvent_t value) { delete value; return 0; }
inline int cudaEventQuery(cudaEvent_t value) { return value->ready ? 0 : cudaErrorNotReady; }
inline int cudaEventRecord(cudaEvent_t value, cudaStream_t) { value->ready = true; return 0; }
inline int cudaEventElapsedTime(float * elapsed, cudaEvent_t, cudaEvent_t) { *elapsed = mock_elapsed_ms; return 0; }
inline int cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return 0; }
inline void ggml_cuda_set_device(int) {}
struct ggml_tensor {
    int op = 0, type = 0;
    int64_t ne[4]{1, 1, 1, 1};
    size_t nb[4]{4, 4, 4, 4};
    char op_params[16]{};
    ggml_tensor * src[2]{};
};
struct ggml_cgraph { int n_nodes; ggml_tensor ** nodes; };
struct ggml_cuda_graph { cudaGraphExec_t instance = nullptr; };
struct ggml_backend_cuda_context {
    int device = 0;
    std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;
    cudaStream_t stream() { return 0; }
};
