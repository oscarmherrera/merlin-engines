#pragma once

#include "common.cuh"
#include "merlin-build.h"
#include "merlin-workload.h"
#include "merlin-dispatch-profile.h"
#include "merlin-dispatch-kernels.h"
#include "merlin-kernel-log.cuh"
#include "merlin-prefill-workspace.cuh"

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>

inline bool merlin_dispatch_calibrating() {
    static const bool enabled = [] {
        const char * value = std::getenv("MERLIN_ENGINE_CALIBRATE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool & merlin_dispatch_reference_active() {
    static thread_local bool active = false;
    return active;
}

struct merlin_dispatch_reference_scope {
    bool previous = merlin_dispatch_reference_active();
    merlin_dispatch_reference_scope() { merlin_dispatch_reference_active() = true; }
    ~merlin_dispatch_reference_scope() { merlin_dispatch_reference_active() = previous; }
};

struct merlin_dispatch_candidate {
    const char * id;
    size_t workspace_bytes;
    void (*launch)(void *);
    void * user;
    size_t shared_bytes_per_block = 0;
};

namespace merlin_dispatch {
constexpr size_t max_output_bytes = size_t(256) * 1024 * 1024;

struct capture_observer {
    bool (*record)(void *, const key &, const char *) = nullptr;
    void * user = nullptr;
};
inline capture_observer & capture_observer_current() {
    static thread_local capture_observer current;
    return current;
}

struct device_state {
    struct drift_history {
        key shape;
        measurement * cost;
        std::array<double, 5> samples{};
        unsigned observed = 0;
        uint64_t submissions = 0;
    };
    struct drift_slot {
        cudaEvent_t start = nullptr, stop = nullptr;
        drift_history * history = nullptr;
        bool pending = false;
    };
    std::mutex mutex;
    profile costs;
    std::set<key> calibrated;
    std::string path, fingerprint;
    std::unique_ptr<float[]> reference_output, candidate_output;
    cudaEvent_t start = nullptr, stop = nullptr;
    std::vector<drift_history> histories;
    std::array<drift_slot, 8> drift{};
    uint64_t epoch = 0;

    explicit device_state(int device) {
        const char * configured = std::getenv("MERLIN_ENGINE_PROFILE");
        if (configured) { path = configured; }
        cudaDeviceProp properties{};
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::ostringstream identity;
        identity << MERLIN_ENGINE_BUILD_ID << '|' << MERLIN_PRISM_REVISION << '|' << MERLIN_CUTLASS_REVISION << '|'
                 << properties.name << '|' << properties.major << '.' << properties.minor << '|'
                 << properties.multiProcessorCount << '|' << properties.totalGlobalMem << '|'
                 << driver << '|' << runtime << '|' << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.'
                 << __CUDACC_VER_BUILD__ << '|' << __VERSION__ << '|';
        for (unsigned char byte : properties.uuid.bytes) {
            identity << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
        }
        fingerprint = identity.str();
        const bool loaded = !path.empty() && costs.load(path, fingerprint);
        if (!merlin_dispatch_calibrating()) {
            for (auto & row : costs.entries) {
                for (auto & item : row.second) {
                    if (item.correct && item.kernel != "prism") { histories.push_back({row.first, &item}); }
                }
            }
        }
        std::fprintf(stderr, "merlin-engine: calibration_profile=%s mode=%s device=%d\n",
            loaded ? "matched" : "missing_or_invalid", merlin_dispatch_calibrating() ? "calibrate" : "inference", device);
    }

    void prepare() {
        if (!reference_output) {
            reference_output.reset(new float[max_output_bytes / sizeof(float)]);
            candidate_output.reset(new float[max_output_bytes / sizeof(float)]);
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
        }
    }

    void prepare_drift() {
        if (histories.empty() || drift[0].start) { return; }
        for (auto & slot : drift) {
            CUDA_CHECK(cudaEventCreate(&slot.start));
            CUDA_CHECK(cudaEventCreate(&slot.stop));
        }
    }

    void poll_drift() {
        bool invalidated = false;
        for (auto & slot : drift) {
            if (!slot.pending) { continue; }
            const cudaError_t ready = cudaEventQuery(slot.stop);
            if (ready == cudaErrorNotReady) { continue; }
            CUDA_CHECK(ready);
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, slot.start, slot.stop));
            auto & history = *slot.history;
            history.samples[history.observed++ % history.samples.size()] = elapsed;
            if (history.cost->correct && history.observed >= history.samples.size() &&
                    material_drift(history.samples, history.cost->median_ms)) {
                history.cost->correct = false;
                invalidated = true;
                std::fprintf(stderr,
                    "merlin-engine: profile_invalidated kernel=%s m=%lld n=%lld k=%lld calibrated_ms=%.6f observed_median_ms=%.6f reason=latency_drift\n",
                    history.cost->kernel.c_str(), (long long)history.shape.m, (long long)history.shape.n,
                    (long long)history.shape.k, history.cost->median_ms, median(history.samples));
            }
            slot.pending = false;
        }
        if (invalidated) { ++epoch; }
        if (invalidated && !costs.save(path, fingerprint)) {
            std::fputs("merlin-engine: drift invalidation is in memory only; profile write failed\n", stderr);
        }
    }

    drift_slot * sample(const key & shape, const char * kernel) {
        if (!drift[0].start) { return nullptr; }
        drift_history * history = nullptr;
        for (auto & item : histories) {
            if (item.shape.fields() == shape.fields() && item.cost->kernel == kernel && item.cost->correct) {
                history = &item;
                break;
            }
        }
        if (!history || ++history->submissions % 128 != 0) { return nullptr; }
        for (auto & slot : drift) {
            if (!slot.pending) {
                slot.history = history;
                return &slot;
            }
        }
        return nullptr;
    }
};

inline device_state & state(int device) {
    static std::mutex creation_mutex;
    static std::map<int, std::unique_ptr<device_state>> devices;
    std::lock_guard<std::mutex> lock(creation_mutex);
    auto & result = devices[device];
    if (!result) { result.reset(new device_state(device)); }
    return *result;
}

inline double measure(device_state & storage, cudaStream_t stream, void (*launch)(void *), void * user) {
    for (int i = 0; i < 2; ++i) { launch(user); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::array<double, 5> samples{};
    for (double & sample : samples) {
        CUDA_CHECK(cudaEventRecord(storage.start, stream));
        launch(user);
        CUDA_CHECK(cudaEventRecord(storage.stop, stream));
        CUDA_CHECK(cudaEventSynchronize(storage.stop));
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, storage.start, storage.stop));
        sample = elapsed;
    }
    return median(samples);
}

inline void copy_output(cudaStream_t stream, float * host, const float * output, size_t count) {
    CUDA_CHECK(cudaMemcpyAsync(host, output, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

inline void calibration_receipt(const key & shape, size_t records, bool reference_valid) {
    static FILE * file = [] {
        const char * path = std::getenv("MERLIN_CALIBRATION_RECEIPT");
        if (!path) { return static_cast<FILE *>(nullptr); }
        FILE * result = std::fopen(path, "wx");
        if (!result) { GGML_ABORT("merlin-engine: cannot create calibration receipt"); }
        return result;
    }();
    if (!file) { return; }
    std::fprintf(file,
        "{\"schema\":3,\"engine_revision\":\"%s\",\"runtime_revision\":\"%s\","
        "\"cutlass_revision\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,"
        "\"sequence_batch\":%lld,\"tokens_in_flight\":%lld,\"phase\":%lld,\"fusion\":%lld,"
        "\"records\":%zu,\"reference_valid\":%s,\"profile_saved\":true}\n",
        MERLIN_ENGINE_BUILD_ID, MERLIN_PRISM_REVISION, MERLIN_CUTLASS_REVISION,
        (long long)shape.m, (long long)shape.n, (long long)shape.k,
        (long long)shape.batch, (long long)shape.tokens_in_flight, (long long)shape.phase, (long long)shape.fusion, records,
        reference_valid ? "true" : "false");
    if (std::fflush(file) || std::ferror(file)) {
        GGML_ABORT("merlin-engine: cannot write calibration receipt");
    }
}

inline const merlin_dispatch_candidate * select(const profile & costs, const key & shape,
        const merlin_dispatch_candidate * candidates, size_t count) {
    GGML_ASSERT(count <= 5);
    const char * names[5]{};
    for (size_t i = 0; i < count; ++i) {
        names[i] = candidates[i].id;
    }
    const int winner = costs.choose(shape, names, count);
    return winner < 0 ? nullptr : &candidates[winner];
}
} // namespace merlin_dispatch

// Sum logical operand footprints; this does not measure cache reuse or DRAM transactions.
inline merlin_logical_bytes merlin_dispatch_logical_bytes(const ggml_tensor * weights,
        const ggml_tensor * input, const ggml_tensor * output,
        const ggml_cuda_mm_fusion_args_host * fusion = nullptr) {
    size_t read = 0;
    const ggml_tensor * operands[] = {weights, input, fusion ? fusion->gate : nullptr,
        fusion ? fusion->x_bias : nullptr, fusion ? fusion->gate_bias : nullptr,
        fusion ? fusion->x_scale : nullptr, fusion ? fusion->gate_scale : nullptr};
    for (const auto * operand : operands) {
        if (!operand) { continue; }
        const size_t bytes = ggml_nbytes(operand);
        if (bytes >= SIZE_MAX - read) { return {SIZE_MAX, ggml_nbytes(output)}; }
        read += bytes;
    }
    return {read, ggml_nbytes(output)};
}

// Callbacks write the same real output. Only explicit startup calibration repeats them.
inline void merlin_dispatch_run(ggml_backend_cuda_context & ctx, const merlin_dispatch::key & shape,
        float * output, size_t elements, void (*reference)(void *), void * reference_user,
        const merlin_dispatch_candidate * candidates, size_t candidate_count, merlin_logical_bytes logical) {
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
    auto & storage = merlin_dispatch::state(ctx.device);
    std::lock_guard<std::mutex> lock(storage.mutex);
    if (!merlin_dispatch_calibrating() && capture == cudaStreamCaptureStatusNone) {
        storage.prepare_drift();
        storage.poll_drift();
    }
    if (merlin_dispatch_calibrating() && capture == cudaStreamCaptureStatusNone && shape.valid() &&
            elements > 0 && elements <= merlin_dispatch::max_output_bytes / sizeof(float) &&
            storage.calibrated.find(shape) == storage.calibrated.end()) {
        storage.prepare();
        auto & rows = storage.costs.entries[shape];
        rows.clear();
        const double reference_ms = merlin_dispatch::measure(storage, ctx.stream(), reference, reference_user);
        merlin_dispatch::copy_output(ctx.stream(), storage.reference_output.get(), output, elements);
        double reference_nmse = 0;
        const bool reference_finite = merlin_dispatch::compare(storage.reference_output.get(),
            storage.reference_output.get(), elements, reference_nmse);
        if (reference_ms > 0) { rows.push_back({"prism", reference_ms, 0, reference_finite}); }
        for (size_t i = 0; i < candidate_count; ++i) {
            const auto & candidate = candidates[i];
            const double elapsed = merlin_dispatch::measure(storage, ctx.stream(), candidate.launch, candidate.user);
            merlin_dispatch::copy_output(ctx.stream(), storage.candidate_output.get(), output, elements);
            double nmse = 0;
            const bool correct = reference_finite && merlin_dispatch::compare(storage.reference_output.get(),
                storage.candidate_output.get(), elements, nmse);
            if (elapsed > 0) { rows.push_back({candidate.id, elapsed, std::isfinite(nmse) ? nmse : 0, correct}); }
            std::fprintf(stderr,
                "merlin-engine: calibration kernel=%s m=%lld n=%lld k=%lld reference_ms=%.6f median_ms=%.6f nmse=%.9g correct=%d\n",
                candidate.id, (long long)shape.m, (long long)shape.n, (long long)shape.k, reference_ms, elapsed, nmse, int(correct));
        }
        storage.calibrated.insert(shape);
        if (!storage.costs.save(storage.path, storage.fingerprint)) {
            GGML_ABORT("merlin-engine: calibration profile could not be persisted");
        }
        merlin_dispatch::calibration_receipt(shape, rows.size(), reference_finite && reference_ms > 0);
    }
    const auto * selected = merlin_dispatch::select(storage.costs, shape, candidates, candidate_count);
    const auto observer = merlin_dispatch::capture_observer_current();
    if (selected && capture != cudaStreamCaptureStatusNone && observer.record &&
            !observer.record(observer.user, shape, selected->id)) {
        selected = nullptr; // A graph never embeds a custom selection the bounded monitor cannot track.
    }
    merlin_dispatch::device_state::drift_slot * sample = nullptr;
    if (selected && !merlin_dispatch_calibrating() && capture == cudaStreamCaptureStatusNone) {
        sample = storage.sample(shape, selected->id);
    }
    if (sample) { CUDA_CHECK(cudaEventRecord(sample->start, ctx.stream())); }
    if (selected) { selected->launch(selected->user); } else { reference(reference_user); }
    if (sample) {
        CUDA_CHECK(cudaEventRecord(sample->stop, ctx.stream()));
        sample->pending = true;
    }
    merlin_kernel_log(merlin_workload_phase_name(shape.phase), selected ? selected->id : "prism", ctx.device,
        shape.m, shape.n, shape.k, selected ? selected->workspace_bytes : SIZE_MAX, shape.fusion != 0,
        selected ? selected->shared_bytes_per_block : SIZE_MAX, logical, merlin_prefill_reserved_bytes(ctx),
        shape.batch, shape.tokens_in_flight,
        ggml_merlin_workload_get().source == GGML_MERLIN_SOURCE_UBATCH ? "llama_ubatch_and_graph_type" :
        ggml_merlin_workload_get().source == GGML_MERLIN_SOURCE_CALIBRATION ? "explicit_calibration_workload" : "unknown");
}
