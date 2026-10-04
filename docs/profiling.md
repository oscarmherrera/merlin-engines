# RTX 8000 custom runtime and diagnostics

Custom packed decode and tiled prefill share the pinned Prism core. The
`13288e4` runtime compiled for `sm_75` and passed the existing 17 direct
RTX8000 CUDA backend cases, but the reported custom decode and prefill shapes
were slower than Prism. Full-model correctness, custom load VRAM,
large-context behavior and a performance gain remain unproven. The endpoint
has been restored to stock Bonsai (`prism-adfffbe`). Until the owner changes
the instruction, all further engine tests are direct runs of a staged runtime;
the endpoint procedures below describe historical baseline capture or future
integration only.

Build directly on go-dev `.30`, from a clean pulled engine commit:

```sh
python3 scripts/build_runtime.py --cuda /home/oscar/.local/merlin-toolchains/cuda-12.8.1 --output /home/oscar/merlin-engine-builds/UNIQUE_BUILD --jobs 6
```

The command checks the pinned source revision, adds both kernel overlays and profiling, and
builds the shared libraries, `llama-bench`, and `test-backend-ops` for `sm_75`.
The server, web UI, and examples are disabled. The standard attention variants
include Q8/Q8; `GGML_CUDA_FA_ALL_QUANTS=OFF` excludes the extra combinations.
It stages them in `runtime/` with CUDA user-space libraries
and a manifest of revisions, compiler identity, and artifact hashes. It never
loads a model, installs the runtime in a service directory, or starts an endpoint.
On `.30`, use `--jobs 6 --scratch /tmp/UNIQUE_BUILD` for its six assigned CPUs
and to avoid charging temporary objects against the home disk quota.
The builder exposes the toolkit's `lib/` through `LD_LIBRARY_PATH` while linking:
the installed `$ORIGIN` search path alone cannot resolve those dependencies
until the final bundle has been staged.
The staged CUDA dependency files are hard links to the toolkit (both must be on
the same filesystem); do not modify toolkit libraries after staging a bundle.
The bundle hashes record their exact content. Extracted download archives can
be removed after verification; keep NVIDIA's manifest.

The build toolkit uses NVIDIA's CUDA 12.8.1 redistributables: nvcc 12.8.93,
cudart/CCCL 12.8.90, and cuBLAS 12.8.4.1. Archive SHA256 values come from
NVIDIA's `redistrib_12.8.1.json`, retained beside the toolkit. GCC/G++ 13 is used.
The component archives use `lib/`; the toolkit also needs a `lib64 -> lib`
symlink for nvcc's linker defaults. If a network clone fails, `--source-cache`
can name a local Git checkout: the builder clones its committed tree and still
requires the exact locked revision before applying the overlay.
Ubuntu 26.04's glibc exposes a known CUDA header incompatibility. The exact
six-declaration correction in pinned Prism `docs/build.md:253–285` is reproduced
by `scripts/cuda_glibc_compat.py`; it saves the original header and hashes.
It changes only this isolated toolkit. A compiler-only `sm_75` smoke test failed
before that correction and passed afterward. This does not establish GPU correctness.

## Capture and report

The first check after a direct custom-model load is VRAM usage, before any inference.
Save `nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv`
and `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`.
Identify the direct process and mapped runtime libraries. Endpoint PID checks
apply only if endpoint testing is later authorized again.
Record available loader weight-buffer and KV-buffer allocation messages and the
engine's `scratch_reserved` records separately. Process/device totals do not by
themselves identify those components; unavailable components remain unknown.

For a diagnostic direct run, set `MERLIN_CUDA_PROFILE` to a **new absolute
file path** when launching the staged runtime. An existing path is refused to preserve evidence.
The environment variable is read once per process. Never set it on production
as an incidental diagnostic action; loading the staged runtime takes a deployment.

```sh
python3 benchmarks/profile_report.py /absolute/path/profile.jsonl
```

CUDA events surround each executed graph operation or fused range on its actual
stream. Skipped views/non-compute nodes have no timing. Records contain the
graph/device/stream, source and destination shapes/types, operation range, and
elapsed milliseconds. They contain no tensor contents or prompt text. The report
groups identical operation sequences and shapes; overlapping or truncated records
fail instead of producing a partial report.

Capture is bounded to the first 128 graph submissions per process and at most
16,384 nodes per graph. Profiling disables CUDA graph capture/replay, synchronizes
completed operation events at graph boundaries, and adds host/event overhead.
The report sums **diagnostic work time**, not wall time: concurrent streams may
overlap, and launch gaps can fall inside event intervals. Do not use this profile
as a kernel calibration table or quote it as end-to-end throughput. Run ordinary
inference with profiling unset for performance comparisons.

## Kernel selection and build progress

Set `MERLIN_KERNEL_LOG` to a new absolute JSONL path before starting the direct runtime process.
The runtime records the first 256 host dispatches in each matrix-column bucket
(N=1, 2, 4, 8, or other), then every 1024. Without the variable, the same sampled records go
to stderr, which the nonverbose endpoint suppresses; direct runs should use the
dedicated file so execution receipts persist. The same independently opened sink also records profile
match status, reference-bypass reasons, workspace lifecycle, drift diagnostics and
GPU completion receipts. Each record identifies calibration or inference mode. Records include the actual microbatch sequence count, tokens in flight,
workload classification and its source, selected kernel, M/N/K and fusion state.
Native decode uses zero scratch. CUTLASS reports used Q8 global scratch, shared
bytes per block, and separately the endpoint context's reserved scratch capacity.
Unknown reference workspace is `null`. Logical operand read/write footprints
are explicitly estimates excluding scratch and repeated/cache traffic; they are
not measured DRAM transactions. No tensor values or prompt text are logged.
An unwritable requested log fails explicitly.

These records prove host dispatch selection, not completed GPU execution. CUDA
graph replays do not repeat host dispatch; `counts_graph_replays: false` makes this
explicit. Separate `merlin_graph_completion` records in that same file time actual graph
launches with CUDA events. They include a process-unique monitor ID, submission,
generation, workload, process ID, submission timestamp, completion latency and
sampling status. The captured custom-operation count is snapshotted per launch;
a positive count with GPU completion establishes sampled custom work. Uncaptured
selected kernels emit `merlin_kernel_completion` after their CUDA stop event is
ready, sampling the first submission and every 128th per key/kernel with an
eight-slot ring. A busy ring drops a sample rather than waiting. Host dispatch
records and calibration records do not establish full-model GPU execution. Two reusable event
pairs per graph avoid blocking inference. Pending observations at teardown are
reported as incomplete, never counted as completed work. Graph latency is not
attributed to an individual kernel. `MERLIN_CUDA_PROFILE` additionally provides
bounded operation completion timings and disables CUDA graphs for that diagnostic
run. Leave it unset for performance measurements. Direct full-model runs need
their own output, timing and runtime-identity evidence; the endpoint's request
logs and `benchmarks/single_request.py` apply to the historical baseline.

The builder prints timestamped START/DONE records and elapsed time for each
command; redirect stdout and stderr to a persistent build log. The final manifest
records the engine/upstream revisions, compiler, jobs, object reuse and file hashes.
`--reuse-scratch --scratch EXISTING` is restricted to the same pinned upstream
revision and exact CMake source path, with the source restored clean before applying
the new overlay. Preserve and verify the prior overlay before restoring it. It is
not a way to reuse objects across different upstream versions.

Remaining hardware evidence: exercise the combined runtime directly on RTX8000,
compare outputs and logits, then measure long-context speed and memory. The profiler does
not measure physical memory traffic. Small matrix widths reach the MMVQ dispatch
hook; larger widths reach the matrix hook. Both use the same measured policy;
this hook routing is not a claimed performance crossover.

## Custom-engine experiment acceptance

`single_request.py` judges retrieval correctness only; its verdict explicitly does
not prove engine viability. Stock baseline correctness remains useful on its own.
The following evaluator is tied to endpoint request windows and is **not** a
current direct-run acceptance procedure. Direct full-model runs require matched
runtime identity, output/logit, memory and timing evidence before any viability
claim. For the historical endpoint workflow, the offline evaluator is:

```sh
python3 benchmarks/engine_acceptance.py --case /absolute/custom-case --baseline /absolute/stock-case --telemetry /absolute/kernels.jsonl --output /absolute/acceptance.json
```

The custom case needs `runtime-identity.json` collected from the actual endpoint:
`pid`, the full 40-character `engine_revision`, `captured_unix` within five minutes
before the request, `evidence_source: "proc_maps_and_sha256"`, and
`runtime_libraries`. The latter contains exactly five `{name, path, sha256}` rows
for `libllama.so`, `libggml.so`, `libggml-base.so`, `libggml-cpu.so` and
`libggml-cuda.so`. Paths must be absolute mapped library paths under
`merlin-engines-<first seven revision characters>` and hashes are SHA256 of those
files. This evidence is captured by the authorized test operator; the evaluator
never infers a runtime from endpoint timing or source checkout state.

The evaluator requires a matching profile event for that PID/revision, distinct
inference-only GPU completion receipts submitted within the request window, a
correct answer and lower TTFT against a cold baseline with matching request,
model/device alias/configuration and token usage. Optional `--minimum-improvement`
sets the minimum fractional improvement. Missing runtime evidence, calibration-only
activity, host submissions, reference-only completions, unmatched baselines or no
benefit cannot pass. One successful case is labeled `custom_case_improved`, not
full engine viability; logits, batching, context range and stability remain separate.

## Logging overhead control

Logging stays enabled by default. `MERLIN_ENGINE_LOG_DISABLE=1` is an explicit
controlled A/B diagnostic: it bypasses logger formatting, counters and file writes,
while retaining the same kernels, calibration profile, CUDA event sampling and
drift decisions. It emits no execution proof and cannot pass custom-engine
acceptance. Change only this setting between matched logging-overhead trials.

`python3 scripts/measure_logging_cost.py` measures the actual production logger
on the host: 100,000 dispatch calls across five buckets plus one completion per
128 calls, alternating three file-enabled and three disabled runs. This measures
formatting/locking/file flushing only. It does not measure GPU event overhead or
full-model throughput; those require matched direct runs on the target GPU.
