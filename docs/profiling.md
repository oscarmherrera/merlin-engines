# RTX 8000 custom runtime and diagnostics

Custom packed decode and tiled prefill share the pinned Prism core.
GPU correctness, calibrated dispatch, and any performance gain remain unproven.
The first build targets only `sm_75`. The endpoint's existing `--lib` loader is
the integration path; its `prism-adfffbe` binding is unchanged.

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

On a separately authorized GPU test, set `MERLIN_CUDA_PROFILE` to a **new absolute
file path** when launching the staged runtime through the existing endpoint or
the staged backend test tool. An existing path is refused to preserve evidence.
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

Set `MERLIN_KERNEL_LOG` to a new absolute JSONL path before starting the endpoint.
The runtime records each phase/batch's first 256 host dispatches, then every 1024;
without the variable, its first 16 dispatches per phase/batch go to stderr.
Records contain timestamp, phase, selected kernel, device, M/N/K, fusion state and
workspace bytes. Decode uses zero scratch; prefill reports shared bytes per CUDA
block and allocates zero global scratch. No prompt, activation or weight values
are logged. An unwritable requested log fails explicitly.

These records prove host dispatch selection, not completed GPU execution. CUDA
graph replays do not repeat host dispatch; `counts_graph_replays: false` makes this
explicit. `MERLIN_CUDA_PROFILE` additionally provides bounded operation completion
timings and disables CUDA graphs for that diagnostic run. Leave profiling unset
for performance measurements. The endpoint's existing request/prefill/decode logs
supply ongoing request progress while graphs replay.

The builder prints timestamped START/DONE records and elapsed time for each
command; redirect stdout and stderr to a persistent build log. The final manifest
records the engine/upstream revisions, compiler, jobs, object reuse and file hashes.
`--reuse-scratch --scratch EXISTING` is restricted to the same pinned upstream
revision and exact CMake source path, with the source restored clean before applying
the new overlay. Preserve and verify the prior overlay before restoring it. It is
not a way to reuse objects across different upstream versions.

Remaining evidence: compile both kernels, exercise them on RTX8000, compare outputs
and logits, then measure long-context speed and memory. The operation profiler does
not measure memory traffic. The fixed N=16 prefill boundary is a supported-shape
boundary, not a measured crossover.
