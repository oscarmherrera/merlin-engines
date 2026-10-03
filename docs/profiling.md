# RTX 8000 diagnostic runtime

This is design step 2: operation instrumentation on the pinned Prism core.
Custom kernels, calibrated dispatch, and any performance gain remain unproven.
The first build targets only `sm_75`. The endpoint's existing `--lib` loader is
the integration path; its `prism-adfffbe` binding is unchanged.

Build directly on go-dev `.30`, from a clean pulled engine commit:

```sh
python3 scripts/build_runtime.py --cuda /home/oscar/.local/merlin-toolchains/cuda-12.8.1 --output /home/oscar/merlin-engine-builds/UNIQUE_BUILD --jobs 8
```

The command checks the pinned source revision, adds the profiling overlay, and
builds the shared libraries, `llama-bench`, and `test-backend-ops` for `sm_75`.
The server, web UI, and examples are disabled. The standard attention variants
include Q8/Q8; `GGML_CUDA_FA_ALL_QUANTS=OFF` excludes the extra combinations.
It stages them in `runtime/` with CUDA user-space libraries
and a manifest of revisions, compiler identity, and artifact hashes. It never
loads a model, installs the runtime in a service directory, or starts an endpoint.
On `.30`, use `--jobs 2 --scratch /tmp/UNIQUE_BUILD` to respect its 12 GB process
memory limit and avoid charging temporary objects against the home disk quota.
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

Remaining evidence: exercise the producer on RTX8000, check outputs against the
reference, distinguish attention/transforms/projections at long context, measure
workspace and memory traffic, then choose a custom kernel target. The current
instrument does not measure bytes moved or workspace and does not identify the
internal kernel selected within a fused operation.
