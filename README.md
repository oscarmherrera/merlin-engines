# Merlin Engines

Experimental resident inference engines integrated through `merlin-endpoint`.

Current scope, in order:

1. Quadro RTX 8000, 48 GB, CUDA `sm_75`.
2. RTX 3090, 24 GB, CUDA `sm_86`.
3. Tesla P40, 24 GB, CUDA `sm_61`; tune its non-Tensor-Core path separately.
4. Radeon AI PRO R9700, 32 GB; implement and tune HIP after the NVIDIA work.

Ryzen, unified-memory engines, and streaming MoE development are outside this
phase. The streaming proposal is retained in `docs/` as reference only.

## Status

Repository setup and the [first RTX 8000 baseline](docs/benchmarks/rtx8000-2026-10-03/README.md)
are complete: 22,583 input tokens, 30.60 seconds to first token, 31.11 seconds
total, and the correct retrieved value. The report preserves a checker false
failure caused by an unspecified JSON key; the request generator now specifies it.
**CUTLASS prefill and measured dispatch compiled for RTX8000 (`5c373dd`, 79.538seconds on `.30`, six jobs). GPU validation has not run.**
Decode specializes batches 1/2/4/8. Prefill uses CUTLASS integer Tensor Core tiles
and the existing Bonsai PQ2/Q8 arithmetic. The old FP16 WMMA prototype was slower
than Prism and has been removed; the endpoint was restored to the reference runtime.
See [decode](docs/decode-kernel.md), [prefill](docs/prefill-kernel.md), and
[runtime logging](docs/profiling.md), [measured dispatch](docs/measured-dispatch.md),
and the [requirement checklist](docs/design-conformance.json). Startup calibration
and persisted measured dispatch are compiled. Remaining design gaps
are explicitly listed in the checklist; implementation is not performance proof.
No speedup is claimed.
The owner has deferred SpecKit work until a viable implementation functions.
The owner has also waived the change gate in this subtree repository during
the experimental phase. Correctness tests and performance validation still
apply; the parent Merlin repository's gate is unchanged.

The design authority is
[the resident engine proposal](docs/merlin_bonsai_hybrid_resident_engine.docx).
`runtime.lock.json` records the initial integration baseline. It does not assert
that this revision is the latest upstream release.

## Integration

The parent Merlin branch is `merlin-engines`, created from
`v1.0.x-bug-fixes` at `7f4081f1f8548ac37cf01fa3bc0b507eb3e70f4a`.
This repository is imported at the parent's `merlin-engines/` subtree prefix.
Engine work lives here; endpoint integration changes belong on that parent
branch. Merge into `v1.0.x-bug-fixes`, then Merlin main, after viability and
formalization. Do not import the entire Merlin repository here.

The endpoint already loads a llama.cpp-compatible shared-library set with
`--lib`. Its Prism binding matches the pinned source. Preserve that ABI and
reuse the existing HTTP interface, scheduling, chat template, sampling, and KV
management. Build runtime libraries from the pinned source plus reviewed engine
changes; do not copy a second Go endpoint or change a live library directory.

The pinned fork already provides PQ2/PTQ packed decode, quantized tiled matrix
kernels, and graph-level Hadamard/normalization fusions. The experimental work
must improve measured behavior beyond those implementations.

## Sequence

1. The single RTX 8000 baseline is recorded. Do not repeat reference-only tests.
2. Build custom packed decode and tiled prefill in parallel, integrate both in the
   pinned runtime, and compile the runtime and existing endpoint on `.30`.
3. After authorized deployment, verify actual custom-kernel selection and outputs,
   then measure context sizes and profile attention, recurrent state and transforms.
4. Add measured dispatch keyed by device/software identity, format, and operation
   shape. Persist profiles, invalidate incompatible profiles, retain the safe
   reference for unvalidated shapes, and prove every selected path is reached.
5. Verify logits/outputs, memory bounds, prefix reuse, continuous batching, and
   long-context stability through the actual endpoint. Never materialize a
   model-sized expanded weight copy.
6. Repeat correctness and end-to-end comparisons on 3090, then P40. Port and tune
   for R9700 last. Hardware-specific results do not transfer by assumption.

## First baseline

Prepare a deterministic synthetic retrieval request locally:

```sh
python3 benchmarks/single_request.py --output /tmp/merlin-resident-baseline
```

This writes the request without contacting the fleet. Inspect it before use.
After the owner's fresh GPU go-ahead, execute that exact prepared request:

```sh
python3 benchmarks/single_request.py --output /tmp/merlin-resident-baseline --execute
```

Execution checks robot's `running_runs.total`, endpoint idleness and the pinned
model/binding before the one inference call. No service stop/start, deployment,
context change, or model load is performed. The 300-second client bound cancels
the request by closing its connection; server-side cancellation still needs
verification from the post-request state. Failures preserve diagnostic results
and return nonzero. Token counts come from the endpoint, never a character
estimate. TTFT excludes SSE keepalive comments. Wall time is not decode speed.

Build deployment artifacts on go-dev `.30` from clean pulled commits. Local
tests may run on the Mac. GPU tests require a fresh explicit go-ahead and an idle
fleet. Results under `results/` are ignored until reviewed for inclusion.
