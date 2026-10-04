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

The standalone repository is at `4bbfd0e`; the parent `merlin-engines` branch
contains that tree locally at `b7958710d`. The standalone commit is pushed. The
parent push is blocked by pre-existing main-repository change-gate failures; its
gate has not been bypassed. The RTX8000 `sm_75` runtime built from `13288e4` on
`.30` and is staged on the RTX8000, but is not installed in the endpoint.

The existing 17-case direct CUDA backend runner passed 17/17 with numerical
references and execution receipts. At the measured `M=17408, K=5120` decode
shape, Q8 custom batches 1/2/4/8 were 2.59–5.91 times slower than Prism; the
Q8 change improved the earlier custom F32 path by 15–19%. At the measured
`M=5120, N=2048, K=17408` prefill shape, the best CUTLASS variant was 1.06
times slower than Prism. These are kernel-shape measurements, not full-model
speed or large-context results. See the [kernel diagnosis](docs/benchmarks/rtx8000-13288e4-kernel-diagnosis.md),
[decode](docs/decode-kernel.md), [prefill](docs/prefill-kernel.md),
[measured dispatch](docs/measured-dispatch.md), [runtime logging](docs/profiling.md),
and [requirement checklist](docs/design-conformance.json). No custom speedup
or viable engine has been demonstrated.

The RTX8000 endpoint was restored to its original Bonsai/Prism binding
`prism-adfffbe` on 2026-10-04. Its model is ready with a 262,144-token configured
context. The 18,430 MiB device usage measured immediately after restoration
is total occupied VRAM, not a separately measured weight allocation. From now
until the owner changes the instruction, engine experiments use **direct
runs of the staged runtime**, not the endpoint. A direct full-model custom run
has not yet been completed. No 3090, P40 or R9700 custom run has begun.
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

1. Stock Bonsai baselines exist at 22,583, 61,666, 128,000 and 226,000 input
   tokens; see the [initial baseline](docs/benchmarks/rtx8000-2026-10-03/README.md)
   and [large-context results](docs/benchmarks/rtx8000-2026-10-03-stock-large-context/baseline-comparison.json).
2. The Q8 packed decode, CUTLASS tiled prefill and measured dispatch are built;
   direct graph cases pass, but the measured custom shapes lose to Prism.
3. Use direct full-model RTX8000 runs against the staged runtime. Record VRAM
   immediately after loading, before inference; then prove custom GPU execution,
   compare outputs/logits and matched speed, and exercise large contexts.
4. Profile the remaining bottlenecks from evidence, improve the custom kernels,
   and repeat the direct measurements. Never materialize a model-sized expanded
   weight copy.
5. After a viable RTX8000 implementation is established, proceed to 3090, P40,
   then R9700. Formalize in SpecKit only after viability, as instructed.

## Historical endpoint baseline

Prepare a deterministic synthetic retrieval request locally:

```sh
python3 benchmarks/single_request.py --output /tmp/merlin-resident-baseline
```

This wrote the historical reference request without contacting the fleet. The
following command documents how that completed endpoint baseline was taken;
it is not the current engine-test procedure:

```sh
python3 benchmarks/single_request.py --output /tmp/merlin-resident-baseline --execute
```

The recorded endpoint baselines are retained for later matched comparisons.
Current custom-engine tests run directly from the staged runtime, with
`running_runs.total` checked before any host load. Direct runs must record the
runtime identity and full-model execution evidence; graph-only tests do not
establish end-to-end or large-context performance.

Build deployment artifacts on go-dev `.30` from clean pulled commits. Local
tests may run on the Mac. GPU tests require an idle fleet and must obey the
owner's current direct-run instruction. Results under `results/` are ignored
until reviewed for inclusion.
