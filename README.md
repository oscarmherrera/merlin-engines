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

Direct tests now show improvements over stock Bonsai on the RTX8000 and
RTX3090. P40 tuning is in progress; R9700 work has not started.

| Device | Prompt + 128-token time reduction | Depth-only decode time reduction |
| --- | ---: | ---: |
| RTX8000, 225K | 18.4% | 28.3% with grouped attention |
| RTX3090, 225K | 13.1% | 22.4% with grouped attention |
| P40, 30K | 11.4% with the retained MMQ tile | No demonstrated gain; native trial rejected |

These are direct model measurements, one run per variant, not endpoint TTFT
or generated-text comparisons. The RTX8000 whole-run measurement predates
its grouped decode change. Production endpoints remain on stock Bonsai.
The current instruction is to use direct runs of isolated staged runtimes.

See the [RTX8000 result](docs/benchmarks/rtx8000-matched-225k-128-2026-10-06.md),
[grouped decode result](docs/benchmarks/grouped-q8-vector-attention-2026-10-06.md),
[RTX3090 result](docs/benchmarks/rtx3090-matched-225k-128-2026-10-06.md),
[P40 experiments](docs/benchmarks/p40-experiments-2026-10-06.md), and
[design checklist](docs/design-conformance.json). P40 prefill matrices improved
20.1–21.9%, but the target of 20% less whole-phase prefill and decode time
has not been reached. The builder accepts CUDA architectures 61, 75 and 86;
SM61 retains stock decode and uses the tuned Bonsai MMQ prefill path.

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
   direct graph cases and matched model measurements are documented above.
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
