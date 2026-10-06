# Grouped Q8 vector attention at 225K, direct operator evidence

The pinned Bonsai vector attention kernel launched one block per query head for one-token decode. Bonsai 2 27B uses six query heads per KV head, so the engine's new scoped path processes two query heads per block and reuses decoded Q8 V values within that block. It retains Q8/Q8 KV, the original attention arithmetic and mask, and the same packed model. The selection is limited to SM75 or SM86, 256-dimensional heads, 6:1 GQA, one query token, Q8/Q8 KV, zero ALiBi bias and no attention sinks; all other shapes use the pinned path.

The direct `test-backend-ops perf` case used 24 query heads, four KV heads, 225,280 KV tokens and one query token. The same diagnostic executable loaded the stock or isolated candidate CUDA library. Each result is one paired benchmark with internal graph replays, not an independent-run distribution:

| Card | Stock Q8 vector attention | Grouped two-head path | Time reduction |
|---|---:|---:|---:|
| Quadro RTX 8000, SM75 | 3,188.50 µs | 2,080.15 µs | 34.8% |
| GeForce RTX 3090, SM86 | 2,049.55 µs | 1,380.34 µs | 32.7% |

The candidate passed two CPU-reference cases on each card at 512 and 4,096 KV tokens with the exact model head geometry and Q8/Q8 KV (2/2 per card). The test logs do not identify whether either case selected split-KV result combining. A separate 256-thread-block trial on the RTX 8000 lost at 3,470.83 µs versus 3,188.50 µs stock and was discarded. The grouped result supports reusing V within a block as a useful mechanism, but these timings alone do not attribute the gain to a measured hardware counter.

The exact tested `fattn-vec.cuh` diff is committed as `engine/backends/cuda/attention/grouped-q8-vector.patch` in engine revision `7d47aaa`. `scripts/apply_runtime.py` applies it to pinned upstream `adfffbe` with zero fuzz. Applying that patch to an untouched pinned source produced the same SHA-256 as the tested scratch source. The clean SM75 runtime from `7d47aaa` built successfully; clean SM86 compilation and clean-bundle GPU verification are pending.

The pre-existing custom engine's measured 225K 128-token decode was 10.043 s versus 10.390 s stock, a 3.3% reduction. **No full-model token gain for the grouped path is established yet.** Its direct 8000 depth-only model run was interrupted during the 225K cache fill when robot `running_runs.total` changed from zero to one; the partial log contains no benchmark result. The 3090 production endpoint leaves only about 4.8 GiB free, so its matched full-model test also remains unmeasured. Production endpoints stayed stock; no fleet config or SpecKit changed.

Raw operator and correctness logs are on this Mac at `/Users/oscar/Projects/merlin-engine-measurements/rtx8000-225k-128-2026-10-06/vector-attention/` and `/Users/oscar/Projects/merlin-engine-measurements/rtx3090-integrated-2026-10-06/vector-attention/`. The two full-model logs used for the pre-existing 3.3% decode result are in the adjacent RTX 8000 evidence directory.
