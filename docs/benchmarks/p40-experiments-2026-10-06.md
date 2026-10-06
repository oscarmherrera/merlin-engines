# P40 approved decode and prefill experiments, 2026-10-06

The owner approved two experiments: enable the existing packed PQ2/Q8 DP4A
decode candidate on SM61 for batches 1/2, and tune Bonsai's Pascal PQ2 MMQ
tile geometry. The target is at least 20% less whole-phase time than stock
Bonsai, with larger gains desirable. No endpoint was changed and no SpecKit
work was performed.

## Measurement conditions

- Tesla P40 on `.40`; production stock endpoint remained resident and idle.
- Pinned runtime `adfffbe41b2cabcd51fff326ab045662265062bb`.
- Engine overlay from committed `286d3f3`, excluding unrelated dirty SM86
  CUTLASS experiments. Decode eligibility and tile variants were applied in
  an isolated scratch build on `.30`.
- Direct `test-backend-ops` comparisons, logging disabled for performance.
  Forced custom decode, no calibration; correctness dispatch logs prove the
  DP4A candidate was selected.
- The operator timer includes quantization and graph execution. CUDA graphs
  are disabled by the runtime on this architecture.
- One timing invocation per layout; each invocation repeats the operator.
  Percentages below mean reduced elapsed time, not increased throughput.

Raw logs, test shapes and exact scratch patches are on the Mac under
`/Users/oscar/Projects/merlin-engine-measurements/p40-2026-10-06/`.
The compact evidence needed to reproduce these conclusions is also retained
in [the report evidence directory](p40-experiments-2026-10-06-evidence/).

## Decode: correctness passes, performance target not met

Both the four-row and two-row warp layouts passed all 15 CPU-reference
cases, including tail rows, all packed codes, fused operations and full model
matrix dimensions. Only the row grouping changed between candidates.

| M × K | Batch | Stock µs | Four-row µs | Two-row µs |
| --- | ---: | ---: | ---: | ---: |
| 17408 × 5120 | 1 | 140.45 | 125.90 | 129.35 |
| 17408 × 5120 | 2 | 139.68 | 161.24 | 173.59 |
| 5120 × 17408 | 1 | 128.02 | 138.23 | 123.26 |
| 5120 × 17408 | 2 | 136.57 | 158.02 | 193.68 |
| 5120 × 5120 | 1 | 44.64 | 47.36 | 43.49 |
| 5120 × 5120 | 2 | 47.55 | 53.31 | 67.49 |

Four-row loses five of six cases. Two-row saves 2.6–7.9% at batch 1 but
costs 24.3–41.9% more time at batch 2. Neither is an acceptable general
P40 decode improvement. These measurements do not prove whether the cause
is register pressure, instruction throughput, or memory scheduling.

The forced two-row full-model depth-30K test took **7.845 s** for 128
generated tokens, versus the earlier uninstrumented stock **7.642 s**:
2.7% more time, not a demonstrated improvement. This is one candidate run
against the earlier stock depth run, not repeated adjacent pairs. The new
combined stock control reproduced the earlier combined time to about 0.3%.
The timed model runs disable dispatch logging; custom selection is directly
proven by the operator correctness logs, not by per-token model receipts.
The experimental SM61 decode eligibility is not retained in the build.

Source inspection confirms stock uses four warps cooperating on a reduction
while the custom kernel assigns each row's reduction to one warp and shares
activation loads across rows. This is a verified implementation difference,
not a measured attribution of the slowdown. Compiled register usage and
stall counters were not collected; the builder lacks `cuobjdump`.

The appended fusion-513 and fusion-2 **performance** cases are invalid for
fused matrix timing: the harness repeats only the final elementwise node,
producing ~2 µs measurements. Those rows are excluded. Their correctness
checks execute the graph, but full-model timing is needed for fused-path
performance.

## Prefill: tile reuse improves both measured matrix shapes

Both candidate layouts passed both full-size CPU-reference cases. The
stock tile is I64/J64; candidates use I64/J128 or I128/J64 with the same
256 threads, occupancy launch bound 2, K256 and packed PQ2/Q8 arithmetic.
Host and device configuration overrides apply only to SM61 and PQ2.
The retained I128/J64 layout also passed 2/2 tail cases, M131/N150/K256
and M67/N193/K512. The earlier N130/N65 cases passed but selected smaller
stock tiles, so they are not evidence for the new tile's boundaries.

| M × N × K | Stock ms | I64/J128 ms | Reduction | I128/J64 ms | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17408 × 2048 × 5120 | 21.888 | 18.111 | 17.3% | 17.102 | 21.9% |
| 5120 × 2048 × 17408 | 22.289 | 18.884 | 15.3% | 17.808 | 20.1% |

The I128/J64 layout is the operator winner. The change increases reuse of
the Q8 activation tile across output rows; it does not expand resident
weights or change quantization. The observed timing establishes the gain,
but does not isolate every hardware bottleneck.

## Whole-model baseline and remaining uncertainty

Stock direct 30K prompt plus 128 generated tokens takes 168.524 seconds;
stock timed 128-token decode after a 30K cache fill takes 7.642 seconds.
Both are uninstrumented, one-run measurements with batch4096, microbatch2048,
Q8/Q8 KV, FA enabled and the same 27B model.

A separate diagnostic trace attributes about 59.6% of prefill to
matrix-containing scopes, 25.9% to attention, and the rest to other work.
Steady decode is approximately 60.8 ms/token: 40.36 ms matrix-containing
scopes, 12.72 ms attention, and 7.7 ms other work. These profiled timings
are attribution evidence, not uninstrumented speed controls.

Consequently, a 20% whole-phase reduction supplied only by matrices needs
about 34% less matrix time for prefill or 30% for decode. A 20–22% matrix
gain alone cannot establish that target. Other unresolved issues are
fused decode performance, consistent batch-2 efficiency and sustained
full-model performance under the card's thermal policy. Stock previously
reached 84°C. The active `asus-p40-thermal` controller varies the power
limit with temperature; the earlier 195 W reading was a snapshot, not a
fixed configuration. No power, cooling or thermal policy was changed.

The isolated I128/J64 full-model run completed at **148.841 seconds** for
30K prompt + 128 tokens, with stock decode selected, versus a fresh stock
control of **168.006 seconds**: **11.4% less whole-run compute time**.
All benchmark JSON settings match except timestamps and measured results.
Each variant was measured once. This timer includes generation and does not
independently measure prefill-only latency or endpoint TTFT. Total GPU memory
peaked at 21,345 MiB including the resident endpoint (1,562 MiB free),
matching stock; sampled temperature peaked at 84°C in both runs. The
generated overlay header is SHA256-identical to the header compiled for
this trial (`3468783872e011ab0738bf2689abd1406ac74baea61de33dda65f40599ca30ba`).
The retained source adds SM61 to the runtime builder and this MMQ override;
the tested artifact is an isolated experimental library, not a newly built
clean release bundle. No 20% whole-phase success is claimed.
