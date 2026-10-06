# P40 engine outcomes, issues, and next measurements

The owner approved two experiments: enable the existing packed PQ2/Q8 DP4A
decode candidate on SM61 for batches 1/2, and tune Bonsai's Pascal PQ2 MMQ
tile geometry. The target is at least 20% less whole-phase time than stock
Bonsai, with larger gains desirable. No endpoint was changed and no SpecKit
work was performed.

## Outcome

**The P40 work is partially successful: retain the prefill improvement,
keep stock Bonsai decode, and continue investigating the remaining gap.**
The target of 20% less time for both prefill and decode has not been met.

| Measurement | Stock Bonsai | Candidate | Outcome |
| --- | ---: | ---: | --- |
| Prefill matrix, 17408 × 2048 × 5120 | 21.888 ms | 17.102 ms | 21.9% less time |
| Prefill matrix, 5120 × 2048 × 17408 | 22.289 ms | 17.808 ms | 20.1% less time |
| Full model, 30K prompt + 128 generated tokens | 168.006 s | 148.841 s | 11.4% less time |
| Full model, 128 generated tokens after a 30K cache fill | 7.642 s | 7.845 s | 2.7% more time; no decode win established |

The first three rows use the retained prefill tile. The last row tests the
separate two-row decode candidate against the earlier stock depth baseline.
The prompt-plus-generation row includes generation; it is not a direct
measurement of prefill-only latency or endpoint time to first token.

Source and evidence are committed in **`98f9da7`** in the standalone
`merlin-engines` repository. The retained implementation enables SM61 builds
and tunes Bonsai's existing PQ2 MMQ tile. Packed weights, quantization and
stock P40 decode are preserved. Validation used isolated experimental
libraries; a clean release bundle from the retained commit is still outstanding.
No production endpoint, hardware setting, parent subtree or SpecKit was changed.

## What we tried and what we kept

| Experiment | Correctness | Performance finding | Disposition |
| --- | --- | --- | --- |
| Existing four-row packed DP4A decode on P40 | 15/15 CPU-reference cases passed | One matrix case improved; five regressed | Not retained for P40 |
| Two-row packed DP4A decode | 15/15 CPU-reference cases passed | Batch 1 matrices improved 2.6–7.9%; batch 2 regressed 24–42%; full-model decode did not improve | Not retained for P40 |
| Bonsai prefill tile I64/J128 | 2/2 full-size cases passed | Both matrices improved 15.3–17.3% | Evidence preserved; I128/J64 performed better |
| Bonsai prefill tile I128/J64 | 2/2 full-size and 2/2 relevant tail cases passed | Both matrices improved 20.1–21.9%; full-model combined time improved 11.4% | Retained |

I is the number of output rows and J the number of prompt columns handled
by a tile. The winning change doubles I from 64 to 128 while keeping J at
64, allowing more output rows to reuse the same activation tile. It improves
the existing Bonsai implementation without adding a new quantization format.

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

The initial stock direct 30K prompt plus 128 generated tokens took 168.524
seconds; stock timed 128-token decode after a 30K cache fill took 7.642 seconds.
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

## Confirmed issues and their impact

### 1. The decode port loses efficiency at batch 2

Both custom layouts are slower than stock on every measured batch-2 shape.
The two-row layout beats stock on the measured batch-1 matrices but loses
at batch 2. Therefore the existing kernel cannot simply be enabled on P40
as a general improvement.
The unsuccessful variants and their exact timings remain in the evidence.

The source shows different work distribution: Bonsai cooperates across four
warps on a reduction; the custom kernel uses one warp per row's reduction
and shares activations across rows. That identifies a concrete comparison
to investigate. It does **not** prove that register pressure, instruction
dependency chains, occupancy or memory stalls caused the loss.

### 2. Small decode matrix wins do not establish a token-generation win

The two-row layout improves the three measured batch-1 matrix shapes, yet
its full-model decode result is worse than the recorded stock result.
The isolated shapes do not represent every operation in a generated token.
In the stock trace, pure matrix scopes account for about 20.9 ms/token;
matrix-containing fused or mixed scopes account for roughly another 19.5 ms.
Those fused paths need valid comparative timing before the discrepancy can
be attributed to a specific operation. The current data does not prove
that fusion is the cause.

### 3. Prefill matrix gains are diluted by the rest of the model

The stock diagnostic trace attributes about 60% of prefill to matrix-related
work. Attention contributes about 26%, with other operations taking the rest.
Reducing matrix time by about 20% consequently cannot be assumed to reduce
the whole phase by 20%. The measured 11.4% combined-run gain is consistent
with this limitation, although the two matrix shapes do not constitute a
complete profile of the improved model.

A 20% reduction on the same combined benchmark would require **134.404 s**;
the retained candidate is **14.437 s** above that mark. This is a benchmark
target, not an independently established prefill-only acceptance threshold.

### 4. Two test fixtures initially measured the wrong thing

- The fused performance cases repeated only the final elementwise node.
  Their roughly 2 µs results are excluded from fused matrix comparisons.
  A valid timer for the complete fused work remains to be established.
- The first small tail cases selected smaller stock tiles. Replacement
  cases with N150 and N193 select the changed J64 tile and passed against
  the CPU reference. Only the replacement cases support the new tile's
  boundary correctness claim.

These were measurement problems, not observed numerical failures in the
kernels. They matter because a passing test or a small timing is useful
only when it exercises the intended path.

### 5. Thermal control and memory constrain the next comparisons

Both matched full-model prefill runs reached 84°C. The existing controller
automatically changes the power limit as temperature rises, so a single
195 W or 206 W reading must not be treated as a fixed test configuration.
The fresh stock comparison used the same unchanged thermal policy.
This does not establish a hardware fault or prove thermal throttling caused
the decode regression.

The co-resident 30K prompt-plus-generation runs left 1,562 MiB free. That is
total device usage including the existing endpoint, not isolated model-weight
VRAM. Larger-context capacity under this arrangement remains untested.

### 6. Build and release validation remain distinct

The builder has 12 GiB of RAM. An earlier 12-job compile was killed; the
successful work used two jobs. Moving completed runtime bundles off tmpfs
while preserving their original paths freed about 1.8 GiB of RAM. Subsequent
tile trials rebuilt the affected PQ2 object and relinked the experimental
library instead of repeating the entire build.

The retained overlay reproduces the tested MMQ header byte-for-byte, which
ties the source change to the experiment. It does not replace a clean build
and GPU validation of the final release bundle. Correctness evidence also
does not yet cover generated-text/logit comparisons, repeated-run variance,
long-duration stability or all P40 context and batch sizes.

## Recommended next measurements

These are diagnostic priorities, not additional implemented changes.

1. **Explain the decode discrepancy.** Compare stock and forced custom
   operation traces at the same depth, separating ordinary matrix work,
   fused matrix work, attention and other operations. Prove path selection
   without synchronous per-operation logging in the performance measurement.
2. **Measure the scheduling cost.** Obtain compiled register/spill information
   and available Pascal-compatible stall or memory measurements. Use those
   observations to decide whether changing the custom reduction layout is
   justified; do not assume register pressure is the answer.
3. **Profile the retained prefill build.** Establish which operations now
   dominate and whether attention offers enough measurable savings to close
   the remaining gap. Stock attribution alone is not a profile of the new build.
4. **Validate a retained candidate as a complete artifact.** Build a clean
   runtime from the committed source, repeat the matched direct comparisons
   and extend context/output validation once the candidate meets its target.

The successful prefill change is preserved independently of further work.
The P40 engine remains incomplete against the requested performance target;
the measurements do not justify declaring the NVIDIA engine program finished.
