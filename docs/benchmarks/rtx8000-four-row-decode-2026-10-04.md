# RTX 8000 direct decode: four-row PQ2/Q8 kernel

The forced custom decode kernel beat the pinned stock Bonsai/Prism path by more than 5% at batches 1, 2, 4, and 8 on three representative PQ2 matrix shapes. These are direct `test-backend-ops perf` operator measurements on the RTX 8000, not full-model token throughput or a 225K-context result. The endpoint remained stock throughout.

| Output rows × input width | Batch | Bonsai µs | Custom µs | Improvement |
|---|---:|---:|---:|---:|
| 17408 × 5120 | 1 | 73.26 | 55.27 | 24.6% |
| 17408 × 5120 | 2 | 71.73 | 61.38 | 14.4% |
| 17408 × 5120 | 4 | 115.84 | 87.15 | 24.8% |
| 17408 × 5120 | 8 | 198.55 | 137.48 | 30.8% |
| 5120 × 17408 | 1 | 66.55 | 52.39 | 21.3% |
| 5120 × 17408 | 2 | 70.71 | 60.03 | 15.1% |
| 5120 × 17408 | 4 | 132.45 | 81.34 | 38.6% |
| 5120 × 17408 | 8 | 229.61 | 160.42 | 30.1% |
| 5120 × 5120 | 1 | 25.56 | 18.16 | 29.0% |
| 5120 × 5120 | 2 | 26.23 | 22.86 | 12.8% |
| 5120 × 5120 | 4 | 40.39 | 29.07 | 28.0% |
| 5120 × 5120 | 8 | 66.70 | 54.81 | 17.8% |

Each custom/stock pair ran the same shape file through the same staged runtime, with only `MERLIN_ENGINE_FORCE_CUSTOM=1` selecting the custom path. The performance log recorded thousands of graph replays per shape; the table is one paired run per shape, not a distribution across independent sessions. The dispatch log recorded the forced `pq2_q8_1_dp4a_warp` kernel. The final staged runtime was `/home/oscar/merlin-engine-stage-batch-four-row-b1` on the RTX 8000 host. No calibration or endpoint deployment was involved.

The kernel keeps resident PQ2 weights and Bonsai's Q8 activation/DP4A arithmetic. One warp now computes four output rows, sharing each activation load. Unfused batch 1 uses the same four-row mapping; fused batch 1 retains its fusion-capable kernel, with byte-permute PQ2 unpack. The earlier two-row batch-8 variant tied Bonsai: 198.14 vs 196.50 µs in one pair and 206.46 vs 206.70 µs in another. An attempted two-iteration loop unroll lost at batch 8 (205.42 vs 196.40 µs) and was removed.

Nsight Compute measured the batch-8 mechanism. Four-row custom versus two-row custom versus stock: L1 global-load requests were 2,001,920 / 3,568,640 / 3,568,640; load sectors were 53,945,889 / 104,076,474 / 104,036,978; load-issue throttle was 2.03 / 16.36 / 7.78 cycles per issued instruction. The four-row kernel used 128 registers per thread versus 64 in the two-row variant. The 44% request and 48% sector reductions outweighed the register increase. These are profiled counters, separate from the unprofiled timing table.

Forced direct CPU-reference correctness passed 26/26 cases on the final runtime: all twelve full-size decode shapes, seven small/tail cases, and seven fusion cases. The 26-case dispatch log recorded 23 native decode and three CUTLASS prefill invocations. A full model run, long-context behavior, and end-to-end token speed remain unmeasured for this revision. The result establishes a direct decode-kernel win on the measured RTX 8000 shapes only.
