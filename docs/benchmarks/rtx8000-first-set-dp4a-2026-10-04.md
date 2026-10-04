# RTX8000 direct kernel iteration — 2026-10-04

The first incremental decode change replaced scalar PQ2 add/sub with packed signed DP4A while retaining resident PQ2 weights and Bonsai Q8 activation scaling. A 128×64 CUTLASS prefill candidate was also tested, then removed because it lost on both major prefill matrix orientations. The endpoint remained stock Prism. No calibration profile or full 225K run was used.

All timings below are one direct `test-backend-ops perf` run per runtime at CUDA0, with identical shape files, `-o MUL_MAT`, and the same RTX8000. `154ac2b` used forced 64×64 single-stage prefill and Q8 add/sub decode; `a3918ec` used forced 128×64 single-stage prefill and Q8 DP4A decode. The Prism column used the `a3918ec` binary without `MERLIN_ENGINE_FORCE_CUSTOM`. Kernel logs confirmed `forced_diagnostic` selected DP4A and 128×64, rather than falling through to Prism. These are graph-case latencies, not end-to-end model results or statistical confidence intervals.

| M × N × K | Prior custom `154ac2b` | New custom `a3918ec` | Prism in `a3918ec` | Result |
|---|---:|---:|---:|---|
| 17408 × 1 × 5120 | 190.79 µs | 70.35 µs | 73.04 µs | DP4A 2.71× faster than prior custom; 3.7% faster than Prism in this run |
| 17408 × 2 × 5120 | 308.75 µs | 103.25 µs | 70.59 µs | DP4A 2.99× faster than prior custom; 46% slower than Prism |
| 17408 × 4 × 5120 | 582.01 µs | 176.95 µs | 113.80 µs | DP4A 3.29× faster than prior custom; 55% slower than Prism |
| 17408 × 8 × 5120 | 1147.22 µs | 349.01 µs | 194.89 µs | DP4A 3.29× faster than prior custom; 79% slower than Prism |
| 17408 × 2048 × 5120 | 8.371 ms | 9.301 ms | 5.978 ms | 128×64 prefill 11% slower than prior custom |
| 5120 × 2048 × 17408 | 8.511 ms | 9.393 ms | 6.415 ms | 128×64 prefill 10% slower than prior custom |

Direct `test-backend-ops test` with forced custom passed 7/7 small all-code matrix cases: decode batches 1/2/4/8 and prefill widths 16/17/65 with a 67-row tail. A second 7/7 direct test passed the seven supported decode fusion combinations. Kernel logs show the intended custom kernel for every one of these cases. These checks compare outputs to the test tool's CPU reference; they do not validate a full 225K model response.

The final source tree after removing the slower prefill candidate is identical to `c49abb1` (DP4A decode only); the removal commit is `2baef00`. The sixth-candidate expansion made for the unsuccessful prefill experiment was removed with it. The final runtime compiled on `.30` in 83.436 seconds and was staged on `.43` without installing it into the endpoint. A forced direct smoke run passed and logged 1,032 DP4A decode dispatches and 167 original 64×64 CUTLASS prefill dispatches. Its five case times were 69.89, 102.60, 176.13, 346.20 µs for decode batches 1/2/4/8, and 8.311 ms for 17408 × 2048 × 5120 prefill. These match the intermediate decode results and restore the previous prefill behavior.

One matched Nsight Compute profile per prefill kernel, with CUDA graphs disabled for profiling only, explains why the rectangular experiment lost. At 5120 × 2048 × 17408, 64×64 used 2,560 blocks, 16.64 KiB shared memory per block, 255 registers per thread, and 24.89% achieved occupancy. The 128×64 candidate used 1,280 blocks, 33.28 KiB shared memory, the same 255 registers, and 24.84% occupancy. Despite half the blocks, it raised raw barrier-stall counts from 364M to 660M (+81%), MIO-throttle counts from 846M to 1,290M (+53%), and math-pipe-throttle counts from 115M to 674M (5.9×); L1/TEX throughput dropped from 82.2% to 60.2%. This supports excess synchronization and pipeline pressure, rather than an occupancy drop, as the cause of this tile's regression. The profiles do not identify the source line responsible for the existing 64×64 kernel's gap to Prism; the binary lacks CUDA line information.

The first set achieved a measured custom-decode improvement. It did not establish an overall 225K improvement over Prism: prefill remains slower and still dominates much of the forced custom penalty. Batch 1's 3.7% lead is only one measurement; batches 2/4/8 remain materially slower. The rectangular tile's regression was profiled, but the existing 64×64 kernel's extra work relative to Prism has not yet been traced to source lines.
