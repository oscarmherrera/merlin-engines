# RTX8000 kernel diagnosis — 13288e4

The staged runtime was tested directly through `test-backend-ops` on the RTX8000. The existing 17-case runner completed 17/17 workloads with schema-3 receipts and correct numerical references. After these measurements, the owner directed that the endpoint be restored to stock Bonsai and that subsequent engine tests use direct runs. The live endpoint now reports binding `prism-adfffbe` and commit `7f4081f1`; the staged custom runtime is not installed there.

## Decode

At `M=17408, K=5120`, the new `pq2_q8_1_addsub_warp` candidate was numerically accepted for every batch:

| batch | Prism (ms) | Q8 add/sub (ms) | Q8 / Prism |
|---:|---:|---:|---:|
| 1 | 0.086400 | 0.223392 | 2.59x |
| 2 | 0.087776 | 0.396832 | 4.52x |
| 4 | 0.143328 | 0.747520 | 5.21x |
| 8 | 0.250336 | 1.478816 | 5.91x |

The Q8 kernel is 15–19% faster than the previous F32 custom kernel, but it remains slower than Prism. The new path applies scales at Prism's 32-value boundary; the old F32 path's excess scaling is no longer the cause. The direct profiler evidence below identifies the current bottleneck.

## Prefill

At `M=5120, N=2048, K=17408`, all candidates were numerically accepted. The vectorized staging patch reduced latency against both the earlier custom runtime and Prism:

| candidate | 13288e4 (ms) | dd4940e (ms) | Prism (ms) | 13288e4 / Prism |
|---|---:|---:|---:|---:|
| 32×32 stage2 | 16.5845 | 30.1754 | 7.5834 | 2.19x |
| 32×32 stage1 | 13.8387 | 27.5825 | 7.5834 | 1.82x |
| 64×64 stage2 | 8.2084 | 128.9851 | 7.5834 | 1.08x |
| 64×64 stage1 | 8.0559 | 94.2461 | 7.5834 | 1.06x |

The compiler resource comparison confirms the staging change removed the prior spills from all four variants. Register use changed from 198/206 to 96/126 for the 32×32 stage1/stage2 kernels; the 64×64 kernels remain at 255 registers but no longer report stack or spill traffic. The direct profiler comparison below shows that the remaining gap is repeated tile/staging work, not absent Tensor Core execution.

## Direct GPU profile, 2026-10-04

Nsight Compute 2024.1.1 profiled the staged `13288e4` runtime through the direct `test-backend-ops perf` CUDA graph cases. Each invocation selected one kernel launch and replayed it for counters. Robot `running_runs.total` was 0 before every direct run. No model or endpoint was loaded for these profiles. Counts are per kernel launch on identical matrix shapes; profiler replay duration is diagnostic and should not replace the unprofiled medians above. A global-load sector is an L1/TEX request, not a DRAM byte count.

| Decode, M=17408 K=5120 N=1 | Custom Q8 | Prism | Custom / Prism |
|---|---:|---:|---:|
| ALU instructions | 24,127,488 | 4,578,304 | 5.27× |
| LSU instructions | 3,794,944 | 1,653,760 | 2.29× |
| Global-load sectors | 98,834,207 | 28,848,685 | 3.43× |
| Excessive global sectors | 95,047,680 (96%) | 25,067,520 (87%) | 3.79× |
| LG queue stall cycles per issued instruction | 15.1 / 19.2 | 9.8 / 18.5 | 1.54× stall component |
| Achieved occupancy | 93.27% | 85.52% | — |

The custom source assigns one warp to each output row and loads Q8 activation values as 32 individual bytes plus eight individual packed-weight bytes per 32-value group. Prism's `vec_dot_pq2_0_q8_1` loads activation values in eight packed four-byte words, weight symbols in four two-byte words, and uses eight `DP4A` instructions for the integer dot. Nsight reports the custom kernel waiting on the load/store instruction queue for 78.6% of cycles between issued instructions; its ALU pipe is not saturated. Thus the severe decode slowdown is a source-level load/instruction-efficiency problem, amplified by the custom one-warp-per-row mapping, rather than missing Q8 quantization, low occupancy or DRAM bandwidth saturation. Packed `DP4A` arithmetic and a more coalesced load mapping are the evidence-backed changes to evaluate; the speedup from either change has not yet been measured.

| Prefill, M=5120 N=2048 K=17408 | Custom 64×64 stage1 | Prism 128×128 | Custom / Prism |
|---|---:|---:|---:|
| Blocks launched | 2,560 | 640 | 4× |
| Integer Tensor Core instructions | 178,257,920 | 178,257,920 | 1.00× |
| FMA instructions | 597,585,920 | 367,185,920 | 1.63× |
| ALU instructions | 331,509,760 | 93,071,360 | 3.56× |
| LSU instructions | 147,230,720 | 110,433,280 | 1.33× |
| Global-load sectors | 303,237,747 | 123,103,798 | 2.46× |
| Excessive global sectors | 183,828,480 (59%) | 51,527,680 (41%) | 3.57× |
| Achieved occupancy | 24.92% | 24.82% | — |

Prism's pinned `mmq-config-ampere.cuh` selects a 128×128 PQ2 tile with 256 threads; the custom CUTLASS kernel uses a 64×64 tile with 128 threads. The custom kernel performs the same number of integer Tensor Core operations and reaches the same 25% occupancy as Prism, so neither lack of Tensor Core use nor occupancy explains its measured 6% deficit. Four times as many blocks stage overlapping operand tiles, and the custom kernel executes extra scale/staging arithmetic and global-load requests. The 32×32 single-stage candidate expands to 10,240 blocks and 619,642,880 global sectors, versus 2,560 blocks and 310,476,800 sectors for custom 64×64; this matches its much larger latency gap. Nsight shows the 64×64 custom kernel's highest warp stall as a fixed execution dependency (1.7 of 5.0 cycles between issued instructions), so the exact latency share of tile duplication versus scalar scaling still needs a controlled variant. Larger tiles and reducing repeated scaling/staging are the evidence-backed directions; no unmeasured speedup is claimed.

## Decision

Measured dispatch selects Prism for these RTX8000 shapes because the custom candidates are slower. The staged `13288e4` runtime has not been installed or loaded by the endpoint. A direct full-model run, logits comparison, and large-context custom comparison remain outstanding; these graph-case results alone do not establish engine viability. No kernel code was changed during this diagnosis.
