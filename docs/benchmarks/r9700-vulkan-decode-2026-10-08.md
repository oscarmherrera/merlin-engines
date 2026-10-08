# R9700 Bonsai decode under Prism Vulkan: budget and levers, 2026-10-08

Direct runs on the Radeon AI PRO R9700 VM (`.41`, RADV GFX1201, Mesa 26.0.3) with
the deployed Prism Vulkan runtime `prism-b10743-adfffbe` (its `libggml-vulkan`
is byte-identical to the Ryzen-built copy, SHA-256 `6b9b2476…`). The P40's
`llama-bench` and `test-backend-ops` binaries were used unchanged; backends resolve
from the library directory. No production endpoint, engine source or SpecKit changed.
Robot `running_runs.total` was 0 before the first load.

## Decode rate and what did not move it

`llama-bench -d <depth> -n 128 -p 0 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa 1`,
one run per arm, GPU alone unless noted.

| Arm | 128-token decode | tok/s |
| --- | ---: | ---: |
| stock, 30K depth (three repeats) | 3.324 / 3.262 s | 38.5 / 39.2 |
| stock, 100K depth | 3.899 s | 32.8 |
| `GGML_VK_DISABLE_COOPMAT=1`, 30K | 3.285 s | 39.0 |
| `RADV_DEBUG=nocompute`, 30K (adjacent stock 39.2) | 3.139 s | 40.8 |
| q4_0 KV / f16 KV / `GGML_VK_DISABLE_MMVQ=1`, 30K, beside the resident endpoint | 4.60 / 4.67 / 4.67 s | 27.8 / 27.4 / 27.4 |
| stock, 30K, beside the resident endpoint | 4.747 s | 27.0 |

**Running beside the resident production endpoint distorts both sides.** With
13 GB of VRAM free, the benchmark's buffers pushed total use past 31 GB and
amdgpu evicted the idle endpoint's 21 GB to host memory (its process stayed up;
uptime and served counts unchanged; no OOM in dmesg). Arms that ran before the
eviction read about 27 tok/s; arms that ran alone read 38.5 to 39.2. KV type and
the integer-dot and cooperative-matrix switches are not levers (within noise).
Attention off is unsupported on this hybrid model. The graphics-queue switch
gained 3.9% in one pair, inside two-run noise of about 2%. Prefill pp2048 was
1,111 tok/s at depth 0 and 669 at 30K, matching the September record.

## Operator timings: the matmuls are already at the floor

`test-backend-ops perf` on the model's real per-token PQ2 shapes, GPU alone,
batch 1: 17408x5120 35.9 µs (617 GB/s of 640), 5120x17408 26.7 µs, 10240x5120
16.7 µs, 5120x6144 14.5 µs, 6144x5120 11.7 µs, 12288x5120 25.9 µs. Batches 4 and
8 fall to 390 to 430 and 245 to 270 GB/s. A 1024x5120 shape read 378 µs in one
file and 5 µs under the harness's other flag, and a 5120x5120 shape read 355 µs
once and 10.7 µs on repeat: those readings are harness artifacts, and neither
shape appears as a separate PQ2 operation in the model's Vulkan graph.

## Per-operation budget of a decode token

`GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1` is a runtime switch in
this fork (the compile option does not exist; a rebuild made on the Ryzen was
unnecessary). Averaged over the last 32 decode graphs at 30K depth, 32 tokens:

| Operation class | ms per token | share | count per token |
| --- | ---: | ---: | ---: |
| PQ2 17408x5120 vector matmul | 6.53 | 20.0% | 128 |
| flash attention, 30K Q8, scalar path | 3.79 | 11.6% | 16 |
| PQ2 5120x17408 with fused add | 3.72 | 11.4% | 64 |
| other PQ2 vector matmuls | 5.37 | 16.4% | 161 |
| about 1,700 small operations at 4 to 12 µs each | 11.2 | 34% | ~1,700 |
| total GPU op time (profiler serializes ops) | 32.7 | 100% | 2,199 |

The small operations are elementwise multiplies (322), the per-head F32
1024x1024 matmuls of the gated-delta-net block (258), norms (177), row gathers
(97), copies and contiguations (160), adds (96), GLU (112), convolutions and
state updates (96). Unprofiled, the big matmuls sum to about 9.4 ms and the
token takes about 26 ms, so roughly a third of every decode token on this card
is per-dispatch cost on tiny kernels, not arithmetic.

## What a 20% decode gain would take

- **Fewer dispatches.** Fusing on the order of 1,000 small operations per token
  inside the Vulkan backend, concentrated in the gated-delta-net block, is the
  only lever large enough on its own. This is GLSL and dispatcher work in the
  pinned fork, not Merlin engine code.
- **Scalar attention.** 237 µs per layer call moves 65 MB, 43% of the bandwidth
  floor; halving it is worth about 7% at 30K and more at depth.
- **In-graph matmul efficiency.** The 17408x5120 product runs at 51 µs inside the
  graph against 36 µs isolated; if that gap is real outside the profiler it is
  worth about 7%.
- **Settings.** `RADV_DEBUG=nocompute` (+3.9%, one pair). The GPU power level
  (`power_dpm_force_performance_level=high`) is reported by the community at
  +13 to 20% decode against -9 to 16% prefill and was not tested; it is a
  system setting. PCIe link power management is a host setting.
- **ROCm.** Not installed in the VM or on any build host. A HIP build would
  bring the CUDA path's graph replay and fusions to the same 2,200 operations;
  community numbers for dense Q4 on this card put HIP ahead on prefill and
  behind on decode, and nothing is measured for PQ2.

Artifacts: Mac `/Users/oscar/Projects/merlin-engine-measurements/r9700-2026-10-08/`
(every arm's JSON, stderr, GPU log, operator logs, the profiler capture and the
run script); compact copies in [`r9700-vulkan-decode-2026-10-08-evidence/`](r9700-vulkan-decode-2026-10-08-evidence/).
Leftovers: the harness directory on `.41`, and an unneeded profiler build on
the Ryzen at `/home/oscar/prism-vulkan-perf-20261008`.
