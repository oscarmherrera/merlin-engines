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

## Fusion 1: sign-fused Hadamard transform on Vulkan (scratch, measured)

The CUDA backend folds the Hadamard sign multiply into its transform kernel
(`fwht_signed`); the Vulkan backend did not, so every rotation cost a separate
multiply dispatch, 258 per token. The port adds a signs binding to `fwht.comp`,
four `fwht_signed*` shader variants, a `MUL + RESHAPE + FWHT-hint MUL_MAT`
fusion predicate and a fused dispatch; the exact change is
[`apply_fwht_signed_vulkan.py`](r9700-vulkan-decode-2026-10-08-evidence/apply_fwht_signed_vulkan.py)
(seventeen exact anchors on the pinned source, built on the Ryzen with the
committed toolchain, library SHA-256 `ded22a5e…`).

| Check | Result |
| --- | --- |
| fork's `MUL_MAT_HADAMARD` CPU-reference cases on Vulkan0 | 27/27 OK, including all `test_fwht_signed` shapes |
| 128-token decode at 30K, fused vs adjacent stock | 3.186 s vs 3.255 s, 2.1% less time (40.2 vs 39.3 tok/s) |
| profiler, dispatches per decode graph | 2,199 → 2,085; 162 of 258 rotations fused |

Removing 162 dispatches saved about 0.55 ms per token, so a small Vulkan
dispatch costs about 3.5 µs unprofiled, far below the 7 to 10 µs the
fence-per-op profiler shows. The 96 unfused rotations are graph-order cases:
the three nodes are not adjacent when the linearization interleaves
independent rotations, one shared rotation feeds three weights through a view,
and some hinted matmuls follow the multiply with no reshape node. A
dispatch-time rule at the multiply could recover most of them for about one
more percent. GPU busy during steady decode was sampled at 92 to 93%, so host
submission gaps are about 2 ms per token.

What this establishes for the 20% target: every small operation on this card
sums to roughly 6 ms of a 25 ms token, so fusion alone cannot reach 20% even
if everything fused; the realistic program is 8 to 12% from fusion, up to 8%
from writing the activation quantization in the transform's epilogue, and
around 7% from the scalar attention path, each a separate kernel change. A HIP
build would inherit the CUDA path's fusions and graph replay; that is
unmeasured and needs a ROCm install first.

## Fusion 2: activation quantize written by the transform (scratch, rejected)

The idea: the transform's subgroup variant also emits the q8_1 blocks of its
output into the backend's shared quantized-activation buffer and records the
reuse memo, so the consuming vector matmul skips its own quantize dispatch;
the memo check was widened so a whole-tensor view of the transform output
counts as the same activation. The change is
[`apply_fwht_q8_vulkan.py`](r9700-vulkan-decode-2026-10-08-evidence/apply_fwht_q8_vulkan.py),
applied after fusion 1 (library SHA-256 `338b347c…`).

| Check | Result |
| --- | --- |
| fork's `MUL_MAT_HADAMARD` cases | 27/27 OK |
| greedy 48-token generation, stock vs fused (`llama-simple`) | byte-identical |
| 128-token decode at 30K, fused vs adjacent stock | 3.531 s vs 3.243 s, **8.9% more time** |
| profiler, GPU op time per decode graph | 31.37 → 30.95 ms (less), transform 10.5 → 14.2 µs, matmuls −2 to −3 µs each |

The quantize pass it removes costs the GPU only 2 to 3 µs per matmul, while
the clustered reductions and lane shuffles the epilogue adds cost 4 to 6 µs
per transform; under the profiler's per-operation fences the trade still
looks slightly positive, in the real pipeline it is a loss. Not retained. The
lesson for the rest of the program: on this card the activation quantize is
not a lever, and a fusion only pays when the fused kernel is no heavier than
the dispatch it removes.
