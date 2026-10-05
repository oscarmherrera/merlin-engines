# Merlin resident-engine findings and attempt ledger — 2026-10-05

This records what we actually built and measured against the pinned stock Bonsai/Prism runtime. **A correctness pass is not a speed win.** Direct `test-backend-ops perf` numbers are graph-compute wall time divided by repeated `MUL_MAT` nodes; they include activation quantization and graph execution. Nsight kernel durations are a different measurement. Endpoint time to first token (TTFT) includes the whole request and prefill. The tables do not exchange one measure for another.

## Where the work stands

| Hardware | Best demonstrated result against stock | Status |
| --- | --- | --- |
| RTX 8000, SM75 | Direct decode beats stock by 12.8–38.6% across the measured batches and shapes; direct prefill beats stock by 12.0%/11.9% on two representative products. An isolated forced-custom endpoint completed the 61,666/128,000/226,000-token requests with 4.57%/3.83%/2.17% lower TTFT. | Operator code committed. Isolated endpoint evidence exists; production endpoint remains stock. |
| RTX 3090, SM86 | The custom CUTLASS candidate is slower than stock. A scratch-only Bonsai MMQ async candidate measured 5.7%/5.1% faster; its prefetch variant measured 6.4%/5.8% faster in one matched run. | Below the 10% target; no winning 3090 engine code committed or endpoint/context test. |
| Tesla P40, SM61 | Stock prefill baseline only. | No custom P40 kernel or performance result. |
| R9700 / Ryzen | Not part of this phase. | Not tested. |

The RTX 8000 isolated endpoint used the same Bonsai 2 27B PQ2 model, Q8/Q8 KV, 262,144-token configured context and request bytes as the saved stock baseline. It listened on a separate port; the production endpoint was not replaced. The original design keeps weights packed and resident, uses Bonsai's PQ2/Q8 arithmetic, CUTLASS/CUDA prefill, optimized decode for batches 1/2/4/8 and measured dispatch with a stock fallback. No SpecKit formalization has been done.

## RTX 8000: baseline, first engine and diagnosis

| Attempt | What passed or failed | What it established |
| --- | --- | --- |
| First stock endpoint request, 22,583 input tokens | HTTP completion and the retrieved `CEDAR-731` value passed; TTFT was 30.601 s. An initial checker rejected the JSON key `verification_code` even though the prompt had not specified a key. | The checker failed, not the inference. This request was a stock reference only, not a custom comparison. [Record](rtx8000-2026-10-03/README.md). |
| First custom F32/add-sub decode and small CUTLASS tiles | The direct numerical suite ran, but candidate decode and prefill lost badly to stock. The later Q8 add/sub candidate at M=17,408, K=5,120 took 0.223/0.397/0.748/1.479 ms at batches 1/2/4/8 versus stock 0.086/0.088/0.143/0.250 ms. The best 64×64 prefill candidate took 8.056 ms versus 7.583 ms on M=5,120, N=2,048, K=17,408. | Preserving Q8 quantization improved the previous F32 decode by 15–19%, but did not fix its data path. The 64×64 prefill tile duplicated staging work. [Direct diagnosis](rtx8000-13288e4-kernel-diagnosis.md). |
| Nsight diagnosis of that build | Custom decode executed 5.27× stock ALU instructions and requested 3.43× global-load sectors on one batch-1 shape. Custom 64×64 prefill launched four times as many blocks as stock 128×128, with 2.46× global-load sectors, while executing the **same** number of integer Tensor Core instructions and reaching essentially the same occupancy. | The gap was extra loading, address/staging and scalar instruction work—not missing Tensor Core use. The profiler did not by itself assign exact time shares to each source line. [Counters](rtx8000-13288e4-kernel-diagnosis.md). |
| Initial measured-dispatch/context exercise | Calibration found no faster candidate in the sampled shapes, and the logged context sweeps selected stock. | Those requests measured the reference path. They could not show how fast the custom engine was. An all-reference execution can pass an answer check while adding no engine value. [Dispatch contract](../measured-dispatch.md). |
| Forced `154ac2b` 225K + 10-token direct model run | Custom prefill and decode selections were logged; the run completed in 799.996 s. On the 109 common full prefill chunks, custom all-ops time was 777.962 s versus 696.759 s stock; matrix work was 275.707 versus 204.441 s. Decode traced 112.54 versus 81.67 ms/token at slightly different final prompt depths. | **Performance failed.** Forcing custom removed calibration/fallback as an explanation. Matrix work, not attention or host logging, accounted for most of this revision's loss. It did not establish generated-output correctness. [Forced trace](rtx8000-154ac2b-forced-225k.md). |

The forced 225K loss belongs to the **earlier** 64×64/add-sub revision. It must not be used as the result of the later four-row decode and 128×128 prefill implementation.

## RTX 8000: decode attempts

| Attempt | Result | Finding / disposition |
| --- | --- | --- |
| Packed signed DP4A, retaining resident PQ2 weights and Bonsai Q8 activations | Correct on seven small matrix and seven fusion cases. For M=17,408, K=5,120, batches 1/2/4/8 took 70.35/103.25/176.95/349.01 µs versus the earlier custom 190.79/308.75/582.01/1,147.22 µs. Stock was 73.04/70.59/113.80/194.89 µs. | Large improvement over custom add/sub, but only batch 1 narrowly beat stock; batches 2/4/8 **failed** the stock comparison. [First set](rtx8000-first-set-dp4a-2026-10-04.md). |
| Explicit unpack reuse, then four-lane and two-lane Q8 block layouts | After fixing a staged-library alias that had initially loaded old code, unpack reuse was effectively tied to the baseline. Four-lane and two-lane variants increased latency despite some lower sector counts. | Fewer sectors alone did not mean fewer load instructions or lower latency; these changes were discarded. The scratch measurements are retained in the local `merlin_engines_direct_measurements_2026-10-04.md` memory checkpoint. |
| Two-row-per-warp reuse | Batch 8 tied stock in two pairs (198.14 versus 196.50 µs; 206.46 versus 206.70 µs). A two-iteration unroll lost (205.42 versus 196.40 µs). | Nsight had shown the one-row custom path issuing almost twice stock's L1 load sectors and much higher load-issue throttle. Two rows helped but did not meet the requested margin. The unroll was removed. [Four-row report](rtx8000-four-row-decode-2026-10-04.md). |
| Four-row-per-warp decode, revision `eb84592` | Forced CPU-reference correctness passed 26/26 cases. Across three matrix shapes, batches 1/2/4/8 all beat stock by at least 12.8%. On M=17,408, K=5,120: stock 73.26/71.73/115.84/198.55 µs; custom 55.27/61.38/87.15/137.48 µs. | **Direct decode target passed.** Sharing each activation load across four output rows reduced batch-8 L1 load requests from 3.57 million to 2.00 million and sectors from 104.08 million to 53.95 million versus the two-row variant. Full-model decode throughput remains unmeasured. [Full twelve-pair table and counters](rtx8000-four-row-decode-2026-10-04.md). |

## RTX 8000: prefill attempts

All prefill candidates in this section retain packed PQ2 model weights, Bonsai Q8 activations/scales and integer Tensor Core arithmetic through CUTLASS. The 10% target was judged on **both** representative products, not on a custom-to-custom change.

| Attempt | Correctness/performance result | Finding / disposition |
| --- | --- | --- |
| 128×64 single-stage tile after the first DP4A decode | Passed the small/tail numerical cases, but took 9.301/9.393 ms on the two products versus the earlier custom 8.371/8.511 ms and stock 5.978/6.415 ms. | **Failed** and removed. Its profiled barrier and math-pipe stalls rose despite fewer blocks; merely widening one tile dimension did not improve the pipeline. [First set](rtx8000-first-set-dp4a-2026-10-04.md). |
| 128×128×128 direct staging, full-tile path and address simplification | Intermediate correct versions reduced first-shape latency from about 6.7 ms to about 6.1 ms. Nsight found 64-bit pointer spills in an early full-tile version; hoisting address terms removed those spills and cut integer address instructions. | Address generation and spills were actionable bottlenecks. Four-K unroll was slower; alternate warp maps also lost. The intermediate measurements are retained in the local `merlin_engines_prefill_2026-10-05.md` memory checkpoint. |
| Parity-swapped shared-memory stores | Correct; direct 5.833/5.759 ms versus matched stock 6.154/6.443 ms at that stage. Measured shared bank conflicts fell from 14.52 million to 3.38 million. | A real improvement, but the first product was only 5.2% ahead of stock. Output padding and XOR transpose reduced conflicts further without reducing time, so both were removed. |
| Vectorized `float4` Q8 scale loads | Correct; first/second product 5.733/5.732 ms. Global-load sectors fell from about 123.2 million to 89.8 million in the corresponding profile. | Improved the first shape but remained short of 10% there. A combined packed-weight L1 prefetch improved by only ~0.006 ms and was discarded as noise. |
| Two-stage K=64, four-byte packed-weight loads, 16 warps, plain Q8 64-bit load | K=64 two-stage was correct but took 9.962 ms; four-byte PQ2 load aborted on misaligned addresses because a packed block is 34 bytes; 16-warp attempt aborted before a correctness result; plain Q8 64-bit load was correct but slowed the first shape to 5.689 ms and raised bank conflicts. | Pipeline size, alignment, resources and bank mapping—not just wider loads—determined viability. These variants were removed. |
| Column-fastest CTA grid and alternating Q8 low/high shared stores; committed `3bc4d03` | Correct on both representative products and one odd M/N tail. Final same-binary custom/stock times: **5.49383/6.24214 ms** for M=17,408, N=2,048, K=5,120 and **5.55573/6.30743 ms** for M=5,120, N=2,048, K=17,408. | **Direct prefill target passed: 12.0% and 11.9% faster.** The grid change and corrected Q8 shared-store order were measured wins. The proposed weight-cache reason for grid order remains an inference, not an isolated cache-hit proof. [Final result and logs](rtx8000-prefill-2026-10-05.md). |

## RTX 8000: isolated endpoint test after the kernel wins

The later isolated endpoint forced the optimized custom path with no calibration. Each request used the exact saved cold stock JSON, produced the expected `{"code":"CEDAR-731"}`, returned HTTP 200 and stopped after ten generated tokens.

| Prompt tokens | Stock TTFT | Forced-custom TTFT | Observed improvement |
| ---: | ---: | ---: | ---: |
| 61,666 | 105.505 s | 100.688 s | 4.57% |
| 128,000 | 297.118 s | 285.741 s | 3.83% |
| 226,000 | 715.553 s | 700.030 s | 2.17% |

**Function and measured TTFT passed at all three sizes**, but the whole-request gain is smaller than the ~12% direct prefill-operator gain and narrows with depth. This test did not collect a complete per-operation trace, so it cannot assign the remaining time to attention versus other work in this final build. The log has 336 CUTLASS prefill and 258 DP4A decode records, all forced custom, with no recorded stock fallback; sampling means these are selection evidence, not total operation counts. Used VRAM rose from 18,552 to 37,106 MiB on model load (+18,554 MiB of **device use**, not an isolated weight allocation) and returned to 18,552 MiB after the isolated process stopped.

The comparison has one request per size, a two-day gap between stock and custom, and an idle stock model co-resident during the custom requests. It has no run-to-run spread. Ten output tokens cannot establish sustained decode speed. Raw SSE, measurements, server and kernel logs are local on this Mac at `/Users/oscar/Projects/merlin-engine-measurements/rtx8000-2026-10-05-custom-context/`; no latest-run Nsight or continuous GPU telemetry was captured. The stage manifest was stale, so the library hash and observed dispatch were used for identity. [Endpoint report](rtx8000-custom-endpoint-context-2026-10-05.md).

## RTX 3090: CUTLASS and native MMQ attempts

The RTX 8000 winner did **not** transfer as a speed win to SM86. The first 3090 custom CUTLASS shape was correct but 4.886/4.930 ms versus stock 3.338/3.427 ms. An early apparent `m16n8k32` profiler result was **invalid**: the staged executable loaded an old `.so.0` while only `.so.0.21.0` had been replaced. After updating all three aliases, the real SM86 `m16n8k32` candidate was correct but took 4.724 ms on the first shape. Nsight confirmed the intended 44.56 million `IMMA.16832` instructions—equal to stock—while showing local spills and excess address work.

| CUTLASS trial | Result against stock / finding |
| --- | --- |
| 64×64 tile | Correct; 4.018 ms first shape versus stock 3.338 ms. It removed spills but repeated loads and address calculations. |
| Best rectangular 64×128×128 | Correct on both shapes; **3.656/3.942 ms versus stock 3.372/3.453 ms**, 8.4%/14.2% slower. The dispatch label still said 128×128 even though the launched kernel was 64×128. This source remains uncommitted. |
| 128×64, Q8 8-byte loads, row-fast block order, `const __restrict__`, 128×128/4×4 warps, two-stage K=64, packed-weight prefetch | All tested runnable variants lost to stock. The 128×128 unbounded launch exceeded resources; two-stage K=128 exceeded static shared memory at compile time. `const __restrict__` produced the stock-style load instruction without a timing gain. See [trial table](rtx3090-prefill-2026-10-05.md). |

A short graph and separate quantizer profile ruled out graph length and the shared quantizer as the ~0.29 ms first-shape gap. Profiling a **later** graph launch measured the matrix kernel at 3.197 ms custom versus 2.931 ms stock, with 132.1 versus 61.2 million L1 global-load miss sectors and 23.0% versus 8.4% long-scoreboard stalls. DRAM reads were close (356.5 versus 346.5 MB). Thus the observed gap is inside matrix work and associated with on-chip load waiting/address work; which exact operand contributes each miss is still unproven. [3090 profiles](rtx3090-prefill-2026-10-05.md).

After owner approval to try the **native Bonsai packed-PQ2/Q8 MMQ** on the 3090, scratch builds on `.30` were tested in isolation on `.42`:

| Native MMQ trial | Correctness and direct result | Finding |
| --- | --- | --- |
| Enable existing Q8 `cp.async` double buffer for SM86 PQ2 | 2/2 correct. **3.14762/3.24848 ms** versus stock 3.33916/3.42299 ms: 5.7%/5.1% faster. | Measured 3090 win, short of the 10% target. `cp.async` requires SM80+, so this exact optimization is unavailable on SM75 RTX 8000. |
| Disable stream-K on top of async | Correct; 3.21962/3.29165 ms. | Slower than async alone; discarded. |
| 256×64 tile | Initial 256-thread build failed correctness because half the output rows were unwritten. The corrected 512-thread build passed 2/2 but took 3.73367/3.78979 ms. | Fixed the coverage bug; still slower than stock, so discarded. |
| One-tile-ahead packed-weight L1 prefetch on async | 2/2 correct; **3.12712/3.22613 ms**, 6.4%/5.8% faster than stock in one paired run. | The ~0.02 ms incremental gain over async alone is not established beyond noise. The 10% target remains unmet. Work paused here for reassessment. |

The async profile sampled stalls at packed-weight unpack instructions, but its DRAM read and global sector totals were nearly stock. That observation motivated the prefetch test; it does not prove that prefetch caused the small measured difference. **None of these MMQ experiments has been committed to engine source or tested through a 3090 endpoint.** The uncommitted CUTLASS files in the Mac checkout are an earlier losing candidate, not this MMQ winner. The copied CUTLASS timing logs and 3090 Nsight reports are in [the evidence directory](rtx3090-prefill-2026-10-05-evidence/); MMQ timing results are recorded in the [3090 benchmark note](rtx3090-prefill-2026-10-05.md).

## P40 and remaining limits

The P40 direct stock Bonsai prefill measured **21.913/22.289 ms** on the same two product dimensions. The dispatch log recorded 256 stock selections, 256 unsupported-custom bypasses and no CUTLASS selection. SM61 has no Tensor Cores, and the current CUTLASS path does not run there. No custom P40 prefill or endpoint result exists. [Baseline and logs](p40-prefill-2026-10-05.md).

The measured RTX 8000 operator wins are real for their shapes, and the isolated endpoint results show correct responses with lower TTFT on all three saved requests. They do **not** establish a 10% whole-request win, sustained decode throughput, run-to-run variability, or the final build's operation-level long-context bottleneck. The 3090 has a correct scratch-only native MMQ lead below target; P40 has no candidate. The production RTX 8000 endpoint currently reports the stock `prism-adfffbe` binding; no production engine or SpecKit change followed these trials.
