# RTX8000 forced custom engine — 225K direct run

On 2026-10-04, staged engine `154ac2b` ran directly through `llama-bench` on the Quadro RTX8000 with `MERLIN_ENGINE_FORCE_CUSTOM=1`. No calibration or engine selection profile was loaded. The prompt was 225,000 tokens followed by 10 generated tokens, with Q8/Q8 KV, batch 4096, microbatch 2048, flash attention, and CUDA operation tracing. The run finished in **799.996 s**. The dispatch log reported `mode=forced_diagnostic`, `profile_match=false`, and only custom selections: `pq2_cutlass_64x64x64_s1` and `pq2_q8_1_addsub_warp`, each with `reason=forced_diagnostic`. The stock endpoint was not changed. GPU memory after model load was 35,168 MiB used, 13,235 MiB free.

The existing direct Prism trace used a 226K prompt. The first 109 full 2048-token prefill chunks have the same workload and context depth in both runs, excluding their different final partial chunks:

| Traced operation, first 109 full chunks | Forced custom | Prism | Custom / Prism |
|---|---:|---:|---:|
| All operations | 777.962 s | 696.759 s | 1.117× |
| Operations containing `MUL_MAT` | 275.707 s | 204.441 s | 1.349× |
| `FLASH_ATTN_EXT` | 441.187 s | 432.931 s | 1.019× |

For the 10 generated tokens, custom matrix operations took **55.35 ms/token** versus Prism **23.83 ms/token** (2.32×), and flash attention took **52.75 versus 53.36 ms/token**. Total traced decode was **112.54 versus 81.67 ms/token**. The 1K difference in final prompt depth limits exact decode matching, but matrix operations account for the observed penalty. Full-run untraced wall overhead was about 11.19 s versus Prism's 11.28 s, so logging overhead does not explain the gap.

An interrupted 225K calibration had produced eight prefill-shape receipts. The custom candidates were numerically accepted but slower; for M=17,408, N=2,048, K=5,120, the fastest custom was 7.612 ms versus Prism 5.501 ms. Forcing the custom path independently confirms that calibration was not masking a faster kernel. The [earlier Nsight profile](rtx8000-13288e4-kernel-diagnosis.md) found extra global-load requests and instructions in custom decode, and four times as many blocks with repeated tile/staging work in custom prefill. The full-model trace localizes the resulting slowdown to matrix operations. It does not validate generated-output correctness or isolate the exact share of each source-level mechanism.

The raw forced trace, dispatch log, partial calibration artifacts, and comparison report are on the local MacBook at `/Users/oscar/Projects/merlin-engine-measurements/rtx8000-2026-10-04/`. The staged runtime remains a direct-test artifact; no production engine code or endpoint was deployed.
