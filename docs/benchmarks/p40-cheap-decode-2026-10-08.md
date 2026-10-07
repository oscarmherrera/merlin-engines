# P40 decode: CUDA graphs and grouped Q8 attention, 2026-10-08

Two low-cost decode changes were tested directly on the Tesla P40 (`.40`, SM61) to
close the 20% decode target. **Both lose or tie; neither is retained.** The production
endpoint stayed resident and untouched; robot `running_runs.total` was 0 before every
load. All candidate libraries came from the isolated SM61 scratch build on `.30` with the
committed decode header (`a23b5a7e…`), so the P40 ran stock Bonsai decode matrices in
every arm. Each candidate was paired with an adjacent stock run under the card's own
thermal policy; the stock runs agree to 0.04% across the session.

| Arm | CUDA library | Timed 128-token decode after a 30K cache fill | Versus adjacent stock |
| --- | --- | ---: | ---: |
| stock0 / stock1 / stock2 | pinned `prism-adfffbe` | 7.522 / 7.524 / 7.525 s (17.01 tok/s) | — |
| build 2: graphs + grouped attention | `1c80ccba…` | 8.331 s | 10.7% more time |
| build 3: CUDA graphs only | `c493a7e5…` | 7.647 s (kernel log enabled) | 1.6% more time |

`llama-bench -d 30000 -n 128 -p 0 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa 1 -r 1 -t 8`,
model `/model/Ternary-Bonsai-2-27B-PQ2_0.gguf`; one repetition per arm; `ldd` of each
arm records which `libggml-cuda.so.0` was loaded.

## Grouped Q8 vector attention does not transfer to Pascal

The committed grouped kernel (`7d47aaa`) cut one-token Q8 attention by 33% on the RTX
8000 and 3090. Its dispatch gate was widened to compute capability 610 in a scratch
build. The 32,768-KV, GQA 6:1, head-dim 256, one-query Q8/Q8 case passed the CPU
reference (1/1), but the direct operator timing was **1,237.88 µs against 877.26 µs
stock, 41% slower**. Build 3 restored the stock gate and reproduced stock at 878.08 µs.
The mechanism was not profiled; the F32 attention path on this card and the halved
block count are the suspects, not a measured attribution.

## CUDA graphs engage on the P40 and change nothing

The pinned runtime disables CUDA graphs below Volta (`ggml-cuda.cu`,
`cc < GGML_CUDA_CC_VOLTA`). Build 3 exempted 610 only. The engine log from that run
contains `merlin_graph_completion` records (sampled; batch 1; whole-graph elapsed
58.6–60.9 ms), proving graph replay ran. The stock run spends 58.8 ms of wall time per
token (7.525 s / 128), the same as the graph's GPU elapsed time. **The P40 decode step
is GPU-execution-bound, not launch-bound**; the ~1,200 small operations per token cost
their 5–17 µs on the device, so removing host launch gaps buys nothing. The 1.6% loss
includes kernel-log overhead and is not distinguishable from noise with one run.

## What this leaves

Decode on this card is 63% weight matmul at 52% of its 346 GB/s read floor, 20%
attention and 17% small operations. With both cheap levers gone, a 20% decode gain
needs about 30% off the matmul alone, which is the Pascal GEMV design the 10-06 report
called for. No such kernel exists; the ported Turing kernel lost at batch 2 on this card.
Nothing was committed to engine source from this trial; the scratch source on `.30` was
restored from byte backups except that its decode header now matches the committed one
and its test file keeps the 32K case.

Evidence: [`p40-cheap-decode-2026-10-08-evidence/`](p40-cheap-decode-2026-10-08-evidence/)
(per-arm JSON, `ldd`, GPU logs, attention operator logs, the build-3 engine log, and the run
script). Full copies are on the Mac at `/Users/oscar/Projects/merlin-engine-measurements/p40-2026-10-08/`.
