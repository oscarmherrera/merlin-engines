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

The Q8 kernel is 15–19% faster than the previous F32 custom kernel, but it remains slower than Prism. Source inspection explains the remaining work: the native path performs 32 scalar integer symbol additions per 32-value group, while Prism uses eight packed DP4A operations. The new path now applies scales at Prism's 32-value boundary; scaling frequency is no longer the old F32 path's eightfold excess. The exact dominant hardware stall was not measured.

## Prefill

At `M=5120, N=2048, K=17408`, all candidates were numerically accepted. The vectorized staging patch reduced latency against both the earlier custom runtime and Prism:

| candidate | 13288e4 (ms) | dd4940e (ms) | Prism (ms) | 13288e4 / Prism |
|---|---:|---:|---:|---:|
| 32×32 stage2 | 16.5845 | 30.1754 | 7.5834 | 2.19x |
| 32×32 stage1 | 13.8387 | 27.5825 | 7.5834 | 1.82x |
| 64×64 stage2 | 8.2084 | 128.9851 | 7.5834 | 1.08x |
| 64×64 stage1 | 8.0559 | 94.2461 | 7.5834 | 1.06x |

The compiler resource comparison confirms the staging change removed the prior spills from all four variants. Register use changed from 198/206 to 96/126 for the 32×32 stage1/stage2 kernels; the 64×64 kernels remain at 255 registers but no longer report stack or spill traffic. The remaining 32×32 performance gap is therefore not explained by spills. Source counts show the custom tile stages fewer activation copies per tile than Prism's MMQ path and performs scalar packed-weight unpacking before the MMA; the relative cache and instruction-stall cost remains unmeasured.

## Decision

Measured dispatch selects Prism for these RTX8000 shapes because the custom candidates are slower. The staged `13288e4` runtime has not been installed or loaded by the endpoint. A direct full-model run, logits comparison, and large-context custom comparison remain outstanding; these graph-case results alone do not establish engine viability.
