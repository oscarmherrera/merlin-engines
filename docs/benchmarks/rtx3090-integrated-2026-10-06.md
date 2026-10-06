# RTX 3090 integrated SM86 runtime, direct checks

Engine revision `38d72ef` built cleanly for SM86 from pinned Bonsai source `adfffbe`. The staged runtime manifest names both revisions and `cuda_architectures: [86]`. Its CUDA library was loaded ahead of the existing pinned test binaries on the RTX 3090; `ldd` confirmed the isolated library path. The stock production endpoint remained unchanged.

The integrated library combines the two-head long-context attention selection with Bonsai's packed PQ2/Q8 MMQ and the narrow SM86 asynchronous Q8 buffer change. It also contains the previously selected SM86 decode mapping: custom batches 1, 4 and 8, stock Bonsai at batch 2. No CUTLASS SM86 prefill candidate is selected.

| Direct operator case | Stock Bonsai | Integrated SM86 | Time reduction |
|---|---:|---:|---:|
| Q8 attention, 225,280 KV, 2,048 queries, 6:1 GQA, D256 | 212.686 ms | 171.495 ms | 19.4% |
| PQ2 MMQ, 17408 × 2048 × 5120 | 3323.65 µs | 3175.79 µs | 4.5% |
| PQ2 MMQ, 5120 × 2048 × 17408 | 3418.99 µs | 3265.04 µs | 4.5% |

These are one paired direct graph-replay benchmark per shape, with no full-model inference. The integrated runtime passed the two targeted attention CPU-reference cases (Q8 and F16 KV, 512 KV, 16 queries) and both targeted PQ2 MMQ CPU-reference cases. The MMQ dispatch log selected Bonsai/Prism for both cases. The 10% MMQ operator target is not met; the attention operator clears the proposed 15% threshold. The exact-shape results are in `/Users/oscar/Projects/merlin-engine-measurements/rtx3090-integrated-2026-10-06/`, and the complete SM86 runtime archive is `/Users/oscar/Projects/merlin-engine-measurements/runtime-38d72ef-sm86.tar.zst`.

This integrated operator test alone did not measure full-model speed. The RTX 3090 had only about 4.8 GiB free with its stock endpoint resident, so the endpoint was later temporarily stopped for matched direct model runs and restored afterward. Robot `running_runs.total` was zero before GPU loads. No production binary, fleet config, or SpecKit was changed.

Subsequent engine revision `7d47aaa` adds [grouped Q8 vector attention for decode](grouped-q8-vector-attention-2026-10-06.md). Its clean SM86 library measured 1.378 ms versus 2.050 ms stock on the exact 225K attention operator and passed the 225K CPU-reference case (1/1). The later [matched 225K direct model comparison](rtx3090-matched-225k-128-2026-10-06.md) measured 13.1% less prompt-plus-generation time and 22.4% less timed depth-decode time than stock. Endpoint time-to-first-token and generated-content equality remain unmeasured.
