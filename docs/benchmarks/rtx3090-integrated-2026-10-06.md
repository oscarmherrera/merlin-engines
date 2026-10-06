# RTX 3090 integrated SM86 runtime, direct checks

Engine revision `38d72ef` built cleanly for SM86 from pinned Bonsai source `adfffbe`. The staged runtime manifest names both revisions and `cuda_architectures: [86]`. Its CUDA library was loaded ahead of the existing pinned test binaries on the RTX 3090; `ldd` confirmed the isolated library path. The stock production endpoint remained unchanged.

The integrated library combines the two-head long-context attention selection with Bonsai's packed PQ2/Q8 MMQ and the narrow SM86 asynchronous Q8 buffer change. It also contains the previously selected SM86 decode mapping: custom batches 1, 4 and 8, stock Bonsai at batch 2. No CUTLASS SM86 prefill candidate is selected.

| Direct operator case | Stock Bonsai | Integrated SM86 | Time reduction |
|---|---:|---:|---:|
| Q8 attention, 225,280 KV, 2,048 queries, 6:1 GQA, D256 | 212.686 ms | 171.495 ms | 19.4% |
| PQ2 MMQ, 17408 × 2048 × 5120 | 3323.65 µs | 3175.79 µs | 4.5% |
| PQ2 MMQ, 5120 × 2048 × 17408 | 3418.99 µs | 3265.04 µs | 4.5% |

These are one paired direct graph-replay benchmark per shape, with no full-model inference. The integrated runtime passed the two targeted attention CPU-reference cases (Q8 and F16 KV, 512 KV, 16 queries) and both targeted PQ2 MMQ CPU-reference cases. The MMQ dispatch log selected Bonsai/Prism for both cases. The 10% MMQ operator target is not met; the attention operator clears the proposed 15% threshold. The exact-shape results are in `/Users/oscar/Projects/merlin-engine-measurements/rtx3090-integrated-2026-10-06/`, and the complete SM86 runtime archive is `/Users/oscar/Projects/merlin-engine-measurements/runtime-38d72ef-sm86.tar.zst`.

There is no measured 3090 full-model prefill, time-to-first-token, or long-context decode gain yet. The RTX 3090 has 24 GiB of VRAM and only about 4.8 GiB free while its stock endpoint is resident, so a second 27B direct model cannot be loaded alongside it. The endpoint must be temporarily taken out of service for a matched direct model test; this report does not infer that result from operator timings. Robot `running_runs.total` was zero before GPU loads. No endpoint, fleet config, or SpecKit was changed.
