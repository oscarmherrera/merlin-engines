# Grouped Q8 vector attention at 225K, direct results

The pinned Bonsai vector attention kernel launched one block per query head for one-token decode. Bonsai 2 27B uses six query heads per KV head, so the engine's new scoped path processes two query heads per block and reuses decoded Q8 V values within that block. It retains Q8/Q8 KV, the original attention arithmetic and mask, and the same packed model. The selection is limited to SM75 or SM86, 256-dimensional heads, 6:1 GQA, one query token, Q8/Q8 KV, zero ALiBi bias and no attention sinks; all other shapes use the pinned path.

The direct attention operator case used 24 query heads, four KV heads, 225,280 KV tokens and one query token. The same diagnostic executable loaded the stock, scratch candidate or clean built CUDA library. Each result is one paired benchmark with internal graph replays:

| Card | Stock Bonsai | Scratch grouped | Clean built `7d47aaa` | Clean time reduction |
|---|---:|---:|---:|---:|
| Quadro RTX 8000, SM75 | 3,188.50 µs | 2,080.15 µs | 2,112.13 µs | 33.8% |
| GeForce RTX 3090, SM86 | 2,049.55 µs | 1,380.34 µs | 1,377.62 µs | 32.8% |

The scratch candidate passed two CPU-reference cases per card at 512 and 4,096 KV tokens. The clean built library passed an additional exact-geometry, **225,280-KV CPU-reference case on each card** (1/1 per card). A separate 256-thread-block trial on the RTX 8000 lost at 3,470.83 µs versus 3,188.50 µs stock and was discarded. Reusing V within a block is the code-level mechanism; no hardware counter isolates how much of the observed gain it caused.

At 225,000-token cache depth, direct `llama-bench -d 225000 -n 128` measured 128 generated tokens on the RTX 8000. The cache fill occurred outside the timed generation section:

| Runtime | Timed 128-token decode | Rate | Time reduction vs stock |
|---|---:|---:|---:|
| Stock Bonsai | 10.390 s | 12.319 tokens/s | — |
| Prior custom engine | 10.043 s | 12.745 tokens/s | 3.3% |
| Custom engine with grouped attention | **7.451 s** | **17.179 tokens/s** | **28.3%** |

The full-model grouped run used the isolated scratch CUDA library whose exact source diff was committed. Both clean runtimes were compiled from engine revision `7d47aaa` and pinned upstream `adfffbe`; their libraries reproduced the operator gain and passed the 225K reference check, but the full-model timer has **not** been repeated with a clean bundled library. The earlier grouped run was interrupted during cache fill when a fleet run started, and its partial log has no result. These are one matched run per variant, not independent-run distributions. `llama-bench` does not compare generated text or measure endpoint time-to-first-token.

The tested source diff is committed as `engine/backends/cuda/attention/grouped-q8-vector.patch`; `scripts/apply_runtime.py` applies it to the pinned upstream file with zero fuzz. Applying the patch to untouched pinned source produced the same SHA-256 as the tested scratch source. The complete SM75 and SM86 archives are `/Users/oscar/Projects/merlin-engine-measurements/runtime-7d47aaa-sm75.tar.zst` and `/Users/oscar/Projects/merlin-engine-measurements/runtime-7d47aaa-sm86.tar.zst`. The build manifests record `gpu_validation: NOT RUN` because the build host has no GPU; the isolated operator and reference tests above supplied direct GPU validation afterward.

The RTX 3090 full-model gain remains unmeasured: its production stock endpoint leaves only about 4.8 GiB free, insufficient for a second 27B direct model load. Production endpoints stayed stock; no fleet config or SpecKit changed. Raw logs are on this Mac at `/Users/oscar/Projects/merlin-engine-measurements/rtx8000-225k-128-2026-10-06/` and `/Users/oscar/Projects/merlin-engine-measurements/rtx3090-integrated-2026-10-06/`.
