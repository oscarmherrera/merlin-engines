# Measured resident dispatch

The existing Prism path, native packed decode, and two CUTLASS tiles compete through one registry. A compatible candidate becomes selectable only after finite, nonzero-energy reference output, full-output NMSE at most 5e-4, and a lower measured median than Prism. Ties keep Prism. The native candidate retains direct F32 activations; it is an experiment against Prism's Q8 activation arithmetic, not an assertion of identical arithmetic. CUTLASS candidates preserve their documented D4 quantization contract.

The key includes compute capability, operation, M/N/K, batch, tokens in flight, weight format, phase and fusion signature. At this tensor boundary, `batch` and `tokens_in_flight` both mean the input column count N. They do **not** claim to measure independent endpoint requests or sequence count. Non-indexed, two-dimensional operators are the supported domain. Fused decode records cannot substitute for unfused records.

Set `MERLIN_ENGINE_CALIBRATE=1` and `MERLIN_ENGINE_PROFILE` to the writable profile file only in the startup calibration process. The existing backend test executable drives representative actual model shapes. Each route gets two warm-up launches and five CUDA-event samples on the actual stream. Medians, numerical error and correctness verdicts are persisted after each shape. Repeated occurrences of the same key in that process reuse its result. Profile write failure aborts calibration so an old file cannot masquerade as new results. Calibration disables CUDA graph capture through the integration hook.

Normal inference does not benchmark requests. It reads the profile and dispatches only compatible, numerically accepted candidates with a measured advantage. Missing, malformed, stale, unmatched or rejected entries use Prism. The fingerprint includes the engine, Prism and CUTLASS revisions, device UUID/name/compute capability/SM count/memory, CUDA driver/runtime, NVCC version and host compiler version. Each target GPU needs its own profile file. Atomic profile replacement uses a temporary file in the same directory, flush/fsync and rename.

Calibration reuses two process-lifetime host arrays, bounded at 256 MiB per output / 512 MiB total. It does not expand weights or allocate device output copies. Shapes above that bound remain reference-only. Kernel workspace is separate: native decode uses zero, CUTLASS reports its Q8 scratch, and unknown Prism scratch is logged as unknown. Normal inference does not allocate these validation arrays.

Outside graph capture, an eight-slot CUDA-event ring samples every 128th selected custom submission **per key and kernel**. Queries are asynchronous; a busy ring drops a sample rather than waiting. A five-sample median exceeding 1.5 times the calibrated median invalidates that candidate in memory and attempts to persist the invalidation. If persistence fails, the log explicitly says the invalidation is in memory only. No user-request calibration or forced graph interruption follows an invalidation.

**Remaining graph boundary:** host dispatch logging and this drift sampler do not observe captured graph replays. Invalidating a profile prevents future host selections but does not replace kernels already captured in a CUDA graph. Complete replay drift handling needs graph-execution timing and an explicit graph invalidation/re-capture integration; this implementation does not claim that coverage. Likewise, full model logits, endpoint quality, context sizes, device timings and GPU correctness remain hardware validations, not results of the CPU profile tests.

The portable test compiles the exact profile/selection code used by the CUDA dispatcher and checks medians, stale/malformed cache rejection, unknown/fused key rejection, faster-compatible selection, slower/incorrect candidate rejection, finite/nonzero numerical controls, and healthy versus slow drift windows:

```sh
python3 -m unittest discover -s scripts -p test_dispatch_profile.py -v
```
