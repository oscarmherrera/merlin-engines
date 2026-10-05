# RTX 8000 isolated custom endpoint, large-context direct requests

On 2026-10-05 an isolated `merlin-engine-endpoint` listened on `.43:18081`
using `/home/oscar/merlin-engine-stage-prefill-q8x8-bank` with
`MERLIN_ENGINE_FORCE_CUSTOM=1`. The versioned CUDA library's SHA-256 was
`5d88715da9d0b121306e424c58d0fa990e8f7da86c9c16734ae2e1857cf67098`.
The stage's copied `manifest.json` did **not** match that library hash, so the
hash and observed dispatch log are the build evidence; the manifest is stale.
The production endpoint on `:8081` was not changed and remained listening.
The isolated process was stopped after the three requests; RTX 8000 VRAM use
returned to its pretest 18,552 MiB.

The endpoint matched the stock baseline's Bonsai 2 27B PQ2 model digest,
`prism-adfffbe` binding, Q8/Q8 KV, 262,144-token context, four slots, 4,096
logical batch, and 2,048 microbatch. Each request was the saved stock JSON
byte-for-byte: cold session, prefix reuse disabled, temperature 0, seed 12345,
128 maximum output tokens. The endpoint returned 10 tokens and the correct
`{"code":"CEDAR-731"}` answer in every run.

| Prompt tokens | Stock TTFT | Custom TTFT | TTFT gain | Custom wall |
| ---: | ---: | ---: | ---: | ---: |
| 61,666 | 105.505 s | 100.688 s | 4.57% | 101.050 s |
| 128,000 | 297.118 s | 285.741 s | 3.83% | 286.254 s |
| 226,000 | 715.553 s | 700.030 s | 2.17% | 700.872 s |

`running_runs.total` was 0 before endpoint load and each request. The custom
endpoint was idle with zero resident KV before the 128K and 226K requests.
VRAM was 18,552 MiB used before loading, 37,106 MiB used after loading
(+18,554 MiB; 11,297 MiB free), and 18,552 MiB after shutdown. The copied
kernel log contains 336 `pq2_cutlass_128x128x128_s1` and 258
`pq2_q8_1_dp4a_warp` dispatch records; all 594 have
`reason=forced_diagnostic`. There were no recorded Prism fallback selections.
The logger recorded far fewer entries on the later requests, so these counts
establish selection but are not complete operator counts or per-request traces.

Raw client `measurement.json` and `response.sse` files for each size, plus the
final `server.log` and `kernels.jsonl`, are local on this Mac at
`/Users/oscar/Projects/merlin-engine-measurements/rtx8000-2026-10-05-custom-context/`.
The stock requests and measurements are under
[`rtx8000-2026-10-03-stock-large-context/`](rtx8000-2026-10-03-stock-large-context/).
No Nsight trace or continuous GPU telemetry was enabled for these runs.
The isolated custom process was co-resident with the idle stock endpoint;
stock baselines used the stock endpoint alone. This and the two-day gap are
comparison limits. There was one request per size, so the timing differences
are observations, not a variability estimate. Ten output tokens do not
establish decode throughput.
The direct-operator prefill gain of about 12% did not become a 10% whole-endpoint
TTFT gain, and the measured endpoint gain narrowed with context depth.
