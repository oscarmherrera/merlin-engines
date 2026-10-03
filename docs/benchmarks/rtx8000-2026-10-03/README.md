# RTX 8000 initial resident-engine baseline

One cold synthetic retrieval request through the existing `merlin-endpoint`,
2026-10-03. This is the reference runtime, not a new hybrid engine.

| Property | Observed value |
| --- | --- |
| Endpoint commit | `7f4081f1` |
| Runtime binding | `prism-adfffbe` |
| Model | Bonsai 2 27B PQ2, digest in `before.json` |
| KV | Q8 / Q8 |
| Configured context | 262,144 tokens |
| Slots / logical batch | 4 / 4,096 |
| Input / output | 22,583 / 17 tokens, reported by endpoint |
| Time to first generated token | 30.6013 seconds |
| Request wall time | 31.1132 seconds |
| Completion | HTTP 200, `stop`, usage and `[DONE]` present |
| Returned JSON | `{"verification_code":"CEDAR-731"}` |
| Endpoint after request | no active generations, busy slots, queue, or resident KV |

Robot reported zero running runs before the request. Prefix reuse was disabled
and the test named only its own session for cleanup. No endpoint restart, model
load, window change, or deployment occurred. Owner authorization covered this
one request with a 300-second client wall bound.

## Checker correction

The original checker returned failure because it expected `{"code":"CEDAR-731"}`.
The original prompt never specified the JSON key. The actual answer contains the
correct retrieved value, so this is a checker false failure, not an inference
failure. `verdict.json` preserves that original output unmodified. The current
request generator explicitly asks for the key `code`; no GPU retry was performed.
The CLI regression reproduces the observed `verification_code` answer whenever
the request omits that key instruction.

The timing/usage measurement is valid for this one successful request. It is not
a repeated performance estimate, a large-context ceiling result, a decode
throughput result, or a comparison against a custom engine. TTFT includes request
processing and prefill; it is not a kernel timer. Attention/matmul/transform
attribution, 64K–262K behavior, and a sustained decode sample remain unmeasured.

The exact synthetic request, raw stream, pre/post state, original verdict, and
measurement are retained here. They contain no production conversation.
