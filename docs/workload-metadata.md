# Workload metadata contract

`llama_context::process_ubatch` publishes the actual `llama_ubatch` immediately before `graph_compute`. The scoped value contains distinct sequence count (`n_seqs_unq`), token count (`n_tokens`), phase and provenance. Per-token sequence memberships are validated and counted with two fixed 256-element arrays. Coupled tokens may belong to multiple sequences, so sequence count may exceed token count.

Phase is an execution workload classification: 0 means every sequence has one token, 1 means every sequence has multiple tokens, 2 means mixed token multiplicities, and 3 means encoder graph type. A one-token prompt and one-token generation cannot be distinguished by this API; labels do not claim that endpoint semantic distinction. Invalid or absent metadata is unknown and selects Prism.

The exported getter/setter and single thread-local storage definition live in ggml-base. At the pinned Prism revision, `graph_compute` calls `ggml_backend_sched_graph_compute_async`, then `ggml_backend_sched_compute_splits`, then each backend graph submission on the calling host thread. GPU execution is asynchronous; dispatch metadata consumption is synchronous. Nested scopes restore previous values, and host threads remain isolated.

Both packed decode and CUTLASS dispatch consume these fields. Profile V2 keys preserve matrix M/N/K separately from sequence batch, tokens and phase; V1 profiles are rejected. The calibration source is excluded from the key so an explicitly described representative workload matches the same real workload. Telemetry records the source as `llama_ubatch_and_graph_type` or `explicit_calibration_workload`.

The backend test file format appends three structured integers after the case name: `sequence_batch tokens_in_flight phase`. Both numerical `eval` and performance `eval_perf` establish this scope. Built-in cases and rows without this suffix remain unknown/reference. Startup calibration generates decode-shaped batches 1/2/4/8 plus multi-token and mixed workloads for actual model matrix dimensions. Receipt schema 2 verifies all six fields M/N/K/sequence batch/tokens/phase; an old receipt or a receipt for a different workload cannot satisfy startup validation.

Local verification passed:

- `python3 -m unittest discover -s scripts -p 'test_calibration*.py' -v`: five producer/receipt tests, including same matrix N with different sequence batches/phases and rejection of mismatched receipts.
- `python3 scripts/verify_workload.py --source <pinned-source>`: overlay anchors in actual source, both backend entry scopes, and compiled real ubatch classification shared through separate dynamic libraries; thread isolation, nested restoration, unknown, coupled and mixed controls.

These checks do not execute a llama model or CUDA kernels. The integrated GPU test must use explicit workload rows and confirm executed candidate calibration records; backend exit zero alone does not establish custom execution. Graph reuse and replay monitoring consume the same workload tuple through the graph integration.
