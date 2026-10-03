# Workload metadata contract

`llama_context::process_ubatch` publishes the actual `llama_ubatch` immediately before `graph_compute`. The scoped value contains distinct sequence count (`n_seqs_unq`), token count (`n_tokens`), phase and provenance. Per-token sequence memberships are validated and counted with two fixed 256-element arrays. Coupled tokens may belong to multiple sequences, so sequence count may exceed token count.

Phase is an execution workload classification: 0 means every sequence has one token, 1 means every sequence has multiple tokens, 2 means mixed token multiplicities, and 3 means encoder graph type. A one-token prompt and one-token generation cannot be distinguished by this API; labels do not claim that endpoint semantic distinction. Invalid or absent metadata is unknown and selects Prism.

The exported getter/setter and single thread-local storage definition live in ggml-base. At the pinned Prism revision, `graph_compute` calls `ggml_backend_sched_graph_compute_async`, then `ggml_backend_sched_compute_splits`, then each backend graph submission on the calling host thread. GPU execution is asynchronous; dispatch metadata consumption is synchronous. Nested scopes restore previous values, and host threads remain isolated.

Both packed decode and CUTLASS dispatch consume these fields. Profile V2 keys preserve matrix M/N/K separately from sequence batch, tokens and phase; V1 profiles are rejected. The calibration source is excluded from the key so an explicitly described representative workload matches the same real workload. Telemetry records the source as `llama_ubatch_and_graph_type` or `explicit_calibration_workload`.

The backend test file format appends five structured integers after the case name: `sequence_batch tokens_in_flight phase fusion packed_fixture`. Both numerical `eval` and performance `eval_perf` establish this scope. Built-in cases and rows without this suffix remain unknown/reference. Startup calibration generates decode-shaped batches 1/2/4/8 plus multi-token and mixed workloads for actual model matrix dimensions. Receipt schema 3 verifies all seven fields M/N/K/sequence batch/tokens/phase/fusion. Every requested key must execute and persist a reference plus at least one candidate measurement; missing, old or mismatched evidence fails startup validation.

Fused calibration reuses the pinned backend test's `test_mul_mat_vec_fusion` graph builder with one channel and sample. It creates actual matmul, bias and GLU nodes, preserving backend fusion detection. Model-sized fused rows come from named FFN up/gate/down and attention-output (`attn_output`/`ssm_out`) matrices. Projection-only shapes are excluded, while dimensions shared with FFN weights retain both workloads. The recorded fused inference cases all used batch=1, tokens=1, phase=0. Supported signatures cover bias-only and SwiGLU/GeGLU/SwiGLU-OAI gates, each with and without both biases. An unfused execution cannot satisfy a fused receipt.

GGUF metadata identifies `output.weight` and tied `token_embd.weight` projections without reading weights. Their N=1/2/4/8 rows use independent microbatch token counts from the representative grid, including the observed 66-token case. Other unseen token counts retain the reference path; this finite grid does not claim exhaustive workload coverage.

Eleven small controls (M=67, K=256) cover native N=1/2/4/8 and the seven fused N=1 signatures. Their initializer directly writes packed PQ2 codes 0/1/2/3 with rotating positions and alternating 0.125/0.25 FP16 block scales. Normal float-to-PQ2 initialization cannot produce code 3 because its scale is the maximum absolute input. Activations and biases retain the existing randomized initializer, and calibration rejects zero-energy reference outputs.

Local verification passed:

- `python3 -m unittest discover -s scripts -p 'test_calibration*.py' -v`: seven producer/receipt tests, including the actual shape-generator CLI, including same matrix N with different sequence batches/phases and rejection of mismatched receipts.
- The modified actual `test-backend-ops.cpp` translation unit passed C++17 syntax checking against pinned headers. This does not execute fusion or kernels.
- `python3 scripts/verify_workload.py --source <pinned-source>`: overlay anchors in actual source, both backend entry scopes, and compiled real ubatch classification shared through separate dynamic libraries; thread isolation, nested restoration, unknown, coupled and mixed controls.

These checks do not execute a llama model or CUDA kernels. The integrated GPU test must use explicit workload rows and confirm executed candidate calibration records; backend exit zero alone does not establish custom execution. Graph reuse and replay monitoring consume the same workload tuple through the graph integration.
