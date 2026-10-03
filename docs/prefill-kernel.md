# CUTLASS resident PQ2 prefill candidates

These candidates implement packed PQ2_0 × Q8 integer matrix products with CUTLASS
4.8.0 (commit `098de2a652cf8f00fd70b2df54051c7eccbb855a`) SM75 Tensor Core operators.
They replace the removed FP16 WMMA prototype. No FP16 activation conversion or
expanded floating-point weight matrix is used.

The caller reuses Prism's exact `quantize_mmq_q8_1_cuda` function, with PQ2's D4
layout: signed 8-bit activation values and one FP32 scale per 32 activations.
Each 128-weight PQ2 block retains its original FP16 scale and all two-bit codes
map to −1, 0, +1, +2. Integer sums reset every 32 elements; accumulation applies
`float(dot) * weight_scale * activation_scale` in that order, as in the existing
PQ2 MMQ arithmetic. Floating-point accumulation order can differ from Prism's
parallel reduction and must pass the measured dispatcher's numeric comparison.
The graph's Hadamard handling remains before dispatch.

The candidates have 32×32 and 64×64 output tiles, both with K=64 and four warps.
Their respective warp tiles are 16×16×64 and 32×32×64. Both use CUTLASS
`DefaultMmaTensorOp`, integer 8×8×16 instructions and
`TensorOpMultiplicandCrosswise<8,64>` shared layouts. Each staged weight is reused
across tile columns; each staged activation is reused across tile rows.

The two-stage software pipeline prefetches the next packed weight/Q8 tile into
registers, consumes the current shared tile, and writes the next stage before one
CTA barrier. It does not use Ampere-only asynchronous-copy instructions. The
larger tile increases reuse and per-thread accumulator/register demand; whether
that improves latency is a calibration result, not an assumption. Shared
workspace is 8,960 bytes per block for the narrow tile and 17,920 for the wide tile.
CUTLASS's accumulator iterator supplies fragment coordinates once per block;
that temporary shared storage is then reused by the two input stages.
A padded shared-memory transpose reuses the same allocation for the final
epilogue, letting consecutive lanes store consecutive rows in ggml's output.

Only ordinary contiguous two-dimensional PQ2/F32→F32 products on NVIDIA compute
capability 7.5 are eligible. N≥1 allows both candidates to be compared against
native decode at batches 1/2/4/8 as well as against reference prefill. K must be a
positive multiple of 128. M/N tails are masked. Same-device buffer ownership,
actual tensor data spans, alignment and nonoverlapping output are checked;
contiguous views are supported. Indexed products, broadcasts, host/split buffers
and fused norm/SwiGLU MMQ contracts remain distinct operations outside this
candidate's supported domain.

The existing backend pool supplies reusable Q8 activation scratch, bounded at
128 MiB per invocation. No expanded weight tensor or extra global output scratch
is created by the candidates. Workspace size is
`N * ceil(K / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING / 128 * 144` bytes.
The dispatcher owns any calibration scratch, candidate selection and logging.
Candidate launch performs no synchronization, route logging or calibration.

APIs in `merlin-prefill.cuh`:

- `merlin_prefill_supported(ctx, weights, input, output)` checks eligibility.
- `merlin_prefill_workspace_bytes(input)` returns bounded Q8 scratch bytes.
- `merlin_prefill_launch(...)` runs the 32×32 candidate.
- `merlin_prefill_launch_wide(...)` runs the 64×64 candidate.

The overlay calls the measured `merlin_dispatch_prefill` hook; there is no
unmeasured default based on N. Its backend test additions exercise real graph
dispatch for M=67, N=16/17/32/65, K=128/5120/17408. Existing batch-1/2/4/8 cases
cover the small-N competitor. Tests must establish actual candidate execution;
an exit-zero run with skipped cases does not validate the kernels.

Status: CUDA source written; compiler, GPU numeric, endpoint and performance
validation pending. No SpecKit or AMD implementation is included.
