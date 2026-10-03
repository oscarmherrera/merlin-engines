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

Four candidates combine 32×32 and 64×64 output tiles with one or two shared stages,
all with K=64 and four warps.
Their respective warp tiles are 16×16×64 and 32×32×64. Both use CUTLASS
`DefaultMmaTensorOp`, integer 8×8×16 instructions and
`TensorOpMultiplicandCrosswise<8,64>` shared layouts. Each staged weight is reused
across tile columns; each staged activation is reused across tile rows.

The two-stage software pipeline prefetches the next packed weight/Q8 tile into
registers, consumes the current shared tile, and writes the next stage before one
CTA barrier. It does not use Ampere-only asynchronous-copy instructions. The
larger tile increases reuse and per-thread accumulator/register demand; whether
that improves latency is a calibration result, not an assumption. Shared
workspace is 8,960 bytes per block for the narrow two-stage tile and 17,920 for the wide tile.
The single-stage variant consumes the tile, barriers before overwrite, then loads
the next tile. Its shared workspace is 4,480 bytes for 32×32 and 16,640 for 64×64;
the wide tile's shared transpose dominates its single-stage allocation. Calibration
compares this lower-storage serial schedule against next-tile prefetch; no speedup
is assumed.
CUTLASS's accumulator iterator supplies fragment coordinates once per block;
that temporary shared storage is then reused by the two input stages.
A padded shared-memory transpose reuses the same allocation for the final
epilogue, letting consecutive lanes store consecutive rows in ggml's output.

Only ordinary contiguous two-dimensional PQ2/F32→F32 products on NVIDIA compute
capability 7.5 are eligible. N≥1 allows all candidates to be compared against
native decode at batches 1/2/4/8 as well as against reference prefill. K must be a
positive multiple of 128. M/N tails are masked. Same-device buffer ownership,
actual tensor data spans, alignment and nonoverlapping output are checked;
contiguous views are supported. Indexed products, broadcasts, host/split buffers
and fused norm/SwiGLU MMQ contracts remain distinct operations outside this
candidate's supported domain.

The endpoint owns stable Q8 scratch addresses from backend initialization until
backend destruction. A matched measured CUTLASS route beating Prism, or explicit
calibration mode, reserves a 128 MiB slab for stream zero on SM75 before inference.
Missing/invalid/reference-only profiles and other architectures reserve nothing.
Before graph capture, additional streams mapped to eligible nodes in the submitted
graph receive their own 128 MiB slab. Eight used streams impose a 1 GiB context
ceiling; ordinary single-stream operation reserves 128 MiB. A new auxiliary stream
can allocate at graph preparation, never during candidate launch or capture.
The backend-free hook synchronizes used streams and frees all slabs. No pointer
is resized or recycled while a captured graph may reference it. Registry access
is mutex protected; a CUDA backend context retains Prism's existing single-host-
submitter lifecycle contract. Each active candidate uses at most 128 MiB. No expanded weight tensor or extra global output scratch
is created by the candidates. Workspace size is
`N * ceil(K / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING / 128 * 144` bytes.
The dispatcher owns any calibration scratch, candidate selection and logging.
Candidate launch performs no synchronization, route logging or calibration.

APIs in `merlin-prefill.cuh`:

- `merlin_prefill_eligible(ctx, weights, input, output)` checks the tensor contract.
- `merlin_prefill_supported(...)` also requires prepared stream scratch.
- `merlin_prefill_workspace_bytes(input)` returns bounded Q8 scratch bytes.
- `merlin_prefill_launch(...)` runs the 32×32 candidate.
- `merlin_prefill_launch_wide(...)` runs the 64×64 two-stage candidate.
- `merlin_prefill_launch_single(...)` and `merlin_prefill_launch_wide_single(...)`
  run the corresponding single-stage candidates.

The overlay calls the measured `merlin_dispatch_prefill` hook; there is no
unmeasured default based on N. Its backend test additions exercise real graph
dispatch for M=67, N=16/17/32/65, K=128/5120/17408. Existing batch-1/2/4/8 cases
cover the small-N competitor. Tests must establish actual candidate execution;
an exit-zero run with skipped cases does not validate the kernels.

Dispatch telemetry reports used global Q8 scratch, total reserved context scratch,
and shared bytes per block separately. Logical read/write bytes are estimates of
operator operand footprints from actual tensor sizes (including fused operands),
exclude internal scratch, and are explicitly not measured DRAM traffic. Reference
workspace/shared usage is unknown (`null`), not a claimed zero.

Status: integrated CUDA compilation passed at engine dd4940e in 84.522 seconds.
The combined scripts suite passes 13 tests, including workspace lifecycle, logger
JSON and graph monitoring. GPU numerical, endpoint, VRAM and performance validation
remain pending. No SpecKit or AMD implementation is included.
