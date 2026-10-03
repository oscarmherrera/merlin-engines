# PQ2 prefill prototype

The experimental runtime routes ordinary contiguous two-dimensional PQ2_0 × F32
matrix products to `pq2_wmma_32x32x128` on NVIDIA compute capability 7.5, for N ≥ 16.
This is an implementation boundary, not a measured performance crossover.
M and N tails are masked; K must be a positive multiple of the format's 128-weight
block. All other supported hardware, formats, layouts, broadcasts and operations
retain their existing algorithms. Decode batches 1, 2, 4 and 8 are separate.
All tensors must have non-null, correctly aligned data inside a CUDA buffer owned
by the dispatch device. Contiguous views are supported using their actual data
span. Host/split buffers and an output overlapping either input are excluded.

The production hook sits in `ggml_cuda_mul_mat` after the existing Hadamard special
operation. It does not alter the graph's separate activation transforms. Fused
normalization/SwiGLU MMQ calls, indexed expert products and multi-device split
products have distinct contracts and are not covered by this first prefill kernel.

Four warps compute one 32×32 output tile. Each K step directly reads 32 packed
128-weight blocks, applies each block's FP16 scale and maps all four two-bit codes
to −1, 0, +1, +2. Only shared FP16 tiles are materialized; WMMA multiplies them with
FP32 accumulation. Output is written as F32 in ggml's column layout.
The block uses 20,480 bytes of static shared workspace and allocates no global
workspace. There is no model-sized expansion, activation-quantization allocation,
per-launch device synchronization or dynamic workspace allocation.

The logger records phase, kernel, device, M/N/K, workspace and fusion state through
the shared `merlin_kernel_log` API. Here workspace denotes shared bytes per CUDA
block, not a global allocation. Kernel selection during CUDA graph capture is a
capture-time record; replay does not invoke host dispatch again.

F32 activations are rounded to FP16. FP16 overflow/underflow, cancellation and
different accumulation order must be evaluated against the CPU reference and
endpoint logits; this prototype does not establish numerical acceptance or speed.
The existing backend test executable gains 12 production graph cases: M=67,
N=16/17/32/65, K=128/5120/17408, covering both output tails and model row lengths.
The runner must verify cases actually execute on CUDA0 and custom-kernel log records
exist; exit zero with skipped cases is not validation.

The implementation follows the CUDA 12.8.1 [WMMA contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#warp-matrix-functions):
32-byte aligned shared tile bases, matching warp-wide calls and valid leading
dimensions. Runtime dispatch and compiled architecture are both checked.

Status: source implementation and production overlay added. CUDA compilation,
GPU numerical validation, endpoint validation and performance measurements pending.
No calibration, measured dispatch, HIP port or SpecKit work is included.
