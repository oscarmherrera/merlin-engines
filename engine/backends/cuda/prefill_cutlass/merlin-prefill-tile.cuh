#pragma once

#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

namespace merlin_prefill {

template<int Tile>
struct tile_config {
    static constexpr int k = 64;
    static constexpr int threads = 128;
    using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<8, k>;
    using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<8, k>;
    using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        cutlass::gemm::GemmShape<Tile/2, Tile/2, k>, cutlass::gemm::GemmShape<8, 8, 16>,
        int8_t, LayoutA, int8_t, LayoutB, int32_t, cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAddSaturate>::Type;
    struct alignas(128) stage {
        int8_t a[Tile*k];
        int8_t b[Tile*k];
        float da[Tile];
        float db[2][Tile];
    };
    struct prefetched {
        uint16_t packed[Tile*k/(8*threads)];
        uint32_t q8[Tile*k/(4*threads)];
        float da;
        float db[2];
    };
};

template<int Tile, int Stages>
union tile_storage {
    static_assert(Stages == 1 || Stages == 2, "unsupported pipeline depth");
    typename tile_config<Tile>::stage stages[Stages];
    float output[Tile*(Tile+1)];
};
constexpr size_t shared_bytes = sizeof(tile_storage<32, 2>);
constexpr size_t shared_bytes_wide = sizeof(tile_storage<64, 2>);
constexpr size_t shared_bytes_single = sizeof(tile_storage<32, 1>);
constexpr size_t shared_bytes_wide_single = sizeof(tile_storage<64, 1>);
static_assert(shared_bytes_single == 4480 && shared_bytes_wide_single == 16640, "single-stage workspace changed");
static_assert(shared_bytes == 8960 && shared_bytes_wide == 17920, "shared workspace changed");

template<int Tile>
static __device__ __forceinline__ typename tile_config<Tile>::prefetched prefetch(
        const block_pq2_0 * weights, const block_q8_1_mmq * input,
        int64_t row0, int64_t col0, int64_t m, int64_t n, int64_t k, int64_t inner0) {
    typename tile_config<Tile>::prefetched next{};
    const int tid = threadIdx.x;
    const int64_t block = inner0 / QK_PQ2_0;
    const int offset = inner0 % QK_PQ2_0;
#pragma unroll
    for (int i = 0; i < Tile*64/(8*128); ++i) {
        const int index = tid + i*128;
        const int row = index / 8;
        const int byte = (index % 8)*2;
        next.packed[i] = row0 + row < m ? *reinterpret_cast<const uint16_t *>(
            weights[(row0 + row)*(k/QK_PQ2_0) + block].qs + offset/4 + byte) : 0x5555;
    }
#pragma unroll
    for (int i = 0; i < Tile*64/(4*128); ++i) {
        const int index = tid + i*128;
        const int col = index / 16;
        next.q8[i] = col0 + col < n ? *reinterpret_cast<const uint32_t *>(
            input[block*n + col0 + col].qs + offset + (index%16)*4) : 0;
    }
    if (tid < Tile) {
        next.da = row0 + tid < m ? __half2float(weights[(row0 + tid)*(k/QK_PQ2_0) + block].d) : 0.0f;
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            next.db[group] = col0 + tid < n ? input[block*n + col0 + tid].d4[offset/32 + group] : 0.0f;
        }
    }
    return next;
}

template<int Tile>
static __device__ __forceinline__ void stage_tile(typename tile_config<Tile>::stage & dst,
                                                  const typename tile_config<Tile>::prefetched & next) {
    const typename tile_config<Tile>::LayoutA layout_a(64);
    const typename tile_config<Tile>::LayoutB layout_b(64);
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < Tile*64/(8*128); ++i) {
        const int index = tid + i*128;
        const int row = index / 8;
        const int inner = (index%8)*8;
        // Prism's packed lookup preserves all four PQ2 symbols, including +2.
        const uint32_t qe = __byte_perm(0x020100ff, 0x020100ff, next.packed[i]);
        const uint32_t qo = __byte_perm(0x020100ff, 0x020100ff, next.packed[i] >> 2);
        *reinterpret_cast<uint32_t *>(dst.a + layout_a({row, inner})) = __byte_perm(qe, qo, 0x5140);
        *reinterpret_cast<uint32_t *>(dst.a + layout_a({row, inner + 4})) = __byte_perm(qe, qo, 0x7362);
    }
#pragma unroll
    for (int i = 0; i < Tile*64/(4*128); ++i) {
        const int index = tid + i*128;
        *reinterpret_cast<uint32_t *>(dst.b + layout_b({(index%16)*4, index/16})) = next.q8[i];
    }
    if (tid < Tile) {
        dst.da[tid] = next.da;
        dst.db[0][tid] = next.db[0];
        dst.db[1][tid] = next.db[1];
    }
}

template<int Tile, int Stages>
static __global__ void pq2_cutlass(const block_pq2_0 * weights, const block_q8_1_mmq * input,
                                  float * output, int64_t m, int64_t n, int64_t k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using Config = tile_config<Tile>;
    using Mma = typename Config::Mma;
    using Instruction = typename Mma::InstructionShape;
    using AccumulatorIterator = typename Mma::IteratorC;
    using Delta = typename AccumulatorIterator::OpDelta;
    constexpr int mma_rows = Mma::MmaIterations::kRow;
    constexpr int mma_cols = Mma::MmaIterations::kColumn;
    static_assert(Instruction::kM == 8 && Instruction::kN == 8 && Instruction::kK == 16 &&
                  Delta::kRow == 1 && Delta::kColumn == 1 &&
                  Mma::FragmentC::kElements == 2*mma_rows*mma_cols, "CUTLASS accumulator mapping changed");
    __shared__ tile_storage<Tile, Stages> shared;
    const int tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int warp_m = (warp/2)*(Tile/2), warp_n = (warp%2)*(Tile/2);
    const int64_t row0 = int64_t(blockIdx.x)*Tile, col0 = int64_t(blockIdx.y)*Tile;

    // CUTLASS 4.8.0 row-major IteratorC: lane quad selects row; each lane owns two columns.
    const int lane_row = lane/4, lane_col = 2*(lane%4);
    cutlass::Array<float, Mma::FragmentC::kElements> sum;
    sum.clear();

    const typename Config::LayoutA layout_a(64);
    const typename Config::LayoutB layout_b(64);
    Mma mma;
    stage_tile<Tile>(shared.stages[0], prefetch<Tile>(weights, input, row0, col0, m, n, k, 0));
    __syncthreads();
    for (int64_t inner0 = 0; inner0 < k; inner0 += 64) {
        const int current = (inner0/64)%Stages;
        typename Config::prefetched next;
        if (Stages == 2 && inner0 + 64 < k) {
            next = prefetch<Tile>(weights, input, row0, col0, m, n, k, inner0 + 64);
        }
        auto & tile = shared.stages[current];
        float weight_scales[mma_rows];
#pragma unroll
        for (int mma_m = 0; mma_m < mma_rows; ++mma_m) {
            weight_scales[mma_m] = tile.da[warp_m + lane_row + 8*mma_m];
        }
        typename Mma::IteratorA iter_a({tile.a, layout_a}, lane);
        typename Mma::IteratorB iter_b({tile.b, layout_b}, lane);
        iter_a.add_tile_offset({warp/2, 0});
        iter_b.add_tile_offset({0, warp%2});
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            typename Mma::FragmentC accum;
            accum.clear();
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                typename Mma::FragmentA a;
                typename Mma::FragmentB b;
                typename Mma::TransformedFragmentA ta;
                typename Mma::TransformedFragmentB tb;
                iter_a.load(a); iter_b.load(b);
                ++iter_a; ++iter_b;
                mma.transform(ta, tb, a, b);
                mma(accum, ta, tb, accum);
            }
#pragma unroll
            for (int mma_n = 0; mma_n < mma_cols; ++mma_n) {
                const int col = warp_n + lane_col + 8*mma_n;
                const float activation_scale0 = tile.db[group][col];
                const float activation_scale1 = tile.db[group][col + 1];
#pragma unroll
                for (int mma_m = 0; mma_m < mma_rows; ++mma_m) {
                    const int i = 2*(mma_n*mma_rows + mma_m);
                    sum[i] += float(accum[i])*weight_scales[mma_m]*activation_scale0;
                    sum[i + 1] += float(accum[i + 1])*weight_scales[mma_m]*activation_scale1;
                }
            }
        }
        if (Stages == 1) {
            // All warps must finish reading the only stage before any thread overwrites it.
            __syncthreads();
            if (inner0 + 64 < k) {
                stage_tile<Tile>(shared.stages[0], prefetch<Tile>(weights, input, row0, col0, m, n, k, inner0 + 64));
            }
        } else if (inner0 + 64 < k) {
            stage_tile<Tile>(shared.stages[1-current], next);
        }
        __syncthreads();
    }
#pragma unroll
    for (int mma_n = 0; mma_n < mma_cols; ++mma_n) {
        const int col = warp_n + lane_col + 8*mma_n;
#pragma unroll
        for (int mma_m = 0; mma_m < mma_rows; ++mma_m) {
            const int row = warp_m + lane_row + 8*mma_m;
            const int i = 2*(mma_n*mma_rows + mma_m);
            shared.output[row*(Tile+1) + col] = sum[i];
            shared.output[row*(Tile+1) + col + 1] = sum[i + 1];
        }
    }
    __syncthreads();
    // A padded shared transpose makes consecutive lanes write consecutive ggml rows.
    for (int i = tid; i < Tile*Tile; i += 128) {
        const int row = i%Tile, col = i/Tile;
        if (row0 + row < m && col0 + col < n) {
            output[(col0 + col)*m + row0 + row] = shared.output[row*(Tile+1) + col];
        }
    }
#else
    GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_UNUSED(m); GGML_UNUSED(n); GGML_UNUSED(k);
#endif
}
} // namespace merlin_prefill
