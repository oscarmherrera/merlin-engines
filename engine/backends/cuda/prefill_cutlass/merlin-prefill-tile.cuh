#pragma once

#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

namespace merlin_prefill {

template<int Rows, int Cols, int WarpsM, int WarpsN, int Ktile>
struct tile_config {
    static constexpr int k = Ktile;
    static constexpr int threads = 32*WarpsM*WarpsN;
    using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<8, k>;
    using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<8, k>;
    using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        cutlass::gemm::GemmShape<Rows/WarpsM, Cols/WarpsN, k>, cutlass::gemm::GemmShape<8, 8, 16>,
        int8_t, LayoutA, int8_t, LayoutB, int32_t, cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAddSaturate>::Type;
    struct alignas(128) stage {
        int8_t a[Rows*k];
        int8_t b[Cols*k];
        float da[Rows];
        float db[k/32][Cols];
    };
    struct prefetched {
        uint16_t packed[Rows*k/(8*threads)];
        uint32_t q8[Cols*k/(4*threads)];
        float da;
        float db[k/32];
    };
};

template<int Rows, int Cols, int WarpsM, int WarpsN, int Ktile, int Stages>
union tile_storage {
    static_assert(Stages == 1 || Stages == 2, "unsupported pipeline depth");
    typename tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>::stage stages[Stages];
    float output[(Rows == 128 && Cols == 128 ? Rows/2 : Rows)*(Cols+1)];
};
constexpr size_t shared_bytes = sizeof(tile_storage<32, 32, 2, 2, 64, 2>);
constexpr size_t shared_bytes_wide = sizeof(tile_storage<64, 64, 2, 2, 64, 2>);
constexpr size_t shared_bytes_single = sizeof(tile_storage<32, 32, 2, 2, 64, 1>);
constexpr size_t shared_bytes_wide_single = sizeof(tile_storage<128, 128, 4, 2, 128, 1>);
static_assert(shared_bytes_single == 4480 && shared_bytes_wide_single == 35328, "single-stage workspace changed");
static_assert(shared_bytes == 8960 && shared_bytes_wide == 17920, "shared workspace changed");

template<int Rows, int Cols, int WarpsM, int WarpsN, int Ktile>
static __device__ __forceinline__ typename tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>::prefetched prefetch(
        const block_pq2_0 * weights, const block_q8_1_mmq * input,
        int row0, int col0, int m, int n, int k, int inner0) {
    using Config = tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>;
    typename Config::prefetched next{};
    const int tid = threadIdx.x;
    const int block = inner0 / QK_PQ2_0;
    const int offset = inner0 % QK_PQ2_0;
#pragma unroll
    for (int i = 0; i < Rows*Config::k/(8*Config::threads); ++i) {
        const int index = tid + i*Config::threads;
        const int row = index / (Config::k/8);
        const int byte = (index % (Config::k/8))*2;
        next.packed[i] = row0 + row < m ? *reinterpret_cast<const uint16_t *>(
            weights[(row0 + row)*(k/QK_PQ2_0) + block].qs + offset/4 + byte) : 0x5555;
    }
#pragma unroll
    for (int i = 0; i < Cols*Config::k/(4*Config::threads); ++i) {
        const int index = tid + i*Config::threads;
        const int col = index / (Config::k/4);
        next.q8[i] = col0 + col < n ? *reinterpret_cast<const uint32_t *>(
            input[block*n + col0 + col].qs + offset + (index%(Config::k/4))*4) : 0;
    }
    if (tid < Rows) {
        next.da = row0 + tid < m ? __half2float(weights[(row0 + tid)*(k/QK_PQ2_0) + block].d) : 0.0f;
    }
    if (tid < Cols) {
#pragma unroll
        for (int group = 0; group < Config::k/32; ++group) {
            next.db[group] = col0 + tid < n ? input[block*n + col0 + tid].d4[offset/32 + group] : 0.0f;
        }
    }
    return next;
}

template<int Rows, int Cols, int WarpsM, int WarpsN, int Ktile>
static __device__ __forceinline__ void stage_tile(
        typename tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>::stage & dst,
        const typename tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>::prefetched & next) {
    using Config = tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>;
    const typename Config::LayoutA layout_a(Config::k);
    const typename Config::LayoutB layout_b(Config::k);
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < Rows*Config::k/(8*Config::threads); ++i) {
        const int index = tid + i*Config::threads;
        const int row = index / (Config::k/8);
        const int inner = (index%(Config::k/8))*8;
        // Prism's packed lookup preserves all four PQ2 symbols, including +2.
        const uint32_t qe = __byte_perm(0x020100ff, 0x020100ff, next.packed[i]);
        const uint32_t qo = __byte_perm(0x020100ff, 0x020100ff, next.packed[i] >> 2);
        *reinterpret_cast<uint32_t *>(dst.a + layout_a({row, inner})) = __byte_perm(qe, qo, 0x5140);
        *reinterpret_cast<uint32_t *>(dst.a + layout_a({row, inner + 4})) = __byte_perm(qe, qo, 0x7362);
    }
#pragma unroll
    for (int i = 0; i < Cols*Config::k/(4*Config::threads); ++i) {
        const int index = tid + i*Config::threads;
        *reinterpret_cast<uint32_t *>(dst.b + layout_b({(index%(Config::k/4))*4, index/(Config::k/4)})) = next.q8[i];
    }
    if (tid < Rows) {
        dst.da[tid] = next.da;
    }
    if (tid < Cols) {
#pragma unroll
        for (int group = 0; group < Config::k/32; ++group) {
            dst.db[group][tid] = next.db[group];
        }
    }
}

template<int Rows, int Cols, int WarpsM, int WarpsN, int Ktile, bool FullTiles>
static __device__ __forceinline__ void stage_tile_direct(
        typename tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>::stage & dst,
        const block_pq2_0 * weights, const block_q8_1_mmq * input,
        int row0, int col0, int m, int n, int k, int inner0) {
    using Config = tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>;
    const typename Config::LayoutA layout_a(Config::k);
    const typename Config::LayoutB layout_b(Config::k);
    const int tid = threadIdx.x;
    const int block = inner0 / QK_PQ2_0;
    const int offset = inner0 % QK_PQ2_0;
    static_assert(Rows == Cols, "direct staging requires square tiles");
    const int row_lane = tid/(Config::k/8);
    const int col_lane = tid/(Config::k/4);
    const int row_stride = Config::threads/(Config::k/8);
    const int col_stride = Config::threads/(Config::k/4);
    const int byte = (tid%(Config::k/8))*2;
    const int inner = (tid%(Config::k/8))*8;
    const int q8_offset = (tid%(Config::k/4))*4;
    const int q8_col_lane8 = tid/(Config::k/8);
    const int q8_col_stride8 = Config::threads/(Config::k/8);
    const int q8_offset8 = (tid%(Config::k/8))*8;
    const int store_swap = row_lane & 1;
#pragma unroll
    for (int i = 0; i < Rows*Config::k/(8*Config::threads); ++i) {
        const int row = row_lane + i*row_stride;
        const int col0_tile = col_lane + 2*i*col_stride;
        const int col1_tile = col0_tile + col_stride;
        const uint16_t packed = (FullTiles || row0 + row < m) ? *reinterpret_cast<const uint16_t *>(
            weights[(row0 + row)*(k/QK_PQ2_0) + block].qs + offset/4 + byte) : 0x5555;
        if (Rows == 128 && Cols == 128 && Config::k == 128 && FullTiles) {
            const int q8_col = q8_col_lane8 + i*q8_col_stride8;
            const uint64_t q8 = *reinterpret_cast<const uint64_t *>(
                input[block*n + col0 + q8_col].qs + offset + q8_offset8);
            const int q8_swap = q8_col & 1;
            *reinterpret_cast<uint32_t *>(dst.b + layout_b({q8_offset8 + 4*q8_swap, q8_col})) =
                q8_swap ? uint32_t(q8 >> 32) : uint32_t(q8);
            *reinterpret_cast<uint32_t *>(dst.b + layout_b({q8_offset8 + 4*(1 - q8_swap), q8_col})) =
                q8_swap ? uint32_t(q8) : uint32_t(q8 >> 32);
        } else {
            const uint32_t q80 = (FullTiles || col0 + col0_tile < n) ? *reinterpret_cast<const uint32_t *>(
                input[block*n + col0 + col0_tile].qs + offset + q8_offset) : 0;
            const uint32_t q81 = (FullTiles || col0 + col1_tile < n) ? *reinterpret_cast<const uint32_t *>(
                input[block*n + col0 + col1_tile].qs + offset + q8_offset) : 0;
            *reinterpret_cast<uint32_t *>(dst.b + layout_b({q8_offset, col0_tile})) = q80;
            *reinterpret_cast<uint32_t *>(dst.b + layout_b({q8_offset, col1_tile})) = q81;
        }
        const uint32_t qe = __byte_perm(0x020100ff, 0x020100ff, packed);
        const uint32_t qo = __byte_perm(0x020100ff, 0x020100ff, packed >> 2);
        const uint32_t low = __byte_perm(qe, qo, 0x5140);
        const uint32_t high = __byte_perm(qe, qo, 0x7362);
        int8_t * a_dst = dst.a + layout_a({row, inner});
        if (Rows == 128 && Cols == 128 && FullTiles) {
            *reinterpret_cast<uint32_t *>(a_dst + 4*store_swap) = store_swap ? high : low;
            *reinterpret_cast<uint32_t *>(a_dst + 4*(1 - store_swap)) = store_swap ? low : high;
        } else {
            *reinterpret_cast<uint32_t *>(a_dst) = low;
            *reinterpret_cast<uint32_t *>(a_dst + 4) = high;
        }
    }
    if (tid < Rows) {
        dst.da[tid] = (FullTiles || row0 + tid < m) ? __half2float(weights[(row0 + tid)*(k/QK_PQ2_0) + block].d) : 0.0f;
    }
    if (tid < Cols) {
        if (Config::k == 128) {
            float4 scales = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (FullTiles || col0 + tid < n) {
                scales = *reinterpret_cast<const float4 *>(input[block*n + col0 + tid].d4);
            }
            dst.db[0][tid] = scales.x;
            dst.db[1][tid] = scales.y;
            dst.db[2][tid] = scales.z;
            dst.db[3][tid] = scales.w;
        } else {
#pragma unroll
            for (int group = 0; group < Config::k/32; ++group) {
                dst.db[group][tid] = (FullTiles || col0 + tid < n) ? input[block*n + col0 + tid].d4[offset/32 + group] : 0.0f;
            }
        }
    }
}

template<int Rows, int Cols, int WarpsM, int WarpsN, int Ktile, int Stages, bool FullTiles>
static __global__ void pq2_cutlass(const block_pq2_0 * weights, const block_q8_1_mmq * input,
                                  float * output, int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using Config = tile_config<Rows, Cols, WarpsM, WarpsN, Ktile>;
    using Mma = typename Config::Mma;
    using Instruction = typename Mma::InstructionShape;
    using AccumulatorIterator = typename Mma::IteratorC;
    using Delta = typename AccumulatorIterator::OpDelta;
    constexpr int mma_rows = Mma::MmaIterations::kRow;
    constexpr int mma_cols = Mma::MmaIterations::kColumn;
    static_assert(Instruction::kM == 8 && Instruction::kN == 8 && Instruction::kK == 16 &&
                  Delta::kRow == 1 && Delta::kColumn == 1 &&
                  Mma::FragmentC::kElements == 2*mma_rows*mma_cols, "CUTLASS accumulator mapping changed");
    __shared__ tile_storage<Rows, Cols, WarpsM, WarpsN, Ktile, Stages> shared;
    const int tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int warp_m = (warp/WarpsN)*(Rows/WarpsM), warp_n = (warp%WarpsN)*(Cols/WarpsN);
    const int row0 = int(blockIdx.y)*Rows, col0 = int(blockIdx.x)*Cols;

    // CUTLASS 4.8.0 row-major IteratorC: lane quad selects row; each lane owns two columns.
    const int lane_row = lane/4, lane_col = 2*(lane%4);
    cutlass::Array<float, Mma::FragmentC::kElements> sum;
    sum.clear();

    const typename Config::LayoutA layout_a(Config::k);
    const typename Config::LayoutB layout_b(Config::k);
    Mma mma;
    if (Stages == 1) {
        stage_tile_direct<Rows, Cols, WarpsM, WarpsN, Ktile, FullTiles>(shared.stages[0],
            weights, input, row0, col0, m, n, k, 0);
    } else {
        stage_tile<Rows, Cols, WarpsM, WarpsN, Ktile>(shared.stages[0],
            prefetch<Rows, Cols, WarpsM, WarpsN, Ktile>(weights, input, row0, col0, m, n, k, 0));
    }
    __syncthreads();
    for (int inner0 = 0; inner0 < k; inner0 += Config::k) {
        const int current = (inner0/Config::k)%Stages;
        typename Config::prefetched next;
        if (Stages == 2 && inner0 + Config::k < k) {
            next = prefetch<Rows, Cols, WarpsM, WarpsN, Ktile>(
                weights, input, row0, col0, m, n, k, inner0 + Config::k);
        }
        auto & tile = shared.stages[current];
        float weight_scales[mma_rows];
#pragma unroll
        for (int mma_m = 0; mma_m < mma_rows; ++mma_m) {
            weight_scales[mma_m] = tile.da[warp_m + lane_row + 8*mma_m];
        }
        typename Mma::IteratorA iter_a({tile.a, layout_a}, lane);
        typename Mma::IteratorB iter_b({tile.b, layout_b}, lane);
        iter_a.add_tile_offset({warp/WarpsN, 0});
        iter_b.add_tile_offset({0, warp%WarpsN});
#pragma unroll
        for (int group = 0; group < Config::k/32; ++group) {
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
            if (inner0 + Config::k < k) {
                stage_tile_direct<Rows, Cols, WarpsM, WarpsN, Ktile, FullTiles>(shared.stages[0],
                    weights, input, row0, col0, m, n, k, inner0 + Config::k);
            }
        } else if (inner0 + Config::k < k) {
            stage_tile<Rows, Cols, WarpsM, WarpsN, Ktile>(shared.stages[1-current], next);
        }
        __syncthreads();
    }
    if (Rows == 128 && Cols == 128) {
        for (int half = 0; half < 2; ++half) {
            if (warp_m/64 == half) {
#pragma unroll
                for (int mma_n = 0; mma_n < mma_cols; ++mma_n) {
                    const int col = warp_n + lane_col + 8*mma_n;
#pragma unroll
                    for (int mma_m = 0; mma_m < mma_rows; ++mma_m) {
                        const int row = warp_m - 64*half + lane_row + 8*mma_m;
                        const int i = 2*(mma_n*mma_rows + mma_m);
                        shared.output[row*(Cols+1) + col] = sum[i];
                        shared.output[row*(Cols+1) + col + 1] = sum[i + 1];
                    }
                }
            }
            __syncthreads();
            for (int i = tid; i < 64*Cols; i += Config::threads) {
                const int row = i%64, col = i/64;
                if (FullTiles || (row0 + 64*half + row < m && col0 + col < n)) {
                    output[(col0 + col)*m + row0 + 64*half + row] = shared.output[row*(Cols+1) + col];
                }
            }
            if (half == 0) { __syncthreads(); }
        }
    } else {
#pragma unroll
        for (int mma_n = 0; mma_n < mma_cols; ++mma_n) {
            const int col = warp_n + lane_col + 8*mma_n;
#pragma unroll
            for (int mma_m = 0; mma_m < mma_rows; ++mma_m) {
                const int row = warp_m + lane_row + 8*mma_m;
                const int i = 2*(mma_n*mma_rows + mma_m);
                shared.output[row*(Cols+1) + col] = sum[i];
                shared.output[row*(Cols+1) + col + 1] = sum[i + 1];
            }
        }
        __syncthreads();
        // A padded shared transpose makes consecutive lanes write consecutive ggml rows.
        for (int i = tid; i < Rows*Cols; i += Config::threads) {
            const int row = i%Rows, col = i/Rows;
            if (FullTiles || (row0 + row < m && col0 + col < n)) {
                output[(col0 + col)*m + row0 + row] = shared.output[row*(Cols+1) + col];
            }
        }
    }
#else
    GGML_UNUSED(weights); GGML_UNUSED(input); GGML_UNUSED(output);
    GGML_UNUSED(m); GGML_UNUSED(n); GGML_UNUSED(k);
#endif
}
} // namespace merlin_prefill
