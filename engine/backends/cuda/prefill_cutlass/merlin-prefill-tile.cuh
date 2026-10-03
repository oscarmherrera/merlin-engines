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
        uint8_t packed[Tile*k/(4*threads)];
        int8_t q8[Tile*k/threads];
        float da;
        float db[2];
    };
};

template<int Tile, int Stages>
union tile_storage {
    static_assert(Stages == 1 || Stages == 2, "unsupported pipeline depth");
    typename tile_config<Tile>::stage stages[Stages];
    int32_t coordinates[Tile*Tile];
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
    for (int i = 0; i < Tile*64/(4*128); ++i) {
        const int index = tid + i*128;
        const int row = index / 16;
        const int byte = index % 16;
        next.packed[i] = row0 + row < m ? weights[(row0 + row)*(k/QK_PQ2_0) + block].qs[offset/4 + byte]
                                      : 0x55;
    }
#pragma unroll
    for (int i = 0; i < Tile*64/128; ++i) {
        const int index = tid + i*128;
        const int col = index / 64;
        next.q8[i] = col0 + col < n ? input[block*n + col0 + col].qs[offset + index%64] : 0;
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
    for (int i = 0; i < Tile*64/(4*128); ++i) {
        const int index = tid + i*128;
        const int row = index / 16;
        const int inner = (index%16)*4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            dst.a[layout_a({row, inner + j})] = int8_t(int((next.packed[i] >> (2*j)) & 3) - 1);
        }
    }
#pragma unroll
    for (int i = 0; i < Tile*64/128; ++i) {
        const int index = tid + i*128;
        dst.b[layout_b({index%64, index/64})] = next.q8[i];
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
    __shared__ tile_storage<Tile, Stages> shared;
    const int tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int warp_m = (warp/2)*(Tile/2), warp_n = (warp%2)*(Tile/2);
    const int64_t row0 = int64_t(blockIdx.x)*Tile, col0 = int64_t(blockIdx.y)*Tile;

    // CUTLASS owns fragment coordinates; reuse this shared storage for input staging.
    for (int i = tid; i < Tile*Tile; i += 128) { shared.coordinates[i] = i; }
    __syncthreads();
    typename Mma::FragmentC coordinates;
    typename Mma::IteratorC coord_iter({shared.coordinates + warp_m*Tile + warp_n,
                                       cutlass::layout::RowMajor(Tile)}, lane);
    coord_iter.load(coordinates);
    __syncthreads();
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
            for (int i = 0; i < Mma::FragmentC::kElements; ++i) {
                const int row = coordinates[i]/Tile, col = coordinates[i]%Tile;
                sum[i] += float(accum[i])*tile.da[row]*tile.db[group][col];
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
    for (int i = 0; i < Mma::FragmentC::kElements; ++i) {
        const int row = coordinates[i]/Tile, col = coordinates[i]%Tile;
        shared.output[row*(Tile+1) + col] = sum[i];
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
