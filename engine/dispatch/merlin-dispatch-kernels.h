#pragma once

namespace merlin_dispatch {
inline constexpr const char * cutlass_narrow = "pq2_cutlass_32x32x64";
inline constexpr const char * cutlass_wide = "pq2_cutlass_64x64x64";
inline constexpr const char * cutlass_narrow_single = "pq2_cutlass_32x32x64_s1";
inline constexpr const char * cutlass_wide_single = "pq2_cutlass_64x64x64_s1";
inline constexpr const char * cutlass_candidates[] = {
    cutlass_narrow, cutlass_wide, cutlass_narrow_single, cutlass_wide_single,
};
}
