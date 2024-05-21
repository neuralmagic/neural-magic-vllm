#include <stddef.h>
#include <torch/extension.h>

#include "scaled_mm_dq_c3x.cuh"
#include "common.hpp"

void cutlass_scaled_mm_dq_sm90(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShapeI8 = Shape<_128, _128, _128>;
  using ClusterShapeI8 = Shape<_1, _2, _1>;
  using KernelScheduleI8 =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueScheduleI8 = typename cutlass::epilogue::TmaWarpSpecialized;

  using TileShapeFP8 = Shape<_128, _128, _128>;
  using ClusterShapeFP8 = Shape<_1, _2, _1>;
  using KernelScheduleFP8 =
      typename cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative;
  using EpilogueScheduleFP8 =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;

  using TileSchedule = cutlass::gemm::PersistentScheduler;
  static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;

  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_scaled_mm_dq_dispatcher<
          cutlass_3x_gemm<int8_t, cutlass::bfloat16_t, TileShapeI8, ClusterShapeI8,
                          KernelScheduleI8, EpilogueScheduleI8, TileSchedule, Mode>>(
          out, a, b, a_scales, b_scales);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);

      return cutlass_scaled_mm_dq_dispatcher<
          cutlass_3x_gemm<int8_t, cutlass::half_t, TileShapeI8, ClusterShapeI8,
                          KernelScheduleI8, EpilogueScheduleI8, TileSchedule, Mode>>(
          out, a, b, a_scales, b_scales);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_scaled_mm_dq_dispatcher<
          cutlass_3x_gemm<cutlass::float_e4m3_t, cutlass::bfloat16_t, TileShapeFP8,
                          ClusterShapeFP8, KernelScheduleFP8, EpilogueScheduleFP8, TileSchedule, Mode>>(
          out, a, b, a_scales, b_scales);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);

      return cutlass_scaled_mm_dq_dispatcher<
          cutlass_3x_gemm<cutlass::float_e4m3_t, cutlass::half_t, TileShapeFP8,
                          ClusterShapeFP8, KernelScheduleFP8, EpilogueScheduleFP8, TileSchedule, Mode>>(
          out, a, b, a_scales, b_scales);
    }
  }
}

