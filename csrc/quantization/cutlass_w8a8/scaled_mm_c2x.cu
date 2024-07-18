#include <stddef.h>
//#include <torch/extension.h>

#include "scaled_mm_c2x.cuh"
#include "common.hpp"

void cutlass_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales,
                               c10::optional<torch::Tensor> const& bias) {
  return cutlass_scaled_mm_sm75_impl(out, a, b, a_scales, b_scales, bias);
}

void cutlass_scaled_mm_sm80(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales,
                               c10::optional<torch::Tensor> const& bias) {
  return cutlass_scaled_mm_sm80_impl(out, a, b, a_scales, b_scales, bias);
}

void cutlass_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales,
                               c10::optional<torch::Tensor> const& bias) {
  using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 5;
  cutlass_scaled_mm_sm89_impl<
      TileShape, WarpShape, InstructionShape,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel, MainLoopStages>(
      out, a, b, a_scales, b_scales, bias);
}

