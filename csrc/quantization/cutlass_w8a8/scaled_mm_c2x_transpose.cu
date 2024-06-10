#include "scaled_mm_c2x_transpose.cuh"

void cutlass_scaled_mm_sm80_transpose(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  cutlass_scaled_mm_i8_transpose<cutlass::arch::Sm80,
                    TileShape,
                    WarpShape,
                    InstructionShape,
                    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
                    cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
                    5>(out, a, b, b_scales, a_scales);
}

