#include "scaled_mm_c2x_transpose.cuh"

#if 0
void cutlass_scaled_mm_sm75_transpose(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;
  cutlass_scaled_mm_i8_transpose<cutlass::arch::Sm75,
                    TileShape,
                    WarpShape,
                    InstructionShape,
                    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
                    cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
                    2>(out, a, b, a_scales, b_scales);
}
#endif

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
                    5>(out, a, b, a_scales, b_scales);
}

#if 0
void cutlass_scaled_mm_sm89_transpose(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales) {

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (a.dtype() == torch::kFloat8_e4m3fn) {
    cutlass_scaled_mm_fp8_transpose<cutlass::arch::Sm89,
                      TileShape,
                      WarpShape,
                      InstructionShape,
                      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
                      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
                      5>(out, a, b, a_scales, b_scales);
  } else {
    cutlass_scaled_mm_i8_transpose<cutlass::arch::Sm89,
                      TileShape,
                      WarpShape,
                      InstructionShape,
                      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
                      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
                      5>(out, a, b, a_scales, b_scales);
  }
}
#endif
