#include <torch/extension.h>

#define AUTOGEN_CUTLASS_DEFS\
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5", &autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5, "autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5"); \


void autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);
