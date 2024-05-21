#include <torch/extension.h>

#define CUTLASS2X_DEFS\
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5", &autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5, "autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4", &autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4, "autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3", &autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3, "autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5", &autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5, "autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4", &autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4, "autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3", &autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3, "autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5", &autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5, "autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4", &autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4, "autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3", &autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3, "autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5", &autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5, "autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4", &autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4, "autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4"); \
 ops.def("autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3", &autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3, "autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3"); \


void autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_5(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass2x_scaled_mm_dq_sm80_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_3(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);
