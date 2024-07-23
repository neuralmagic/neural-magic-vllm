#pragma once
#define CUTLASS2X_DEFS\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_64x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_64x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_64x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_64x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_64x64x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_64x64x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_64x64x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_64x64x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_2_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_2_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_2_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_2_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x64x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x64x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x64x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x64x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_64x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_64x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_64x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_64x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8);\
 ops.def("autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(Tensor! out, Tensor a, Tensor b,Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8); \
 ops.impl("autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8", torch::kCUDA, &autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8);\


void autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x64x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_64x64x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_2_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_64x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_64x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAdd_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);

void autogen_cutlass2x_scaled_mm_sm89_32x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales,
                c10::optional<torch::Tensor> const& bias);
