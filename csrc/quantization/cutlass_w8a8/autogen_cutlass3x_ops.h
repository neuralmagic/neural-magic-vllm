#include <torch/extension.h>

#define CUTLASS3X_DEFS\
 ops.def("autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8", &autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8, "autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8"); \
 ops.def("autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8", &autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8, "autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8"); \
 ops.def("autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8", &autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8, "autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8"); \
 ops.def("autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8", &autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8, "autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8"); \


void autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperativeFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);

void autogen_cutlass3x_scaled_mm_dq_sm90_128x128x128_1x2x1_KernelTmaWarpSpecializedCooperative_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_half_t_fp8(torch::Tensor &out, torch::Tensor const &a,
                torch::Tensor const &b,
                torch::Tensor const &a_scales,
                torch::Tensor const &b_scales);
