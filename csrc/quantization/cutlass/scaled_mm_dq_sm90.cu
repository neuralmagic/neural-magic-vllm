#include <torch/extension.h>

#include <iostream>
#include <sstream>
#include <vector>

// clang-format will break include orders
// clang-format off
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "common.hpp"
// clang-format on 

/////////////////////////////////////////
// Begin automatically generated section

template<typename ElementAB, typename ElementD, typename ElementAcc>
struct sm90_int8_gemm
{
using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  cute::Shape<cute::_128, cute::_128, cute::_128>, cutlass::epilogue::collective::EpilogueTileAuto,
  ElementD, ElementD,
  cutlass::epilogue::TmaWarpSpecialized
>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using ScaleBDescriptor = cutlass::epilogue::collective::detail::RowBroadcastDescriptor<EpilogueDescriptor, float>;

using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
    ScaleBDescriptor::Stages, typename EpilogueDescriptor::TileShape,
    typename ScaleBDescriptor::Element, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    ScaleB,
    Accum>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, ElementD, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    ScaleA,
    EVTCompute0>;

using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using ElementC = void;
using StrideC = StrideD;



using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, float,
    ElementC, StrideC, 4,
    ElementD, StrideD, 4,
    cutlass::epilogue::TmaWarpSpecialized,
    EVTCompute1
  >::CollectiveOp;

static constexpr size_t CEStorageSize = sizeof(typename CollectiveEpilogue::SharedStorage);

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementAB, cutlass::layout::RowMajor, 16,
    ElementAB, cutlass::layout::ColumnMajor, 16,
    ElementAcc,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(CEStorageSize)>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_bf16_bf16_128x128x128_2x1x1_0_tnt_align16_warpspecialized_pingpong_epi_tma
using cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_bf16_bf16_128x128x128_2x1x1_0_tnt_align16_warpspecialized_pingpong_epi_tma_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler
>;

// Define named type
struct GemmKernel :
  public cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_bf16_bf16_128x128x128_2x1x1_0_tnt_align16_warpspecialized_pingpong_epi_tma_base { };

};

// End automatically generated section
/////////////////////////////////////////

using StrideA = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using StrideB = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

template <typename Gemm, typename ElementIn, typename ElementOut>
void cutlass_scaled_mm_dq_dispatcher(torch::Tensor &out, torch::Tensor const &a,
                                     torch::Tensor const &b,
                                     torch::Tensor const &a_scales,
                                     torch::Tensor const &b_scales) {

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideC = typename Gemm::StrideC;
  StrideA a_stride{lda, cute::Int<1>{}, cute::Int<0>{}};
  StrideB b_stride{ldb, cute::Int<1>{}, cute::Int<0>{}};
  StrideC c_stride{ldc, cute::Int<1>{}, cute::Int<0>{}};

  using GemmKernel = typename Gemm::GemmKernel;
  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  auto a_ptr = static_cast<ElementIn *>(a.data_ptr());
  auto b_ptr = static_cast<ElementIn *>(b.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                 b_stride};

  auto c_ptr = static_cast<ElementOut *>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                prob_shape, mainloop_args, epilogue_args};

  using ScaleA_Args = typename Gemm::ScaleA::Arguments;
  using ScaleB_Args = typename Gemm::ScaleB::Arguments;
  ScaleA_Args a_args = a_scales.numel() == 1
                           ? ScaleA_Args{nullptr, a_scales.item<float>(), {}}
                           : ScaleA_Args{a_scales.data_ptr<float>(), {}, {}};

  ScaleB_Args b_args = b_scales.numel() == 1
                           ? ScaleB_Args{nullptr, b_scales.item<float>(), {}}
                           : ScaleB_Args{b_scales.data_ptr<float>(), {}, {}};

  args.epilogue.thread = {a_args, {b_args}};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op.run(args);
  CUTLASS_CHECK(status);
}

void cutlass_scaled_mm_dq_sm90(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales) {
  if (a.dtype() == torch::kInt8) {

    return cutlass_scaled_mm_dq_dispatcher<
        sm90_int8_gemm<int8_t, cutlass::bfloat16_t, int32_t>, int8_t,
        cutlass::bfloat16_t>(out, a, b, a_scales, b_scales);
  } else {

    return cutlass_scaled_mm_dq_dispatcher<
        sm90_int8_gemm<cutlass::float_e4m3_t, cutlass::bfloat16_t, float>,
        cutlass::float_e4m3_t, cutlass::bfloat16_t>(out, a, b, a_scales,
                                                    b_scales);
  }
}

