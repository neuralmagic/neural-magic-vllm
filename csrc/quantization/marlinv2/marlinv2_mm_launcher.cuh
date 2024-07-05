#pragma once

#include <torch/all.h>
#include <Python.h>

#include "marlinv2_mm_kernel.cuh"
#include "cutlass_extensions/torch_utils.hpp"

namespace marlinv2 {

struct PytorchArguments {
  torch::Tensor const A;
  torch::Tensor const B;
  c10::optional<torch::Tensor> const& scales;
  c10::optional<torch::Tensor> const& zeros;
  c10::optional<int64_t> group_size;
  c10::optional<torch::Tensor> const& C;
  c10::optional<double> alpha;
  c10::optional<double> beta;
  c10::optional<std::string> schedule;
};

template <typename KernelSpeacialization>
torch::Tensor run_impl(PytorchArguments args) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(args.A));

  auto device = args.A.device();
  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  using ElementA = typename KernelSpeacialization::ElementA;
  using ElementB = typename KernelSpeacialization::ElementB;
  using ElementC = typename KernelSpeacialization::ElementC;
  using ElementD = typename KernelSpeacialization::ElementD;
  using ElementScale = typename KernelSpeacialization::ElementScale;
  using ElementZero = typename KernelSpeacialization::ElementZero;

  using LayoutA = typename KernelSpeacialization::LayoutA;
  using LayoutB = typename KernelSpeacialization::LayoutB;
  using LayoutC = typename KernelSpeacialization::LayoutC;
  using LayoutD = typename KernelSpeacialization::LayoutD;
  using LayoutScale = typename KernelSpeacialization::LayoutScale;
  using LayoutZero = typename KernelSpeacialization::LayoutScale;

  int M = args.A.size(0);
  int N = args.B.size(1);
  int K = args.A.size(1);

  // Allocate output
  torch::Tensor D = args.B.new_empty({M, N}, torch::kF16);

  auto A_ptr = data_ptr<ElementA const, LayoutA>(args.A, "A");
  auto B_ptr = data_ptr<ElementB const, LayoutB>(args.B, "B");
  auto D_ptr = data_ptr<ElementD, LayoutD>(D, "D");
  auto C_ptr = maybe_data_ptr<ElementC const, LayoutC>(args.C, "C");
  auto scales_ptr =
      maybe_data_ptr<ElementScale const, LayoutScale>(args.scales, "scales");
  auto zeros_ptr =
      maybe_data_ptr<ElementZero const, LayoutZero>(args.zeros, "zeros");

  auto arguments = KernelSpeacialization::create_arguments(
      stream, M, N, K, A_ptr, B_ptr, C_ptr, D_ptr, scales_ptr, zeros_ptr,
      args.alpha.value_or(1), args.beta.value_or(0),
      args.group_size.value_or(K));

  TORCH_CHECK(KernelSpeacialization::can_implement(arguments),
              "Marlinv2 kernel cannot be run with these arguments");

  size_t workspace_size = KernelSpeacialization::get_workspace_size(arguments);
  torch::Tensor workspace = args.B.new_empty(workspace_size, torch::kU8);

  KernelSpeacialization::run(arguments, workspace.data_ptr(), stream);

  return D;
};

template <typename ElementA, typename ElementB, typename ElementD,
          typename AccumulatorT = float, typename ScaleT = cutlass::half_t,
          typename ZeroT = cutlass::half_t>
struct KernelDispatcher {
  static torch::Tensor dispatch(PytorchArguments args);
};

};  // namespace marlinv2