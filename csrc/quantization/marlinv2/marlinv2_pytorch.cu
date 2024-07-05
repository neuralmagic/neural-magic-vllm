#include "marlinv2_mm_launcher.cuh"
#include "marlinv2_prepack_launcher.cuh"

namespace marlinv2 {

torch::Tensor marlinv2_mm(torch::Tensor const A, torch::Tensor const B,
                          bool B_signed, int64_t B_bits,
                          c10::optional<torch::Tensor> const& scales,
                          c10::optional<torch::Tensor> const& zeros,
                          c10::optional<int64_t> group_size,
                          c10::optional<torch::Tensor> const& C,
                          c10::optional<double> alpha,
                          c10::optional<double> beta,
                          c10::optional<std::string> schedule) {
  TORCH_CHECK(B_bits == 4);  // only supports 4bit for now

  auto args = PytorchArguments{.A = A,
                               .B = B,
                               .scales = scales,
                               .zeros = zeros,
                               .group_size = group_size,
                               .C = C,
                               .alpha = alpha,
                               .beta = beta,
                               .schedule = schedule};

  if (B_bits == 4 && !B_signed) {
    return KernelDispatcher<cutlass::half_t, cutlass::uint4b_t,
                            cutlass::half_t>::dispatch(args);
  } else if (B_bits == 4 && B_signed) {
    return KernelDispatcher<cutlass::half_t, cutlass::int4b_t,
                            cutlass::half_t>::dispatch(args);
  } else {
    TORCH_CHECK(false, "Not implemented");
  }
}

torch::Tensor marlinv2_prepack_B(torch::Tensor const B, bool B_signed,
                                 int64_t B_bits) {
  if (B_bits == 4 && !B_signed) {
    return PrepackDispatcher<cutlass::half_t, cutlass::uint4b_t,
                             cutlass::half_t>::dispatch(B);
  } else if (B_bits == 4 && B_signed) {
    return PrepackDispatcher<cutlass::half_t, cutlass::int4b_t,
                             cutlass::half_t>::dispatch(B);
  } else {
    TORCH_CHECK(false, "Not implemented");
  }
}

};  // namespace marlinv2

// TORCH_LIBRARY_IMPL(nm_ops, CUDA, m) {
//   m.impl("marlinv2_mm", &marlinv2::marlinv2_mm);
//   m.impl("marlinv2_prepack_B", &marlinv2::marlinv2_prepack_B);
// }