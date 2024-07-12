#include "marlinv2_mm_launcher.cuh"
#include "marlinv2_prepack_launcher.cuh"
#include "scalar_type.hpp"

namespace marlinv2 {

using namespace vllm;

//
//  Utils (type dispatching)
//

template <typename Fn>
static auto scalar_type_dispatch(ScalarType const& type, Fn fn) {
  if (type == vllm::kU4) {
    return fn(cutlass::uint4b_t{});
  } else if (type == vllm::kS4) {
    return fn(cutlass::int4b_t{});
  } else {
    TORCH_CHECK(false, "Unsupported type ", type.str());
  }
}

#define AT_DISPATCH_CASE_SUPPORTED_COMPUTE_TYPES(...) \
  AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__)

#define AT_DISPATCH_SUPPORTED_COMPUTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                             \
                     AT_DISPATCH_CASE_SUPPORTED_COMPUTE_TYPES(__VA_ARGS__))

//
//  Interface
//

std::vector<ScalarTypeTorchPtr> supported_types() {
  return {c10::make_intrusive<ScalarTypeTorch>(vllm::kU4),
          c10::make_intrusive<ScalarTypeTorch>(vllm::kS4)};
}

std::vector<std::string> supported_schedules(ScalarTypeTorchPtr const& btype) {
  return scalar_type_dispatch(*btype, [&](auto BType) {
    return KernelDispatcher<half_t, decltype(BType)>::supported_schedules();
  });
}

torch::Tensor gemm(torch::Tensor const A, torch::Tensor const B,
                   ScalarTypeTorchPtr const& btype,
                   c10::optional<torch::Tensor> const& scales,
                   c10::optional<torch::Tensor> const& zeros,
                   c10::optional<int64_t> group_size,
                   c10::optional<torch::Tensor> const& C,
                   c10::optional<double> alpha, c10::optional<double> beta,
                   c10::optional<std::string> schedule) {
  TORCH_CHECK(btype->size_bits() == 4);  // only supports 4bit for now

  auto args = PytorchArguments{.A = A,
                               .B = B,
                               .scales = scales,
                               .zeros = zeros,
                               .group_size = group_size,
                               .C = C,
                               .alpha = alpha,
                               .beta = beta,
                               .schedule = schedule};

  return scalar_type_dispatch(*btype, [&](auto BType) {
    return AT_DISPATCH_SUPPORTED_COMPUTE_TYPES(
        A.scalar_type(), "marlinv2_gemm", [&] {
          using ComputeType = equivalent_cutlass_type_t<scalar_t>;
          return KernelDispatcher<ComputeType, decltype(BType)>::dispatch(args);
        });
  });
}

torch::Tensor prepack_B(torch::Tensor const B,
                        ScalarTypeTorchPtr const& btype) {
  return scalar_type_dispatch(*btype, [&](auto BType) {
    return PrepackDispatcher<half_t, decltype(BType), half_t>::dispatch(B);
  });
}

};  // namespace marlinv2
