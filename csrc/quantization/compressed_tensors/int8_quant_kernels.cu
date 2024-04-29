#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <assert.h>

#include "../../dispatch_utils.h"

static inline __device__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
}

namespace vllm {

template <typename scalar_t, typename scale_type>
__global__ void quant_kernel(
  const scalar_t* __restrict__ input,
  int8_t* __restrict__ out,
  scale_type scale,
  const int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] =
        float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / scale);
  }
}
} // namespace vllm

void quant_per_tensor(
  torch::Tensor& out,   // [..., hidden_size]
  torch::Tensor& input, // [..., hidden_size]
  float scale) {
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
    vllm::quant_kernel<scalar_t, float><<<grid, block, 0, stream>>>(
      input.data_ptr<scalar_t>(),
      out.data_ptr<int8_t>(),
      scale,
      hidden_size);
  });
}
