#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <assert.h>

#include "../../dispatch_utils.h"
#include "../../reduction_utils.cuh"
#include "quant_utils.cuh"

namespace vllm {

template <typename scalar_t, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel(
  const scalar_t* __restrict__ input,
  int8_t* __restrict__ out,
  scale_type scale,
  const int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx * hidden_size + i];
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = block_amax_val / 127.0f;
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      out[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      out[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / scale);
    }
  }
}
} // namespace vllm

void quant(
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
    vllm::quant_kernel<scalar_t, float, false><<<grid, block, 0, stream>>>(
      input.data_ptr<scalar_t>(),
      out.data_ptr<int8_t>(),
      scale,
      hidden_size);
  });
}

void quant(
  torch::Tensor& out,   // [..., hidden_size]
  torch::Tensor& input, // [..., hidden_size]
  torch::Tensor& scale) { // [num_tokens]
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
    vllm::quant_kernel<scalar_t, float*, true><<<grid, block, 0, stream>>>(
      input.data_ptr<scalar_t>(),
      out.data_ptr<int8_t>(),
      scale.data_ptr<float>(),
      hidden_size);
  });
}