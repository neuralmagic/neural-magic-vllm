#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../cuda_compat.h"
#include "../../dispatch_utils.h"
#include "../../reduction_utils.cuh"
// #include "quant_utils.cuh"

namespace vllm {


static inline __device__ int8_t float_to_int8_rn(float x) {
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t &>(dst);
}

template<typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template <typename scalar_t>
__global__ void silu_and_mul_quant_kernel(
  int8_t* __restrict__ out,          // [..., d]
  const scalar_t* __restrict__ input, // [..., 2 * d]
  const int d,
  float* __restrict__ scale, // [num_tokens]
  float* __restrict__ tmp) { 
  const int64_t token_idx = blockIdx.x;
  float amax_val = 0.0f;

  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    // const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    // scalar_t t = silu(x) * y;
    // input[token_idx * 2 * d + idx] = t;
    // amax_val = fmaxf(amax_val, fabsf((float) t));
    const float x = (float) VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const float y = (float) VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    float t = silu(x) * y;
    tmp[token_idx * d + idx] = t;
    amax_val = fmaxf(amax_val, fabsf(t));
  }

  __shared__ float s_amax;
  amax_val = blockReduceMax(amax_val);
  if (threadIdx.x == 0) {
    s_amax = amax_val;
    scale[blockIdx.x] = amax_val / 127.0f;
  }
  __syncthreads();

  float tmp_scale = 127.0f / s_amax;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // out[token_idx * d + idx] =
    //     float_to_int8_rn(tmp_scale * (float) input[token_idx * 2 * d + idx]);
    out[token_idx * d + idx] =
        float_to_int8_rn(tmp_scale * tmp[token_idx * d + idx]);
  }
}
} // namespace vllm


void silu_and_mul_quant(
  torch::Tensor& out,   // [..., d]
  torch::Tensor const& input, // [..., 2 * d]
  torch::Tensor& scale, // [num_tokens]
  torch::Tensor& tmp    // [..., d]
) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(                                                           \
    input.scalar_type(),                                                                  \
    "silu_and_mul_quant_kernel",                                                          \
    [&] {                                                                                 \
      vllm::silu_and_mul_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(              \
        out.data_ptr<int8_t>(),                                                           \
        input.data_ptr<scalar_t>(),                                                       \
        d,                                                                                \
        scale.data_ptr<float>(),                                                          \
        tmp.data_ptr<float>());                                                           \
    });
}