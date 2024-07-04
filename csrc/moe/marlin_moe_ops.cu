/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "quantization/gptq_marlin/quantized_marlin_common.cuh"

#include <iostream>

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace marlin_moe {

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

}  // namespace marlin_moe

torch::Tensor marlin_gemm_moe(
    const torch::Tensor& a, const torch::Tensor& b_q_weights,
    const torch::Tensor& sorted_ids, const torch::Tensor& topk_weights,
    const torch::Tensor& b_scales, const torch::Tensor& g_idx,
    const torch::Tensor& perm, const torch::Tensor& expert_offsets,
    torch::Tensor& workspace, int64_t size_m, int64_t size_n, int64_t size_k,
    bool is_k_full, int64_t num_tokens_post_padded, int64_t num_experts,
    int64_t topk, int64_t moe_block_size, bool replicate_input,
    bool apply_weights) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

#else

  #define __CALL_IF_MOE(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,     \
                        HAS_ACT_ORDER, GROUP_BLOCKS, NUM_THREADS)              \
    else if (thread_m_blocks == THREAD_M_BLOCKS &&                             \
             thread_n_blocks == THREAD_N_BLOCKS &&                             \
             thread_k_blocks == THREAD_K_BLOCKS &&                             \
             has_act_order == HAS_ACT_ORDER && group_blocks == GROUP_BLOCKS && \
             num_threads == NUM_THREADS) {                                     \
      cudaFuncSetAttribute(                                                    \
          gptq_marlin::Marlin<half, 4, NUM_THREADS, THREAD_M_BLOCKS,           \
                              THREAD_N_BLOCKS, THREAD_K_BLOCKS,                \
                              gptq_marlin::pipe_stages, HAS_ACT_ORDER,         \
                              GROUP_BLOCKS, true>,                             \
          cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);        \
      gptq_marlin::Marlin<half, 4, NUM_THREADS, THREAD_M_BLOCKS,               \
                          THREAD_N_BLOCKS, THREAD_K_BLOCKS,                    \
                          gptq_marlin::pipe_stages, HAS_ACT_ORDER,             \
                          GROUP_BLOCKS, true>                                  \
          <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                   \
              A_ptr, B_ptr, C_ptr, sorted_ids_ptr, topk_weights_ptr, s_ptr,    \
              g_idx_ptr, num_groups, num_tokens_post_padded, expert_idx,       \
              num_experts, topk, prob_m, prob_n, prob_k, tot_m, locks,         \
              replicate_input, apply_weights);                                 \
    }

  #define CALL_IF_MOE(N_BLOCKS, K_BLOCKS, NUM_THREADS)           \
    __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
    __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
    __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
    __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
                                                                 \
    __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)  \
                                                                 \
    __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)  \
                                                                 \
    __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)  \
                                                                 \
    __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)

void marlin_mm_moe_f16i4(const void* A, const void* B, void* C,
                         const void* sorted_ids, const void* topk_weights,
                         const void* s, const void* g_idx, const void* perm,
                         void* a_tmp, const void* expert_offsets, int prob_m,
                         int prob_n, int prob_k, void* workspace,
                         bool has_act_order, bool is_k_full, int num_groups,
                         int group_size, int num_tokens_post_padded,
                         int num_experts, int topk, int moe_block_size, int dev,
                         cudaStream_t stream, int thread_k, int thread_n,
                         int sms, int max_par, bool replicate_input,
                         bool apply_weights) {
  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  if (sms == -1) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  // Set thread config
  gptq_marlin::exec_config_t exec_cfg = gptq_marlin::get_and_check_exec_config(
      prob_m, prob_n, prob_k, 4, has_act_order, is_k_full, group_size, thread_k,
      thread_n, max_shared_mem);

  int group_blocks = gptq_marlin::get_and_check_group_blocks(
      prob_k, has_act_order, is_k_full, group_size);

  int num_threads = exec_cfg.tb_cfg.num_threads;
  thread_k = exec_cfg.tb_cfg.thread_k;
  thread_n = exec_cfg.tb_cfg.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  int blocks = sms;

  int tot_m = prob_m;

  const long* expert_offsets_ptr = (const long*)expert_offsets;

  bool do_permute_a = has_act_order;

  // If we have a full K, then we can run the non-act-order version of Marlin
  // (since the weight rows are reordered by increasing group ids, and by
  // having a full K, we have full original groups)
  if (is_k_full) {
    has_act_order = false;
  }

  for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const int4* A_ptr = (const int4*)A;
    int4* a_tmp_ptr = (int4*)a_tmp;
    const int4* B_ptr = (const int4*)B + (prob_n * prob_k / 32) * expert_idx;
    int4* C_ptr = (int4*)C;
    const float* topk_weights_ptr = (const float*)topk_weights;
    const int* sorted_ids_ptr =
        (const int*)sorted_ids + expert_offsets_ptr[expert_idx];
    const int4* s_ptr =
        (const int4*)s +
        (((group_size == -1 || group_size == 0) ? 1 : prob_k / group_size) *
         prob_n / 8) *
            expert_idx;

    const int* g_idx_ptr = (const int*)g_idx + prob_k * expert_idx;
    const int* perm_ptr = (const int*)perm + prob_k * expert_idx;
    int* locks = (int*)workspace;

    if (do_permute_a) {
      // Permute A columns
      int topk_rows = replicate_input ? tot_m : tot_m * topk;
      int block_rows = div_ceil(topk_rows, blocks);
      gptq_marlin::permute_cols_kernel<<<blocks, num_threads, 0, stream>>>(
          A_ptr, perm_ptr, a_tmp_ptr, topk_rows, prob_k, block_rows);
      A_ptr = a_tmp_ptr;
    }

    int tot_its = expert_offsets_ptr[expert_idx + 1] -
                  expert_offsets_ptr[expert_idx];  // prob_m;
    if (tot_its == 0) {
      continue;
    }
    int tot_m_blocks = div_ceil(tot_its, 16);
    int pad = 16 * tot_m_blocks - tot_its;

    // Main loop
    for (int i = 0; i < tot_m_blocks; i += exec_cfg.max_m_blocks) {
      int thread_m_blocks = tot_m_blocks - i;
      prob_m = tot_its - 16 * i;
      int par = 1;
      if (thread_m_blocks > exec_cfg.max_m_blocks) {
        // Note that parallel > 1 currently only works for inputs without any
        // padding
        par = (16 * thread_m_blocks - pad) / (16 * exec_cfg.max_m_blocks);
        if (par > max_par) par = max_par;
        prob_m = (16 * exec_cfg.max_m_blocks) * par;
        i += exec_cfg.max_m_blocks * (par - 1);
        thread_m_blocks = exec_cfg.max_m_blocks;
      }

      // Define kernel configurations

      if (false) {
      }
      CALL_IF_MOE(16, 4, 256)
      CALL_IF_MOE(8, 8, 256)
      CALL_IF_MOE(8, 4, 128)
      CALL_IF_MOE(4, 8, 128)
      else {
        TORCH_CHECK(false, "Unsupported shapes: MNK = [" + str(prob_m) + ", " +
                               str(prob_n) + ", " + str(prob_k) + "]" +
                               ", has_act_order = " + str(has_act_order) +
                               ", num_groups = " + str(num_groups) +
                               ", group_size = " + str(group_size) +
                               ", thread_m_blocks = " + str(thread_m_blocks) +
                               ", thread_n_blocks = " + str(thread_n_blocks) +
                               ", thread_k_blocks = " + str(thread_k_blocks));
      }

      sorted_ids_ptr += 16 * thread_m_blocks * par;
    }
  }
}

}  // namespace marlin_moe

torch::Tensor marlin_gemm_moe(
    const torch::Tensor& a, const torch::Tensor& b_q_weights,
    const torch::Tensor& sorted_ids, const torch::Tensor& topk_weights,
    const torch::Tensor& b_scales, const torch::Tensor& g_idx,
    const torch::Tensor& perm, const torch::Tensor& expert_offsets,
    torch::Tensor& workspace, int64_t size_m, int64_t size_n, int64_t size_k,
    bool is_k_full, int64_t num_tokens_post_padded, int64_t num_experts,
    int64_t topk, int64_t moe_block_size, bool replicate_input,
    bool apply_weights) {
  int max_par = 4;

  int dev = a.get_device();

  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::zeros({size_m, topk, size_n}, options);
  torch::Tensor a_tmp = replicate_input
                            ? torch::zeros({size_m, size_k}, options)
                            : torch::zeros({size_m, topk, size_k}, options);

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;
  bool has_act_order = g_idx.size(1) != 0;

  int b_rank = b_scales.sizes().size();
  TORCH_CHECK(b_rank == 3, "b_scales rank = ", b_rank, " is not 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(1);

  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  marlin_moe::marlin_mm_moe_f16i4(
      a.data_ptr(), b_q_weights.data_ptr(), c.data_ptr(), sorted_ids.data_ptr(),
      topk_weights.data_ptr(), b_scales.data_ptr(), g_idx.data_ptr(),
      perm.data_ptr(), a_tmp.data_ptr(), expert_offsets.data_ptr(), size_m,
      size_n, size_k, workspace.data_ptr(), has_act_order, is_k_full,
      num_groups, group_size, num_tokens_post_padded, num_experts, topk,
      moe_block_size, dev, at::cuda::getCurrentCUDAStream(dev), thread_k,
      thread_n, sms, max_par, replicate_input, apply_weights);
  return c;
}

#endif
