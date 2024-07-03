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

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#include "gptq_marlin.cuh"
#include "gptq_marlin_dtypes.cuh"
#include "cuda_marlin_common.cuh"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace gptq_marlin {

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

}  // namespace gptq_marlin

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

#else

  #define __CALL_IF(NUM_BITS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,                \
                    THREAD_K_BLOCKS, HAS_ACT_ORDER, GROUP_BLOCKS, NUM_THREADS) \
    else if (num_bits == NUM_BITS && thread_m_blocks == THREAD_M_BLOCKS &&     \
             thread_n_blocks == THREAD_N_BLOCKS &&                             \
             thread_k_blocks == THREAD_K_BLOCKS &&                             \
             has_act_order == HAS_ACT_ORDER && group_blocks == GROUP_BLOCKS && \
             num_threads == NUM_THREADS) {                                     \
      cudaFuncSetAttribute(                                                    \
          marlin_common_cu::Marlin<scalar_t, NUM_BITS, NUM_THREADS, THREAD_M_BLOCKS,             \
                 THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER, \
                 GROUP_BLOCKS, false>,                                                \
          cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);        \
      marlin_common_cu::Marlin<scalar_t, NUM_BITS, NUM_THREADS, THREAD_M_BLOCKS,                 \
             THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER,     \
             GROUP_BLOCKS, false><<<blocks, NUM_THREADS, max_shared_mem, stream>>>(   \
          A_ptr, B_ptr, C_ptr, nullptr, nullptr, s_ptr, g_idx_ptr, num_groups,  \
          0, 0, 0, 0, prob_m, prob_n, prob_k, prob_m, locks, false, false);                                                      \
    }

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

typedef struct {
  int max_m_blocks;
  thread_config_t tb_cfg;
} exec_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},

};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * pipe_stages * 2;     // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;

  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

bool is_valid_cache_size(thread_config_t const& th_config, int max_m_blocks,
                         int prob_m, int prob_n, int prob_k, int num_bits,
                         int scales_cache_size, int max_shared_mem) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;

  int b_size = (tb_k * tb_n / pack_factor) * 4;

  // Get A size
  int m_blocks = div_ceil(prob_m, 16);
  int tb_max_m = 16;

  while (true) {
    if (m_blocks >= max_m_blocks) {
      tb_max_m *= max_m_blocks;
      break;
    }

    max_m_blocks--;
    if (max_m_blocks == 0) {
      TORCH_CHECK(false, "Unexpected m_blocks = ", m_blocks);
    }
  }

  int a_size = (tb_max_m * tb_k) * 2;

  float pipe_size = (a_size + b_size) * pipe_stages;

  TORCH_CHECK(max_shared_mem / 2 > scales_cache_size);  // Sanity

  return pipe_size < 0.95f * (max_shared_mem - scales_cache_size);
}

bool is_valid_config(thread_config_t const& th_config, int max_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  //  Determine cache for scales
  int scales_cache_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full);

  // Check that pipeline fits into cache
  if (!is_valid_cache_size(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                           num_bits, scales_cache_size, max_shared_mem)) {
    return false;
  }

  return true;
}

exec_config_t determine_thread_config(int prob_m, int prob_n, int prob_k,
                                      int num_bits, int group_size,
                                      bool has_act_order, bool is_k_full,
                                      int max_shared_mem) {
  int max_m_blocks = 4;
  while (max_m_blocks > 0) {
    if (prob_m <= 16) {
      for (auto th_config : small_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    } else {
      for (auto th_config : large_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    }

    max_m_blocks--;  // Process less M blocks per invocation to reduce cache
                     // usage
  }

  return exec_config_t{0, {-1, -1, -1}};
}

  #define CALL_IF(NUM_BITS, N_BLOCKS, K_BLOCKS, NUM_THREADS)           \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS)   \
                                                                       \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)  \
                                                                       \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)  \
                                                                       \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)  \
                                                                       \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS) \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS)  \
    __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS)

template <typename scalar_t>
void marlin_mm_f16i4(const void* A, const void* B, void* C, void* s,
                     void* g_idx, void* perm, void* a_tmp, int prob_m,
                     int prob_n, int prob_k, void* workspace, int num_bits,
                     bool has_act_order, bool is_k_full, int num_groups,
                     int group_size, int dev, cudaStream_t stream, int thread_k,
                     int thread_n, int sms, int max_par) {
  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  int tot_m = prob_m;
  int tot_m_blocks = div_ceil(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  // Set thread config
  exec_config_t exec_cfg;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    exec_cfg =
        exec_config_t{4, thread_config_t{thread_k, thread_n, default_threads}};
  } else {
    // Auto config
    exec_cfg =
        determine_thread_config(prob_m, prob_n, prob_k, num_bits, group_size,
                                has_act_order, is_k_full, max_shared_mem);
  }

  TORCH_CHECK(exec_cfg.max_m_blocks > 0 &&
                  is_valid_config(exec_cfg.tb_cfg, exec_cfg.max_m_blocks,
                                  prob_m, prob_n, prob_k, num_bits, group_size,
                                  has_act_order, is_k_full, max_shared_mem),
              "Invalid thread config: max_m_blocks = ", exec_cfg.max_m_blocks,
              ", thread_k = ", exec_cfg.tb_cfg.thread_k,
              ", thread_n = ", exec_cfg.tb_cfg.thread_n,
              ", num_threads = ", exec_cfg.tb_cfg.num_threads, " for MKN = [",
              prob_m, ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
              ", group_size = ", group_size,
              ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
              ", max_shared_mem = ", max_shared_mem);

  int num_threads = exec_cfg.tb_cfg.num_threads;
  thread_k = exec_cfg.tb_cfg.thread_k;
  thread_n = exec_cfg.tb_cfg.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  int blocks = sms;

  TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
              " is not divisible by thread_n = ", thread_n);
  TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
              " is not divisible by thread_k = ", thread_k);

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }

  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  const int4* s_ptr = (const int4*)s;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;

  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, blocks);
    marlin_common_cu::permute_cols_kernel<<<blocks, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, block_rows, default_threads);
    A_ptr = a_tmp_ptr;
  }

  // If we have a full K, then we can run the non-act-order version of Marlin
  // (since the weight rows are reordered by increasing group ids, and by having
  // a full K, we have full original groups)
  if (is_k_full) {
    has_act_order = false;
  }

  // Main loop
  for (int i = 0; i < tot_m_blocks; i += exec_cfg.max_m_blocks) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
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
    CALL_IF(4, 32, 2, 256)
    CALL_IF(4, 16, 4, 256)
    CALL_IF(4, 8, 8, 256)
    CALL_IF(4, 8, 4, 128)
    CALL_IF(4, 4, 8, 128)
    CALL_IF(8, 32, 2, 256)
    CALL_IF(8, 16, 4, 256)
    CALL_IF(8, 8, 8, 256)
    CALL_IF(8, 8, 4, 128)
    CALL_IF(8, 4, 8, 128)
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

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }
}

}  // namespace gptq_marlin

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full) {
  // Verify num_bits
  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  int pack_factor = 32 / num_bits;

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(size_k % gptq_marlin::tile_size == 0, "size_k = ", size_k,
              " is not divisible by tile_size = ", gptq_marlin::tile_size);
  TORCH_CHECK((size_k / gptq_marlin::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k, ", tile_size = ", gptq_marlin::tile_size);
  TORCH_CHECK(b_q_weight.size(1) % gptq_marlin::tile_size == 0,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not divisible by tile_size = ", gptq_marlin::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / gptq_marlin::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
  TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");

  TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
  TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::empty({size_m, size_n}, options);
  torch::Tensor a_tmp = torch::empty({size_m, size_k}, options);

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  // Verify g_idx and perm
  TORCH_CHECK((g_idx.size(0) == 0 && perm.size(0) == 0) ||
                  (g_idx.size(0) == size_k && perm.size(0) == size_k),
              "Unexpected g_idx.size(0) = ", g_idx.size(0),
              " and perm.size(0) = ", perm.size(0),
              ", where size_k = ", size_k);

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;
  bool has_act_order = g_idx.size(0) != 0;

  int b_rank = b_scales.sizes().size();
  TORCH_CHECK(b_rank == 2, "b_scales rank = ", b_rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

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

  // Verify workspace size
  TORCH_CHECK(
      size_n % gptq_marlin::min_thread_n == 0, "size_n = ", size_n,
      ", is not divisible by min_thread_n = ", gptq_marlin::min_thread_n);
  int min_workspace_size =
      (size_n / gptq_marlin::min_thread_n) * gptq_marlin::max_par;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  if (a.scalar_type() == at::ScalarType::Half) {
    gptq_marlin::marlin_mm_f16i4<half>(
        a.data_ptr<at::Half>(), b_q_weight.data_ptr(), c.data_ptr<at::Half>(),
        b_scales.data_ptr<at::Half>(), g_idx.data_ptr(), perm.data_ptr(),
        a_tmp.data_ptr<at::Half>(), size_m, size_n, size_k,
        workspace.data_ptr(), num_bits, has_act_order, is_k_full, num_groups,
        group_size, dev, at::cuda::getCurrentCUDAStream(dev), thread_k,
        thread_n, sms, gptq_marlin::max_par);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    gptq_marlin::marlin_mm_f16i4<nv_bfloat16>(
        a.data_ptr<at::BFloat16>(), b_q_weight.data_ptr(),
        c.data_ptr<at::BFloat16>(), b_scales.data_ptr<at::BFloat16>(),
        g_idx.data_ptr(), perm.data_ptr(), a_tmp.data_ptr<at::BFloat16>(),
        size_m, size_n, size_k, workspace.data_ptr(), num_bits, has_act_order,
        is_k_full, num_groups, group_size, dev,
        at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms,
        gptq_marlin::max_par);
  } else {
    TORCH_CHECK(false, "gpt_marlin_gemm only supports bfloat16 and float16");
  }

  return c;
}

#endif
