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

#include "cuda_marlin_common.cuh"

#include <iostream>

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace marlin_moe {

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is
// documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;  // quantization scales

// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Asynchronous global->shared copy
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b,
                           FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// Given 2 floats multiply by 2 scales (halves)
__device__ inline void scale_float(float* c, FragS& s) {
  __half* s_ptr = reinterpret_cast<__half*>(&s);
  c[0] = __fmul_rn(c[0], __half2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], __half2float(s_ptr[1]));
}

// Same as above, but for act_order (each K is multiplied individually)
__device__ inline void scale4(FragB& frag_b, FragS& frag_s_1, FragS& frag_s_2,
                              FragS& frag_s_3, FragS& frag_s_4, int i) {
  __half2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<__half*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<__half*>(&frag_s_2)[i];

  __half2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<__half*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<__half*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

#endif

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int USER_THREADS =
    256;               // Note: This is only used with user-provided thread_k/n
const int STAGES = 4;  // 4 pipeline stages fit into shared memory
// const int SHARED_MEM =
//     96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

#define __CALL_IF_MOE(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,      \
                      HAS_ACT_ORDER, GROUP_BLOCKS, NUM_THREADS)               \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                              \
           thread_n_blocks == THREAD_N_BLOCKS &&                              \
           thread_k_blocks == THREAD_K_BLOCKS &&                              \
           has_act_order == HAS_ACT_ORDER && group_blocks == GROUP_BLOCKS &&  \
           num_threads == NUM_THREADS) {                                      \
    cudaFuncSetAttribute(                                                     \
        marlin_common_cu::Marlin<half, 4, NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,        \
                  THREAD_K_BLOCKS, STAGES, HAS_ACT_ORDER, GROUP_BLOCKS, true>,\
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);         \
    marlin_common_cu::Marlin<half, 4, NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
              STAGES, HAS_ACT_ORDER, GROUP_BLOCKS, true>                            \
        <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                    \
            A_ptr, B_ptr, C_ptr, sorted_ids_ptr, topk_weights_ptr, s_ptr,     \
            g_idx_ptr, num_groups, num_tokens_post_padded, expert_idx,        \
            num_experts, topk, prob_m, prob_n, prob_k, tot_m, locks,          \
            replicate_input, apply_weights);                                  \
  }

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},  // Default
    {128, 64, 128},   // Reduce N 2X, same K
    {64, 256, 256},   // Reduce K 2X, increase N 2X
    {64, 128, 128},   // Reduce K 2X, same N
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},   // Default
    {128, 128, 256},  // Reduce N 2X, increase K 2X
    {64, 128, 128},   // Reduce N 2X, same K
    {128, 64, 128},   // Reduce N 4X, increase K 2X
};

bool is_valid_config(thread_config_t const& th_config, int prob_m, int prob_n,
                     int prob_k) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // thread_k can be only 128 or 64 (because it must be less than groupsize
  // which is 128)
  if (th_config.thread_k != 128 && th_config.thread_k != 64) {
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

  return true;
}

thread_config_t determine_thread_config(int prob_m, int prob_n, int prob_k) {
  if (prob_m <= 16) {
    for (auto th_config : small_batch_thread_configs) {
      if (is_valid_config(th_config, prob_m, prob_n, prob_k)) {
        return th_config;
      }
    }

  } else {
    for (auto th_config : large_batch_thread_configs) {
      if (is_valid_config(th_config, prob_m, prob_n, prob_k)) {
        return th_config;
      }
    }
  }

  return thread_config_t{-1, -1, -1};
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

  // Set thread config
  thread_config_t th_config;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    th_config = thread_config_t{thread_k, thread_n, USER_THREADS};
  } else {
    // Auto config
    th_config = determine_thread_config(prob_m, prob_n, prob_k);
  }

  TORCH_CHECK(is_valid_config(th_config, prob_m, prob_n, prob_k),
              "Invalid thread config: thread_k = " + str(th_config.thread_k) +
                  ", thread_n = " + str(th_config.thread_n) +
                  ", num_threads = " + str(th_config.num_threads) +
                  " for MKN = [" + str(prob_m) + ", " + str(prob_k) + ", " +
                  str(prob_n) + "]");

  int num_threads = th_config.num_threads;
  thread_k = th_config.thread_k;
  thread_n = th_config.thread_n;

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

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

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
      int block_rows = ceildiv(topk_rows, blocks);
      marlin_common_cu::permute_cols_kernel<<<blocks, num_threads, 0, stream>>>(
          A_ptr, perm_ptr, a_tmp_ptr, topk_rows, prob_k, block_rows,
          USER_THREADS);
      A_ptr = a_tmp_ptr;
    }

    int tot_its = expert_offsets_ptr[expert_idx + 1] -
                  expert_offsets_ptr[expert_idx];  // prob_m;
    if (tot_its == 0) {
      continue;
    }
    int tot_m_blocks = ceildiv(tot_its, 16);
    int pad = 16 * tot_m_blocks - tot_its;

    // Main loop
    for (int i = 0; i < tot_m_blocks; i += 4) {
      int thread_m_blocks = tot_m_blocks - i;
      prob_m = tot_its - 16 * i;
      int par = 1;
      if (thread_m_blocks > 4) {
        // Note that parallel > 1 currently only works for inputs without any
        // padding
        par = (16 * thread_m_blocks - pad) / 64;
        if (par > max_par) par = max_par;
        prob_m = 64 * par;
        i += 4 * (par - 1);
        thread_m_blocks = 4;
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
