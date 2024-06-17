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

// #include "marlin_moe_ops.h"

#include <iostream>
// #include <torch/extension.h>

template <typename T> inline std::string str(T x) { return std::to_string(x); }

namespace marlin_moe {

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template <typename T, int n> struct Vec {
  T elems[n];
  __device__ T &operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>; // quantization scales

// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
__device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   .reg .pred p;\n"
               "   setp.ne.b32 p, %0, 0;\n"
               "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
               "}\n" ::"r"((int)pred),
               "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Asynchronous global->shared copy
__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   cp.async.cg.shared.global [%0], [%1], %2;\n"
               "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n> __device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32 output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float*          c = reinterpret_cast<float*>(&frag_c);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                 "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in
// tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a    = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t  smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler
// does not seem to automatically recognize it in all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
// We mostly follow the strategy in the link below, with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB`
  // and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB     frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used only for grouped
// quantization.
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s   = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// Given 2 floats multiply by 2 scales (halfs)
__device__ inline void scale_float(float* c, FragS& s) {
  __half* s_ptr = reinterpret_cast<__half*>(&s);
  c[0]          = __fmul_rn(c[0], __half2float(s_ptr[0]));
  c[1]          = __fmul_rn(c[1], __half2float(s_ptr[1]));
}

// Same as above, but for act_order (each K is multiplied individually)
__device__ inline void scale4(FragB& frag_b, FragS& frag_s_1, FragS& frag_s_2, FragS& frag_s_3,
                              FragS& frag_s_4, int i) {
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
      // Guarantee that subsequent writes by this threadblock will be visible globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
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
    // Make sure that all writes since acquiring this barrier are visible globally, while releasing
    // the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

template <const int threads,         // number of threads in a threadblock
          const int thread_m_blocks, // number of 16x16 blocks in the m dimension (batchsize) of the
                                     // threadblock
          const int  thread_n_blocks, // same for n dimension (output)
          const int  thread_k_blocks, // same for k dimension (reduction)
          const int  stages,        // number of stages for the async global->shared fetch pipeline
          const int  group_blocks = -1 // number of consecutive 16x16 blocks with a separate
                                       // quantization scale
          >
__global__ void
MarlinMoE(const int4* __restrict__ A,       // fp16 input matrix of shape mxk
       const int4* __restrict__ B,          // 4bit quantized weight matrix of shape kxn /// TODO offset B to the beginning of right expert and use this as the func argument
       int4* __restrict__ C,                // fp16 output buffer of shape mxn
       int* __restrict__ sorted_ids,        // int32 sorted ids of experts
       float* __restrict__ topk_weights,    // float topk weights
       const int4* __restrict__ scales_ptr, // fp16 quantization scales of shape (k/groupsize)xn
       int  num_groups,                     // number of scale groups per output channel
       int  num_tokens_post_padded,         // scales_ptrs size with padding
       int  expert_idx,                     // idx of current expert
       int  num_experts,                    // number of experts
       int  topk,                           // topk parameter of moe
       int  prob_m,                         // batch dimension m
       int  prob_n,                         // output dimension n
       int  prob_k,                         // reduction dimension k
       int  tot_m,                          // total number of rows in A and C
       int* locks,                          // extra global storage for barrier synchronization
       bool replicate_input,                // do we use the same input for each expert?
       bool apply_weights                   // apply weights to output
) {

  // Each threadblock processes one "stripe" of the B matrix with (roughly) the same size, which
  // might involve multiple column "slices" (of width 16 * `thread_n_blocks`). Stripes are defined
  // as shown in the 3x3 matrix 5 SM example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it ensures good
  // utilization of all SMs for many kinds of shape and GPU configurations, while requiring as few
  // slow global cross-threadblock reductions as possible.

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
    // printf("sajdhajkshdjkashdjkashdjkahsdjkashdjk\n");
    // printf("sorted ids: %d %d %d %d\n", sorted_ids[0], sorted_ids[1], sorted_ids[2], sorted_ids[3]);
  // }

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a better partitioning
  // with less reductions
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m   = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters   = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

  if constexpr (group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the groupsize; this avoids
      // an annoying special case where a stripe starts in the middle of group.
      iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("XXXXX");
  //   const int* bhalf = reinterpret_cast<const int*>(&B[0]);
  //   const __half* shalf = reinterpret_cast<const __half*>(&scales_ptr[0]);
  //   printf("FIRST B: %d, FIRST SCALE: %f\n", bhalf[0], __half2float(shalf[0]));
  // }

  int slice_row     = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col     = slice_col_par;
  int slice_iters;     // number of threadblock tiles in the current slice
  int slice_count = 0; // total number of active threadblocks in the current slice
  int slice_idx;       // index of threadblock in current slice; numbered bottom to top

  // We can easily implement parallel problem execution by just remapping indices and advancing
  // global pointers
  if (slice_col_par >= n_tiles) {
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
    sorted_ids += (slice_col_par / n_tiles) * 16 * thread_m_blocks;
  }

  // Compute all information about the current slice which is required for synchronization.
  auto init_slice = [&]() {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count   = 1;
    slice_idx     = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      sorted_ids += 16 * thread_m_blocks;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = prob_k / 8;
  // stride of an A matrix tile in shared memory
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // between shared memory tile reads
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  // within a shared memory tile
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

  // B sizes/strides
  int           b_gl_stride     = 16 * prob_n / 32;
  constexpr int b_sh_stride     = 32 * thread_n_blocks / 4;
  int           b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int           b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta   = threads;
  constexpr int b_sh_rd_delta   = threads;
  constexpr int b_sh_stage      = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters   = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides
  int           s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_tb_groups =
      group_blocks < thread_k_blocks ? thread_k_blocks / group_blocks : 1;
  constexpr int s_sh_stage    = s_tb_groups * s_sh_stride;
  int           s_gl_rd_delta = s_gl_stride;
  constexpr int tb_k        = 16 * thread_k_blocks;

  constexpr int sorted_sh_stride = threads;
  constexpr int sorted_gl_stride = threads;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  // For act_order
  constexpr int k_iter_size                = tb_k / b_sh_wr_iters;
  int           slice_k_start              = tb_k * slice_row;
  int           slice_k_start_shared_fetch = slice_k_start;

  // No act_order
  int s_gl_rd;
  if constexpr (group_blocks == -1 || group_blocks == 0) {
    s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
  } else {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
        s_sh_stride * slice_col + threadIdx.x;
  }
  int  s_sh_wr      = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // We use a different scale layout for grouped and column-wise quantization as we scale a `half2`
  // tile in column-major layout in the former and in row-major in the latter case.
  int s_sh_rd;
  if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;

  int shs_size = group_blocks > 0 ? stages * s_sh_stage : threads;

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines.
  int4* sh_a      = sh;
  int4* sh_b      = sh_a + (stages * a_sh_stage);
  int4* sh_s      = sh_b + (stages * b_sh_stage);
  // printf("%f %f %f\n", __half2float(reinterpret_cast<__half*>(sh_a)[0]), __half2float(reinterpret_cast<__half*>(sh_b)[0]), __half2float(reinterpret_cast<__half*>(sh_s)[0]));
  int*  sh_sorted = (int*)(sh_s + shs_size);

  // Precompute which thread should not read memory in which iterations; this is needed if there are
  // more threads than required for a certain tilesize or when the batchsize is not a multiple
  // of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  // int mcols = replicate_input ? 1 : topk;
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++) {
    int a_idx = a_sh_wr_delta * i + a_sh_wr;
    int row = a_idx / a_gl_rd_delta_o;
    if (row >= prob_m) {
      a_sh_wr_pred[i] = false;
    }
    else {
      // if (threadIdx.x == 0) {
      //   int sorted_row = sorted_ids[row] / (replicate_input ? topk : 1);
      //   int new_idx = sorted_row * a_gl_rd_delta_o + a_idx % a_gl_rd_delta_o;
      //   bool onevar = sorted_row >= 0 && sorted_row < tot_m * mcols && new_idx < a_gl_stride * tot_m * mcols;
      //   bool twovar = a_sh_wr_delta * i + a_sh_wr < a_gl_stride * prob_m;
      //   if (onevar != twovar)
      //     printf("%d vs. %d\n", onevar, twovar);
      // }
      a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
      // a_sh_wr_pred[i] = sorted_row < tot_m * mcols && new_idx < a_sh_stride * tot_m * mcols;
    }
  }

  // To ensure that writing and reading A tiles to/from shared memory, the latter in fragment
  // format, is fully bank conflict free, we need to use a rather fancy XOR-based layout. The key
  // here is that neither reads nor writes of the 16-byte `int4` blocks of 8 consecutive threads
  // involve the same shared memory banks. Further, it seems (based on NSight-Compute) that each
  // warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main loop unrolls, all
  // shared memory accesses are static, we simply precompute both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at runtime; we break
  // dependicies between subsequent accesses with a tile by maintining multiple pointers (we have
  // enough registers), a tiny optimization.
  const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4    frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  // Zero accumulators.
  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next shared memory pipeline
  // location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        int a_idx = a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off;
        int row = a_idx / a_gl_stride;
        int sorted_row = replicate_input ? sorted_ids[row] / topk : sorted_ids[row];
        int new_idx = sorted_row * a_gl_stride + a_idx % a_gl_stride;
        // if (threadIdx.x < 8 && blockIdx.x == 80) {
        //     // int mcols = replicate_input ? 1 : topk;
        //     // bool check = sorted_row >= 0 && sorted_row < tot_m * mcols && new_idx < a_gl_stride * tot_m * mcols;
        //     // printf("%d vs. %d\n", check, a_sh_wr_pred[i]);
        //     printf("row: %d -> %d\n", row, sorted_row);
        //     // printf("row: %d -> %d, sh: %d -> %d ? %d // %d, %d, %d\n", row, sorted_row, i,
        //     //     a_sh_wr_trans[i], a_sh_wr_pred[i], tot_m * (replicate_input ? 1 : topk), a_sh_wr_iters, stages * a_sh_stage);
        //   }
        if (sorted_row < tot_m * (replicate_input ? 1 : topk)
            && new_idx < a_gl_stride * tot_m * (replicate_input ? 1 : topk)) {
          cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]],
                         &A[new_idx],
                         a_sh_wr_pred[i]);
          // if (threadIdx.x == 0) {
          //   printf("reached pred (%d, %d)\n", blockIdx.x, i);
          // }
          // if (a_sh_wr_pred[i]) {
          //   if (threadIdx.x == 0) {
          //     printf("reached inside condition (%d, %d)\n", blockIdx.x, i);
          //   }
          //   // int4 a_elem = A[new_idx];
          //   // if (threadIdx.x == 0) {
          //   //   __half* elem = reinterpret_cast<__half*>(&a_elem);
          //   //   printf("A elem: %f (%d, %d)\n", __half2float(elem[0]), blockIdx.x, i);
          //   // }
          //   // int trans_elem = a_sh_wr_trans[i];
          //   // if (threadIdx.x == 0) {
          //   //   printf("Trans elem: %d (%d, %d)\n", trans_elem, blockIdx.x, i);
          //   // }
          //   sh_a_stage[a_sh_wr_trans[i]] = A[new_idx];
          //   // if (threadIdx.x == 0) {
          //   //   __half* elem = reinterpret_cast<__half*>(&sh_a_stage[a_sh_wr_trans[i]]);
          //   //   printf("Done: %f (%d, %d)\n", __half2float(elem[0]), blockIdx.x, i);
          //   // }
          // }
        }
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }

      if constexpr (group_blocks != -1) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        if constexpr (group_blocks >= thread_k_blocks) {
          // Only fetch scales if this tile starts a new group
          if (pipe % (group_blocks / thread_k_blocks) == 0) {
            if (s_sh_wr_pred) {
              cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
            }
            s_gl_rd += s_gl_rd_delta;
          }
        } else {
          for (int i = 0; i < s_tb_groups; i++) {
            if (s_sh_wr_pred) {
              cp_async4(&sh_s_stage[i * s_sh_stride + s_sh_wr], &scales_ptr[s_gl_rd]);
            }
            s_gl_rd += s_gl_rd_delta;
          }
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that waiting is also
    // correct at this point.
    cp_async_fence();
  };

  auto fetch_sorted_ids_to_shared = [&]() {
    const int mpt = ceildiv(prob_m, threads);
    for (int i = 0; i < mpt; i++) {
      if ((i * sorted_gl_stride) + threadIdx.x < prob_m) {
        // printf("load %d -> %d to shared sorted, eid: %d  (%d)\n", (i * sorted_gl_stride) + threadIdx.x,
        //     sorted_ids[(i * sorted_gl_stride) + threadIdx.x], expert_idx, (i * sorted_sh_stride) + threadIdx.x);
        sh_sorted[(i * sorted_sh_stride) + threadIdx.x] =
            sorted_ids[(i * sorted_gl_stride) + threadIdx.x];
        // printf("load %d -> %d to shared sorted, eid: %d  (%d / %d)\n", (i * sorted_gl_stride) + threadIdx.x,
        //     sorted_ids[(i * sorted_gl_stride) + threadIdx.x], expert_idx, (i * sorted_sh_stride) + threadIdx.x,
        //         sh_sorted[(i * sorted_sh_stride) + threadIdx.x]);
      }
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering and can only issue the
    // next fetch when it is guaranteed that the previous shared memory load is fully complete (as
    // it may otherwise be overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe into the current
  // register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] =
        *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;

    if constexpr (group_blocks != -1) {
      if constexpr (group_blocks >= thread_k_blocks) {
        int4* sh_s_stage = sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                                                (pipe / (group_blocks / thread_k_blocks)));
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
      } else {
        int warp_id = threadIdx.x / 32;
        int n_warps = thread_n_blocks / 4;

        int warp_row = warp_id / n_warps;

        int cur_k = warp_row * 16;
        cur_k += k_iter_size * (k % b_sh_wr_iters);

        int k_blocks     = cur_k / 16;
        int cur_group_id = k_blocks / group_blocks;

        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
            sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
      }
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  auto matmul = [&](int k) {
// We have the m dimension as the inner loop in order to encourage overlapping dequantization and
// matmul operations.
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant       = frag_b_quant[k % 2][j];
      int b_quant_shift = b_quant >> 8;

      FragB frag_b0 = dequant(b_quant);

      // Apply scale to frag_b0
      if constexpr (group_blocks != -1) {
        scale(frag_b0, frag_s[k % 2][j], 0);
      }

      FragB frag_b1 = dequant(b_quant_shift);

      // Apply scale to frag_b1
      if constexpr (group_blocks != -1) {
        scale(frag_b1, frag_s[k % 2][j], 1);
      }

#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the number of warps while
  // keeping the n dimension of a tile reasonable, we have multiple warps that accumulate their
  // partial sums of the same output location; which we have to reduce over in the end. We do in
  // shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int           red_idx       = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta  = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any unnecessary read or
      // write iterations, e.g., for two warps we write only once by warp 1 and read only once by
      // warp 0.

#pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
#pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we finally have to
  // globally reduce over the results. As the striped partioning minimizes the number of such
  // reductions and our outputs are usually rather small, we perform this reduction serially in L2
  // cache.
  auto global_reduce = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to maximize L2 cache
    // utilization in this step. To do this, we write out results in FP16 (but still reduce with
    // FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride     = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr =
          c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int           c_sh_wr       = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
// Interestingly, doing direct global accesses here really seems to mess up the compiler and lead to
// slowdowns, hence we also use async-copies even though these fetches are not actually
// asynchronous.
#pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          int c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
          int sorted_row = sorted_ids[c_idx / c_gl_stride];
          int new_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
          cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i],
                        &C[new_idx],
                        sorted_row < tot_m * topk
                            && (8 * (i / 2) + row < prob_m && (i < (thread_m_blocks - 1) * 4
                                || sorted_ids[8 * (i / 2) + row] < tot_m * topk)));
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

#pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (8 * (i / 2) + row < prob_m && (i < (thread_m_blocks - 1) * 4
            || sorted_ids[8 * (i / 2) + row] < tot_m * topk)) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                  __half2float(reinterpret_cast<__half*>(&c_red)[j]);
            }
          }
          if (!last) {
            int4 c;
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<__half*>(&c)[j] = __float2half(
                  reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]);
            }
            int c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
            int row = sorted_ids[c_idx / c_gl_stride];;
            if (row < tot_m * topk) {
              int new_idx = row * c_gl_stride + c_idx % c_gl_stride;
              C[new_idx] = c;
            }
          }
        }
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually reshuffle matrix
  // fragments in this step, the reduction above is performed in fragment layout.
  auto write_result = [&]() {
    int           c_gl_stride   = prob_n / 8;
    constexpr int c_sh_stride   = 2 * thread_n_blocks + 1;
    int           c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr =
        c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd =
        c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s) {
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));

      // For per-column quantization we finally apply the scale here
      if constexpr (group_blocks == -1) {
        res = __hmul2(res, s[0]);
      }

      ((half2*)sh)[idx] = res;
    };
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1],
                frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3],
                frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1],
                frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3],
                frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        int row = sorted_ids[c_gl_wr / c_gl_stride];
        if (row < tot_m * topk) {
          int off = row * c_gl_stride + c_gl_wr % c_gl_stride;
          // TODO make function that multiplies everything in sh[] by scalar topk_weights
          if (!apply_weights) {
            // if (c_gl_wr % c_gl_stride == 0) {
            //   printf("wr w/o apply %d -> %d\n", c_gl_wr / c_gl_stride, row);
            // }
            C[off] = sh[c_sh_rd];
          } else {
            // if (c_gl_wr % c_gl_stride == 0) {
            //   printf("wr w/ apply %d -> %d\n", c_gl_wr / c_gl_stride, row);
            // }
            __half* ctrg = reinterpret_cast<__half*>(&C[off]);
            __half* csrc = reinterpret_cast<__half*>(&sh[c_sh_rd]);
            // if (/*expert_idx == 0 && */row < 64 && c_gl_wr % c_gl_stride >= 250 && c_gl_wr % c_gl_stride < 260) {
            // printf("add to row: %d -> %d and col: %d (b: %d, t: %d, e: %d) --- %f += %f\n",
            //     c_gl_wr / c_gl_stride, row, c_gl_wr % c_gl_stride, blockIdx.x, threadIdx.x, expert_idx,
            //     __half2float(ctrg[0]), __half2float(csrc[0]));
            // }
            for (int j = 0; j < 8; ++j) {
              // __half old = ctrg[j];
              ctrg[j] = __float2half(topk_weights[row] * __half2float(csrc[j]));
            }
          }
          c_gl_wr += c_gl_wr_delta;
          c_sh_rd += c_sh_rd_delta;
        }
        else {
          if (expert_idx == 0) {
          // printf("don't add to row: %d -> %d and col: %d (b: %d, t: %d, e: %d)\n",
          //     c_gl_wr / c_gl_stride, row, c_gl_wr % c_gl_stride, blockIdx.x, threadIdx.x, expert_idx);
          }
        }
      }
    }
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {

  // fetch_sorted_ids_to_shared();
  __syncthreads();

#pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    slice_k_start_shared_fetch += tb_k * (stages - 1);
  };
  if (slice_iters) {
    start_pipes();
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to ensure all shared
    // memory accesses are static. Note that both pipelines have even length meaning that the next
    // iteration will always start at index 0.
#pragma unroll
    for (int pipe = 0; pipe < stages;) {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd += a_gl_rd_delta_o * stages;
    slice_k_start += tb_k * stages;
    slice_k_start_shared_fetch += tb_k * stages;

    // Process results and, if necessary, proceed to the next column slice. While this pattern may
    // not be the most readable, other ways of writing the loop seemed to noticeably worse
    // performance after compliation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before write-out
      if constexpr (group_blocks == -1) {
        if (last) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();
      if constexpr (group_blocks == -1) {
        if (last) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
          }
        }
      }
      if (slice_count > 1) { // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last) // only the last block in a slice actually writes the result
        write_result();
      slice_row = 0;
      slice_col_par++;
      
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
#pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= b_gl_stride;
        }

        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

#else

template <const int threads,         // number of threads in a threadblock
          const int thread_m_blocks, // number of 16x16 blocks in the m
                                     // dimension (batchsize) of the threadblock
          const int thread_n_blocks, // same for n dimension (output)
          const int thread_k_blocks, // same for k dimension (reduction)
          const int stages, // number of stages for the async global->shared
                            // fetch pipeline
          const int group_blocks = -1 // number of consecutive 16x16 blocks with
                                      // a separate quantization scale
          >
__global__ void
MarlinMoE(const int4* __restrict__ A,       // fp16 input matrix of shape mxk
       const int4* __restrict__ B,          // 4bit quantized weight matrix of shape kxn /// TODO offset B to the beginning of right expert and use this as the func argument
       int4* __restrict__ C,                // fp16 output buffer of shape mxn
       int* __restrict__ sorted_ids,        // int32 sorted ids of experts
       float* __restrict__ topk_weights,    // float topk weights
       const int4* __restrict__ scales_ptr, // fp16 quantization scales of shape (k/groupsize)xn
       int  num_groups,                     // number of scale groups per output channel
       int  num_tokens_post_padded,         // scales_ptrs size with padding
       int  expert_idx,                     // idx of current expert
       int  num_experts,                    // number of experts
       int  topk,                           // topk parameter of moe
       int  prob_m,                         // batch dimension m
       int  prob_n,                         // output dimension n
       int  prob_k,                         // reduction dimension k
       int  tot_m,                          // total number of rows in A and C
       int* locks,                          // extra global storage for barrier synchronization
       bool replicate_input,                // do we use the same input for each expert?
       bool apply_weights                   // apply weights to output
) {
  // Marlin is not implemented yet for SM < 8.0
  assert(false);
  return;
}

#endif

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int USER_THREADS = 256;              // Note: This is only used with user-provided thread_k/n
const int STAGES = 4; // 4 pipeline stages fit into shared memory
// const int SHARED_MEM =
//     96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

#define __CALL_IF_MOE(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS,  \
                  NUM_THREADS)                                                                     \
  else if (thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&             \
           thread_k_blocks == THREAD_K_BLOCKS &&                 \
           group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {                           \
    cudaFuncSetAttribute(MarlinMoE<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,    \
                                STAGES, GROUP_BLOCKS>,                         \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);             \
    MarlinMoE<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES,            \
           GROUP_BLOCKS><<<blocks, NUM_THREADS, max_shared_mem, stream>>>(          \
        A_ptr, B_ptr, C_ptr, sorted_ids_ptr, topk_weights_ptr, s_ptr, \
        num_groups, num_tokens_post_padded, expert_idx, num_experts, topk, \
        prob_m, prob_n, prob_k, tot_m, locks, replicate_input, apply_weights);         \
  }

  typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256}, // Default
    {128, 64, 128},  // Reduce N 2X, same K
    {64, 256, 256},  // Reduce K 2X, increase N 2X
    {64, 128, 128},  // Reduce K 2X, same N
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},  // Default
    {128, 128, 256}, // Reduce N 2X, increase K 2X
    {64, 128, 128},  // Reduce N 2X, same K
    {128, 64, 128},  // Reduce N 4X, increase K 2X
};

bool is_valid_config(thread_config_t const &th_config, int prob_m, int prob_n,
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

  #define CALL_IF_MOE(N_BLOCKS, K_BLOCKS, NUM_THREADS)                                                   \
  __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                         \
  __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, 2, NUM_THREADS)                                          \
  __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, 4, NUM_THREADS)                                          \
  __CALL_IF_MOE(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                                          \
                                                                                                   \
  __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                         \
  __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, 2, NUM_THREADS)                                          \
  __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, 4, NUM_THREADS)                                          \
  __CALL_IF_MOE(2, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                                          \
                                                                                                   \
  __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                         \
  __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, 2, NUM_THREADS)                                          \
  __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, 4, NUM_THREADS)                                          \
  __CALL_IF_MOE(3, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                                          \
                                                                                                   \
  __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                         \
  __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, 2, NUM_THREADS)                                          \
  __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, 4, NUM_THREADS)                                          \
  __CALL_IF_MOE(4, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)

void marlin_mm_moe_f16i4(const void* A, const void* B, void* C, void* sorted_ids, void* topk_weights, void* s,
                     int* expert_offsets, int prob_m, int prob_n, int prob_k,
                     void* workspace, int num_groups, int group_size,
                     int num_tokens_post_padded, int num_experts, int topk, int moe_block_size,
                     int dev, cudaStream_t stream, int thread_k, int thread_n, int sms, int max_par,
                     bool replicate_input, bool apply_weights) {

  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ",
              prob_k, "]");

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
              "Invalid thread config: thread_k = " + str(th_config.thread_k) + ", thread_n = " +
                  str(th_config.thread_n) + ", num_threads = " + str(th_config.num_threads) +
                  " for MKN = [" + str(prob_m) + ", " + str(prob_k) + ", " + str(prob_n) + "]");

  int num_threads = th_config.num_threads;
  thread_k        = th_config.thread_k;
  thread_n        = th_config.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  int blocks = sms;

  TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
              " is not divisible by thread_n = ", thread_n);
  TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
              " is not divisible by thread_k = ", thread_k);

  int group_blocks = 0;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                " is not divisible by group_blocks = ", group_blocks);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  int tot_m = prob_m;

  // printf("run loop for %d %d %d and topk: %d\n", prob_m, prob_n, prob_k, topk);

  for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const int4* A_ptr     = (const int4*)A;
    const int4* B_ptr     = (const int4*)B + (prob_n * prob_k / 32) * expert_idx;
    int4*       C_ptr     = (int4*)C;
    int*        sorted_ids_ptr  = (int*)sorted_ids + expert_offsets[expert_idx];
    float*      topk_weights_ptr = (float*)topk_weights;
    const int4* s_ptr     = (const int4*)s + (((group_size == -1 || group_size == 0) ? 1 : prob_k / group_size) * prob_n / 8) * expert_idx;

    // printf("%d * %d vs. %d * %d\n", prob_n, prob_k / 32, prob_k / group_size, prob_n / 8);

    // printf("expert offset: %d, group offset: %d, (mult: %d), expert: %d, gs: %d\n",
    //     (prob_n * prob_k / 32) * expert_idx,
    //     (((group_size == -1 || group_size == 0) ? 1 : prob_k / group_size) * prob_n / 8) * expert_idx,
    //     (prob_n * prob_k / 32) * expert_idx * group_size / 8,
    //     expert_idx, group_size);

    int* locks = (int*)workspace;

    int tot_its        = expert_offsets[expert_idx + 1] - expert_offsets[expert_idx]; //prob_m;
    if (tot_its == 0) {
      continue;
    }
    int tot_m_blocks = ceildiv(tot_its, 16);
    int pad          = 16 * tot_m_blocks - tot_its;

    // Main loop
    for (int i = 0; i < tot_m_blocks; i += 4) {
      int thread_m_blocks = tot_m_blocks - i;
      prob_m              = tot_its - 16 * i;
      int par             = 1;
      if (thread_m_blocks > 4) {
        // Note that parallel > 1 currently only works for inputs without any padding
        par = (16 * thread_m_blocks - pad) / 64;
        if (par > max_par)
          par = max_par;
        prob_m = 64 * par;
        i += 4 * (par - 1);
        thread_m_blocks = 4;
      }
      // printf("main loop it: %d/%d (tot its: %d)\n", i, tot_m_blocks, tot_its);

      // Define kernel configurations

      if (false) {
      }
      CALL_IF_MOE(16, 4, 256)
      CALL_IF_MOE(8, 8, 256)
      CALL_IF_MOE(8, 4, 128)
      CALL_IF_MOE(4, 8, 128)
      else {
        TORCH_CHECK(false, "Unsupported shapes: MNK = [" + str(prob_m) + ", " + str(prob_n) + ", " +
                              str(prob_k) + "]" +
                              ", num_groups = " + str(num_groups) + ", group_size = " +
                              str(group_size) + ", thread_m_blocks = " + str(thread_m_blocks) +
                              ", thread_n_blocks = " + str(thread_n_blocks) +
                              ", thread_k_blocks = " + str(thread_k_blocks));
      }

      // A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
      // C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
      sorted_ids_ptr += 16 * thread_m_blocks * par;
    }
  }
}

} // namespace marlin_moe

torch::Tensor marlin_gemm_moe(torch::Tensor& a, torch::Tensor& b_q_weights, torch::Tensor& sorted_ids, torch::Tensor& topk_weights,
                        torch::Tensor& b_scales, torch::Tensor& expert_offsets, torch::Tensor& workspace, int64_t size_m, int64_t size_n, int64_t size_k,
                        int64_t num_tokens_post_padded, int64_t num_experts, int64_t topk, int64_t moe_block_size, bool replicate_input, bool apply_weights)
{
  int max_par = 4;

  int dev = a.get_device();

  auto          options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c       = torch::empty({size_m, topk, size_n}, options); 

  // // thread_k: `k` size of a thread_tile in `weights` (can usually be left as auto -1)
  // int thread_k = -1;
  // // thread_n: `n` size of a thread_tile in `weights` (can usually be left as auto -1)
  // int thread_n = -1;
  // // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  // int sms = -1;

  // // Detect groupsize and act_order
  // int  num_groups    = -1;
  // int  group_size    = -1;

  // int b_rank = b_scales.sizes().size();
  // TORCH_CHECK(b_rank == 3, "b_scales rank = ", b_rank, " is not 3");
  // TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2),
  //             " is not size_n = ", size_n);
  // num_groups = b_scales.size(1);
  // // printf("NUM GROUPS: %d\n", num_groups);

  // if (num_groups > 1) {
  //   TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
  //               ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
  //   group_size = size_k / num_groups;
  // } else {
  //   group_size = -1;
  // }

  // int* eoff_f = (int*)(expert_offsets.data_ptr());

  // printf("offf: %d\n", eoff_f[0]);

  // marlin_moe::marlin_mm_moe_f16i4(a.data_ptr(), b_q_weights.data_ptr(), c.data_ptr(),
  //               sorted_ids.data_ptr(), topk_weights.data_ptr(), b_scales.data_ptr(),
  //               expert_offsets.data_ptr(), size_m, size_n, size_k,
  //               workspace.data_ptr(), num_groups, group_size,
  //               num_tokens_post_padded, num_experts, topk, moe_block_size,
  //               dev, at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms,
  //               max_par, replicate_input, apply_weights);
  return c;
}
