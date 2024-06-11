#include <Python.h>

#include "moe_ops.h"
#include "marlin_moe_ops.h"

#include <torch/extension.h>

#include <pybind11/numpy.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &topk_softmax, "Apply topk softmax to the gating outputs.");
  m.def("marlin_gemm_moe", [](torch::Tensor& a, torch::Tensor& b_q_weights, torch::Tensor& sorted_ids, torch::Tensor& topk_weights,
                        torch::Tensor& b_scales, py::array_t<int>& expert_offsets, torch::Tensor& workspace, int64_t size_m, int64_t size_n, int64_t size_k,
                        int64_t num_tokens_post_padded, int64_t num_experts, int64_t topk, int64_t moe_block_size, bool replicate_input, bool apply_weights){
    py::buffer_info expert_offsets_bo = expert_offsets.request(); 
    return marlin_gemm_moe(a, b_q_weights, sorted_ids, topk_weights, b_scales,
                    static_cast<int*>(expert_offsets_bo.ptr),
                    workspace, size_m, size_n, size_k, num_tokens_post_padded, 
                    num_experts, topk, moe_block_size, replicate_input, apply_weights);
  }, "Marlin gemm moe kernel.");
}

// // // should be enough for a unit test?
// TORCH_LIBRARY(nm_ops, m) {
//   // m.def("marlin_gemm_moe() -> Tensor");
//   // m.def("marlin_gemm_moe(Tensor a, Tensor b_q_weights, Tensor sorted_ids, Tensor b_scales, Tensor g_idx, Tensor perm, "
//   //       "Tensor workspace, int size_m, "
//   //       "int size_n, int size_k, bool is_k_full, int num_tokens_post_padded, int num_experts, int moe_block_size) -> Tensor");
// }
