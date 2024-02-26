"""Checks the memory usage of the sparse model is < memory usage of the
dense model by checking that the number of KV cache blocks is
bigger for the sparse model rather than the dense model.

Run `pytest tests/models/test_sparse_memory.py --forked`.
"""

import gc
import pytest
import torch

model_format_pairs = [
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16"),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned2.4",
     "semi_structured_sparse_w16a16"),
]

KV_CACHE_SIZE_INCREASE_THRESHOLD = 0.2


@pytest.mark.parametrize("model_format_pairs", model_format_pairs)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner_sparse,
    example_prompts,
    model_format_pairs,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_name, sparsity = model_format_pairs

    sparse_model = vllm_runner_sparse(model_name=model_name,
                                      sparsity=sparsity,
                                      dtype=dtype,
                                      max_model_len=1024)
    sparse_num_kv_blocks = sparse_model.model.llm_engine.scheduler.block_manager.gpu_allocator.num_blocks

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del sparse_model.model.llm_engine.driver_worker
    del sparse_model
    torch.cuda.empty_cache()
    gc.collect()

    dense_model = vllm_runner_sparse(model_name,
                                     sparsity=None,
                                     dtype=dtype,
                                     max_model_len=1024)
    dense_num_kv_blocks = dense_model.model.llm_engine.scheduler.block_manager.gpu_allocator.num_blocks

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del dense_model.model.llm_engine.driver_worker
    del dense_model
    torch.cuda.empty_cache()
    gc.collect()

    assert sparse_num_kv_blocks > dense_num_kv_blocks * (
        1 + KV_CACHE_SIZE_INCREASE_THRESHOLD
    ), (f"Test{model_name}: Sparse model KV cache size not bigger than dense model KV cache size"
        )
