"""Compare the outputs of a GPTQ model to a Marlin model.
Note: sparse kernels do not have bitwise correctness vs the dense models. 
As a result, in this test, we just confirm that the top selected tokens of the 
sparse models are in the top 3 selections of same model running dense.
Run `pytest tests/models/test_sparse.py --forked`.
"""

import time
import gc
import pytest
import torch
from compare_utils import check_logprobs_close
from dataclasses import dataclass

KV_CACHE_SIZE_INCREASE_THRESHOLD = 0.2

model_format_pairs = [
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16")
]

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

    sparse_model = vllm_runner_sparse(model_name=model_name, sparsity=sparsity, dtype=dtype, max_model_len=4096)
    sparse_num_kv_blocks = sparse_model.model.llm_engine.scheduler.block_manager.gpu_allocator.num_blocks
    sparse_outputs = sparse_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del sparse_model.model.llm_engine.driver_worker
    del sparse_model
    torch.cuda.empty_cache()
    gc.collect()

    dense_model = vllm_runner_sparse(model_name, sparsity=None, dtype=dtype, max_model_len=4096)
    dense_num_kv_blocks = dense_model.model.llm_engine.scheduler.block_manager.gpu_allocator.num_blocks
    dense_outputs = dense_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del dense_model.model.llm_engine.driver_worker
    del dense_model

    assert sparse_num_kv_blocks > dense_num_kv_blocks * (1 + KV_CACHE_SIZE_INCREASE_THRESHOLD), (
        f"Test{model_name}: Sparse model KV cache size not bigger than dense model KV cache size"
    )

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=dense_outputs,
        outputs_1_lst=sparse_outputs,
        name_0="dense",
        name_1="sparse",
    )
