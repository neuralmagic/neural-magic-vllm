"""Checks the memory usage of the sparse model is < memory usage of the
dense model by checking that the number of KV cache blocks is
bigger for the sparse model rather than the dense model. vLLM pre-allocates
the memory for the KV-cache after checking availability once the model
is loaded. This implies that using a compressed model should give more space
for the KV cache and thus more allocated blocks.

Run `pytest tests/models/test_sparse_memory.py --forked`.
"""

import gc

import pytest
import torch

MODEL_FORMAT_EXTRABLOCKS = [
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16", 1500),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned2.4",
     "semi_structured_sparse_w16a16", 1500),
]


@pytest.mark.parametrize("model_format_extrablocks", MODEL_FORMAT_EXTRABLOCKS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    vllm_runner,
    model_format_extrablocks,
    dtype: str,
) -> None:
    model_name, sparsity, num_extra_blocks = model_format_extrablocks
    dense_model = vllm_runner(model_name=model_name,
                              enforce_eager=True,
                              sparsity=None,
                              dtype=dtype,
                              max_model_len=1024)
    dense_gpu_alloc = (
        dense_model.model.llm_engine.scheduler.block_manager.gpu_allocator)
    dense_num_kv_blocks = dense_gpu_alloc.num_blocks

    del dense_model
    torch.cuda.empty_cache()
    gc.collect()

    sparse_model = vllm_runner(
        model_name=model_name,
        enforce_eager=True,
        sparsity=sparsity,
        dtype=dtype,
        max_model_len=1024,
    )
    sparse_gpu_alloc = (
        sparse_model.model.llm_engine.scheduler.block_manager.gpu_allocator)
    sparse_num_kv_blocks = sparse_gpu_alloc.num_blocks

    del sparse_model
    torch.cuda.empty_cache()
    gc.collect()

    assert sparse_num_kv_blocks > dense_num_kv_blocks + num_extra_blocks, (
        f"Test{model_name}: Sparse model KV cache size {sparse_num_kv_blocks} "
        f"not bigger than dense model KV cache size {dense_num_kv_blocks} + "
        f"expected num_extra_blocks {num_extra_blocks}")
