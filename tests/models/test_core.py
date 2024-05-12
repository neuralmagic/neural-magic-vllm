"""Compare the outputs of core models against baseline implementations.

Run `pytest tests/models/test_core.py`.
"""
import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.utils_skip import should_skip_models_test_group
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

MODEL_MAX_LEN = 1024

MODELS_FP16 = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


@pytest.mark.parametrize("model", MODELS_FP16)
@pytest.mark.parametrize("dtype", ["bfloat16", "half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_fp16_models(
    vllm_runner_nm,
    hf_runner_nm,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    hf_model = hf_runner_nm(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy_logprobs_nm(example_prompts,
                                                      max_tokens, num_logprobs)

    del hf_model

    vllm_model = vllm_runner_nm(model,
                                dtype=dtype,
                                max_model_len=MODEL_MAX_LEN)
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)

    del vllm_model

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )


MODEL_FORMAT_PAIRS = [
    ("nm-testing/TinyLlama-1.1B-Chat-v1.0-pruned2.4",
     "semi_structured_sparse_w16a16"),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16"),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned2.4",
     "semi_structured_sparse_w16a16"),
]


@pytest.mark.skipif(should_skip_models_test_group(),
                    reason="Current job configured to skip this test group")
@pytest.mark.parametrize("model_format_pairs", MODEL_FORMAT_PAIRS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_sparse_models(
    vllm_runner,
    example_prompts,
    model_format_pairs,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_name, sparsity = model_format_pairs

    sparse_model = vllm_runner(model_name=model_name,
                               sparsity=sparsity,
                               dtype=dtype,
                               max_model_len=MODEL_MAX_LEN)
    sparse_outputs = sparse_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    del sparse_model

    dense_model = vllm_runner(model_name=model_name,
                              sparsity=None,
                              dtype=dtype,
                              max_model_len=MODEL_MAX_LEN)
    dense_outputs = dense_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    del dense_model

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=dense_outputs,
        outputs_1_lst=sparse_outputs,
        name_0="dense",
        name_1="sparse",
    )


MODEL_FORMAT_EXTRABLOCKS = [
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16", 2000),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned2.4",
     "semi_structured_sparse_w16a16", 2000),
]


@pytest.mark.skipif(should_skip_models_test_group(),
                    reason="Current job configured to skip this test group")
@pytest.mark.parametrize("model_format_extrablocks", MODEL_FORMAT_EXTRABLOCKS)
@pytest.mark.parametrize("dtype", ["half"])
def test_sparse_models_memory(
    vllm_runner,
    model_format_extrablocks,
    dtype: str,
) -> None:
    model_name, sparsity, num_extra_blocks = model_format_extrablocks
    dense_model = vllm_runner(model_name=model_name,
                              enforce_eager=True,
                              sparsity=None,
                              dtype=dtype,
                              max_model_len=MODEL_MAX_LEN)
    dense_gpu_alloc = (
        dense_model.model.llm_engine.scheduler.block_manager.gpu_allocator)
    dense_num_kv_blocks = dense_gpu_alloc.num_blocks

    del dense_model

    sparse_model = vllm_runner(model_name=model_name,
                               enforce_eager=True,
                               sparsity=sparsity,
                               dtype=dtype,
                               max_model_len=MODEL_MAX_LEN)
    sparse_gpu_alloc = (
        sparse_model.model.llm_engine.scheduler.block_manager.gpu_allocator)
    sparse_num_kv_blocks = sparse_gpu_alloc.num_blocks

    del sparse_model

    assert sparse_num_kv_blocks > dense_num_kv_blocks + num_extra_blocks, (
        f"Test{model_name}: Sparse model KV cache size {sparse_num_kv_blocks} "
        f"not bigger than dense model KV cache size {dense_num_kv_blocks} + "
        f"expected num_extra_blocks {num_extra_blocks}")


capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
gptq_marlin_not_supported = (
    capability < QUANTIZATION_METHODS["gptq_marlin"].get_min_capability())

MODELS_QUANT = [
    # act_order==False, group_size=channelwise
    ("robertgshaw2/zephyr-7b-beta-channelwise-gptq", "main"),
    # act_order==True, group_size=128
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "main"),
    # 8-bit, act_order==True, group_size=channelwise
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit--1g-actorder_True"),
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(should_skip_models_test_group(),
                    reason="Current job configured to skip this test group")
@pytest.mark.skipif(gptq_marlin_not_supported,
                    reason="gptq_marlin is not supported on this GPU type.")
@pytest.mark.parametrize("model", MODELS_QUANT)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner,
    example_prompts,
    model,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_name, revision = model

    # Run marlin.
    gptq_marlin_model = vllm_runner(model_name=model_name,
                                    revision=revision,
                                    dtype=dtype,
                                    quantization="marlin",
                                    max_model_len=MODEL_MAX_LEN)

    gptq_marlin_outputs = gptq_marlin_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del gptq_marlin_model

    # Run gptq.
    gptq_model = vllm_runner(model_name=model_name,
                             revision=revision,
                             dtype=dtype,
                             quantization="gptq",
                             max_model_len=MODEL_MAX_LEN)
    gptq_outputs = gptq_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)
    del gptq_model

    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=gptq_marlin_outputs,
        name_0="gptq",
        name_1="gptq_marlin",
    )
