"""Compares the outputs of gptq vs gptq_marlin 

Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.

Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.

Note: This test currently fails running with --forked with the following:
    RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    To use CUDA with multiprocessing, you must use the 'spawn' start method

Run `pytest tests/models/test_gptq_marlin.py`.
"""

import random

import pytest
import torch
from compare_utils import check_logprobs_close

from tests.conftest import cleanup
from vllm.model_executor.layers.quantization import (
    _QUANTIZATION_CONFIG_REGISTRY)

MAX_MODEL_LEN = 1024

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
gptq_marlin_not_supported = (
    capability <
    _QUANTIZATION_CONFIG_REGISTRY["gptq_marlin"].get_min_capability())

test_prompts = [
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",  # noqa: E501
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",  # noqa: E501
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",  # noqa: E501
    "Describe the basic components of a neural network and how it can be trained.",  # noqa: E501
    "Write a short story about a robot that dreams for the first time.",  # noqa: E501
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",  # noqa: E501
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",  # noqa: E501
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

models = [
    # act_order==False, group_size=channelwise
    ("robertgshaw2/zephyr-7b-beta-channelwise-gptq", "main"),
    # act_order==False, group_size=128
    ("TheBloke/Llama-2-7B-GPTQ", "main"),

    # act_order==True, group_size=128
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "main"),
    # act_order==True, group_size=64
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-64g-actorder_True"),
    # act_order==True, group_size=32
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-32g-actorder_True"),
]


def gen_test_prompts():
    # Randomly shuffle prompts
    prompts = []
    for i in range(8):
        idx = random.randint(0, len(test_prompts) - 1)
        prompts.append(test_prompts[idx])

    return prompts


def run_test(vllm_runner_nm, prompts, model, dtype: str, max_tokens: int,
             num_logprobs: int, num_gpus: int, enforce_eager: bool) -> None:
    # Run gptq_marlin
    gptq_marlin_model = vllm_runner_nm(model_name=model[0],
                                       model_revision=model[1],
                                       dtype=dtype,
                                       quantization="gptq_marlin",
                                       max_model_len=MAX_MODEL_LEN,
                                       tensor_parallel_size=num_gpus,
                                       enforce_eager=enforce_eager)
    gptq_marlin_outputs = gptq_marlin_model.generate_greedy_logprobs(
        prompts, max_tokens, num_logprobs)

    del gptq_marlin_model
    cleanup()

    # Run gptq
    gptq_model = vllm_runner_nm(model_name=model[0],
                                model_revision=model[1],
                                dtype=dtype,
                                quantization="gptq",
                                max_model_len=MAX_MODEL_LEN,
                                tensor_parallel_size=num_gpus,
                                enforce_eager=enforce_eager)
    gptq_outputs = gptq_model.generate_greedy_logprobs(prompts, max_tokens,
                                                       num_logprobs)

    del gptq_model
    cleanup()

    # Compare
    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=gptq_marlin_outputs,
        name_0="gptq",
        name_1="gptq_marlin",
    )


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(gptq_marlin_not_supported,
                    reason="gptq_marlin is not supported on this GPU type.")
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_1_gpu(
    vllm_runner_nm,
    model,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    prompts = gen_test_prompts()
    run_test(vllm_runner_nm,
             prompts,
             model,
             dtype,
             max_tokens,
             num_logprobs,
             num_gpus=1,
             enforce_eager=True)


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(gptq_marlin_not_supported,
                    reason="gptq_marlin is not supported on this GPU type.")
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_2_gpu(
    vllm_runner_nm,
    model,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    prompts = gen_test_prompts()
    run_test(vllm_runner_nm,
             prompts,
             model,
             dtype,
             max_tokens,
             num_logprobs,
             num_gpus=2,
             enforce_eager=True)
