"""
Compare the outputs from identical models:
- one that is loaded from uncompressed safetensors
- one that is loaded form `compressed-tensors`.
The expectation is for the inference result in same behavior
"""
from typing import Tuple

import pytest
from compare_utils import check_logprobs_close

MODEL_MAX_LEN = 1024

# pair of same models with weight safed as
# compressed-tensors (compressed)
# and ordinary safetensors (uncompressed)
MODELS = [("neuralmagic/llama2.c-stories110M-pruned50",
           "nm-testing/llama2.c-stories110M-pruned50-compressed-tensors")]


@pytest.mark.parametrize("model_pair", MODELS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner_nm,
    example_prompts,
    model_pair: Tuple[str, str],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    model_uncompressed, model_compressed = model_pair

    vllm_model_0 = vllm_runner_nm(model_uncompressed,
                                  dtype=dtype,
                                  max_model_len=MODEL_MAX_LEN)

    vllm_outputs_0 = vllm_model_0.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    del vllm_model_0

    vllm_model_1 = vllm_runner_nm(model_compressed,
                                  dtype=dtype,
                                  max_model_len=MODEL_MAX_LEN)

    vllm_outputs_1 = vllm_model_1.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    del vllm_model_1

    # make sure that both weight types result in same
    # models i.e. same outputs
    check_logprobs_close(
        outputs_0_lst=vllm_outputs_0,
        outputs_1_lst=vllm_outputs_1,
        name_0="vllm_model_from_uncompressed_weights",
        name_1="vllm_model_from_compressed_weights",
    )
