"""Compare the outputs of a GPTQ model to a Marlin_24 model.

Note: GPTQ and Marlin_24 do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.

Run `pytest tests/models/test_marlin_24.py`.
"""
from dataclasses import dataclass

import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

if should_skip_test_group(group_name="TEST_MODELS"):
    pytest.skip("TEST_ALL_MODELS=0, skipping model test group",
                allow_module_level=True)

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
marlin_not_supported = (capability <
                        QUANTIZATION_METHODS["marlin"].get_min_capability())


@dataclass
class ModelPair:
    model_marlin: str
    model_gptq: str


model_pairs = [
    # 4-bit, group_size == 128
    ModelPair(model_marlin="alexm-nm/tinyllama-24-marlin24-4bit-g128",
              model_gptq="alexm-nm/tinyllama-24-gptq-4bit-g128"),
    # 4-bit, group_size == channelwise
    ModelPair(model_marlin="alexm-nm/tinyllama-24-marlin24-4bit-channelwise",
              model_gptq="alexm-nm/tinyllama-24-gptq-4bit-channelwise"),

    # 8-bit, group_size == 128
    ModelPair(model_marlin="alexm-nm/tinyllama-24-marlin24-8bit-g128",
              model_gptq="alexm-nm/tinyllama-24-gptq-8bit-g128"),
    # 8-bit, group_size == channelwise
    ModelPair(model_marlin="alexm-nm/tinyllama-24-marlin24-8bit-channelwise",
              model_gptq="alexm-nm/tinyllama-24-gptq-8bit-channelwise"),
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(marlin_not_supported,
                    reason="Marlin24 is not supported on this GPU type.")
@pytest.mark.parametrize("model_pair", model_pairs)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner,
    example_prompts,
    model_pair: ModelPair,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    marlin_24_model = vllm_runner(model_pair.model_marlin,
                                  dtype=dtype,
                                  quantization="gptq_marlin_24")
    marlin_24_outputs = marlin_24_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del marlin_24_model

    gptq_model = vllm_runner(model_pair.model_gptq,
                             dtype=dtype,
                             quantization="gptq")
    gptq_outputs = gptq_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)
    del gptq_model

    check_logprobs_close(
        outputs_0_lst=gptq_outputs,
        outputs_1_lst=marlin_24_outputs,
        name_0="gptq",
        name_1="marlin_24",
    )
