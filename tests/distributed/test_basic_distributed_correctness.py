"""Compare the outputs of HF and distributed vLLM when using greedy sampling.
vLLM will allocate all the available memory, so we need to run the tests one
by one. The solution is to pass arguments (model name) by environment
variables.
Run:
```sh
cd $VLLM_PATH/tests

TEST_DIST_MODEL=facebook/opt-125m pytest \
    distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf \
    distributed/test_basic_distributed_correctness.py
```
"""
# UPSTREAM SYNC: We can only run one model per invocation of the test.
#   Otherwise, we have duplicate ray.init() calls which fails.
#   Rather than ruining .github/scripts/run-tests to pass via env
#   variables, we just run llama which is sufficient for smoke test.
import os

import pytest

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.utils import cuda_device_count_stateless

from ..models.utils import check_outputs_equal

if should_skip_test_group(group_name="TEST_DISTRIBUTED"):
    pytest.skip("TEST_DISTRIBUTED=DISABLE, skipping distributed test group",
                allow_module_level=True)

MODELS = [
    "meta-llama/Llama-2-7b-hf",
]
DISTRIBUTED_EXECUTOR_BACKEND = "DISTRIBUTED_EXECUTOR_BACKEND"


@pytest.mark.skip("Upstream test that compares 'golden' results from fp16 "
                  "model with TP, which is an invalid test strategy due to "
                  "numerical precision on GPU.")
@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    distributed_executor_backend = os.getenv(DISTRIBUTED_EXECUTOR_BACKEND)

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=2,
                     distributed_executor_backend=distributed_executor_backend
                     ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
