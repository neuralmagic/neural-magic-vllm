"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""
import pytest
import torch

from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from tests.utils import is_quant_method_supported

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="FP8 is not supported on this GPU type.")
def test_load_fp16_model(vllm_runner) -> None:
    llm = vllm_runner("facebook/opt-125m", quantization="fp8")

    model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model
    fc1 = model.model.decoder.layers[0].fc1
    assert isinstance(fc1.quant_method, Fp8LinearMethod)
    assert fc1.weight.dtype == torch.float8_e4m3fn
