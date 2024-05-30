"""Test model set-up and weight loading for sparseml-quantized models.

Run `pytest tests/quantization/test_compressed_tensors.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod, CompressedTensorsW4A16,
    CompressedTensorsW8A8StaticTensor)


def test_compressed_tensors_w8a8_static_setup(vllm_runner):
    model_path = "nm-testing/tinyllama-one-shot-static-quant-test-compressed"
    llm = vllm_runner(model_path, quantization="sparseml", enforce_eager=True)
    model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model
    layer = model.model.layers[0]

    qkv_proj = layer.self_attn.qkv_proj
    o_proj = layer.self_attn.o_proj
    gate_up_proj = layer.mlp.gate_up_proj
    down_proj = layer.mlp.down_proj

    assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
    assert isinstance(o_proj.quant_method, CompressedTensorsLinearMethod)
    assert isinstance(gate_up_proj.quant_method, CompressedTensorsLinearMethod)
    assert isinstance(down_proj.quant_method, CompressedTensorsLinearMethod)

    assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8StaticTensor)

    assert qkv_proj.weight.dtype is torch.int8
    assert o_proj.weight.dtype is torch.int8
    assert gate_up_proj.weight.dtype is torch.int8

    assert qkv_proj.weight_scale.shard_splitter is not None
    assert qkv_proj.weight_scale.logical_widths is not None
    assert qkv_proj.input_scale.dtype is torch.float32


@pytest.fixture(params=[
    ("nm-testing/tinyllama-one-shot-w4a16-channel-packed", "channel", None),
    ("nm-testing/tinyllama-one-shot-w4a16-group128-packed", "group", 128)
])
def test_compressed_tensors_w4a16(vllm_runner, model: str, strategy: str,
                                  group: int):
    llm = vllm_runner(model, quantization="sparseml", enforce_eager=True)
    model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model
    layer = model.model.layers[0]

    qkv_proj = layer.self_attn.qkv_proj
    assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
    assert isinstance(qkv_proj.scheme, CompressedTensorsW4A16)

    assert qkv_proj.scheme.strategy == strategy
    assert qkv_proj.scheme.group_size == group

    assert qkv_proj.weight_packed.dtype is torch.int32
    assert qkv_proj.weight_scale.dtype is torch.float16
    assert qkv_proj.weight_packed.pack_factor == 8
