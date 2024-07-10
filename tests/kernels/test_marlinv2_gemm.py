"""Tests for the marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_gemm.py`.
"""

from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm import common_types as vllm_type
from vllm._custom_classes import VLLMType
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    is_marlin_supported)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack, quantize_weights)

MNK_SHAPES = [
    (1, 128, 128),
    (1, 512, 1024),
    (1, 4096, 4096),
    (13, 8192, 4096),
    (26, 4096, 4224),
    (1, 4096, 4096),
    (257, 128, 4096),
    (257, 4224, 4096),
    (257, 4096, 4096),
    (64, 4096, 4096),
]

ACT_TYPES = [torch.float16]  # torch.bfloat16 TODO
WEIGHT_TYPES = [vllm_type.int4, vllm_type.uint4]
GROUP_SIZES = [128, None]


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


def marlinv2_quantize_and_pack(b_weight, wtype: VLLMType, group_size: int):
    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, _, _ = quantize_weights(
        b_weight,
        wtype.size_bits,
        group_size,
        False,
        zero_point="symmetric" if wtype.signed else 0,
    )

    if wtype.signed:
        # quantize_weights uses the midpoint as the zero-point,
        #  and always produces unsigned values
        zero_point = (2**wtype.size_bits) // 2
        q_w -= zero_point

    assert wtype.integer, "TODO: support floating point weights"

    # Pack to GPTQ format
    q_w_gptq = gptq_pack(q_w, wtype.size_bits, *b_weight.shape)
    q_w_gptq = q_w_gptq.t().contiguous().t()
    q_w_marlinv2 = ops.marlinv2_prepack_B(q_w_gptq, wtype)

    return w_ref, q_w_marlinv2, s


@pytest.mark.skipif(not is_marlin_supported(),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype", WEIGHT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_marlinv2_all_schedules(shape, atype: torch.dtype, wtype: VLLMType,
                                group_size: Optional[int]):
    size_m, size_k, size_n = shape

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Normalize group_size
    if group_size is None:
        group_size = size_k
    assert group_size <= size_k

    a_input = rand_data((size_m, size_k), atype)
    b_weight = rand_data((size_k, size_n))

    w_ref, q_w_marlinv2, s = marlinv2_quantize_and_pack(
        b_weight, wtype, group_size)

    output_ref = torch.matmul(a_input, w_ref)

    for schedule in ops.marlinv2_supported_schedules(wtype):
        output = ops.marlinv2_gemm(
            a_input,
            b_q_weight=q_w_marlinv2,
            b_type=wtype,
            b_scales=s,
            b_group_size=group_size,
            schedule=schedule,
        )
        assert torch.allclose(output, output_ref, rtol=1e-1,
                              atol=1), f"Schedule failed {schedule}"


@pytest.mark.skipif(not is_marlin_supported(),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype", WEIGHT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_marlinv2_heuristic(shape, atype: torch.dtype, wtype: VLLMType,
                            group_size: Optional[int]):
    size_m, size_k, size_n = shape

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Normalize group_size
    if group_size is None:
        group_size = size_k
    assert group_size <= size_k

    a_input = rand_data((size_m, size_k), atype)
    b_weight = rand_data((size_k, size_n))

    w_ref, q_w_marlinv2, s = marlinv2_quantize_and_pack(
        b_weight, wtype, group_size)

    output_ref = torch.matmul(a_input, w_ref)

    output = ops.marlinv2_gemm(
        a_input,
        b_q_weight=q_w_marlinv2,
        b_type=wtype,
        b_scales=s,
        b_group_size=group_size,
    )
    assert torch.allclose(output, output_ref, rtol=1e-1, atol=1)
