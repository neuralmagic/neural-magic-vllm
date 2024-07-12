"""Tests for the marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_gemm.py`.
"""

from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm import scalar_type
from vllm._custom_classes import ScalarType
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    is_marlinv2_supported)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_weights_into_int32, quantize_weights)

MNK_SHAPES = [
    (1, 128, 128),
    (1, 512, 1024),
    (1, 4096, 4096),
    (13, 8192, 4096),
    (26, 4096, 8192),
    (1, 4096, 4096),
    (257, 128, 4096),
    (257, 4224, 4096),
    (257, 4096, 4096),
    (64, 4096, 4096),
]

ACT_TYPES = [torch.float16, torch.bfloat16]
WEIGHT_TYPES = [scalar_type.s4, scalar_type.u4]
GROUP_SIZES = [128, None]


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


def marlinv2_quantize_and_pack(w, wtype: ScalarType, group_size: int):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s = quantize_weights(w, wtype, group_size)
    w_q = pack_weights_into_int32(w_q, wtype)
    w_q_marlinv2 = ops.marlinv2_prepack_B(w_q, wtype)

    return w_ref, w_q_marlinv2, w_s


@pytest.mark.skipif(not is_marlinv2_supported(),
                    reason="MarlinV2 is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype", WEIGHT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_marlinv2_all_schedules(shape, atype: torch.dtype, wtype: ScalarType,
                                group_size: Optional[int]):
    size_m, size_k, size_n = shape

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Normalize group_size
    if group_size is None:
        group_size = size_k
    assert group_size <= size_k

    a = rand_data((size_m, size_k), atype)
    w = rand_data((size_k, size_n), atype)

    w_ref, w_q_marlinv2, w_s = marlinv2_quantize_and_pack(w, wtype, group_size)

    output_ref = torch.matmul(a, w_ref)
    
    print(a.dtype, output_ref.dtype)

    for schedule in ops.marlinv2_supported_schedules(wtype):
        output = ops.marlinv2_gemm(
            a,
            b_q=w_q_marlinv2,
            b_type=wtype,
            b_scales=w_s,
            b_group_size=group_size,
            schedule=schedule,
        )
        print("output:", output.dtype, output_ref.dtype)
        assert torch.allclose(output, output_ref, rtol=1e-1, atol=1e-1
                              ), f"Schedule failed {schedule}"


@pytest.mark.skipif(not is_marlinv2_supported(),
                    reason="MarlinV2 is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype", WEIGHT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_marlinv2_heuristic(shape, atype: torch.dtype, wtype: ScalarType,
                            group_size: Optional[int]):
    size_m, size_k, size_n = shape

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Normalize group_size
    if group_size is None:
        group_size = size_k
    assert group_size <= size_k

    a_input = rand_data((size_m, size_k), atype)
    b_weight = rand_data((size_k, size_n), atype)

    w_ref, w_q_packed, w_s = marlinv2_quantize_and_pack(
        b_weight, wtype, group_size)

    output_ref = torch.matmul(a_input, w_ref)

    output = ops.marlinv2_gemm(
        a_input,
        b_q=w_q_packed,
        b_type=wtype,
        b_scales=w_s,
        b_group_size=group_size,
    )
    assert torch.allclose(output, output_ref, rtol=1e-1, atol=1e-1)
