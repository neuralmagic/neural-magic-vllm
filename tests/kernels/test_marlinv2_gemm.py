"""Tests for the marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_gemm.py`.
"""

from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types, ScalarType
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_weights_into_int32, quantize_weights)
from vllm.platforms import current_platform

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
WEIGHT_TYPES = [scalar_types.s4, scalar_types.u4]
GROUP_SIZES = [128, None]
# TODO: in future PR refactor this and `is_quant_method_supported` in the kernel
#  unit tests to a common utility function. Currently the use of 
#  `is_quant_method_supported` conflates kernels with quantization methods
#  an assumption which is breaking down as quantizations methods can have 
#  have kernels and some kernels support multiple quantization methods.
IS_SUPPORTED_BY_GPU = current_platform.get_device_capability()[0] >= 9


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


def marlinv2_quantize_and_pack(w, wtype: ScalarType, group_size: int):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s = quantize_weights(w, wtype, group_size)
    w_q = pack_weights_into_int32(w_q, wtype)
    w_q_marlinv2 = ops.marlinv2_prepack_B(w_q, wtype)

    return w_ref, w_q_marlinv2, w_s


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
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


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
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
