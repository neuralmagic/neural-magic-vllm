"""Tests for the marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_gemm.py`.
"""

from typing import Optional, Tuple

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_weights_into_int32, quantize_weights)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

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
WTYPE_ZEROPOINTS = [
    # GPTQ style
    (scalar_types.uint4b8, False), 
    (scalar_types.uint8b128, False),
    # AWQ style
    (scalar_types.uint4, True),
    (scalar_types.uint8, True),
]
    

# TODO: in future PR refactor this and `is_quant_method_supported` in the kernel
#  unit tests to a common utility function. Currently the use of
#  `is_quant_method_supported` conflates kernels with quantization methods
#  an assumption which is breaking down as quantizations methods can have
#  have kernels and some kernels support multiple quantization methods.
IS_SUPPORTED_BY_GPU = current_platform.get_device_capability()[0] >= 9


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


def maybe_convert_zeropoints(zps: Optional[torch.Tensor], s: torch.Tensor, dtype):
    print(zps)
    return zps if zps is None else -1*s*(zps.to(dtype))


def marlinv2_quantize_and_pack(w, wtype: ScalarType, group_size: int, 
                               zero_points: bool = False):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(w, wtype, group_size, 
                                          zero_points=zero_points)
    w_q = pack_weights_into_int32(w_q, wtype)
    w_q_marlinv2 = ops.marlinv2_prepack_B(w_q, wtype)

    return w_ref, w_q_marlinv2, w_s, w_zp


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="MarlinV2 is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype_zeropoints", WTYPE_ZEROPOINTS)
@pytest.mark.parametrize("group_size", [128, None])
def test_marlinv2_all_schedules(shape, atype: torch.dtype, 
                                     wtype_zeropoints: Tuple[ScalarType, bool],
                                     group_size: Optional[int]):
    size_m, size_k, size_n = shape
    wtype, zero_points = wtype_zeropoints

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Normalize group_size
    if group_size is None:
        group_size = size_k
    assert group_size <= size_k

    a = rand_data((size_m, size_k), atype)
    w = rand_data((size_k, size_n), atype)

    w_ref, w_q_marlinv2, w_s, w_zp = marlinv2_quantize_and_pack(
        w, wtype, group_size, zero_points)

    output_ref = torch.matmul(a, w_ref)

    for schedule in ops.marlinv2_supported_schedules(wtype):
        output = ops.marlinv2_gemm(
            a,
            b_q=w_q_marlinv2,
            b_type=wtype,
            b_scales=w_s,
            b_zeros=maybe_convert_zeropoints(w_zp, w_s, a.dtype),
            b_group_size=group_size,
            schedule=schedule,
        )
        print(w)
        print(torch.max(w))
        print(torch.min(w))
        idx = torch.argmax(torch.abs(output - output_ref), keepdim=True)
        idx2 = torch.argmax(torch.abs(output - output_ref) / torch.abs(output_ref))
        idx3 = torch.argmax((torch.abs(output - output_ref) - 0.1) / torch.abs(output_ref))
        print(torch.max(torch.abs(w_s)), torch.max(torch.abs(output - output_ref)))
        print(output.flatten()[idx], output_ref.flatten()[idx])
        print(torch.max((torch.abs(output - output_ref) - 0.1) / torch.abs(output_ref)))
        print(output.flatten()[idx2], output_ref.flatten()[idx2])
        print(output.flatten()[idx3], output_ref.flatten()[idx3])
        print(torch.allclose(output, output_ref, rtol=1e-1,
                              atol=1e-1))
        print(maybe_convert_zeropoints(w_zp, w_s, a.dtype))
        assert torch.allclose(output, output_ref, rtol=1e-1,
                              atol=1e-1), f"Schedule failed {schedule}"


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="MarlinV2 is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype_zeropoints", WTYPE_ZEROPOINTS)
@pytest.mark.parametrize("group_size", [128, None])
def test_marlinv2_heuristic(shape, atype: torch.dtype, 
                                 wtype_zeropoints: Tuple[ScalarType, bool],
                                 group_size: Optional[int]):
    size_m, size_k, size_n = shape
    wtype, zero_points = wtype_zeropoints

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Normalize group_size
    if group_size is None:
        group_size = size_k
    assert group_size <= size_k

    a = rand_data((size_m, size_k), atype)
    b_weight = rand_data((size_k, size_n), atype)

    w_ref, w_q_packed, w_s, w_zp = marlinv2_quantize_and_pack(
        b_weight, wtype, group_size, zero_points)

    output_ref = torch.matmul(a, w_ref)

    output = ops.marlinv2_gemm(
        a,
        b_q=w_q_packed,
        b_type=wtype,
        b_scales=w_s,
        b_zeros=maybe_convert_zeropoints(w_zp, w_s, a.dtype),
        b_group_size=group_size,
    )
    print(maybe_convert_zeropoints(w_zp, w_s, a.dtype))
    assert torch.allclose(output, output_ref, rtol=1e-1, atol=1e-1)
