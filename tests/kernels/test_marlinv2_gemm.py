"""Tests for the marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_gemm.py`.
"""

import matplotlib.pyplot as plt
import numpy as np

import pytest
import torch
import random as rand
import numpy

from vllm import common_types as vllm_type
from vllm._custom_classes import VLLMType
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQ_MARLIN_MAX_PARALLEL,
    GPTQ_MARLIN_MIN_THREAD_N,
    GPTQ_MARLIN_SUPPORTED_GROUP_SIZES,
    GPTQ_MARLIN_SUPPORTED_NUM_BITS,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL,
    GPTQ_MARLIN_24_MIN_THREAD_N,
    GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES,
    GPTQ_MARLIN_24_SUPPORTED_NUM_BITS,
)
from vllm.model_executor.layers.quantization.utils.marlin_perms import marlin_perm
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MarlinWorkspace,
    compute_max_diff,
    is_marlin_supported,
    marlin_24_quantize,
    marlin_quantize,
    marlin_weights,
    pack_fp8_to_int32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    quantize_weights,
    sort_weights,
)


@pytest.fixture
def random():
    torch.manual_seed(0)
    rand.seed(0)
    numpy.random.seed(0)


GROUP_SIZES = [128]

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
WEIGHT_TYPES = [vllm_type.int4]  # , vllm_type.uint4


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(
    not is_marlin_supported(), reason="Marlin is not supported on this GPU type."
)
@pytest.mark.parametrize("shape", MNK_SHAPES, ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype", WEIGHT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_marlinv2(shape, atype: torch.dtype, wtype: VLLMType, group_size: int):
    torch.manual_seed(42)
    rand.seed(0)
    numpy.random.seed(0)

    size_m, size_k, size_n = shape
    # torch.cuda.set_sync_debug_mode(1)

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Normalize group_size
    if group_size is None:
        group_size = size_k
    assert group_size <= size_k

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, _ = quantize_weights(
        b_weight, wtype.size_bits, group_size, False
    )

    assert wtype.integer, "TODO: support floating point weights"
    # quantize_weights uses the midpoint as the zero-point,
    #  and always produces unsigned values
    zero_point = (2**wtype.size_bits) // 2

    if wtype.signed:
        q_w -= zero_point
    else:
        w_ref += zero_point

    # Pack to GPTQ format
    q_w_gptq = gptq_pack(q_w, wtype.size_bits, size_k, size_n)
    q_w_gptq = q_w_gptq.t().contiguous().t()
    q_w_marlinv2 = ops.marlinv2_prepack_B(q_w_gptq, wtype)

    output = ops.marlinv2_gemm(
        a_input,
        b_q_weight=q_w_marlinv2,
        b_type=wtype,
        b_scales=s,
        b_group_size=group_size,
        schedule="128x128_1x1x1_TmaMI_TmaCoop_void",
    )

    output_ref = torch.matmul(a_input, w_ref)
    assert torch.allclose(output, output_ref, rtol=1e-1, atol=1)
