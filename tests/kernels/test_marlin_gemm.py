"""Tests for the marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_gemm.py`.
"""
import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL, GPTQ_MARLIN_24_MIN_THREAD_N,
    GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES, GPTQ_MARLIN_24_SUPPORTED_NUM_BITS)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
    GPTQ_MARLIN_SUPPORTED_GROUP_SIZES, GPTQ_MARLIN_SUPPORTED_NUM_BITS,
    marlin_permute_scales)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    pack_fp8_to_int32)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace, get_weight_perm, marlin_quantize, marlin_weights,
    awq_marlin_quantize)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    marlin_24_quantize)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack, quantize_weights, sort_weights)

ACT_ORDER_OPTS = [False, True]
K_FULL_OPTS = [False, True]

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 128, 256]

MARLIN_24_K_CHUNKS = [128]
MARLIN_24_N_CHUNKS = [512]

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
]

DTYPES = [torch.float16, torch.bfloat16]


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("num_bits", GPTQ_MARLIN_SUPPORTED_NUM_BITS)
@pytest.mark.parametrize("group_size", GPTQ_MARLIN_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("act_order", ACT_ORDER_OPTS)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_repack(k_chunk, n_chunk, num_bits, group_size, act_order,
                            mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Create input
    b_weight = rand_data((size_k, size_n))

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = quantize_weights(b_weight, num_bits,
                                                       group_size, act_order)

    # Pack to GPTQ format
    q_w_gptq = gptq_pack(q_w, num_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Pack to Marlin format
    weight_perm = get_weight_perm(num_bits)
    marlin_q_w_1 = marlin_weights(q_w, size_k, size_n, num_bits, weight_perm)

    # Run Marlin repack GPU kernel
    marlin_q_w_2 = ops.gptq_marlin_repack(
        q_w_gptq,
        sort_indices,
        size_k,
        size_n,
        num_bits,
    )
    torch.cuda.synchronize()

    assert torch.allclose(marlin_q_w_1, marlin_q_w_2)

@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("num_bits", GPTQ_MARLIN_SUPPORTED_NUM_BITS)
@pytest.mark.parametrize("group_size", [128])#GPTQ_MARLIN_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("act_order", ACT_ORDER_OPTS)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_awq_marlin_repack(k_chunk, n_chunk, num_bits, group_size, act_order,
                            mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    print(f"MNK = {size_m} {size_n} {size_k}")

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Create input
    b_weight = rand_data((size_k, size_n))

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = quantize_weights(b_weight, num_bits,
                                                       group_size, act_order)

    # Pack to GPTQ format
    q_w_gptq = gptq_pack(q_w, num_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Pack to Marlin format
    weight_perm = get_weight_perm(num_bits)
    marlin_q_w_1 = marlin_weights(q_w, size_k, size_n, num_bits, weight_perm)

    # Run Marlin repack GPU kernel
    marlin_q_w_2 = ops.gptq_marlin_repack(
        q_w_gptq,
        sort_indices,
        size_k,
        size_n,
        num_bits,
    )
    torch.cuda.synchronize()

    assert torch.allclose(marlin_q_w_1, marlin_q_w_2)


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("num_bits", GPTQ_MARLIN_SUPPORTED_NUM_BITS)
@pytest.mark.parametrize("group_size", GPTQ_MARLIN_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("act_order", ACT_ORDER_OPTS)
@pytest.mark.parametrize("is_k_full", K_FULL_OPTS)
def test_gptq_marlin_gemm(
    k_chunk,
    n_chunk,
    num_bits,
    group_size,
    mnk_factors,
    act_order,
    is_k_full,
):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    print(f"MNK = {size_m} {size_n} {size_k}")
    print(f"groupsize = {group_size}")

    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight, num_bits, group_size, act_order)

    workspace = MarlinWorkspace(size_n, GPTQ_MARLIN_MIN_THREAD_N,
                                GPTQ_MARLIN_MAX_PARALLEL)

    output = ops.gptq_marlin_gemm(
        a_input,
        marlin_q_w,
        marlin_s,
        g_idx,
        sort_indices,
        workspace.scratch,
        num_bits,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full,
    )
    output_ref = torch.matmul(a_input, w_ref)

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)
    print("max_diff = {}".format(max_diff))

    assert max_diff < 0.04


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_24_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_24_N_CHUNKS)
@pytest.mark.parametrize("num_bits", GPTQ_MARLIN_24_SUPPORTED_NUM_BITS)
@pytest.mark.parametrize("group_size", GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_24_gemm(k_chunk, n_chunk, num_bits, group_size,
                             mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    print(f"MNK = {size_m} {size_n} {size_k}")
    print(f"groupsize = {group_size}")

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    (w_24_ref, marlin_24_q_w_comp, marlin_24_meta,
     marlin_24_s) = marlin_24_quantize(b_weight, num_bits, group_size)

    workspace_24 = MarlinWorkspace(size_n, GPTQ_MARLIN_24_MIN_THREAD_N,
                                   GPTQ_MARLIN_24_MAX_PARALLEL)

    output_ref = torch.matmul(a_input, w_24_ref)

    output = ops.gptq_marlin_24_gemm(
        a_input,
        marlin_24_q_w_comp,
        marlin_24_meta,
        marlin_24_s,
        workspace_24.scratch,
        num_bits,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
    )

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)
    print("max_diff = {}".format(max_diff))

    assert max_diff < 0.04


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("num_bits", [8])
@pytest.mark.parametrize("group_size", [-1])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fp8_marlin_gemm(
    k_chunk,
    n_chunk,
    num_bits,
    group_size,
    mnk_factors,
    dtype,
):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    print(f"MNK = {size_m} {size_n} {size_k}")
    print(f"groupsize = {group_size}")

    a_input = rand_data((size_m, size_k), dtype=dtype)
    b_weight = rand_data((size_k, size_n), dtype=dtype)

    # WEIGHTS
    fp8_weight, weight_scale = ops.scaled_fp8_quant(b_weight, scale=None)
    # Repack weights to gptq format (packed int32 elements)
    packed_gptq_qweight = pack_fp8_to_int32(fp8_weight)
    # Repack weights to marlin format
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=packed_gptq_qweight,
        perm=torch.empty(0, dtype=torch.int, device="cuda"),
        size_k=size_k,
        size_n=size_n,
        num_bits=8,
    )

    # WEIGHT SCALES
    # Currently Marlin doesn't support per-tensor scales, so we
    # expand it to channelwise
    scales = weight_scale.repeat(1, size_n).to(a_input.dtype).to("cuda")
    # Permute scales
    marlin_scales = marlin_permute_scales(s=scales,
                                          size_k=size_k,
                                          size_n=size_n,
                                          group_size=-1)

    workspace = MarlinWorkspace(size_n, GPTQ_MARLIN_MIN_THREAD_N,
                                GPTQ_MARLIN_MAX_PARALLEL)

    output = ops.fp8_marlin_gemm(
        a=a_input,
        b_q_weight=marlin_qweight,
        b_scales=marlin_scales,
        workspace=workspace.scratch,
        num_bits=num_bits,
        size_m=a_input.shape[0],
        size_n=b_weight.shape[1],
        size_k=a_input.shape[1],
    )
    output_ref = torch.matmul(a_input, b_weight)

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)
    print("max_diff = {}".format(max_diff))

    assert max_diff < 0.04


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("num_bits", GPTQ_MARLIN_SUPPORTED_NUM_BITS)
@pytest.mark.parametrize("group_size", GPTQ_MARLIN_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_awq_marlin_gemm(
    k_chunk,
    n_chunk,
    num_bits,
    group_size,
    mnk_factors,
):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    print(f"MNK = {size_m} {size_n} {size_k}")
    print(f"groupsize = {group_size}")

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
        b_weight, num_bits, group_size)

    g_idx = torch.empty(0, dtype=torch.int, device=marlin_q_w.device)
    sort_indices = torch.empty(0, dtype=torch.int, device=marlin_q_w.device)
    is_k_full = True
    has_zp = True

    workspace = MarlinWorkspace(size_n, GPTQ_MARLIN_MIN_THREAD_N,
                                GPTQ_MARLIN_MAX_PARALLEL)

    output = ops.gptq_marlin_gemm(
        a_input,
        marlin_q_w,
        marlin_s,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace.scratch,
        num_bits,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full,
        has_zp,
    )
    output_ref = torch.matmul(a_input, w_ref)

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)
    print("max_diff = {}".format(max_diff))

    assert max_diff < 0.04
