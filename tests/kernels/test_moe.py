"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""

import pytest
import torch
from typing import List
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe, fused_marlin_moe, single_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_quantize
from vllm.model_executor.models.mixtral import MixtralMoE

if should_skip_test_group(group_name="TEST_KERNELS"):
    pytest.skip("TEST_KERNELS=DISABLE, skipping kernels test group",
                allow_module_level=True)


def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                 a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)

def torch_moe_single(a, w, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    _, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.view(-1)
    for i in range(w.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] =  a[mask] @ w[i].transpose(0, 1)
    return (out.view(B, -1, w.shape[1])).sum(dim=1)

# UPSTREAM SYNC: breaks NM automation.
@pytest.mark.skip("C compiler not installed in NM automation. "
                  "This codepath follows a triton pathway, which "
                  "JITs using clang or gcc. Since neither are installed "
                  "in our test instances, we need to skip this for now.")
@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
    torch_output = torch_moe(a, w1, w2, score, topk)

    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)


# UPSTREAM SYNC: breaks NM automation.
# @pytest.mark.skip("C compiler not installed in NM automation. "
#                   "This codepath follows a triton pathway, which "
#                   "JITs using clang or gcc. Since neither are installed "
#                   "in our test instances, we need to skip this for now.")
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mixtral_moe(dtype: torch.dtype):
    "Make sure our Mixtral MoE implementation agrees with the one from"
    "huggingface."

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        params_dtype=dtype,
        tp_size=1,
    ).cuda()

    # Load the weights
    vllm_moe.gate.weight.data[:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.w13_weight[i][:] = torch.cat(weights, dim=0)
        vllm_moe.w2_weight[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    hf_inputs = torch.randn((1, 64, config.hidden_size)).to(dtype).to("cuda")
    # vLLM uses 1D query [num_tokens, hidden_dim]
    vllm_inputs = hf_inputs.flatten(0, 1)

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(hf_inputs)
    vllm_states = vllm_moe.forward(vllm_inputs)

    mixtral_moe_tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    assert torch.allclose(hf_states.flatten(0, 1),
                          vllm_states,
                          rtol=mixtral_moe_tol[dtype],
                          atol=mixtral_moe_tol[dtype])

def stack_and_dev(tensors: List[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)

def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))

# UPSTREAM SYNC: breaks NM automation.
@pytest.mark.skip("C compiler not installed in NM automation. "
                  "This codepath follows a triton pathway, which "
                  "JITs using clang or gcc. Since neither are installed "
                  "in our test instances, we need to skip this for now.")
@pytest.mark.parametrize("m", [64, 512, 222, 33, 1])
@pytest.mark.parametrize("n", [128, 2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024, 512])
@pytest.mark.parametrize("e", [4, 8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_fused_marlin_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
):
    if topk > e:
        return

    num_bits = 4
    dtype = torch.float16
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10
    for i in range(w2.shape[0]):
        w2[0] = torch.eye(k, n, device='cuda', dtype=dtype)

    w_refs1 = []
    qweights1 = []
    scaless1 = []

    for i in range(w1.shape[0]):
        w_ref1, qweight1, scales1, _, _, _ = marlin_quantize(w1[i].transpose(1, 0), num_bits, group_size, False)
        w_refs1.append(w_ref1)
        qweights1.append(qweight1)
        scaless1.append(scales1)

    w_ref1 = stack_and_dev(w_refs1)
    qweight1 = stack_and_dev(qweights1).contiguous()
    scales1 = stack_and_dev(scaless1)

    w_refs2 = []
    qweights2 = []
    scaless2 = []

    for i in range(w2.shape[0]):
        w_ref2, qweight2, scales2, _, _, _ = marlin_quantize(w2[i].transpose(1, 0), num_bits, group_size, False)
        w_refs2.append(w_ref2)
        qweights2.append(qweight2)
        scaless2.append(scales2)

    w_ref2 = stack_and_dev(w_refs2)
    qweight2 = stack_and_dev(qweights2).contiguous()
    scales2 = stack_and_dev(scaless2)

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    triton_output = fused_moe(a, w_ref1.transpose(1, 2).contiguous(),
                              w_ref2.transpose(1, 2).contiguous(),
                              score, topk, renormalize=False)
    marlin_output = fused_marlin_moe(a, qweight1, qweight2, score, topk,
                                    renormalize=False,
                                    w1_scale=scales1, w2_scale=scales2)

    assert(compute_max_diff(marlin_output, triton_output) < 4e-2)


# UPSTREAM SYNC: breaks NM automation.
@pytest.mark.skip("C compiler not installed in NM automation. "
                  "This codepath follows a triton pathway, which "
                  "JITs using clang or gcc. Since neither are installed "
                  "in our test instances, we need to skip this for now.")
@pytest.mark.parametrize("m", [64, 512, 222, 33, 1])
@pytest.mark.parametrize("n", [128, 2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024, 512])
@pytest.mark.parametrize("e", [4, 8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_single_marlin_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
):
    if topk > e:
        return

    num_bits = 4
    dtype = torch.float16
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w = torch.randn((e, n, k), device='cuda', dtype=dtype) / 10

    w_refs = []
    qweights = []
    scaless = []

    for i in range(w.shape[0]):
        w_ref, qweight, scales, _, _, _ = marlin_quantize(w[i].transpose(1, 0), num_bits, group_size, False)
        w_refs.append(w_ref)
        qweights.append(qweight)
        scaless.append(scales)

    w_ref = stack_and_dev(w_refs)
    qweight = stack_and_dev(qweights).contiguous()
    scales = stack_and_dev(scaless)

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    marlin_output = single_marlin_moe(a, qweight, scales, score, topk, renormalize=False)
    torch_output = torch_moe_single(a, w_ref.transpose(1, 2), score, topk)

    assert(compute_max_diff(marlin_output, torch_output) < 1e-2)
