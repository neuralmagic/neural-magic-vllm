"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""

import pytest
import torch
import numpy
import random
from typing import List
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe, fused_marlin_moe, single_marlin_moe
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.model_executor.models.mixtral_quant import MixtralMoE as MixtralMoEQuant

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
            simul = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1))
            out[mask] = simul @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)

def torch_moe_small(a, w, score, topk):
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

def get_marlin_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = numpy.array(perm)
    interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


def marlin_permute_weights(q_w, size_k, size_n, num_bits, perm):
    tile = 16
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    # print("r1:", q_w.size())
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))
    # print("r2:", q_w.size())
    
    # print("NUMEL", perm.numel())
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    # print("r3:", q_w.size())

    return q_w

def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, num_bits, perm)

    # Pack
    pack_factor = 32 // num_bits
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)
    q_packed = numpy.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                           dtype=numpy.uint32)

    # print("PACKED:", q_w.shape, ">>", q_packed.shape)

    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(numpy.int32)).to(orig_device)

    return q_packed


def marlin_permute_scales(s, size_k, size_n, group_size, scale_perm, scale_perm_single):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

def quantize_weights(w: torch.Tensor, num_bits: int, group_size: int):
    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2

    # print("")
    # print("pre reshape w:", w.size())

    # Reshape to [groupsize, -1]
    if group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / max_q_val  # 2 => symmetric

    # print("pre quantize w:", w.size())

    # Quantize
    q_w = torch.round(w / s).int()
    q_w += half_q_val
    q_w = torch.clamp(q_w, 0, max_q_val)

    # print("pre reshape q_w:", q_w.size())
    # print("pre reshape s:", s.size())

    # Compute ref (dequantized)
    w_ref = (q_w - half_q_val).half() * s

    # Restore original shapes
    if group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        # print("mid reshape q_w:", q_w.size())
        # print("mid reshape w_ref:", w_ref.size())
        # print("mid reshape s:", s.size())
        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

    s = s.reshape((-1, size_n)).contiguous()
    # print("post reshape q_w:", q_w.size())
    # print("post reshape w_ref:", w_ref.size())
    # print("post reshape s:", s.size())

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        s.to(device=orig_device),
    )

# TODO rewrite these transformations for multi expert
def marlin_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
):
    # print("START:", w.size(), num_bits, group_size)
    perm, scale_perm, scale_perm_single = get_marlin_perms()

    # print("SHAPE:", w.shape)
    #TODO experts dim
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize
    w_ref, q_w, s = quantize_weights(w, num_bits, group_size)
    # print("interm:", w_ref.size(), q_w.size(), s.size())

    #TODO experts
    # Reformat to marlin
    marlin_q_w = marlin_weights(q_w, size_k, size_n, num_bits, perm)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size, scale_perm, scale_perm_single)

    marlin_q_w = marlin_q_w

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list

""" END MARLIN PREP FUNCTIONS """

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
# @pytest.mark.parametrize("m", [64]) #[64, 512, 222, 33, 1])
# @pytest.mark.parametrize("n", [256]) #[128, 2048, 256, 1024])
# @pytest.mark.parametrize("k", [128]) #[128, 1024, 512])
# @pytest.mark.parametrize("e", [4]) #[4, 8, 64])
# @pytest.mark.parametrize("topk", [2])
# @pytest.mark.parametrize("group_size", [-1]) #[-1, 32, 64, 128])
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
        w_ref1, qweight1, scales1 = marlin_quantize(w1[i].transpose(1, 0), num_bits, group_size)
        w_refs1.append(w_ref1)
        qweights1.append(qweight1)
        scaless1.append(scales1)

    w_ref1 = stack_and_dev(w_refs1)
    qweight1 = stack_and_dev(qweights1)
    scales1 = stack_and_dev(scaless1)

    w_refs2 = []
    qweights2 = []
    scaless2 = []

    for i in range(w2.shape[0]):
        w_ref2, qweight2, scales2 = marlin_quantize(w2[i].transpose(1, 0), num_bits, group_size)
        w_refs2.append(w_ref2)
        qweights2.append(qweight2)
        scaless2.append(scales2)

    w_ref2 = stack_and_dev(w_refs2)
    qweight2 = stack_and_dev(qweights2)
    scales2 = stack_and_dev(scaless2)

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    triton_output = fused_moe(a, w_ref1.transpose(1, 2), w_ref2.transpose(1, 2), score, topk, renormalize=False)
    marlin_output = fused_marlin_moe(a, qweight1, qweight2, score, topk,
                                    renormalize=False, w1_scale=scales1, w2_scale=scales2)

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
    # torch.manual_seed(4000)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w = torch.randn((e, n, k), device='cuda', dtype=dtype) / 10
    # gran = group_size
    # sav_w = w[0][0][0:gran]
    # sav_w = torch.ones((1, 1, gran), device='cuda', dtype=dtype)
    # inc = 0.01
    # for ii in range(w.shape[0]):
    #     # for jj in range(w.shape[1] // gran):
    #     #     w[ii][jj*gran:(jj+1)*gran] = sav_w
    #     for jj in range(w.shape[1]):
    #         for kk in range(w.shape[2] // gran):
    #             # if ii == 0 and jj == 0:
    #             #     print(kk*gran,(kk+1)*gran)
    #             w[ii][jj][kk*gran:(kk+1)*gran] = sav_w * inc 
    #             inc += 0.01

    w_refs = []
    qweights = []
    scaless = []

    for i in range(w.shape[0]):
        w_ref, qweight, scales = marlin_quantize(w[i].transpose(1, 0), num_bits, group_size)
        w_refs.append(w_ref)
        qweights.append(qweight)
        scaless.append(scales)
        # print("wrefs size:", w_ref.size())
        # print("qweights size:", qweight.size())
        # print("scaless size:", scales.size())

    w_ref = stack_and_dev(w_refs)
    qweight = stack_and_dev(qweights)
    scales = stack_and_dev(scaless)

    # print("MNK:", M, N, K)
    # print("A size:", hidden_states.size())
    # print("wref size:", w_ref.size())
    # print("qweight size:", qweight.size())
    # print("scales size:", scales.size())
    # print("scales:", scales)
    # print("b quant:", qweight)

    # print(w.size(), "->", qweight.size())

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    marlin_output = single_marlin_moe(a, qweight, scales, score, topk, renormalize=False)
    torch_output = torch_moe_small(a, w_ref.transpose(1, 2), score, topk)

    # print(marlin_output.size(), torch_output.size())

    # print(marlin_output)
    # print(torch_output)

    # for i in range(m):
    #     print(i, marlin_output[i][0].item(), torch_output[i][0].item())
    # print("---")
    # for i in range(n):
    #     print(i, marlin_output[m - 1][i].item(), torch_output[m - 1][i].item())

    flat_m = marlin_output.flatten()
    flat_t = torch_output.flatten()

    # if not torch.allclose(marlin_output, torch_output, atol=1e-1):
    #     for mm, tt in zip(enumerate(flat_m), enumerate(flat_t)):
    #         if abs(mm[1].item() - tt[1].item()) > 1e-1:
    #             print(mm[0], mm[0] // n, mm[0] // n * n, "+", mm[0] % n, mm[1].item(), tt[1].item())

    assert(compute_max_diff(marlin_output, torch_output) < 1e-2)

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.distributed.parallel_state import initialize_model_parallel, init_distributed_environment
import os

@torch.inference_mode()
def test_forward():
    m = 8
    n = 256
    k = 128
    # e = 2
    # topk = 2
    group_size = -1
    dtype = torch.float16

    init_distributed_environment(1, 0, "tcp://192.168.198.114:60519", 0, "nccl")
    initialize_model_parallel()

    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w = torch.randn((n, k), device='cuda', dtype=dtype) / 10

    quant_config = GPTQMarlinConfig(4, group_size, False, True)
    layer = ReplicatedLinear(k, n, bias=False, quant_config=quant_config, params_dtype=dtype)
    print(layer.qweight.data)
    # layer.weight = w

    # making w:
    # GPTQMarlinLinearMethod.create_weights(self, self.input_size,
    #                                      [self.output_size], self.input_size,
    #                                      self.output_size, self.params_dtype)

    # config = MixtralConfig()
    # hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    # vllm_moe = MixtralMoEQuant(
    #     config, quant_config
    # ).cuda()

    # for i in range(1):
    #     w1 = vllm_moe.experts[i].w1.weight.data
    #     w3 = vllm_moe.experts[i].w3.weight.data

    # out = layer(a)
    # print(out)
