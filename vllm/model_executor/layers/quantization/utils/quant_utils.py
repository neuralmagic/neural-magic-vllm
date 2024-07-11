"""This file is used for /tests and /benchmarks"""

from typing import Union

import numpy
import torch

from vllm._custom_classes import VLLMType

SUPPORTED_NUM_BITS = [4, 8]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


def get_pack_factor(num_bits):
    assert num_bits in SUPPORTED_NUM_BITS, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def permute_rows(q_w: torch.Tensor, w_ref: torch.Tensor, group_size: int):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size, ), dtype=torch.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = torch.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def quantize_weights(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
    act_order: bool,
    zero_point: Union[int, str] = "symmetric",
):
    if zero_point == "symmetric":
        zero_point = 2**num_bits // 2

    if isinstance(zero_point, str):
        raise ValueError("zero_point must be an integer or 'symmetric'")

    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"
    assert num_bits in SUPPORTED_NUM_BITS, f"Unsupported num_bits = {num_bits}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    max_q_val = 2**num_bits - 1
    max_q_val_pos = max_q_val - zero_point
    max_q_val_neg = int(zero_point)

    # Reshape to [groupsize, -1]
    if group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_pos_val = torch.abs(torch.max(w, 0, keepdim=True).values)
    max_neg_val = torch.abs(torch.min(w, 0, keepdim=True).values)

    # If the zero_point is such that there are no possible negative/positive
    #  values, set the max value to inf to avoid divide by 0
    max_q_val_pos = max_q_val_pos if max_q_val_pos > 0 else torch.inf
    max_q_val_neg = max_q_val_neg if max_q_val_neg > 0 else torch.inf
    s = torch.max(max_pos_val / max_q_val_pos, max_neg_val / max_q_val_neg)

    # Quantize
    q_w = torch.round(w / s).int()
    q_w += zero_point
    q_w = torch.clamp(q_w, 0, max_q_val)

    # Compute ref (dequantized)
    w_ref = (q_w - zero_point).half() * s

    # Restore original shapes
    if group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

    s = s.reshape((-1, size_n)).contiguous()

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert (
            group_size < size_k
        ), "For act_order, groupsize = {} must be less than size_k = {}".format(
            group_size, size_k)

        w_ref, q_w, g_idx, rand_perm = permute_rows(q_w, w_ref, group_size)

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        s.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(
        dtype=torch.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )


def gptq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res


def pack_weights_into_int32(q_w: torch.Tensor, wtype: VLLMType, dim: int = 0):
    orig_device = q_w.device

    # move dim to pack to the end
    perm = (*[i for i in range(len(q_w.shape)) if i != dim], dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    q_w_perm = q_w.permute(perm)

    q_w_perm = q_w_perm.cpu().numpy().astype(numpy.uint32)
    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1
    
    new_shape_perm = list(q_w_perm.shape)
    new_shape_perm[-1] //= pack_factor
    assert new_shape_perm[-1] % pack_factor == 0
    
    q_res = numpy.zeros(new_shape_perm, dtype=numpy.uint32)
    for i in range(pack_factor):
        q_res |= (q_w_perm[..., i::pack_factor] & mask) << wtype.size_bits * i
        
    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.permute(inv_perm)

    return q_res