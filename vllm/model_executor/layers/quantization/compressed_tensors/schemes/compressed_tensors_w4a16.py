from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm._C import ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

from enum import Enum
import numpy
import torch.nn.functional as F

__all__ = ["CompressedTensorsW4A16"]

def reshape_to_group(w, s, size_k, size_n, group_size):
    assert w.shape[0] == size_k, "w.shape = {}, size_k/n = {}".format(w.shape, (size_k, size_n))
    assert w.shape[1] == size_n, "w.shape = {}, size_k/n = {}".format(w.shape, (size_k, size_n))
    assert w.dtype == torch.half or w.dtype == torch.int32

    if group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))
    s = s.reshape((1, -1))

    return w, s


def reshape_from_group(w, s, size_k, size_n, group_size):
    assert w.shape[0] == group_size, "w.shape[0] = {}, group_size = {}".format(w.shape[0], group_size)
    assert w.dtype == torch.half or w.dtype == torch.int32, "w.dtype = {}".format(w.dtype)

    if group_size < size_k:
        w = w.reshape((group_size, -1, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((size_k, size_n)).contiguous()

    s = s.reshape((-1, size_n)).contiguous()

    return w, s


def dequant(w, s, size_k, size_n, num_bits, group_size):
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2

    # Reshape to [groupsize, -1]
    w, s = reshape_to_group(w, s, size_k, size_n, group_size)

    # Dequantize
    w_norm = w - half_q_val
    res = w_norm.half() * s

    # Restore shapes
    res, s = reshape_from_group(res, s, size_k, size_n, group_size)

    return res

def unpack_gptq(w_gptq, size_k, size_n, num_bits):
    import numpy as np
    pack_factor = 32 // num_bits

    assert w_gptq.shape[0] * pack_factor == size_k
    assert w_gptq.shape[1] == size_n
    assert w_gptq.is_contiguous()

    res = np.zeros((size_k, size_n), dtype=np.uint32)
    w_gptq_cpu = w_gptq.cpu().numpy().astype(np.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = w_gptq_cpu & mask
        w_gptq_cpu >>= num_bits
        res[i::pack_factor, :] = vals

    res = torch.from_numpy(res.astype(np.int32)).to(w_gptq.device)

    return res

def _get_perms():
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
    perm = perm.reshape((-1, 8))[:, interleave].ravel()  # type: ignore
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()

class GPTQMarlinState(Enum):
    import enum

    REPACK = enum.auto()
    READY = enum.auto()

def marlin_permute_scales(s, size_k, size_n, group_size):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    else:
        s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

class CompressedTensorsW4A16(CompressedTensorsScheme):
    def __init__(self, strategy: str = None, group_size: int = None):
        self.strategy = strategy
        self.group_size = group_size

        if self.strategy == "group" and self.group_size is None:
            raise ValueError("group_size must be given when using strategy group")
        

    def create_weights(self, layer: torch.nn.Module,
                    input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable, layer_name: str,
                    **kwargs):


        pack_factor = 8 # the only one we support for now
        # for group size, 128 things next to each other in memory, 2nd dimension for things next to each other in memory
        output_size_per_partition = sum(output_partition_sizes)

        weight_scale_dim = None 
        scales_and_zp_size = input_size // self.group_size

        if input_size != input_size_per_partition:
            weight_scale_dim = 1
            scales_and_zp_size = input_size_per_partition // self.group_size


        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0, "packed_dim": 1, "pack_factor": 8, "ignore_warning": True})
        set_weight_attrs(weight, {"weight_loader": weight_loader})
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                1,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        set_weight_attrs(weight_scale, {"weight_loader": weight_loader})
        set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0, "ignore_warning": True})
        layer.register_parameter("weight_scale", weight_scale)
        

        weight_zero_point = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                1,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight_zero_point", weight_zero_point)
        set_weight_attrs(weight_zero_point, {"weight_loader": weight_loader})
        set_weight_attrs(weight_zero_point, {"input_dim": weight_scale_dim, "output_dim": 0})

        weight_shape = Parameter(torch.empty(2,
                                            device="cuda",
                                            dtype=torch.int64),
                                requires_grad=False)

        layer.register_parameter("weight_shape", weight_shape)
        set_weight_attrs(weight_shape, {"weight_loader": weight_loader})

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        """
        layer.input_size = input_size
        layer.marlin_state = GPTQMarlinState.REPACK
        layer.is_k_full = True

        max_workspace_size = (output_size_per_partition // 64) * 16

        workspace = torch.zeros(max_workspace_size,
                                dtype=torch.int,
                                requires_grad=False)
        layer.workspace = workspace
        """

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        
        """
        reshaped_x = x.reshape(-1, x.shape[-1])

        size_m = reshaped_x.shape[0]
        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        full_size_k = layer.input_size

        out_shape = x.shape[:-1] + (part_size_n, )
        
        if layer.marlin_state == GPTQMarlinState.REPACK:
            layer.marlin_state = GPTQMarlinState.READY
            print("Inside Repack", layer)

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                # It is important to use resize_() here since it ensures
                # the same buffer is reused
                getattr(layer, name).resize_(new_t.shape)
                getattr(layer, name).copy_(new_t)
                del new_t

            cur_device = layer.weight.device

            # Reset g_idx related tensors
            layer.g_idx = Parameter(torch.empty(0,
                                                dtype=torch.int,
                                                device=cur_device),
                                    requires_grad=False)
            layer.g_idx_sort_indices = Parameter(torch.empty(
                0, dtype=torch.int, device=cur_device),
                                                    requires_grad=False)

            # Repack weights
            weight_temp = layer.weight.t().contiguous()
            print("Weight temp", weight_temp.shape)
            marlin_qweight = ops.gptq_marlin_repack(
                weight_temp,
                layer.g_idx_sort_indices,
                part_size_k,
                part_size_n,
                4
            )
            print("marlin weight", marlin_qweight.shape)
            
            replace_tensor("weight", marlin_qweight)

            # Permute scales
            scales_size_k = part_size_k
            scales_size_n = part_size_n

            
            weight_scale_temp = layer.weight_scale.squeeze().t().contiguous()
            print("scales", weight_scale_temp.shape)
            marlin_scales = marlin_permute_scales(weight_scale_temp, scales_size_k,
                                                  scales_size_n,
                                                  self.group_size)
            replace_tensor("weight_scale", marlin_scales)
            print("marlin scale", marlin_scales.shape)
            print("\n")

        output = ops.gptq_marlin_gemm(reshaped_x, layer.weight, layer.weight_scale,
                                      layer.g_idx, layer.g_idx_sort_indices,
                                      layer.workspace, 4, size_m, part_size_n,
                                      part_size_k, layer.is_k_full)

        #print(output)
        return output.reshape(out_shape)
        """
        size_k = layer.input_size_per_partition
        size_n = layer.output_size_per_partition

        weight_temp = layer.weight.squeeze().t().contiguous()
        scale_temp = layer.weight_scale.squeeze().t().contiguous()


        w_unpacked = unpack_gptq(weight_temp, size_k, size_n, 4)
        #print("    w_unpacked: shape = {} type = {}".format(w_unpacked.shape, w_unpacked.type()))

        w = dequant(w_unpacked, scale_temp, size_k, size_n, 4, self.group_size)
        #print("    w dequant: shape = {} type = {}".format(w.shape, w.type()))
                    
        return torch.matmul(x, w)