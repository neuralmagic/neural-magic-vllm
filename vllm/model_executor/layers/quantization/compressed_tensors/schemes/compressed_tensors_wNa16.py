from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear, marlin_make_empty_g_idx, marlin_make_workspace,
    marlin_permute_scales, replace_tensor, verify_gptq_marlin_supported,
    verify_marlin_supports_shape)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter, vLLMParameter)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_BITS = [4, 8]


class CompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None):
        self.num_bits = num_bits
        self.pack_factor = 32 // self.num_bits
        self.strategy = strategy

        self.group_size: int
        if group_size is None:
            if self.strategy != "channel":
                raise ValueError(
                    "Marlin kernels require group quantization or "
                    "channelwise quantization, but found no group "
                    "size and strategy is not channelwise.")
            self.group_size = -1
        else:
            self.group_size = group_size

        # Verify supported on platform.
        verify_gptq_marlin_supported(num_bits=self.num_bits,
                                     group_size=self.group_size,
                                     is_sym=True)

    def get_min_capability(self) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)

        # If group_size is -1, we are in channelwise case.
        group_size = input_size if self.group_size == -1 else self.group_size

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size)

        scales_and_zp_size = input_size // group_size

        if (input_size != input_size_per_partition
                and self.group_size is not None):
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition //
                                         pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }
        if self.group_size is not None:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
        else:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = vLLMParameter(data=torch.empty(2, dtype=torch.int64),
                                     weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.group_size = group_size

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from marlin format. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device

        # Allocate marlin workspace.
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Act-order not supported in compressed-tensors yet, so set to empty.
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.weight_zp = marlin_make_empty_g_idx(device)

        # Repack weights from compressed-tensors format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.weight_packed.t().contiguous(),
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.num_bits)
        replace_tensor(layer, "weight_packed", marlin_qweight)

        # Permute scales from compressed-tensors format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.weight_scale.squeeze().t().contiguous(),
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=layer.group_size)
        replace_tensor(layer, "weight_scale", marlin_scales)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:

        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.weight_packed,
            weight_scale=layer.weight_scale,
            weight_zp=layer.weight_zp,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            num_bits=self.num_bits,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=True,
            bias=bias)
