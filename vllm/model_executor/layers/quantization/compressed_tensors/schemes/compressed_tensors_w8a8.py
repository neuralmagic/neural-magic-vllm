from typing import Callable, List, Union

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    QuantizationStrategy, QuantizationType)
from vllm.model_executor.utils import set_weight_attrs


class CompressedTensorsW8A8(CompressedTensorsScheme):

    def __init__(self, strategy: QuantizationStrategy,
                 quant_type: QuantizationType):
        self.strategy = strategy
        self.quant_type = quant_type

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # If not per tensor or not a "fused" (QKV/MLP) module, do nothing.
        if (self.strategy != QuantizationStrategy.TENSOR
                or len(self.logical_widths) == 1):
            return

        # Cutlass kernels support only per-tensor and per-channel cases.
        # For a fused module (QKV, MLP) with N per tensor scales, we do:
        #   > int8): requantize with single scale
        #   > fp8 ): convert N per tensor scales >> channelwise

        # int8 case -> convert the N per-tensor scales into channelwise.
        if self.quant_type == QuantizationType.INT:
            weight_scale_channel = torch.empty(
                (sum(self.logical_widths), 1),
                dtype=torch.float32,
                device=layer.weight_scale.device)
            start = 0
            for idx, logical_width in enumerate(self.logical_widths):
                end = start + logical_width
                weight_scale_channel[start:end, :] = layer.weight_scale[idx]
                start = end
            layer.weight_scale = Parameter(weight_scale_channel,
                                           requires_grad=False)

        # fp8 case -> convert the N per-tensor scales to 1 by requantizing.
        else:
            max_w_scale = layer.weight_scale.max()
            start = 0
            for idx, logical_width in enumerate(self.logical_widths):
                end = start + logical_width
                weight_dq = per_tensor_dequantize(layer.weight[start:end, :],
                                                  layer.weight_scale[idx])
                layer.weight[start:end, :] = per_tensor_quantize(
                    weight_dq, layer.weight_scale.max())
                start = end
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        self.logical_widths = output_partition_sizes

        # WEIGHT SCALE
        shape = (sum(self.logical_widths),
                 1) if self.strategy == QuantizationStrategy.CHANNEL else (len(
                     self.logical_widths), )

        weight_scale = Parameter(torch.empty(*shape, dtype=torch.float32),
                                 requires_grad=False)
        layer.register_parameter("weight_scale", weight_scale)
        if self.strategy == QuantizationStrategy.CHANNEL:
            set_weight_attrs(weight_scale, {
                "weight_loader": weight_loader,
                "output_dim": 0,
            })
        else:
            set_weight_attrs(weight_scale, {
                "weight_loader": weight_loader,
                "is_per_tensor_scale": True,
            })

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn if self.quant_type
                        == QuantizationType.FLOAT else torch.int8)
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=weight_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
        })


def per_tensor_quantize(tensor: torch.Tensor,
                        inv_scale: Union[float, torch.Tensor]) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight
