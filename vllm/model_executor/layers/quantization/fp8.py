from typing import Any, Dict, List, Optional

import torch
from magic_wand import CompressedStorageFormat

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.parameters import LazyCompressedParameter
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


def fp8_quantize(
    weight,
    qdtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale


@CompressedStorageFormat._dataclass
class FP8StorageFormat(CompressedStorageFormat):
    values: torch.Tensor
    scale: torch.Tensor
    dtype: torch.dtype

    @classmethod
    def compress(cls, uncompressed: torch.Tensor):
        dtype = uncompressed.dtype
        compressed, scale = fp8_quantize(uncompressed)

        return cls(values=compressed, scale=scale, dtype=dtype)

    def decompress(self):
        return self.values.to(self.dtype) * self.scale.to(self.dtype)


class FP8Config(QuantizationConfig):
    """Config class for FP8.

    Reference: https://arxiv.org/abs/2209.05433
    """

    def __init__(self, qdtype=torch.float8_e4m3fn) -> None:
        self.qdtype = qdtype

        supported_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2]
        if self.qdtype not in supported_dtypes:
            raise ValueError(
                f"Currently, only {supported_dtypes} are supported types "
                "for fp8.")

    def __repr__(self) -> str:
        return (f"FP8Config(qdtype={self.qdtype})")

    def get_name(self) -> str:
        return "fp8"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        # FP8 hardware support is required because
        # torch._scaled_mm is only supported on CUDA devices with
        # compute capability >= 9.0 or 8.9, or ROCm MI300+");
        return 89

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
        return cls()

    def get_linear_method(self) -> "FP8LinearMethod":
        return FP8LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class FP8LinearMethod(LinearMethodBase):
    """Linear method for FP8.

    Args:
        quant_config: The FP8 quantization config.
    """

    def __init__(self, quant_config: FP8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.dtype = params_dtype

        weight = LazyCompressedParameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            # For create_weights(..), we initialize an empty tensor to
            # save GPU memory. When the parameter will be loaded from
            # disk it will be copied into this tensor
            is_empty=True,
            storage_format_cls=FP8StorageFormat,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w: LazyCompressedParameter = layer.weight

        qx, xscale = fp8_quantize(x)

        output, _ = torch._scaled_mm(
            qx,
            w.compressed_data.values,
            out_dtype=self.dtype,
            scale_a=xscale,
            scale_b=w.compressed_data.scale,
            bias=bias,
        )

        return output
