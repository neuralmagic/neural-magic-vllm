from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from torch._tensor import Tensor
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
import threading

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant

    Reference: https://github.com/mit-han-lab/smoothquant
    """

    def __init__(self,
                 weight_bits: int = 8,
                 quant_map: dict[str:str] = None) -> None:
        self.weight_bits = weight_bits
        self.quant_map = quant_map

        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only w8a8 quantization is supported for "
                f"SmoothQuant, but got {self.weight_bits} bits.")
        if self.quant_map is None or self.quant_map == {}:
            raise ValueError(
                'Quant_map for SmoothQuant should not be None or an empty dict. '
                'For example, when using llama, you should set a quant_config.json in model directory, like '
                '{ "qkv": "per-tensor", "out": "per-token", "fc1": "per-tensor", "fc2": "per-token" }'
            )

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(weight_bits={self.weight_bits}, "
                f"quant_map={self.quant_map})")

    def get_name(self) -> str:
        return "smoothquant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.float]

    def get_min_capability(self) -> int:
        # The smoothquant kernel only supports Ampere or newer GPUs.
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        try:
            weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        except ValueError as e:
            weight_bits = 8
            print(str(e) + " Set weight_bits = 8 by default.")

        quant_map = {}
        for key, value in config.items():
            if value in ["per-tensor", "per-token"]:
                quant_map[key] = value
        return cls(weight_bits, quant_map)

    def get_linear_method(self) -> "SQLinearMethod":
        return SQLinearMethod()

    def get_scaled_act_names(self) -> List[str]:
        return []


class Int8GEMM(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, "i8cugemm"):
            self.i8cugemm = ops.I8CUGEMM()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Int8GEMM, "_instance"):
            with Int8GEMM._instance_lock:
                if not hasattr(Int8GEMM, "_instance"):
                    Int8GEMM._instance = object.__new__(cls)
        return Int8GEMM._instance

    def get_i8cugemm(self):
        return self.i8cugemm

class SQLinearMethod(LinearMethodBase):
    """Linear method for AutoSmoothQuant.
    """

    def __init__(self,):
        self.i8cugemm = Int8GEMM().get_i8cugemm()

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        logical_widths: Optional[List[int]] = None,
        per_token_quant:bool = False,
    ) -> Dict[str, Tensor]:
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="cuda",
                dtype=torch.int8,
            ), requires_grad=False,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })

        dequant_scale = Parameter(
            torch.tensor(
                [1.0] * len(logical_widths), 
                dtype=params_dtype,
                device='cuda'
            ), requires_grad=False
        )

        return {
            "weight": weight,
            "dequant_scale": dequant_scale,
            "logical_widths": logical_widths,
            "per_token_quant": per_token_quant,
        }


    def _dequantize(self, x_q, weight_scales, activation_scales, logical_widths):
        x_dq = torch.empty_like(x_q, dtype=self.dequant_dtype)

        # Split into shards.
        x_q_split = x_q.split(logical_widths, dim=-1)
        x_dq_split = x_dq.split(logical_widths, dim=-1)

        # If QuantType is Static per Tensor:
        if activation_scales is None:
            for xdq, xq, weight_scale in zip(x_dq_split, x_q_split, weight_scales):
                ops.dequant(xdq, xq, weight_scale)
        
        # If QuantType is Dynamic per Token:
        else:
            for xdq, xq, weight_scale, activation_scale in zip(
                x_dq_split, x_q_split, weight_scales, activation_scales):
                ops.dequant(xdq, xq, activation_scale, weight_scale)

        # Return dequantized activation.
        return x_dq


    def _quantize(self, x, per_token_quant: bool):
        x_q = torch.empty_like(x, dtype=self.quant_dtype)
        
        # Compute activation scale if per token.
        if per_token_quant:
            activation_scale = torch.empty(
                x.numel() // x.shape[-1],
                dtype=torch.float32,
                device=x.device)
            ops.quant(x_q, x, activation_scale)
        # Set activation scale if per tensor. TODO: why 1.0? << static?
        else:   
            activation_scale = None
            ops.quant(x_q, x, 1.0)
        
        return x_q, activation_scale


    def apply_weights(self,
                      weights: Dict[str, Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> Tensor:
        assert bias is None
        weight = weights["weight"]
        dequant_scale = weights["dequant_scale"]
        logical_widths = weights["logical_widths"]
        per_token_quant = weights["per_token_quant"]

        # Q
        x_q, activation_scale = self._quantize(x, per_token_quant)
        
        # GEMM
        x_q = x_q.view(-1, x_q.shape[-1])
        out_q = torch.empty(
            (x_q.shape[0], weight.shape[0]),
            dtype=torch.int32, device=x.device)
        
        self.i8cugemm.linear_a8_w8_o32_(x_q, weight, out_q)
        out_q = out_q.view(*x_q.shape[:-1], -1)
        
        # DQ
        return self._dequantize(out_q, dequant_scale, activation_scale, logical_widths)
