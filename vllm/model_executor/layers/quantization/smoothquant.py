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
        return SQLinearMethod(Int8GEMM)

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
    """Linear method for SmoothQuant.
    """

    def __init__(self, gemm):
        i8_gemm = gemm()
        self.i8cugemm = i8_gemm.get_i8cugemm()

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       logical_widths=None) -> Dict[str, Tensor]:
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="cuda",
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })
        # q k v dequant_scales are used in QKVParallelLinear
        q_dequant_scale = Parameter(
            torch.tensor(1.0, dtype=torch.float32, device='cpu'),
            requires_grad=False,
        )
        k_dequant_scale = Parameter(
            torch.tensor(1.0, dtype=torch.float32, device='cpu'),
            requires_grad=False,
        )
        v_dequant_scale = Parameter(
            torch.tensor(1.0, dtype=torch.float32, device='cpu'),
            requires_grad=False,
        )
        # gate up dequant_scales are used in MergedColumnParallelLinear
        gate_dequant_scale = Parameter(
            torch.tensor(1.0, dtype=torch.float32, device='cpu'),
            requires_grad=False,
        )
        up_dequant_scale = Parameter(
            torch.tensor(1.0, dtype=torch.float32, device='cpu'),
            requires_grad=False,
        )
        # dequant_scale is used in RowParallelLinear
        dequant_scale = Parameter(
            torch.tensor(1.0, dtype=torch.float32, device='cpu'),
            requires_grad=False,
        )
        return {
            "weight": weight,
            "q_dequant_scale": q_dequant_scale,
            "k_dequant_scale": k_dequant_scale,
            "v_dequant_scale": v_dequant_scale,
            "gate_dequant_scale": gate_dequant_scale,
            "up_dequant_scale": up_dequant_scale,
            "dequant_scale": dequant_scale
        }

    def apply_weights(self,
                      weights: Dict[str, Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> Tensor:
        assert bias is None
        weight = weights["weight"]
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], weight.shape[0]),
                        dtype=torch.int32,
                        device=x.device)
        self.i8cugemm.linear_a8_w8_o32_(x, weight, y)
        y = y.view(*x_shape[:-1], -1)
        return y


class FunSQLinearMethod(LinearMethodBase):
    """Linear method for SmoothQuant.
    """

    def __init__(
        self, 
        per_token_quant,
        quant_dtype,
        dequant_dtype,
    ):
        self.per_token_quant = per_token_quant
        self.quant_dtype = quant_dtype
        self.dequant_dtype = dequant_dtype
        self.i8cugemm = Int8GEMM().get_i8cugemm()

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        logical_widths: Optional[List[int]] = None,
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
                dtype=torch.float32,
                device='cpu'
            ), requires_grad=False
        )

        return {
            "weight": weight,
            "dequant_scale": dequant_scale,
            "logical_widths": logical_widths,
        }


    def _dequantize(self, x_q, weight_scales, activation_scales, logical_widths):
        x_dq = torch.empty_like(x_q, dtype=self.dequant_dtype)

        # Split into shards.
        x_q_split = x_q.split(logical_widths, dim=-1)
        x_dq_split = x_dq.split(logical_widths, dim=-1)

        # Dequantize each shard.
        for xq, weight_scale, activation_scale, xdq in zip(
            x_q_split, weight_scales, activation_scales, x_dq_split):
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

        # Q
        x_q, activation_scale = self._quantize(x, self.per_token_quant)
        
        # GEMM
        x_q = x_q.view(-1, x_q.shape[-1])
        out_q = torch.empty(
            (x_q.shape[0], weight.shape[0]),
            dtype=torch.int32, device=x.device)
        
        self.i8cugemm.linear_a8_w8_o32_(x_q, weight, out_q)
        out_q = out_q.view(*x_q.shape[:-1], -1)
        
        # DQ
        return self._dequantize(out_q, dequant_scale, activation_scale, logical_widths)


class SQLinearMethodQKV(SQLinearMethod):

    def __init__(self,
                 gemm,
                 qkv_sizes : Tuple[int, int, int],
                 quant_dtype : torch.dtype = torch.int8,
                 dequant_dtype : torch.dtype = torch.float):
        super().__init__(gemm)
        self.qkv_sizes = qkv_sizes
        self.quant_dtype = quant_dtype
        self.dequant_dtype = dequant_dtype

    def quantize(self, x):
        assert x.dtype != self.quant_dtype
        x_q = torch.empty_like(x, dtype=self.quant_dtype)
        ops.quant(x_q, x, 1.0)
        return x_q

    def dequantize(self, x_q, weights : Dict[str, Tensor]):
        # split to get the quantized qkv
        q_q, k_q, v_q = x_q.split(list(self.qkv_sizes), dim=-1)

        # create dequant qkv buffer and split to get the individual dequant qkv
        # buffers
        qkv = torch.empty_like(x_q, dtype=self.dequant_dtype)
        q, k, v = qkv.split(list(self.qkv_sizes), dim=-1)

        q_scale, k_scale, v_scale = (weights['q_dequant_scale'],
                                     weights['k_dequant_scale'],
                                     weights['v_dequant_scale'])
        ops.dequant(q, q_q, q_scale)
        ops.dequant(k, k_q, k_scale)
        ops.dequant(v, v_q, v_scale)

        return qkv

    def apply_weights(self,
                      weights: Dict[str, Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> Tensor:
        x_q = self.quantize(x)
        y_q = super().apply_weights(weights, x_q, bias)
        return self.dequantize(y_q, weights)
        
class SQLinearMethodOProj(SQLinearMethod):

    def __init__(self,
                 gemm,
                 use_per_token_quant:bool,
                 quant_dtype : torch.dtype = torch.int8,
                 dequant_dtype : torch.dtype = torch.float):
        super().__init__(gemm)
        self.use_per_token_quant = use_per_token_quant
        self.quant_dtype = quant_dtype
        self.dequant_dtype = dequant_dtype

    def quantize(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x is the paged-attention output
        assert x.dtype != self.quant_dtype
        act_scale = None
        x_q = torch.empty_like(x, dtype=self.quant_dtype)
        if self.use_per_token_quant:
            act_scale = torch.empty(x.numel() // x.shape[-1],
                                dtype=torch.float32,
                                device=x.device)
            ops.quant(x_q, x, act_scale)
        else:
            ops.quant(x_q, x, 1.0)
        return x_q, act_scale

    def dequantize(self, x_q, weights : Dict[str, Tensor], act_scale : torch.Tensor) -> torch.Tensor:
        o_dequant_scale = weights['dequant_scale']
        x = torch.empty_like(
            x_q,
            dtype=self.dequant_dtype,
            device=x_q.device)
        ops.dequant(x, x_q, act_scale, o_dequant_scale)
        return x

    def apply_weights(self,
                      weights: Dict[str, Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> Tensor:
        pass
        x_q, act_scale = self.quantize(x)
        y_q = super().apply_weights(weights, x_q, bias)
        return self.dequantize(y_q, weights, act_scale)

class SQLinearMethodGateUpProj(SQLinearMethod):

    def __init__(self,
                 gemm,
                 quant_dtype : torch.dtype = torch.int8,
                 dequant_dtype : torch.dtype = torch.float):
        super().__init__(gemm)
        self.quant_dtype = quant_dtype
        self.dequant_dtype = dequant_dtype

    def quantize(self, x) -> torch.Tensor: 
        # x is the attention output
        assert x.dtype != self.quant_dtype
        x_q = torch.empty_like(x, dtype=self.quant_dtype, device=x.device)
        ops.quant(x_q, x, 1.0)
        return x_q

    def dequantize(self, gate_up_q: torch.Tensor, weights : Dict[str, Tensor]) -> torch.Tensor:

        def split_gate_up(gate_up : torch.Tensor):
            d = gate_up.shape[-1]
            return (torch.narrow(gate_up, 1, 0, d//2), 
                    torch.narrow(gate_up, 1, d//2, d//2))

        # create a dequant gate_up buffer and split it into constituent parts.
        gate_up = torch.empty_like(gate_up_q,
                                   dtype=self.dequant_dtype,
                                   device=gate_up_q.device) 

        # split quantized gate_up into constituent parts.
        gate_q, up_q = split_gate_up(gate_up_q)
        # split output gate_up buffer into constituent parts. 
        gate, up = split_gate_up(gate_up)

        gate_scale, up_scale = (weights['gate_dequant_scale'],
                                weights['up_dequant_scale'])
        ops.dequant(gate, gate_q, gate_scale)
        ops.dequant(up, up_q, up_scale)

        return gate_up

    def apply_weights(self,
                      weights: Dict[str, Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> Tensor:
        x_q = self.quantize(x)
        gate_up_q = super().apply_weights(weights, x_q, bias)
        return self.dequantize(gate_up_q, weights)

class SQLinearMethodDownProj(SQLinearMethod):

    def __init__(self,
                 gemm,
                 quant_dtype : torch.dtype = torch.int8,
                 dequant_dtype : torch.dtype = torch.float):
        super().__init__(gemm)
        self.quant_dtype = quant_dtype
        self.dequant_dtype = dequant_dtype

    def quantize(self, x) -> Tuple[torch.Tensor, torch.Tensor]: 
        assert x.dtype != self.quant_dtype
        # TODO (varun) : This is per-token quant - Read from config
        x_q = torch.empty_like(x, dtype=self.quant_dtype) 
        scale = torch.empty(x.numel() // x.shape[-1],
                            dtype=torch.float32,
                            device=x.device)
        ops.quant(x_q, x, scale)
        return x_q, scale

    def dequantize(self, x_q, weights : Dict[str, Tensor], act_scale : torch.Tensor) -> torch.Tensor:
        down_dequant_scale = weights['dequant_scale']
        x = torch.empty_like(
            x_q,
            dtype=self.dequant_dtype,
            device=x_q.device)
        ops.dequant(x, x_q, act_scale, down_dequant_scale)
        return x

    def apply_weights(self,
                      weights: Dict[str, Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_q, act_scale = self.quantize(x)
        y_q = super().apply_weights(weights, x_q, bias)
        return self.dequantize(y_q, weights, act_scale)