from typing import Any, Dict, List, Tuple, Type, Optional, Union
import threading

import cutlass
from cutlass import Tensor as FakeTensor
import cutlass.epilogue
import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.smoothquant.formats import (
    SmoothQuantFormat,
    SmoothQuantDynamicPerToken,
    SmoothQuantStaticPerTensor,
)

LAYER_KEYS = ["qkv", "out", "fc1", "fc2"]
FORMAT_REGISTRY = {
    "per-token": SmoothQuantDynamicPerToken,
    "per-tensor": SmoothQuantStaticPerTensor,
}

def get_sq_format_cls(format_key: str) -> Type[SmoothQuantFormat]:
    if format_key not in FORMAT_REGISTRY:
        raise ValueError(f"Invalid smoothquant format: {format_key}")
    return FORMAT_REGISTRY[format_key]

class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant.

    Reference: https://github.com/mit-han-lab/smoothquant
    """
    def __init__(self,
                 layer_format_map: Dict[str, str]) -> None:
        self.layer_format_map = layer_format_map

        for key, format in self.layer_format_map.items():
            if key not in LAYER_KEYS:
                raise ValueError(
                    f"Found key of {key} in {self.layer_format_map}, " 
                    f"but key must be one of {LAYER_KEYS}"
                )
            if format not in FORMAT_REGISTRY:
                raise ValueError(
                    f"Found format of {format} in {self.layer_format_map}, "
                    f"but format must be one of {FORMAT_REGISTRY}"
                )
        for key in LAYER_KEYS:
            if key not in self.layer_format_map:
                raise ValueError(
                    f"Could not find {key} in {layer_format_map}"
                )

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(layer_format_map={self.layer_format_map})")

    def get_name(self) -> str:
        return "smoothquant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        # TODO: check if we support fp16 / bf16 as well.
        return [torch.float]

    def get_min_capability(self) -> int:
        # TODO: check if this is right.
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        return [
            "quant_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        layer_format_map: Dict[str, str] = {}
        for layer_key, format in config.items():
            if format in FORMAT_REGISTRY:
                layer_format_map[layer_key] = format
        return cls(layer_format_map)
    
    def get_linear_method(self) -> "SmoothQuantLinearMethod":
        return SmoothQuantLinearMethod(self)


# TODO: why is this needed?
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


class SmoothQuantLinearMethod(LinearMethodBase):
    def __init__(self, sq_config: SmoothQuantConfig) -> None:
        self.sq_config = sq_config
        self.sq_type = None
        self.i8cugemm = Int8GEMM().get_i8cugemm()    

    def maybe_update_loaded_weight_name(self, 
                                        name: str) -> str:
        """Convert serialized name k_dequant_scale to dequant_scale.

        This function is called by model_cls.load_weights() during the weight
        loading process to match on disk state dict to vllm state dict.
        """
        if "dequant_scale" in name:
            suffix = name.split('.')[-1]
            name.replace(suffix, "dequant_scale")
        return name

    def scales_shard_splitter(self, 
                              param: torch.Tensor,
                              loaded_weight: torch.Tensor, 
                              shard_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index into param for for loading.
        
        This function is called by QKVColumnLinear and MergedColumnParallelLinear
        during weight loading to put the scales from disk in the right spot.
        """
        if type(shard_id) == str:
            qkv_idxs = { "q": 0, "k": 1, "v": 2 }
            if shard_id not in qkv_idxs:
                raise ValueError(f"Invalid shard_id {shard_id}")
            shard_id = qkv_idxs[shard_id]
        elif type(shard_id) != int:
            raise ValueError(f"Invalid shard id {shard_id}")

        return param[shard_id], loaded_weight

    def get_layer_format(self, layer_name: str) -> SmoothQuantFormat:
        """
        Gets the SmoothQuantFormat for a specific layer.

        SmoothQuantLinearMethod uses SmoothQuantType to support non-uniform quantization 
        (where each layer has a different format). To determine the SmoothQuantFormat 
        for a layer, we match the layer_name to the layer_keys=["qkv","out","fc1","fc2"] 
        and use layer_format_map to to determine the SQFormat.
        
        Args:
            layer_name: Name of the layer we are creating the LinearMethod for.
        Returns
            sq_linear_method: SmoothQuantLinearMethod with the right SQFormat.
        """      
        # Note: AutoSmoothQuant Serialization is not very good yet.
        #
        # It looks like the following (which does not map to layer names in the model):
        # {
        #   "qkv": "per-tensor",
        #   "out": "per-token",
        #   "fc1": "per-tensor",
        #   "fc2": "per-token"
        # }
        #
        # So, this is a hack for llama now. But with the SparseMLConfig, we can make robust,
        # where we actually use the layer_name in the model to look up what the format is
        # based on the config.
        #
        # What it would actually look like:
        # layer_config is None
        # for supported_key in SUPPORTED_LAYER_KEYS:
        #     if supported_key in layer_name:
        #         sq_format = self.layer_mapping[lookup_key]
        #           return get_sq_format_cls(sq_format)()

        HACKED_REMAP_FOR_LLAMA = {
            "qkv": "qkv", 
            "o_proj": "out", 
            "gate_up": 
            "fc1", "down": "fc2",
        }

        for match_key, lookup_key in HACKED_REMAP_FOR_LLAMA.items():
            if match_key in layer_name:
                sq_format = self.sq_config.layer_format_map[lookup_key]
                return get_sq_format_cls(sq_format)()
            
        raise ValueError
        
    def create_weights(self, 
                       layer_name: str,
                       input_size_per_partition: int,
                       output_sizes_per_partition: int,
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        del input_size, output_size

        # Statically Quantized Weights.
        weight = Parameter(
            torch.empty(
                sum(output_sizes_per_partition),
                input_size_per_partition,
                device="cuda", dtype=torch.int8,
            ), requires_grad=False,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })

        # Static scale for each logical weight (e.g. 3 for QKV).
        dequant_scale = Parameter(
            torch.empty(
                len(output_sizes_per_partition), 
                 device='cuda', dtype=params_dtype,
            ), requires_grad=False
        )
        set_weight_attrs(dequant_scale, {
            "shard_splitter": self.scales_shard_splitter,
        })

        return {
            "weight": weight,
            "dequant_scale": dequant_scale,
            "logical_widths": output_sizes_per_partition,
            "sq_format": self.get_layer_format(layer_name)
        }

    def _quantize(self,
                  x: torch.Tensor,
                  sq_format: SmoothQuantFormat) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantize activations.

        Args:
            x: Activation at floating point precision.
        Returns:
            x_q: Quantized activation at INT8
            activation_scales: Optional dynamic scales for each token.
        """
        x_q = torch.empty_like(x, dtype=torch.int8)
        x_q, activation_scales = sq_format.quantize_op(x, x_q)
        return x_q, activation_scales

    def _dequantize(self, 
                    x_q: torch.Tensor, 
                    dynamic_scales: Optional[torch.Tensor],
                    static_scales: torch.Tensor,
                    logical_widths: List[int],
                    dtype: torch.dtype,
                    sq_format: SmoothQuantFormat) -> torch.Tensor:
        """Dequantize activations.

        Args:
            x_q: quantized activations.
            dynamic_scales: Optional dynamic scales.
            static_scales: Static dequantization scales.
            logical_widths: Width of each logical activation (for QKV case).
            dtype: Datatype to dequantize to.
        Returns:
            x_dq: dequantized activation at output_dtype precision
        """
        # Split X_q and X_dq buffer into logical activations (for QKV case).
        x_q_split = x_q.split(logical_widths, dim=-1)
        x_dq = torch.empty_like(x_q, dtype=dtype)
        x_dq_split = x_dq.split(logical_widths, dim=-1)
        # Dequantize in place and return.
        sq_format.dequantize_op(x_q_split, x_dq_split, dynamic_scales, static_scales)
        return x_dq

    def cutlass_gemm(self,
                     x_q : torch.Tensor,
                     w_q : torch.Tensor,
                     o_q : torch.Tensor) -> torch.Tensor:

        print (f"cutlass gemm : x_q {x_q.shape} {x_q.dtype} \n \
                w_q {w_q.shape} {w_q.dtype} \n \
                o_q {o_q.shape} {o_q.dtype}")

        plan = cutlass.op.Gemm(element_A=x_q.dtype, element_B=w_q.dtype,
                               element_C=o_q.dtype, element_D=o_q.dtype,
                               layout_A=cutlass.LayoutType.RowMajor, 
                               layout_B=cutlass.LayoutType.ColumnMajor,
                               layout_C=cutlass.LayoutType.RowMajor, 
                               element_accumulator=torch.int32,
                               # TODO (varun) : lets not have kernel cc here please.
                               kernel_cc=80)

        plan.run(x_q, w_q.t(), o_q, o_q, alpha=1, beta=0, print_module=False)
        return o_q

    def setup_dequant_epilogue(self,
                               plan : cutlass.op.Gemm,
                               dq: torch.Tensor, 
                               static_scales: Optional[torch.Tensor],
                               activation_scales: Optional[torch.Tensor]) -> Tuple[cutlass.op.Gemm, Dict]:

        if all([static_scales is None, activation_scales is None]):
            return plan, None
        assert static_scales is not None

        def epilog_with_scales_and_act_scales(accum, scales, act_scales):
            D = accum * scales * act_scales
            return D

        def epilog_with_scales(accum, scales):
            D = accum * scales
            return D

        epilog_tensors =  {
            'scales' : static_scales,
            'D' : dq
        }
        epilogue_trace_tensors = {
            "accum": FakeTensor(element=torch.int32, shape=dq.shape,
                                layout_tag=cutlass.LayoutType.RowMajor),
            'scales' : static_scales,
            'D' : dq,
        }
        epilog_fn = epilog_with_scales

        if activation_scales is not None:
            epilog_tensors['act_scales'] = activation_scales
            epilogue_trace_tensors['act_scales'] = activation_scales
            epilog_fn = epilog_with_scales_and_act_scales

        plan.epilogue_visitor = cutlass.epilogue.trace(epilog_fn, epilogue_trace_tensors)
        return plan, epilog_tensors 

    def cutlass_gemm_dq(self,
                     x_q : torch.Tensor,
                     w_q : torch.Tensor,
                     static_scales: torch.Tensor,
                     # TODO : move optional to end
                     activation_scales: Optional[torch.Tensor],
                     dtype: torch.dtype) -> torch.Tensor:

        dq = torch.empty((x_q.shape[0], w_q.shape[0]),
                          dtype=dtype, device="cuda")

        print(f"cutlass gemm : x_q {x_q.shape} {x_q.dtype} \n \
                w_q {w_q.shape} {w_q.dtype} \n \
                a_q {dq.shape} {dq.dtype} \n \
                static scales {static_scales} {static_scales.shape} \n")
        if activation_scales is not None:
            print (f"activation_scales - {activation_scales} {activation_scales.shape}")

        plan = cutlass.op.Gemm(element_A=x_q.dtype, element_B=w_q.dtype,
                               element_C=dq.dtype, element_D=dq.dtype,
                               layout_A=cutlass.LayoutType.RowMajor, 
                               layout_B=cutlass.LayoutType.ColumnMajor,
                               layout_C=cutlass.LayoutType.RowMajor, 
                               element_accumulator=torch.int32,
                               # TODO (varun) : lets not have kernel cc here please.
                               kernel_cc=80)

        plan, visitor_args = self.setup_dequant_epilogue(plan, dq, static_scales, activation_scales)

        plan.run(x_q, w_q.t(), dq, dq, alpha=1, beta=0,
                 visitor_args=visitor_args, print_module=False)

        dq = dq.view(*x_q.shape[:-1], -1)
        return dq

    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward method. Computes Q --> GEMM --> DQ.

        Args:
            weigths: Dictionary of weights, scales, and metadata.
            x: Input in floating point precision.
            bias: Optional bias.
        Returns:
            a_dq: Dequantized activation at floating point precision.
        """
        if bias is not None:
            raise NotImplementedError
        weight_q = weights["weight"]
        static_scales = weights["dequant_scale"]
        logical_widths = weights["logical_widths"]
        sq_format = weights["sq_format"]

        # Q
        x_q, activation_scales = self._quantize(x, sq_format)

        if isinstance(sq_format, SmoothQuantStaticPerTensor):
            # TODO (varun) : Move scale_to_broadcast to create_weights
            scale_to_broadcast = torch.zeros((sum(logical_widths)))
            filled_idx = 0 
            for scale, size in zip(static_scales, logical_widths):
                indices = torch.arange(filled_idx, filled_idx + size)
                scale_to_broadcast.index_fill_(0, indices, scale)
                filled_idx += size

            return self.cutlass_gemm_dq(x_q, weight_q,
                                        scale_to_broadcast,
                                        None,
                                        x.dtype)
        else:

            # GEMM
            x_q = x_q.view(-1, x_q.shape[-1])
            a_q = torch.empty((x_q.shape[0], weight_q.shape[0]), dtype=torch.int32, device="cuda")
            print (f"a_q {a_q.shape} {a_q.dtype} - activation_scales {activation_scales.shape} {activation_scales.dtype} \n")
            self.i8cugemm.linear_a8_w8_o32_(x_q, weight_q, a_q)
            a_q = a_q.view(*x_q.shape[:-1], -1)
            # DQ
            dq = self._dequantize(a_q, activation_scales, static_scales,
                                  logical_widths, x.dtype, sq_format)

            # TODO (varun) : Move scale_to_broadcast to create_weights
            scale_to_broadcast = torch.zeros((sum(logical_widths)))
            filled_idx = 0 
            for scale, size in zip(static_scales, logical_widths):
                indices = torch.arange(filled_idx, filled_idx + size)
                scale_to_broadcast.index_fill_(0, indices, scale)
                filled_idx += size

            activation_scales = activation_scales[:, None]

            cutlass_dq = self.cutlass_gemm_dq(x_q, weight_q, scale_to_broadcast,
                                              activation_scales, x.dtype)
            
            torch.allclose(dq, cutlass_dq)
            return cutlass_dq

