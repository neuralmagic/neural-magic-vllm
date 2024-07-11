from abc import abstractmethod
from typing import Optional

import torch

from vllm import _custom_ops as ops

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe, fused_marlin_moe
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig, GPTQMarlinState)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    @abstractmethod
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True) -> torch.Tensor:
        raise NotImplementedError

class MarlinFusedMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Currently assuming is_k_full is always True
        # (input size per partition is the same as full input size)
        # Supports only sym for now (no zp)

        #TODO scales g_idx etc.
        #also do marlin transformations

        # print("*")
        # print("group_size:", self.quant_config.group_size)
        # print("hidden_size:", hidden_size)
        # print("intermediate_size:", intermediate_size)

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            scales_size2 =  intermediate_size // self.quant_config.group_size
        else:
            scales_size13 = 1
            scales_size2 = 1

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(torch.empty(num_experts,
                                                    hidden_size // self.quant_config.pack_factor,
                                                    2 * intermediate_size,
                                                    # hidden_size * 4,
                                                    # intermediate_size * 8,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(torch.empty(num_experts,
                                                    intermediate_size // self.quant_config.pack_factor,
                                                    hidden_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        # up_proj scales
        w13_scales = torch.nn.Parameter(torch.empty(num_experts,
                                                    scales_size13,
                                                    2 * intermediate_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        # down_proj scales
        w2_scales = torch.nn.Parameter(torch.empty(num_experts,
                                                   scales_size2,
                                                   hidden_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # print("w13g shape:", w13_g_idx.shape)
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.marlin_state = GPTQMarlinState.REPACK

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True) -> torch.Tensor:
        
        # TODO translate qweights into Marlin format
        if layer.marlin_state == GPTQMarlinState.REPACK:
            layer.marlin_state = GPTQMarlinState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                # It is important to use resize_() here since it ensures
                # the same buffer is reused
                getattr(layer, name).resize_(new_t.shape)
                getattr(layer, name).copy_(new_t)
                del new_t

            # Process act_order
            if self.quant_config.desc_act:
                # Get sorting based on g_idx
                w13_g_idx_sort_indices = torch.argsort(layer.w13_g_idx).to(torch.int)
                w2_g_idx_sort_indices = torch.argsort(layer.w2_g_idx).to(torch.int)
    
                w13_sorted_g_idx = layer.w13_g_idx[w13_g_idx_sort_indices]
                w2_sorted_g_idx = layer.w2_g_idx[w2_g_idx_sort_indices]

                replace_tensor("w13_g_idx", w13_sorted_g_idx)
                replace_tensor("w2_g_idx", w2_sorted_g_idx)
                replace_tensor("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
                replace_tensor("w2_g_idx_sort_indices", w2_g_idx_sort_indices)

            else:
                # Reset g_idx related tensors
                layer.w13_g_idx = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int),
                    requires_grad=False,
                )
                layer.w2_g_idx = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int),
                    requires_grad=False,
                )
                layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int),
                    requires_grad=False,
                )
                layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.int),
                    requires_grad=False,
                )

            print(layer.w13_qweight.shape)
            print(layer.w2_qweight.shape)
            print(x.shape)

            print("weight type:", layer.w13_qweight.dtype)

            # Repack weights
            marlin_w13_qweight = ops.gptq_marlin_moe_repack(
                layer.w13_qweight,
                layer.w13_g_idx_sort_indices,
                layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
                layer.w13_qweight.shape[2],
                self.quant_config.weight_bits,
            )
            replace_tensor("w13_qweight", marlin_w13_qweight)
            # marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            #     layer.w2_qweight,
            #     layer.w2_g_idx_sort_indices,
            #     layer.w2_qweight.shape[1] * 8,
            #     layer.w2_qweight.shape[2] // 2,
            #     self.quant_config.weight_bits,
            # )
            # replace_tensor("w2_qweight", marlin_w2_qweight)
            # TODO scales

            print(layer.w13_qweight.shape)
            print(layer.w2_qweight.shape)
            print(x.shape)

        return fused_marlin_moe(x,
                                layer.w13_qweight,
                                layer.w2_qweight,
                                router_logits,
                                layer.w13_g_idx,
                                layer.w2_g_idx,
                                layer.w13_g_idx_sort_indices,
                                layer.w2_g_idx_sort_indices,
                                top_k,
                                renormalize=renormalize,
                                w1_scale=layer.w13_scales,
                                w2_scale=layer.w2_scales)

class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """MoE method without quantization."""

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size,
                                                    hidden_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True) -> torch.Tensor:

        return fused_moe(x,
                         layer.w13_weight,
                         layer.w2_weight,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True)


# TODO should work from this
class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj / 
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (tp_size if tp_size is not None else
                        get_tensor_model_parallel_world_size())
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize

        # TODO we need to rewrite to QuantizedFusedMoEMethod
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod())
        else:
            # TODO assert GPTQ quant config
            self.quant_method: Optional[QuantizeMethodBase] = (
                MarlinFusedMoEMethod(quant_config))
            # self.quant_method = quant_config.get_quant_method(self)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: int, expert_id: int, is_quantized: bool = False):
        param_data = param.data
        # print("param_data shape:", param_data.shape)
        # print("loaded weight shape:", loaded_weight.shape)
        # TODO why is param_data[expert_id].shape == 7 * loaded_weight.shape?

        # print(param_data[expert_id])

        if is_quantized:
            if "_qweight" in weight_name or "_scales" in weight_name:
                if "w13" in weight_name:
                    shard_size = self.intermediate_size_per_partition
                    # print("shard size:", shard_size)
                    if shard_id == 0:
                        param_data[expert_id, :, :shard_size]  = loaded_weight
                    elif shard_id == 1:
                        param_data[expert_id, :, shard_size:] = loaded_weight
                    else:
                        ValueError("wrong shard:", shard_id)
                elif "w2" in weight_name:
                    param_data[expert_id][:] = loaded_weight
                else:
                    ValueError("what is this?", weight_name)
            elif "_g_idx" in weight_name:
                if "w13" not in weight_name and "w2" not in weight_name:
                    ValueError("what is this?", weight_name)
                param_data[expert_id] = loaded_weight
            else:
                ValueError("what is this?", weight_name)
        else:
            # FIXME(robertgshaw2-neuralmagic): Overfit to Mixtral.
            # Follow up PR to enable fp8 for other MoE models.
            if "input_scale" in weight_name or "w2.weight_scale" in weight_name:
                if param_data[expert_id] != 1 and (param_data[expert_id] -
                                                loaded_weight).abs() > 1e-5:
                    raise ValueError(
                        "input_scales of w1 and w3 of a layer "
                        f"must be equal. But got {param_data[expert_id]} "
                        f"vs. {loaded_weight}")
                param_data[expert_id] = loaded_weight
            # FIXME(robertgshaw2-neuralmagic): Overfit to Mixtral.
            # Follow up PR to enable fp8 for other MoE models.
            elif "weight_scale" in weight_name:
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                assert "w1" in weight_name or "w3" in weight_name
                shard_id = 0 if "w1" in weight_name else 1
                param_data[expert_id][shard_id] = loaded_weight
            else:
                tp_rank = get_tensor_model_parallel_rank()
                shard_size = self.intermediate_size_per_partition
                shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)

                # w1, gate_proj case: Load into first shard of w13.
                if shard_id == 0:
                    param_data[expert_id,
                            0:shard_size, :] = loaded_weight[shard, :]
                # w3, up_proj case: Load into second shard of w13.
                elif shard_id == 2:
                    param_data[expert_id, shard_size:2 *
                            shard_size, :] = loaded_weight[shard, :]
                # w2, down_proj case: Load into only shard of w2.
                elif shard_id == 1:
                    param_data[expert_id, :, :] = loaded_weight[:, shard]
                else:
                    raise ValueError(
                        f"Shard id must be in [0,1,2] but got {shard_id}")

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize)

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states
