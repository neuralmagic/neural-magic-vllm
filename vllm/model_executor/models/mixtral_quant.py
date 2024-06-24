# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import MixtralConfig

from vllm import _custom_ops as ops

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import fused_marlin_moe, fused_marlin_moe_2
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput

from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales_2


class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   quant_config=quant_config)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   quant_config=quant_config)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   quant_config=quant_config)

        # print("config:", quant_config)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print("sizes:",
        #     hidden_states.size(),
        #     self.w1.get_parameter("qweight").size(),
        #     self.w2.get_parameter("qweight").size(),
        #     self.w3.get_parameter("qweight").size())
        # print("ws:",
        #     self.w1.get_parameter("qweight"),
        #     self.w2.get_parameter("qweight"),
        #     self.w3.get_parameter("qweight"))

        # first mmm
        w1_out, _ = self.w1(hidden_states)
        # print("inter1:", w1_out)
        # silu
        w1_out = self.act_fn(w1_out)
        # 
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        # print("outs:", w1_out.size(), "*", w3_out.size(), "=", current_hidden_states.size())

        # print(w1_out[0][0].item(), w3_out[0][0].item(), current_hidden_states[0][0].item())
        # print(w1_out[0][1].item(), w3_out[0][1].item(), current_hidden_states[0][1].item())
        # print(w1_out[0][2].item(), w3_out[0][2].item(), current_hidden_states[0][2].item())
        # print("---")
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # Mixtral MoE creates a MixtralMLP for each expert
        # -> extract weights from these
        print("Mixtral MoE")
        self.config = config
        self.quant_config = quant_config
        print("config:", quant_config)
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        # self.quant_method = quant_config.get_quant_method(self)
        # assert self.quant_method is not None
        # self.w1 = torch.empty(sum(output_partition_sizes),
        #                                input_size_per_partition,
        #                                dtype=params_dtype)
        # set_weight_attrs(self.w1, {"input_dim": 1, "output_dim": 0})
        # print(self.w1)

        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}.")
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(range(
            self.num_total_experts), self.tp_size)[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(
                f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList([
            MixtralMLP(self.num_total_experts,
                       config.hidden_size,
                       config.intermediate_size,
                       quant_config=quant_config)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     quant_config=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print("forward")
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        qweights13 = []
        scaless13 = []
        qweights2 = []
        scaless2 = []
        qweights1 = []
        scaless1 = []
        qweights3 = []
        scaless3 = []

        new_version = True

        # no_shape_match = hidden_states.shape[1] != self.experts[0].w1.get_parameter("qweight").shape[0] * 16
        # print("NSM", no_shape_match)

        if new_version:
            for i in range(len(self.experts)):
                w1_qw = self.experts[i].w1.get_parameter("qweight")
                w3_qw = self.experts[i].w3.get_parameter("qweight")
                w1_s = self.experts[i].w1.get_parameter("scales")
                w3_s = self.experts[i].w3.get_parameter("scales")
                w2_qw = self.experts[i].w2.get_parameter("qweight")
                w2_s = self.experts[i].w2.get_parameter("scales")
                if (i == 0):
                    print("LAYERS", self.experts[i].w1.quant_method, self.experts[i].w2.quant_method, self.experts[i].w3.quant_method)
                    print("h:", hidden_states.shape, "qws:", w1_qw.shape, w2_qw.shape, w3_qw.shape)

                # w_ref = torch.zeros((8), dtype=torch.float)
                # q_item = w1_qw[0][0].item()
                # s_item = w1_s[0][0].item()
                # for i in range(8):
                #     deq_el = ((q_item >> (4 * i)) & 15)
                #     w_ref[i] = deq_el * s_item
                # print("w ref 1:", w_ref)

                w13_qw = torch.cat((w1_qw, w3_qw), 1)
                w13_s = torch.cat((w1_s, w3_s), 1)
                # size_k = w13_qw.shape[0] * 8
                # size_n = w13_qw.shape[1]
                # if no_shape_match:
                size_k = hidden_states.shape[1]
                size_n = w13_qw.shape[1]
                # print(hidden_states.shape, "*", w13_qw.shape, "(", size_k, size_n, ")")
                g_idx_sort_idx = torch.empty(0, dtype=torch.int, device=w13_qw.device)
                w13_qw = ops.gptq_marlin_repack(w13_qw, g_idx_sort_idx, size_k,
                                                size_n, self.quant_config.weight_bits)
                w13_s =  marlin_permute_scales_2(w13_s, size_k, size_n,
                                                 self.quant_config.group_size,
                                                 self.quant_config.weight_bits).half()

                # w_ref = torch.zeros((8), dtype=torch.float)
                # q_item = w13_qw[0][0].item()
                # s_item = w13_s[0][0].item()
                # for i in range(8):
                #     deq_el = ((q_item >> (4 * i)) & 15)
                #     w_ref[i] = deq_el * s_item
                # print("w ref 2:", w_ref)

                size_k = w2_qw.shape[0] * 8
                size_n = w2_qw.shape[1]
                w2_qw = ops.gptq_marlin_repack(w2_qw, g_idx_sort_idx, size_k,
                                               size_n, self.quant_config.weight_bits)
                w2_s =  marlin_permute_scales_2(w2_s, size_k, size_n,
                                                self.quant_config.group_size,
                                                self.quant_config.weight_bits).half()
                # print(hidden_states.shape, ",", w1_qw.shape, ",", w2_qw.shape, ",", w3_qw.shape, "(", size_k, size_n, ")")

                size_k = w1_qw.shape[0] * 8
                size_n = w1_qw.shape[1]
                # if (i == 0):
                #     print("unmarlin: ", w1_qw)
                #     print("scales:", w1_s)
                w1_qw = ops.gptq_marlin_repack(w1_qw, g_idx_sort_idx, size_k,
                                               size_n, self.quant_config.weight_bits)
                w1_s =  marlin_permute_scales_2(w1_s, size_k, size_n,
                                                self.quant_config.group_size,
                                                self.quant_config.weight_bits).half()
                
                w1_qw= torch.full(w1_qw.shape, -1180006009).to(w1_qw.device)
                w1_s= torch.full(w1_s.shape, 0.0054).to(w1_qw.device)
                # if (i == 0):
                #     print("marlin: ", w1_qw)
                #     print("scales:", w1_s)
                #     print("---")
                size_k = w3_qw.shape[0] * 8
                size_n = w3_qw.shape[1]
                w3_qw = ops.gptq_marlin_repack(w3_qw, g_idx_sort_idx, size_k,
                                               size_n, self.quant_config.weight_bits)
                w3_s =  marlin_permute_scales_2(w3_s, size_k, size_n,
                                                self.quant_config.group_size,
                                                self.quant_config.weight_bits).half()

                torch.cuda.synchronize()
                qweights13.append(w13_qw)
                scaless13.append(w13_s)
                qweights2.append(w2_qw)
                scaless2.append(w2_s)
                qweights1.append(w1_qw)
                scaless1.append(w1_s)
                qweights3.append(w3_qw)
                scaless3.append(w3_s)

            hidden_states= torch.full(hidden_states.shape, 1, dtype=hidden_states.dtype).to(hidden_states.device)
            
                # print(w1_s.shape, ":", w3_s.shape, ",", w2_s.shape)

            # print(bin(qweights2[0][0][0].item()), bin(qweights2[0][0][1].item()))

            qweight13 = torch.stack(qweights13, dim=0).to(qweights13[0].device)
            scales13 = torch.stack(scaless13, dim=0).to(scaless13[0].device)
            qweight2 = torch.stack(qweights2, dim=0).to(qweights2[0].device)
            scales2 = torch.stack(scaless2, dim=0).to(scaless2[0].device)
            qweight1 = torch.stack(qweights1, dim=0).to(qweights1[0].device)
            scales1 = torch.stack(scaless1, dim=0).to(scaless1[0].device)
            qweight3 = torch.stack(qweights3, dim=0).to(qweights3[0].device)
            scales3 = torch.stack(scaless3, dim=0).to(scaless3[0].device)

            # print(hidden_states.device, router_logits.device, qweight13.device, scales13.device, qweight2.device, scales2.device)

            # final_hidden_states = fused_marlin_moe(
            #     hidden_states,
            #     qweight13,
            #     qweight2,
            #     router_logits,
            #     self.top_k,
            #     renormalize=True,
            #     w1_scale=scales13,
            #     w2_scale=scales2,
            # )

            final_hidden_states = fused_marlin_moe_2(
                hidden_states,
                qweight1,
                qweight2,
                qweight3,
                router_logits,
                self.top_k,
                renormalize=True,
                w1_scale=scales1,
                w2_scale=scales2,
                w3_scale=scales3,
            )
            print("out size:", final_hidden_states.shape, "->", num_tokens, hidden_dim)
            for x in range(10):
                print(final_hidden_states[0][x].item(), end=' ')
            print('')
            assert(False)
            assert not final_hidden_states[0][0].isnan()
            return final_hidden_states

        torch.cuda.synchronize()
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        # TODO bring this back to all experts
        for expert_idx in self.expert_indicies[:1]:
            expert_layer = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)

            current_hidden_states = expert_layer(hidden_states).mul_(
                expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states).view(
            num_tokens, hidden_dim)

        print("out size:", final_hidden_states.shape, "->", num_tokens, hidden_dim)

        # if final_hidden_states.count_nonzero() > 0:
        #     for x in range(final_hidden_states.shape[0]):
        #         for y in range(final_hidden_states.shape[1]):
        #             if final_hidden_states[x][y].item() != 0:
        #                 print("nonzero:", final_hidden_states[x][y].item())
        #                 return final_hidden_states

        for x in range(10):
                print(final_hidden_states[0][x].item(), end=' ')
        print('')
        # for x in range(10):
        #         print(final_hidden_states[1][x].item(), end=' ')
        # print('')
        # for x in range(10):
        #         print(final_hidden_states[2][x].item(), end=' ')
        # print('')
        # for x in range(10):
        #         print(final_hidden_states[3][x].item(), end=' ')
        # print('')
        assert(False)
        return final_hidden_states


class MixtralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config)
        self.block_sparse_moe = MixtralMoE(config=config,
                                           quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config,
                                cache_config,
                                quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], attn_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("block_sparse_moe.experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
