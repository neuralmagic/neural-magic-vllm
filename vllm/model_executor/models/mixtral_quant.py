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
import re

from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import fused_marlin_moe, FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               FusedLinearMarlin,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_permute_scales_numbits)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput


class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        experimental_fused_moe: bool,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size
        self.experimental_fused_moe = experimental_fused_moe

        # TODO
        # print("hidden dim:", self.hidden_dim, "ffn_dim:", self.ffn_dim)
        if self.experimental_fused_moe:
            self.ws = FusedLinearMarlin(self.hidden_dim, self.ffn_dim,
                                        quant_config=quant_config)
        else:
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

            # TODO: Use vllm's SiluAndMul
            self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.experimental_fused_moe:
            current_hidden_states = self.ws(hidden_states)
            return current_hidden_states
        else:
            w1_out, _ = self.w1(hidden_states)
            w1_out = self.act_fn(w1_out)
            w3_out, _ = self.w3(hidden_states)
            current_hidden_states = w1_out * w3_out
            current_hidden_states, _ = self.w2(current_hidden_states)
            return current_hidden_states


class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        experimental_fused_moe: bool,
        old_code: bool,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.experimental_fused_moe = experimental_fused_moe
        self.old_code = old_code
        self.quant_config = quant_config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
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

        if self.old_code:
            self.experts = nn.ModuleList([
                MixtralMLP(self.num_total_experts,
                        config.hidden_size,
                        config.intermediate_size,
                        self.experimental_fused_moe,
                        quant_config=quant_config)
                if idx in self.expert_indicies else None
                for idx in range(self.num_total_experts)
            ])
        else:
            # TODO type
            params_dtype = torch.float16
            self.experts = FusedMoE(num_experts=self.num_total_experts,
                                    top_k=self.top_k,
                                    hidden_size=config.hidden_size,
                                    intermediate_size=config.intermediate_size,
                                    params_dtype=params_dtype,
                                    reduce_results=True,
                                    renormalize=True,
                                    quant_config=quant_config,
                                    tp_size=self.tp_size)
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     quant_config=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits, _ = self.gate(hidden_states)

        if self.experimental_fused_moe:

            if not self.old_code:
                return self.experts(hidden_states.half(), router_logits).bfloat16()

            qweight13_l = []
            scales13_l = []
            qweight2_l = []
            scales2_l = []
            g_idx13_l = []
            g_idx2_l = []
            g_idx_sort_idx13_l = []
            g_idx_sort_idx2_l = []

            for i in range(len(self.experts)):
                current_expert = self.experts[i].ws
                current_expert(hidden_states)
                # print("get weights")
                # w1_qw = current_expert.get_parameter("qweight1").int()
                # w3_qw = current_expert.get_parameter("qweight3").int()
                # w1_s = current_expert.get_parameter("scales1").half()
                # w3_s = current_expert.get_parameter("scales3").half()
                w2_qw = current_expert.get_parameter("qweight2").int()
                w2_s = current_expert.get_parameter("scales2").half()
                w13_qw = current_expert.get_parameter("qweight13").int()
                w13_s = current_expert.get_parameter("scales13").half()
                # w1_qw = self.experts[i].w1.get_parameter("qweight").int()
                # w3_qw = self.experts[i].w3.get_parameter("qweight").int()
                # w1_s = self.experts[i].w1.get_parameter("scales").half()
                # w3_s = self.experts[i].w3.get_parameter("scales").half()
                # w2_qw = self.experts[i].w2.get_parameter("qweight").int()
                # w2_s = self.experts[i].w2.get_parameter("scales").half()
                if self.quant_config.desc_act:
                    # g_idx13 = self.experts[i].w1.get_parameter("g_idx")
                    # g_idx2 = self.experts[i].w2.get_parameter("g_idx")
                    g_idx13 = current_expert.get_parameter("g_idx13")
                    g_idx2 = current_expert.get_parameter("g_idx2")
                    g_idx_sort_idx13 = current_expert.get_parameter("g_idx_sort_indices13")
                    g_idx_sort_idx2 = current_expert.get_parameter("g_idx_sort_indices2")
                else:
                    g_idx13 = torch.empty(0, device=w13_qw.device)
                    g_idx2 = torch.empty(0, device=w2_qw.device)
                    g_idx_sort_idx13 = torch.empty(0, device=w13_qw.device)
                    g_idx_sort_idx2 = torch.empty(0, device=w2_qw.device)
                # g_idx_sort_idx13 = torch.argsort(g_idx13).int()
                # g_idx_sort_idx2 = torch.argsort(g_idx2).int()

                # w13_qw = torch.cat((w1_qw, w3_qw), 0)
                # w13_s = torch.cat((w1_s, w3_s), 0)
                # w13_qw = torch.cat((w1_qw, w3_qw), 1)
                # w13_s = torch.cat((w1_s, w3_s), 1)
                # size_k = hidden_states.shape[1]
                # size_n = w13_qw.shape[1]
                # print("do repack 13", w13_qw.shape, g_idx_sort_idx13.shape)
                # w13_qw = ops.gptq_marlin_repack(w13_qw, g_idx_sort_idx13, size_k,
                #                                 size_n,
                #                                 self.quant_config.weight_bits)
                # w13_s = marlin_permute_scales_numbits(
                #     w13_s, size_k, size_n, self.quant_config.group_size,
                #     self.quant_config.weight_bits)

                # size_k = w2_qw.shape[0] * 8
                # size_n = w2_qw.shape[1]
                # print("do repack 2", w2_qw.shape, g_idx_sort_idx2.shape)
                # w2_qw = ops.gptq_marlin_repack(w2_qw, g_idx_sort_idx2, size_k,
                #                             size_n,
                #                             self.quant_config.weight_bits)
                # w2_s = marlin_permute_scales_numbits(w2_s, size_k, size_n,
                #                                     self.quant_config.group_size,
                #                                     self.quant_config.weight_bits)

                qweight13_l.append(w13_qw)
                scales13_l.append(w13_s)
                qweight2_l.append(w2_qw)
                scales2_l.append(w2_s)
                g_idx13_l.append(g_idx13)
                g_idx2_l.append(g_idx2)
                g_idx_sort_idx13_l.append(g_idx_sort_idx13)
                g_idx_sort_idx2_l.append(g_idx_sort_idx2)

            qweight13 = torch.stack(qweight13_l, dim=0).to(qweight13_l[0].device)
            scales13 = torch.stack(scales13_l, dim=0).to(scales13_l[0].device)
            qweight2 = torch.stack(qweight2_l, dim=0).to(qweight2_l[0].device)
            scales2 = torch.stack(scales2_l, dim=0).to(scales2_l[0].device)
            g_idx13 = torch.stack(g_idx13_l, dim=0).to(g_idx13_l[0].device)
            g_idx2 = torch.stack(g_idx2_l, dim=0).to(g_idx2_l[0].device)
            g_idx_sort_idx13 = torch.stack(g_idx_sort_idx13_l,
                                        dim=0).to(g_idx_sort_idx13_l[0].device)
            g_idx_sort_idx2 = torch.stack(g_idx_sort_idx2_l,
                                        dim=0).to(g_idx_sort_idx2_l[0].device)

            final_hidden_states = fused_marlin_moe(
                hidden_states.half(),
                qweight13,
                qweight2,
                router_logits,
                g_idx13,
                g_idx2,
                g_idx_sort_idx13,
                g_idx_sort_idx2,
                self.top_k,
                renormalize=True,
                w1_scale=scales13,
                w2_scale=scales2,
            )

            return final_hidden_states.bfloat16()

        else:

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights,
                                                            self.top_k,
                                                            dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            final_hidden_states = None
            for expert_idx in self.expert_indicies:
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

            return tensor_model_parallel_all_reduce(final_hidden_states).view(
                                                    num_tokens, hidden_dim)

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
        experimental_fused_moe: bool,
        old_code: bool,
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
                                           experimental_fused_moe=experimental_fused_moe,
                                           old_code=old_code,
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
        experimental_fused_moe: bool,
        old_code: bool,
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
                                experimental_fused_moe,
                                old_code,
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

        self.experimental_fused_moe = True
        self.old_code = False

        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(config, self.experimental_fused_moe, self.old_code, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                    config.hidden_size,
                                    quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
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

        # weight names: [w[0] for w in weights]
        # 'model.layers.0.block_sparse_moe.experts.0.w1.bias',
        # 'model.layers.0.block_sparse_moe.experts.0.w1.g_idx',
        # 'model.layers.0.block_sparse_moe.experts.0.w1.qweight',
        # 'model.layers.0.block_sparse_moe.experts.0.w1.qzeros',
        # 'model.layers.0.block_sparse_moe.experts.0.w1.scales',
        # 'model.layers.0.block_sparse_moe.experts.0.w2.bias',
        # 'model.layers.0.block_sparse_moe.experts.0.w2.g_idx',
        # 'model.layers.0.block_sparse_moe.experts.0.w2.qweight',
        # 'model.layers.0.block_sparse_moe.experts.0.w2.qzeros',
        # 'model.layers.0.block_sparse_moe.experts.0.w2.scales',
        # 'model.layers.0.block_sparse_moe.experts.0.w3.bias',
        # 'model.layers.0.block_sparse_moe.experts.0.w3.g_idx',
        # 'model.layers.0.block_sparse_moe.experts.0.w3.qweight',
        # 'model.layers.0.block_sparse_moe.experts.0.w3.qzeros',
        # 'model.layers.0.block_sparse_moe.experts.0.w3.scales',
        # 'model.layers.0.block_sparse_moe.experts.1.w1.bias'
        # ...

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

                if self.old_code:
                    if self.experimental_fused_moe:
                        if("block_sparse_moe.experts." in name
                           and ".w1." not in name and ".w2." not in name
                           and ".w3." not in name 
                           and name not in params_dict):
                            continue
                
                        if (".qzeros" in name):
                            continue

                        has_weight_or_scale = (".qweight" in name or ".scales" in name)
                        has_g_idx = ".g_idx" in name
                        if (has_weight_or_scale and ".w1." in name):
                            name = name.replace(".w1.", ".ws.")
                            name += "1"
                        if ((has_weight_or_scale or has_g_idx) and ".w2." in name):
                            name = name.replace(".w2.", ".ws.")
                            name += "2"
                        if (has_weight_or_scale and ".w3." in name):
                            name = name.replace(".w3.", ".ws.")
                            name += "3"
                        if (has_g_idx and ".w1." in name):
                            name = name.replace(".w1.", ".ws.")
                            name += "13"
                        if (has_g_idx and ".w3." in name):
                            name = name.replace(".w3.", ".ws.")
                            name += "13"

                    else:
                        if("block_sparse_moe.experts." in name
                           and name not in params_dict):
                            continue

                    param = params_dict[name]
                    # print("load", name, "into", param.shape)
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

                else:

                    if self.experimental_fused_moe:
                        if("block_sparse_moe.experts." in name
                        and ".w1." not in name and ".w2." not in name
                        and ".w3." not in name 
                        and name not in params_dict):
                            continue
                
                        if (".qzeros" in name):
                            continue

                        shard_id = None
                        expert_id = 0

                        # print("process:", name)

                        has_any_numbered = (".qweight" in name or ".scales" in name or ".g_idx" in name)
                        if (has_any_numbered and (".w1." in name)):
                            name = name.replace(".w1.", ".w13_")
                            shard_id = 0
                        if (has_any_numbered and (".w2." in name)):
                            name = name.replace(".w2.", ".w2_")
                            shard_id = 0
                        if (has_any_numbered and (".w3." in name)):
                            name = name.replace(".w3.", ".w13_")
                            shard_id = 1

                        exp_string = re.search(r"\.experts\.\d+.", name)
                        if exp_string:
                            exp_string = exp_string.group(0)
                            # print("Exp string:", exp_string)
                            expert_id = int(exp_string.split(".")[2])
                            # print("I found:", expert_shard, "in", name)
                            name = name.replace(exp_string, ".experts.")

                    else:
                        if("block_sparse_moe.experts." in name
                        and name not in params_dict):
                            continue

                    param = params_dict[name]
                    
                    # print("load", name, "into", param.shape)
                    if shard_id is not None:
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        # print("load:", name, "with shard", shard_id)
                        weight_loader(param, loaded_weight, name, shard_id, expert_id, True)
                    else:
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        # print("load:", name, "without shard")
                        weight_loader(param, loaded_weight)
                        
