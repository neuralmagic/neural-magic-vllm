# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/t5/modeling_t5.py
# Copyright 2023 The vLLM team.
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model."""
from typing import List, Optional, Tuple

import math
import copy

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sequence import SamplerOutput

import torch.nn.functional as F

import torch
from torch import nn
from transformers import T5Config

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.enc_dec_attention import (
    EncoderAttention,
    DecoderAttention,
    CrossAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    RowParallelLinear,
)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size, )
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. 
        No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # T5 uses a layer_norm which only scales and doesn't shift,
        # which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467
        # thus variance is calculated
        # w/o mean and there is no bias. Additionally we want
        # to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Module):

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_states, _ = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = ColumnParallelLinear(config.d_model,
                                         config.d_ff,
                                         bias=False)
        self.wi_1 = ColumnParallelLinear(config.d_model,
                                         config.d_ff,
                                         bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states)[0])
        hidden_linear, _ = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states

def to_block_diagonal_nested(tensors):
    # Base case: if tensors is a list of tensors, create a block diagonal tensor
    if all(isinstance(t, torch.Tensor) for t in tensors):
        row_total_size = sum(t.size(0) for t in tensors)
        col_total_size = sum(t.size(1) for t in tensors)
        block_diagonal = torch.zeros(row_total_size, col_total_size)
        
        row_offset = 0
        col_offset = 0
        for t in tensors:
            n_rows = t.size(0)
            n_cols = t.size(1)
            block_diagonal[row_offset:row_offset+n_rows, col_offset:col_offset+n_cols] = t
            row_offset += n_rows
            col_offset += n_cols
            
        return block_diagonal
    # Recursive case: if tensors is a nested list, apply function to each sublist
    else:
        return [to_block_diagonal_nested(sublist) for sublist in tensors]

class T5Attention(nn.Module):

    def __init__(
        self,
        config: T5Config,
        is_cross: bool,
        has_relative_attention_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.relative_attention_num_buckets = \
            config.relative_attention_num_buckets
        self.relative_attention_max_distance = \
            config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        total_num_heads = config.num_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size(
        )
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.n_heads = total_num_heads // tensor_model_parallel_world_size
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.k = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.v = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)

        self.is_cross = is_cross
        if self.is_decoder:
            if self.is_cross:
                self.attn = CrossAttention(self.n_heads,
                                           self.key_value_proj_dim, 1)
            else:
                self.attn = DecoderAttention(self.n_heads,
                                             self.key_value_proj_dim, 1)
        else:
            self.attn = EncoderAttention(self.n_heads, self.key_value_proj_dim,
                                         1)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative 
        attention. The relative position is defined as
        memory_position - query_position, i.e. the distance 
        in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative 
        positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for 
        larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All 
        relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to 
        longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, 
             containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(
                torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position,
                                           torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically
        # bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_position,
                                        relative_position_if_large)
        return relative_buckets

    '''
    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length,
                                        dtype=torch.long,
                                        device="cuda")[:, None]
        memory_position = torch.arange(key_length,
                                       dtype=torch.long,
                                       device="cuda")[None, :]
        relative_position = (memory_position - context_position
                             )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # shape (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1])
        return values
    '''

    def compute_bias(self, query_lens, key_lens, dtype, device):
        biases = [[] for _ in range(self.n_heads)]
        
        for query_length, key_length in zip(query_lens,key_lens):
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position
            
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            
            values = self.relative_attention_bias(relative_position_bucket)
            values = values.permute(2, 0, 1)  # Rearrange to (num_heads, seq_len, seq_len)
            
            for head in range(self.n_heads):
                biases[head].append(values[head][:,:key_length])
        
        biases = to_block_diagonal_nested(biases) # List of per-head block-diagonal relative position encoding matrices
        biases = torch.stack(biases).unsqueeze(0).to(dtype).to(device).contiguous() # 1 x (# heads) x (num_tokens) x (num_tokens)

        # xFormers attn kernel (possibly flash_attn too?) requires stride(-2) to be divisible by 8; force this
        num_k_tokens = biases.shape[-1]
        padded_num_k_tokens = (num_k_tokens + 7) // 8 * 8
        if padded_num_k_tokens-num_k_tokens > 0:
            # Enforce right-most attention bias stride is a multiple of 8
            padding = (0,padded_num_k_tokens-num_k_tokens,0,0,0,0,0,0,)
            biases = F.pad(biases, padding, "constant", 0)
            biases = biases[:,:,:,:num_k_tokens]

        return [biases] # vLLM Attention wrapper expects biases as a list
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q, _ = self.q(hidden_states)

        prompt_lens = input_metadata.prompt_lens
        context_lens = [max(context_len.item(),1) for context_len in input_metadata.context_lens]

        key_cache = None
        value_cache = None
        if kv_cache is not None:
            key_cache, value_cache = kv_cache

        if not self.is_decoder:
            assert kv_cache is None
            # Encoder self attention, no cache operations

            k, _ = self.k(hidden_states)
            v, _ = self.v(hidden_states)
            print("*- v computed:",v.sum())
            #print("*- hash hidden:",hash_tensor_floating_point(hidden_states, decimals=5))

            print("*- v xform:",self.v.linear_weights["weight"].sum(),self.v.linear_weights["weight"].shape)
            print("*- hidden_states.shape:",hidden_states.shape)
            print("*- k computed:",k.sum())


            if input_metadata.attn_bias is None:
                input_metadata.attn_bias = self.compute_bias(
                    prompt_lens, prompt_lens, dtype=q.dtype, device=q.device)

            attn_output = self.attn(q, k, v, input_metadata)
        elif not self.is_cross:
            # Decoder self attention
            k, _ = self.k(hidden_states)
            v, _ = self.v(hidden_states)

            if input_metadata.attn_bias is None:
                # Paged attention does not expect a list of attention biases
                input_metadata.attn_bias = self.compute_bias(
                    context_lens, context_lens, dtype=q.dtype, device=q.device)

            attn_output = self.attn(q, k, v, key_cache, value_cache,
                                    input_metadata)

        else:
            # Cross attention
            if input_metadata.attn_bias is None:
                input_metadata.attn_bias = "not_causal"

            if input_metadata.is_prompt:
                assert encoder_hidden_states is not None
                k, _ = self.k(encoder_hidden_states)
                v, _ = self.v(encoder_hidden_states)
                attn_output = self.attn(q, k, v, key_cache, value_cache,
                                        input_metadata)
            else:
                attn_output = self.attn(q, None, None, key_cache, value_cache,
                                        input_metadata)

        attn_output, _ = self.o(attn_output)
        return attn_output


class T5LayerSelfAttention(nn.Module):

    def __init__(
        self,
        config,
        has_relative_attention_bias,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            is_cross=False,
            has_relative_attention_bias=has_relative_attention_bias,
            linear_method=linear_method,
        )
        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            hidden_states=normed_hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            encoder_hidden_states=None,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5LayerCrossAttention(nn.Module):

    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config,
            is_cross=True,
            has_relative_attention_bias=False,
            linear_method=linear_method,
        )
        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states=normed_hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5Block(nn.Module):

    def __init__(
        self,
        config,
        has_relative_attention_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                linear_method=linear_method,
            ))
        if self.is_decoder:
            self.layer.append(
                T5LayerCrossAttention(config, linear_method=linear_method))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata_dict: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        self_input_metadata = None
        cross_input_metadata = None
        if self.is_decoder:
            self_input_metadata: InputMetadata = input_metadata_dict["self"]
            cross_input_metadata: InputMetadata = input_metadata_dict["cross"]
        else:
            self_input_metadata = input_metadata_dict 

        print("- Encoder block in:",hidden_states.sum())

        hidden_states = self.layer[0](
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=self_input_metadata,
        )

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)

        print("- Encoder block out:",hidden_states.sum())

        if self.is_decoder:
            hidden_states = self.layer[1](
                hidden_states,
                kv_cache=kv_cache,
                input_metadata=cross_input_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states,
                                            min=-clamp_value,
                                            max=clamp_value)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        return hidden_states


class T5Stack(nn.Module):

    def __init__(
        self,
        config: T5Config,
        embed_tokens: torch.Tensor,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.embed_tokens = embed_tokens

        self.block = nn.ModuleList([
            T5Block(
                config,
                has_relative_attention_bias=(i == 0),
                linear_method=linear_method,
            ) for i in range(config.num_layers)
        ])

        self.final_layer_norm = T5LayerNorm(config.d_model,
                                            eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata_dict: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        print("input_ids:",sum(input_ids))
        hidden_states = self.embed_tokens(input_ids)
        print(hidden_states.shape)
        print("Embed tokens:",hidden_states)
        print("Embed tokens:",hidden_states.sum())

        for i, layer_module in enumerate(self.block):
            kv_cache = kv_caches[i] if self.is_decoder else None

            layer_outputs = layer_module(
                hidden_states,
                kv_cache=kv_cache,
                input_metadata_dict=input_metadata_dict,
                encoder_hidden_states=encoder_hidden_states,
            )

            hidden_states = layer_outputs

            print(i)
            print(self.is_decoder)
            print(hidden_states.sum())

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states

def batch_input_ids(input_ids:torch.Tensor, input_metadata:InputMetadata):
    # Initialize an empty list to hold the batched input_ids
    device=input_ids.device
    input_ids=input_ids.tolist()
    batch_input_ids = []

    # Starting index for slicing input_ids
    start_idx = 0
    for length in input_metadata.prompt_lens:
        # Extract the prompt's input_ids
        prompt_input_ids = input_ids[start_idx:start_idx + length]
        
        # Pad the prompt_input_ids to max_prompt_len
        padded_prompt_input_ids = prompt_input_ids + [0] * (max(input_metadata.prompt_lens) - length)
        
        # Add the padded_prompt_input_ids to batch_input_ids
        batch_input_ids.append(padded_prompt_input_ids)
        
        # Update the start_idx for the next prompt
        start_idx += length

    return torch.Tensor(batch_input_ids).to(device).long()

def unbatch_input_ids(batched_input_ids: torch.Tensor, input_metadata: InputMetadata):
    # List to hold the unpacked, variable-length sequences
    packed_sequences = []

    # Iterate over each sequence in the batch
    for i, length in enumerate(input_metadata.prompt_lens):
        # Extract the sequence for the current batch item and its true length (remove padding)
        true_sequence = batched_input_ids[i, :length, :]
        packed_sequences.append(true_sequence)
    
    # Concatenate all the true sequences into a single packed tensor
    packed_tensor = torch.cat(packed_sequences, dim=0)

    return packed_tensor

class T5ForConditionalGeneration(nn.Module):

    def __init__(self,
                 config: T5Config,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, linear_method)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared, linear_method)

        self.unpadded_vocab_size = config.vocab_size
        #if lora_config:
        #    self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        #self.sampler = Sampler(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        assert(input_metadata.cross_input_metadata is not None)
        assert((not input_metadata.is_prompt) or "encoder" in input_metadata.cross_input_metadata)
        assert("decoder" in input_metadata.cross_input_metadata)

        # Extract input metadata for encoder self-attention and decoder
        # self-/cross-attention
        is_prompt=input_metadata.is_prompt

        cross_decoder_input_metadata: InputMetadata = input_metadata.cross_input_metadata['decoder']
        self_decoder_input_metadata: InputMetadata = input_metadata
        
        decoder_input_ids = input_ids
        # decoder_input_ids = batch_input_ids(input_ids, self_decoder_input_metadata)

        if is_prompt:
            # prompt run, need to run encoder once
            self_encoder_input_metadata: InputMetadata = input_metadata.cross_input_metadata['encoder']
            encoder_input_ids = input_metadata.cross_input_metadata["encoder_input_tokens"]   
            #encoder_input_ids = batch_input_ids(encoder_input_ids, self_encoder_input_metadata)
            print("encoder_input_ids:",encoder_input_ids)
            print("encoder_input_ids:",encoder_input_ids.shape)
            print("decoder_input_ids:",decoder_input_ids.shape)
  
            hidden_states: torch.Tensor = self.encoder(encoder_input_ids, kv_caches, self_encoder_input_metadata,
                                         None)
        else:
            hidden_states = None

        print("Pre-decoder hidden states:",None if hidden_states is None else hidden_states.sum())
        if kv_caches[0][0] is not None:  # Skip decoder for profiling run
            hidden_states = self.decoder(decoder_input_ids, kv_caches, 
                                         {"self":self_decoder_input_metadata,
                                          "cross":cross_decoder_input_metadata},
                                         hidden_states)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim**-0.5)

        #hidden_states = unbatch_input_ids(hidden_states, input_metadata)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.shared.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    # def sample(self, hidden_states: torch.Tensor,
    #            sampling_metadata: SamplingMetadata):
    #     next_tokens = self.sampler(self.shared.weight, hidden_states,
    #                                sampling_metadata)
    #     return next_tokens

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "EncDecAttention.relative_attention_bias" in name:
                continue

            assert name in params_dict, f"{name} not in params_dict"
            param = params_dict[name]
            assert param.shape == loaded_weight.shape, (
                f"{name} shape mismatch between model and checkpoint: "
                f"{param.shape} != {loaded_weight.shape}")
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
