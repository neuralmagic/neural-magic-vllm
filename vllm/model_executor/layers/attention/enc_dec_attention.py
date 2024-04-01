"""Multi-head attention for encoder-decoder models."""
from typing import List, Optional

import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip
from vllm.model_executor.layers.attention.attention import Attention

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]

class EncDecAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")


class EncoderAttention(EncDecAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads, head_size, scale)
        self.attn: Attention = Attention(num_heads, head_size, scale, 
                                         num_heads if num_kv_heads is None else num_kv_heads, 
                                         alibi_slopes, sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Encoder attention forward pass.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            custom_bias: Custom bias tensor.

        Returns:
            Output tensor.
        """
        # query: [batch_size, seq_len, num_heads * head_size]
        # key: [batch_size, seq_len, num_heads * head_size]
        # value: [batch_size, seq_len, num_heads * head_size]
        # custom_bias: [batch_size, seq_len, seq_len]
        # output: [batch_size, seq_len, num_heads * head_size]
        assert input_metadata.is_prompt

        # Reshape the query, key, and value tensors.
        #batch_size = len(input_metadata.prompt_lens)
        #seq_len = query.shape[0]//batch_size
        #query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        #key = key.view(batch_size, seq_len, self.num_heads, self.head_size)
        #value = value.view(batch_size, seq_len, self.num_heads, self.head_size)
        # if input_metadata.attn_bias is None:
        #     input_metadata.attn_bias = BlockDiagonalCausalMask.from_seqlens(
        #         [seq_len] * batch_size)

        # Attention wrapper wants a list of biases
        #input_metadata.attn_bias = [input_metadata.attn_bias] # input_metadata.attn_bias[:, :, :, :seq_len]

        # Normal attention
        # out = xops.memory_efficient_attention_forward(
        #     query,
        #     key,
        #     value,
        #     attn_bias=input_metadata.attn_bias,
        #     p=0.0,
        #     scale=self.scale,
        #     op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
        #     (is_hip()) else None,
        # )

        print("-- Inner query in:",query.sum())
        print("-- Inner key in:",key.sum())
        print("-- Inner value in:",value.sum())

        out: torch.Tensor = self.attn(
                query,
                key,
                value,
                None,
                None,
                input_metadata, # Should have nonzero attention bias
            )

        print("-- Inner out:",out.sum())

        #output = out.view(batch_size, seq_len, hidden_size)
        return out


class DecoderAttention(EncDecAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads, head_size, scale)
        self.attn: Attention = Attention(num_heads, head_size, scale, 
                                         num_heads if num_kv_heads is None else num_kv_heads, 
                                         alibi_slopes, sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ):
        """Decoder attention forward pass.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_cache: Key cache tensor.
            value_cache: Value cache tensor.
            custom_bias: Custom bias tensor.

        Returns:
            Output tensor.
        """

        #batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        #query = query.view(-1, self.num_heads, self.head_size)
        #key = key.view(-1, self.num_heads, self.head_size)
        #value = value.view(-1, self.num_heads, self.head_size)
        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        #if key_cache is not None and value_cache is not None:
        #
        #    PagedAttentionImpl.reshape_and_cache(key, value, key_cache,
        #                                         value_cache, input_metadata)

        # max_prompt_len = max(input_metadata.prompt_lens)
        # block_size = value_cache.shape[3]
        # prompt_table_len = (max_prompt_len + block_size - 1) // block_size
        # self_attn_block_tables = input_metadata.block_tables[:,
        #                                                      prompt_table_len:].contiguous(
        #                                                      )

        '''
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        '''

        output = self.attn(
                query,
                key,
                value,
                key_cache,
                value_cache,
                input_metadata,
            )

        '''
        output = PagedAttentionImpl.forward_decode(
            query,
            key_cache,
            value_cache,
            input_metadata,
            self.num_heads,
            self.scale,
            None,)
        '''

        return output
        #return output.view(batch_size, seq_len, hidden_size)


class CrossAttention(EncDecAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads, head_size, scale)
        self.attn: Attention = Attention(num_heads, head_size, scale, 
                                         num_heads if num_kv_heads is None else num_kv_heads, 
                                         alibi_slopes, sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ):
        """Cross attention forward pass.
        Args:
            query: Query tensor.
            key_cache: Key cache tensor.
            value_cache: Value cache tensor.
            input_metadata: Input metadata.
            key: Key tensor. Only needed in the first pass.
            value: Value tensor. Only needed in the first pass.
            custom_bias: Custom bias tensor.
        Returns:
            Output tensor.
        """

        #batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        # query = query.view(-1, self.num_heads, self.head_size)
        # if key is not None:
        #     key = key.view(-1, self.num_heads, self.head_size)
        # if value is not None:
        #     value = value.view(-1, self.num_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # It only happens during the first pass.
        # if (input_metadata.is_prompt and key_cache is not None
        #         and value_cache is not None):
        #     assert key is not None and value is not None
        #     PagedAttentionImpl.reshape_and_cache(key, value, key_cache,
        #                                          value_cache, input_metadata)

        #block_size = value_cache.shape[3]
        #prompt_table_len = (max_prompt_len + block_size - 1) // block_size
        # cross_attn_block_tables = input_metadata.block_tables[:, :
        #                                                       prompt_table_len].contiguous(
        #                                                       )

        # Cross-attention decode run.
        output = self.attn(
                query,
                key,
                value,
                key_cache,
                value_cache,
                input_metadata,
            )

        '''
        output = PagedAttentionImpl.forward_decode(
            query,
            key_cache,
            value_cache,
            input_metadata,
            self.num_heads,
            self.scale,
            None,  # No alibi slopes
            apply_attn_bias=False,
            override_context_lens=input_metadata.prompt_lens.int(),
            override_max_context_len=max_prompt_len,
            override_block_tables=cross_attn_block_tables)
        '''
            
        return output #.view(batch_size, seq_len, hidden_size)
