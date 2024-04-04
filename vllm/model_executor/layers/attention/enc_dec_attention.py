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

def pre_attn_reshape(query,key,value):
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    query = query.reshape((query.shape[0]*query.shape[1],-1))
    if key is not None:
        key = key.reshape((key.shape[0]*key.shape[1],-1))
    if value is not None:
        value = value.reshape((value.shape[0]*value.shape[1],-1))

    return query, key, value, batch_size, seq_len

def post_attn_reshape(out,batch_size,seq_len):
    return out.reshape((batch_size,seq_len,-1))



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
        #query,key,value,batch_size,seq_len=pre_attn_reshape(query,key,value)
        # query: [batch_size, seq_len, num_heads * head_size]
        # key: [batch_size, seq_len, num_heads * head_size]
        # value: [batch_size, seq_len, num_heads * head_size]
        # custom_bias: [batch_size, seq_len, seq_len]
        # output: [batch_size, seq_len, num_heads * head_size]
        assert input_metadata.is_prompt

        out: torch.Tensor = self.attn(
                query,
                key,
                value,
                None,
                None,
                input_metadata, # Should have nonzero attention bias
            )

        #out = post_attn_reshape(out,batch_size,seq_len)
        
        print("-- Inner out:",out.sum())
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

        #query,key,value,batch_size,seq_len=pre_attn_reshape(query,key,value)

        #batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        #query = query.view(-1, self.num_heads, self.head_size)
        #key = key.view(-1, self.num_heads, self.head_size)
        #value = value.view(-1, self.num_heads, self.head_size)
        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.

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
        #output = post_attn_reshape(output,batch_size,seq_len)

        return output


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

        #query,key,value,batch_size,seq_len=pre_attn_reshape(query,key,value)

        #batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        # query = query.view(-1, self.num_heads, self.head_size)
        # if key is not None:
        #     key = key.view(-1, self.num_heads, self.head_size)
        # if value is not None:
        #     value = value.view(-1, self.num_heads, self.head_size)

        # Cross-attention decode run.
        output = self.attn(
                query,
                key,
                value,
                key_cache,
                value_cache,
                input_metadata,
            )

        #output = post_attn_reshape(output,batch_size,seq_len)
            
        return output
