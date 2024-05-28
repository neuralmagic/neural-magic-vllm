"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "xformers"

    @staticmethod
    def get_impl_cls() -> Type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "XFormersMetadata":
        return XFormersMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # Self-attention prefill/decode metadata cache
    _self_cached_prefill_metadata: Optional["XFormersMetadata"] = None
    _self_cached_decode_metadata: Optional["XFormersMetadata"] = None
    # Cross-attention prefill/decode metadata cache
    _cross_cached_prefill_metadata: Optional["XFormersMetadata"] = None
    _cross_cached_decode_metadata: Optional["XFormersMetadata"] = None

    # Begin cross-attention fields...

    # If True, prefill_metadata() and decode_metadata() will return
    # seqlen & memory-mapping data structures for cross-attention;
    # otherwise, self-attention data structures will be returned.
    is_cross_attn: bool = False

    # (batch_size,). The "cross-sequence-length" per sequence,i.e. the key/value
    # sequence length (usually encoder sequence length) in the cross-attention
    # computation. None if this is self-attention
    cross_seq_lens: Optional[List[int]] = None
    cross_seq_lens_tensor: Optional[torch.Tensor] = None

    # The maximum cross-sequence-length, if cross_seq_lens is specified.
    # Note that for cross-attention there is no difference in key/value
    # sequence length between prefill and decode
    max_cross_seq_len: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None

    @property
    def has_valid_cross_attn_metadata(self):
        # No cross-attention metadata is present whatsoever
        no_md = (self.cross_seq_lens is
                 None) and (self.cross_slot_mapping is
                            None) and (self.cross_block_tables is None)
        # If any cross-attention metadata is present, it is invalid
        invalid_md_if_not_no_md = (self.cross_seq_lens is None) or (
            self.cross_slot_mapping is None) or (self.cross_block_tables is
                                                 None)

        if no_md:
            return False

        assert (
            not invalid_md_if_not_no_md), "Invalid cross-attention metadata"

        return True

    @property
    def do_cross_attn(self):
        return self.is_cross_attn

    @do_cross_attn.setter
    def do_cross_attn(self, state: bool):

        if state:
            assert self.has_valid_cross_attn_metadata, \
            "Must have self.cross_seq_lens not None " + \
            "in order to enable cross-attention"

            # Infer implicit cross-attention fields
            # from user-provided fields, if needed
            if self.cross_seq_lens_tensor is None:
                assert self.seq_lens_tensor is not None
                self.cross_seq_lens_tensor = torch.tensor(
                    self.cross_seq_lens,
                    dtype=self.seq_lens_tensor.dtype,
                    device=self.seq_lens_tensor.device)
            if self.max_cross_seq_len is None:
                assert self.cross_seq_lens is not None
                self.max_cross_seq_len = max(self.cross_seq_lens)

            self.is_cross_attn = True
        else:
            self.is_cross_attn = False

    @property
    def prefill_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_prefills == 0:
            return None

        if not self.do_cross_attn:
            # Self-attention prefill

            if self._self_cached_prefill_metadata is not None:
                return self._self_cached_prefill_metadata

            assert self.seq_lens is not None
            assert self.seq_lens_tensor is not None
            assert self.query_start_loc is not None
            assert self.context_lens_tensor is not None
            assert self.block_tables is not None

            self._self_cached_prefill_metadata = XFormersMetadata(
                num_prefills=self.num_prefills,
                num_prefill_tokens=self.num_prefill_tokens,
                num_decode_tokens=0,
                slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
                seq_lens=self.seq_lens[:self.num_prefills],
                seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
                max_query_len=self.max_query_len,
                max_prefill_seq_len=self.max_prefill_seq_len,
                max_decode_seq_len=0,
                query_start_loc=self.query_start_loc[:self.num_prefills + 1],
                seq_start_loc=None,
                context_lens_tensor=self.context_lens_tensor[:self.
                                                             num_prefills],
                block_tables=self.block_tables[:self.num_prefills],
                use_cuda_graph=False,
                is_cross_attn=False,  # Begin cross-attention fields below...
                cross_seq_lens=None,
                cross_seq_lens_tensor=None,
                max_cross_seq_len=None,
                cross_block_tables=None,
                cross_slot_mapping=None)
            return self._self_cached_prefill_metadata

        else:
            # Cross-attention prefill

            if self._cross_cached_prefill_metadata is not None:
                return self._cross_cached_prefill_metadata

            assert self.seq_lens is not None
            assert self.seq_lens_tensor is not None
            assert self.query_start_loc is not None
            assert self.context_lens_tensor is not None
            assert self.block_tables is not None

            self._cross_cached_prefill_metadata = XFormersMetadata(
                num_prefills=self.num_prefills,
                num_prefill_tokens=self.num_prefill_tokens,
                num_decode_tokens=0,
                slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
                seq_lens=self.seq_lens[:self.num_prefills],
                seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
                max_query_len=self.max_query_len,
                max_prefill_seq_len=self.max_prefill_seq_len,
                max_decode_seq_len=0,
                query_start_loc=self.query_start_loc[:self.num_prefills + 1],
                seq_start_loc=None,
                context_lens_tensor=self.context_lens_tensor[:self.
                                                             num_prefills],
                block_tables=self.block_tables[:self.num_prefills],
                use_cuda_graph=False,
                is_cross_attn=True,  # Begin cross-attention fields below...
                cross_seq_lens=self.cross_seq_lens,
                cross_seq_lens_tensor=self.cross_seq_lens_tensor,
                max_cross_seq_len=self.max_cross_seq_len,
                cross_slot_mapping=self.cross_slot_mapping,
                cross_block_tables=self.cross_block_tables)
            return self._cross_cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if not self.do_cross_attn:
            # Self-attention decode

            if self._self_cached_decode_metadata is not None:
                return self._self_cached_decode_metadata
            assert self.block_tables is not None
            assert self.seq_lens_tensor is not None

            self._self_cached_decode_metadata = XFormersMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=self.num_decode_tokens,
                slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
                seq_lens=None,
                seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
                max_query_len=None,
                max_prefill_seq_len=0,
                max_decode_seq_len=self.max_decode_seq_len,
                query_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=self.block_tables[self.num_prefills:],
                use_cuda_graph=self.use_cuda_graph,
                is_cross_attn=False,  # Begin cross-attention fields below...
                cross_seq_lens=None,
                cross_seq_lens_tensor=None,
                max_cross_seq_len=None,
                cross_block_tables=None,
                cross_slot_mapping=None)
            return self._self_cached_decode_metadata

        else:
            # Cross-attention decode

            if self._cross_cached_decode_metadata is not None:
                return self._cross_cached_decode_metadata
            assert self.block_tables is not None
            assert self.seq_lens_tensor is not None

            self._cross_cached_decode_metadata = XFormersMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=self.num_decode_tokens,
                slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
                seq_lens=None,
                seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
                max_query_len=None,
                max_prefill_seq_len=0,
                max_decode_seq_len=self.max_decode_seq_len,
                query_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=self.block_tables[self.num_prefills:],
                use_cuda_graph=self.use_cuda_graph,
                is_cross_attn=True,  # Begin cross-attention fields below...
                cross_seq_lens=self.cross_seq_lens,
                cross_seq_lens_tensor=self.cross_seq_lens_tensor,
                max_cross_seq_len=self.max_cross_seq_len,
                cross_slot_mapping=self.cross_slot_mapping,
                cross_block_tables=self.cross_block_tables)
            return self._cross_cached_decode_metadata


class XFormersImpl(AttentionImpl[XFormersMetadata]):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert blocksparse_params is None, ValueError(
            "XFormer does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor],
        attn_metadata: "XFormersMetadata",
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * XFormersImpl.forward() may be invoked for both self- and cross-
          attention layers.
        * For self-attention: query, key and value must be non-None.
        * For cross-attention:
            * Query must be non-None
            * During prefill, key and value must be non-None; key and value
              get cached for use during decode.
            * During decode, key and value may be None, since:
              (1) key and value tensors were cached during prefill, and
              (2) cross-attention key and value tensors do not grow during
                  decode
        
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size)

        # Self-attention vs. cross-attention will impact
        # which KV cache memory-mapping & which
        # seqlen datastructures we utilize
        do_cross_attn = attn_metadata.do_cross_attn

        if (kv_cache is not None):
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):

                if do_cross_attn:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                    value_cache,
                                                    updated_slot_mapping,
                                                    self.kv_cache_dtype,
                                                    kv_scale)

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        assert do_cross_attn or (key.shape[0]
                                 == num_prefill_tokens + num_decode_tokens)
        assert do_cross_attn or (value.shape[0]
                                 == num_prefill_tokens + num_decode_tokens)

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]

        if not do_cross_attn and key is not None and value is not None:
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # normal attention.
                # block tables are empty if the prompt does not have a cached
                # prefix.
                out = self._run_memory_efficient_xformers_forward(
                    query, key, value, prefill_meta)
                assert out.shape == output[:num_prefill_tokens].shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                #
                # TODO(afeldman-nm): support cross-attention
                out = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            if do_cross_attn:
                # Paged attention against cross-attention KV cache
                seq_lens_arg = decode_meta.cross_seq_lens_tensor
                max_seq_len_arg = decode_meta.max_cross_seq_len
                block_tables_arg = decode_meta.cross_block_tables
            else:
                # Paged attention against self-attention KV cache
                seq_lens_arg = decode_meta.seq_lens_tensor
                max_seq_len_arg = decode_meta.max_decode_seq_len
                block_tables_arg = decode_meta.block_tables

            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersMetadata,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        """
        assert attn_metadata.seq_lens is not None
        original_query = query
        if self.num_kv_heads != self.num_heads:
            # GQA/MQA requires the shape [B, M, G, H, K].
            # Note that the output also has the same shape (which is different
            # from a spec from the doc).
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])
        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        if attn_metadata.attn_bias is None:
            if self.alibi_slopes is None:
                if attn_metadata.is_cross_attn:
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens, attn_metadata.cross_seq_lens)
                else:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        attn_metadata.seq_lens)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_metadata.attn_bias = [attn_bias]
            else:
                attn_metadata.attn_bias = _make_alibi_bias(
                    self.alibi_slopes, self.num_kv_heads, query.dtype,
                    attn_metadata.seq_lens)

        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_metadata.attn_bias[0],
                p=0.0,
                scale=self.scale)
            return out.view_as(original_query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        output = torch.empty_like(original_query)
        start = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            end = start + seq_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += seq_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> LowerTriangularMaskWithTensorBias:
    attn_biases = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases
