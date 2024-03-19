from dataclasses import dataclass, fields
from typing import Optional, Any, Dict

import torch


@dataclass
class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Args:
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
        kv_cache_dtype: Data type to store kv cache.
    """

    is_prompt: bool
    slot_mapping: torch.Tensor
    prompt_lens: Optional[torch.Tensor]
    max_seq_len: Optional[int]
    start_loc: Optional[torch.Tensor]
    max_context_len: Optional[int]
    context_lens: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    use_cuda_graph: bool
    kv_cache_dtype: str

    def __post_init__(self):
        # will not appear in the __repr__ and __init__
        self.attn_bias = None

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"is_prompt={self.is_prompt}, "
                f"slot_mapping={self.slot_mapping}, "
                f"prompt_lens={self.prompt_lens}, "
                f"max_seq_len={self.max_seq_len}, "
                f"start_loc={self.start_loc}, "
                f"max_context_len={self.max_context_len}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables}, "
                f"use_cuda_graph={self.use_cuda_graph}, "
                f"kv_cache_dtype={self.kv_cache_dtype})")
    
    def asdict_zerocopy(self) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
