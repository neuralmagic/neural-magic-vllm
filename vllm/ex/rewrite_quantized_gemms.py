import torch

from .utils import ModuleInputGenerator, graph_print_tabular, is_call, call_method_class

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.subgraph_rewriter import replace_pattern
from torch.fx import symbolic_trace, subgraph_rewriter
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor


from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

from vllm.logger import init_logger
from vllm import _custom_ops as custom_ops

import traceback

logger = init_logger(__name__)

###############################################################################
#
# Rewrite quantized gemms
#
###############################################################################

def pattern(x,  weight, weight_scale):
    f_x = x.to(torch.float32)
    f_w = weight.to(torch.float32) * weight_scale
    f_out = torch.nn.functional.linear(f_x, f_w.transpose(1, 0))
    return f_out.to(x.dtype)

def pattern2(f_x, f_w):
    f_w_t = f_w.transpose(1, 0)
    f_out = torch.nn.functional.linear(f_x, f_w_t)
    return f_out.to(f_x.dtype)


def pattern3(x, w, w_scale, xtype):
    f_x = x.to(torch.float32)
    f_w = w.to(torch.float32) * w_scale
    f_out = torch.nn.functional.linear(f_x, f_w.transpose(1, 0))
    return f_out.to(xtype)


def replacement(self, x, act_scale, w, w_scale, xtype):
    x_q = self._quantize_single(x, act_scale[0].item())
    return custom_ops.cutlass_scaled_mm_dq(x_q, w, act_scale, w_scale, xtype)


def rewrite_quantized_gemms(
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor]
) -> torch.fx.GraphModule:
    pattern_graph = symbolic_trace(pattern3).graph

#    replace_graph = symbolic_trace(replacement, {'xtype': torch.int8}).graph  # provide sample inputs?
#    rep_matches = replace_pattern(mod, pattern_graph, replacement)
#    print(f"root MATCHES {rep_matches}")

    matcher = SubgraphMatcher(pattern_graph)
    matches = matcher.match(mod.graph)
    print(f"MATCHES {matches}")

#    for name, subm in mod.named_modules():
#        matches = matcher.match(subm.graph)
#        print(f"sub {name} MATCHES {matches}")

    return mod
