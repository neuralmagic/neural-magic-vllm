import copy
import torch
import unittest.mock

from .code_cache import CodeCache
from .fusion import FusedOpGenerator, pointwise_fusion
from .register import SUPPORTED
from .utils import extract_node_tensor_meta, extract_node_type, ModuleInputGenerator

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

from vllm.logger import init_logger

import traceback

logger = init_logger(__name__)

###############################################################################
#
# Partitioning
#
###############################################################################

def is_node_supported(
    submodules: Mapping[str, torch.nn.Module],
    node: torch.fx.Node,
) -> bool:
    if node.op == 'call_function':
        return get_node_target(submodules, node) in SUPPORTED
    else:
        return False


def partition_graph(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor]
) -> Tuple[torch.fx.GraphModule, List[Partition]]:
    """
    Partition 'gm' into submodules based on the 'is_node_supported' callback.
    Modules containing "supported" nodes will be optimized by the backend.
    """
    support = create_op_support(is_node_supported)

    #
    # hacking
    #
    holders = dict()
    for n in gm.graph.nodes:
        if (
            n.op == "placeholder" and
            (val := n.meta.get("example_value")) is not None and
            isinstance(val, torch.SymInt)
        ):
            holders[str(val)] = n
    #
    # end hacking
    #

    p = CapabilityBasedPartitioner(
        gm,
        support,
        allows_single_node_partition=False,
        non_compute_ops=None,
        allowed_single_node_partition_ops=None
    )
    parts = p.propose_partitions()

    #
    # hacking
    #
    def ff(n):
        return f"{n.format_node()}"# "{n.meta}"

    for i, pp in enumerate(parts):
        syms = []
        newline = "  \n"
        logger.info(f"PART{i}: {newline.join([ff(n) for n in pp.nodes])}")
        for n in pp.nodes:
            if n.meta.get('example_value') is not None:
                val = n.meta['example_value']
                logger.info(f"example_value {val} {type(val)}")
                if isinstance(val, FakeTensor):
                    logger.info(f"FAKE_TENSOR {val.size()}{any([isinstance(d, torch.SymInt) for d in val.size()])}")
                    for d in val.size():
                        if isinstance(d, torch.SymInt) and not d in syms:
                            syms.append(d)

        logger.info(f"SYMS: {syms}")
        for s in syms:
            continue
            #pp.nodes.add(holders[str(s)])
    #
    # end hacking
    #

    return p.fuse_partitions(parts), parts


###############################################################################
#
# Inline
#
###############################################################################

def inline_submodules(mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    #mod.graph = torch.fx.symbolic_trace(mod)
    mod.graph = torch.fx.Tracer().trace(mod)
    return mod


###############################################################################
#
# Backend
#
###############################################################################

# torch._inductor.fx_passes.joint_graph.joint_graph_passes
def optimize(
    cc: CodeCache,
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> torch.fx.GraphModule:
    mod = pointwise_fusion(cc, fgen, mod, example_inputs)
    # TODO: should we re-trace here to inline?  or will inductor handle it?
    # mod = inline_submodules(mod)
    return mod


# names should be unique, so this is ok
def node_in_module(n: torch.fx.Node, m: torch.fx.GraphModule) -> bool:
    return n.name in [nn.name for nn in m.graph.nodes]


def module_in_partitions(parts: List[Partition], m: torch.fx.GraphModule) -> bool:
    for p in parts:
        if node_in_module(next(iter(p.nodes)), m):
            return True
    return False


def backend_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    backend: str ='inductor'
) -> Callable:
    try:
        backend = lookup_backend(backend)
        logger.info(f"attempting {backend}")
        backend_compiled = backend(gm, example_inputs)
        if backend_compiled is not None:
            logger.info(f"{backend} COMPILED!")
            return backend_compiled
    except Exception as ex:
        logger.info(f"EX '{ex}'")
        tb = ex.__traceback__
        print(f"EX TRACE")
        traceback.print_tb(tb)
        pass

    return gm.forward


# See https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50
# maybe useful https://github.com/huggingface/optimum/blob/main/optimum/fx/optimization/transformations.py
# TODO: see if transforms can work here
#gm = AnnotateTypesWithSchema(gm).transform()
# TODO: see schema_type_annotation.py/AnnotateTypesWithSchema

class backend_class:
    def __init__(self, final='inductor'):
        self.final = final

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
        # Must make a copy so that inductor backend doesn't choke.
        gm = copy.copy(gm)

        logger.info(f"ORIGINAL {gm.graph}")

        # hack to get around https://github.com/pytorch/pytorch/issues/108446
        # probably not a good long term solution.
        logger.info(f"inputs: {[type(inp) for inp in example_inputs]}")
        for node in gm.graph.nodes:
            if node.op == 'placeholder' and 'example_value' in node.meta:
                val = node.meta['example_value']
                if isinstance(val, FakeTensor) and any([isinstance(d, torch.SymInt) for d in val.size()]):
                    logger.info(f"FAKE_TENSOR {val.size()}{any([isinstance(d, torch.SymInt) for d in val.size()])}")
                    return gm

        part_gm, parts = partition_graph(gm, example_inputs)

        logger.info(f"BEFORE forward: {part_gm.forward}")

        logger.info(f"part_gm: {part_gm.print_readable(False)}")
        logger.info(f"parts: {parts}")
        newline = "\n"
        logger.info(f"children: {newline.join([f'{cname}: {cm.print_readable(False)}' for cname, cm in part_gm.named_children()])}")
        logger.info(f"end children")

        # get the current FakeTensorMode (there should be one since we are in a backend)
        fake_mode = torch._guards.detect_fake_mode()

        # There should be an existing fake_mode but double check
        if not fake_mode:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        # Is this ok?  probably should save/restore at least
        #fake_mode.allow_non_fake_inputs = True
        with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
            #example_inputs = [fake_mode.from_tensor(input) for input in example_inputs]

            logger.info(f"fake mode = {fake_mode}")

            # use FakeTensorProp-like class to get example inputs for submodules
            # static_shapes can be applied here
            mig = ModuleInputGenerator(part_gm, fake_mode)
            mig.propagate(*example_inputs)

            logger.info(f"mod args = {mig.module_args}")

            # TODO: store this in the root module state dictionary so that code for
            # all sub-modules is shared?
            cc = CodeCache()
            fgen = FusedOpGenerator()

            mods_to_compile = []

            for name, m in part_gm.named_modules():
                if module_in_partitions(parts, m):
                    assert name in mig.module_args
                    module_inputs = mig.module_args[name][0]

                    # TODO: make this smarter
                    if not module_inputs:
                        logger.info(f"SKIPPING {name} FOR NOW (multiple callers): {m.print_readable(False)}")
                        continue

                    logger.info(f"OPTIMIZE! {name}: {m.print_readable(False)}")
                    m = optimize(cc, fgen, m, module_inputs)
                    setattr(part_gm, name, m)

                    logger.info(f"POST OPTIMIZE! {name}: {m.print_readable(False)}")

                    # TODO: don't really need to recompile if nothing happened.
                    m.recompile()

                    logger.info(f"mod inputs {module_inputs}")
                    #logger.info(f"fake mode={torch._guards.detect_fake_mode(module_inputs)}")
                    if self.final != None:
                        m.forward = backend_compile(m, module_inputs, backend=self.final)

        #part_gm.recompile()
        #part_gm = inline_submodules(part_gm)

        part_gm.recompile()

        logger.info(f"FULL FINAL GRAPH: {part_gm.print_readable(False)}")
        # Add option for backend for this graph?
        #return backend_compile(part_gm, example_inputs)
        return part_gm.forward


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    return backend_class()(gm, example_inputs)


def make_backend(final: str = 'inductor') -> backend_class:
    return backend_class(final)

