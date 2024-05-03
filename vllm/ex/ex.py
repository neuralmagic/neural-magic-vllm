import copy
import torch
import unittest.mock

from .code_cache import CodeCache
from .fusion import FusedOpGenerator, pointwise_fusion
from .register import SUPPORTED
from .utils import ModuleInputGenerator

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
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
# Inliner
#
###############################################################################

def inline_submodules(mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # TODO: double check that this will actually inline
    #mod.graph = torch.fx.symbolic_trace(mod)
    mod.graph = torch.fx.Tracer().trace(mod)
    return mod


###############################################################################
#
# Backend
#
###############################################################################

"""
Run optimizer on the given module.
"""
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


"""
Compile a module with the given backend.
"""
def backend_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    backend: str ='inductor'
) -> Callable:
    try:
        backend = lookup_backend(backend)
        logger.info(f"attempting {backend} on {gm.name}")
        backend_compiled = backend(gm, example_inputs)
        if backend_compiled is not None:
            logger.info(f"{backend} compiled {gm.name}.")
            return backend_compiled
    except Exception as ex:
        logger.info(f"backend_compile failed: {ex}")
        logger.info(f"Trace: {traceback.format_tb(ex.__traceback__)}")
        pass

    return gm.forward


def node_in_module(n: torch.fx.Node, m: torch.fx.GraphModule) -> bool:
    #    return n.graph.owning_module == m
    # names should be unique, so this is ok
    return n.name in [nn.name for nn in m.graph.nodes]


def module_in_partitions(parts: List[Partition], m: torch.fx.GraphModule) -> bool:
    for p in parts:
        if node_in_module(next(iter(p.nodes)), m):
            return True
    return False


class backend_class:
    """
    A custom backend for torch.compile.

    This backend works by partitioning the provided module into supported/unsupported
    sub-modules.  The supported submodules are passed to an optimizer and then compiled
    via an optional "final" backend.
    """
    def __init__(self, backend: Optional[str] = 'inductor'):
        self.backend = backend

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
        # Must make a copy so that inductor backend doesn't choke.
        gm = copy.copy(gm)

        logger.info(f"Original module {gm.graph}")
        logger.info(f"input_types: {[type(inp) for inp in example_inputs]}")

        # Temporary hack to get around https://github.com/pytorch/pytorch/issues/108446
        # probably not a good long term solution.
        for node in gm.graph.nodes:
            if node.op == 'placeholder' and 'example_value' in node.meta:
                val = node.meta['example_value']
                if (isinstance(val, FakeTensor) and
                    any([isinstance(d, torch.SymInt) for d in val.size()])):
                    return gm
        #
        # end hack
        #

        # TODO: store these in the root module state dictionary so that code for
        # all sub-modules is shared?  Should CodeCache be a global?
        cc = CodeCache()
        fgen = FusedOpGenerator()

        part_gm, parts = partition_graph(gm, example_inputs)

        logger.info(f"Partitioned module: {part_gm.print_readable(False)}")
        logger.info(f"parts: {parts}")

        # Get the current FakeTensorMode (there should be one since we are in
        # a backend).
        fake_mode = torch._guards.detect_fake_mode()

        # There should be an existing fake_mode but double check.
        if not fake_mode:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        # Ensure that allow_non_fake_inputs is True so that original
        # 'example_inputs' can be used.
        with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
            # Determine example inputs for submodules.
            # Note: static_shapes can be applied here
            mig = ModuleInputGenerator(part_gm, fake_mode)
            mig.propagate(*example_inputs)

            for name, m in part_gm.named_modules():
                if module_in_partitions(parts, m):
                    assert name in mig.module_args
                    module_inputs = mig.module_args[name][0]

                    # If one of the partitioned modules is called in multiple
                    # places, we skip it.  This should not happen though.
                    if not module_inputs:
                        logger.info(f"SKIPPING {name}: multiple callers.")
                        continue

                    logger.info(f"Optimizing {name}.")
                    m = optimize(cc, fgen, m, module_inputs)
                    setattr(part_gm, name, m)

                    logger.info(f"Optimized {name}: {m.print_readable(False)}")

                    # TODO: don't really need to recompile if nothing happened.
                    m.recompile()

                    if self.backend != None:
                        m.forward = backend_compile(m, module_inputs, backend=self.backend)

        part_gm.recompile()

        logger.info(f"Final module: {part_gm.print_readable(False)}")

        # TODO: Add option for backend for the final graph?
        return part_gm.forward


"""
The default custom backend function for use with torch.compile.
"""
def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    return backend_class()(gm, example_inputs)


"""
Construct a custom torch.compile backend with optional 'final' backend for
optimized subgraphs. The default 'final' backend is the inductor. None can
be used instead to leave optimized subgraphs as interpreted.
"""
def make_backend(backend: Optional[str] = 'inductor') -> backend_class:
    return backend_class(backend)
