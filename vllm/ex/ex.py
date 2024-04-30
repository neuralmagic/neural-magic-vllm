import copy
import torch

from .fusion import FusedOpGenerator, pointwise_fusion
from .utils import extract_type, extract_node_tensor_meta, extract_node_type, ModuleInputGenerator

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

import traceback

###############################################################################
#
# Partitioning
#
###############################################################################

# TODO: make this smarter/add registration mechanism
def is_node_supported(submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
    return node.op == 'call_function' and (get_node_target(submodules, node) == '_operator.add' or
                                           get_node_target(submodules, node) == '_operator.mul' or
                                           get_node_target(submodules, node) == '_operator.getitem' or
                                           get_node_target(submodules, node) == 'torch.matmul' or
                                           get_node_target(submodules, node) == 'torch.relu' or
                                           get_node_target(submodules, node) == 'torch.nn.functional.silu' or
                                           get_node_target(submodules, node) == 'torch._C._nn.linear' or
                                           get_node_target(submodules, node) == 'torch.ops.vllm.silu_and_mul')


# See: torch.fx.passes.infra.partitioner
def partition_graph(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Tuple[torch.fx.GraphModule, List[Partition]]:
    support = create_op_support(is_node_supported)

    holders = dict()
    for n in gm.graph.nodes:
        if (
            n.op == "placeholder" and
            (val := n.meta.get("example_value")) is not None and
            isinstance(val, torch.SymInt)
        ):
            holders[str(val)] = n

    p = CapabilityBasedPartitioner(
        gm,
        support,
        allows_single_node_partition=False, #True
        non_compute_ops=None,
        allowed_single_node_partition_ops=None
    )
    parts = p.propose_partitions()

    def ff(n):
        return f"{n.format_node()}"# "{n.meta}"

    for i, pp in enumerate(parts):
        syms = []
        newline = "  \n"
        print(f"PART{i}: {newline.join([ff(n) for n in pp.nodes])}")
        for n in pp.nodes:
            if n.meta.get('example_value') is not None:
                val = n.meta['example_value']
                print(f"example_value {val} {type(val)}")
                if isinstance(val, FakeTensor):
                    print(f"FAKE_TENSOR {val.size()}{any([isinstance(d, torch.SymInt) for d in val.size()])}")
                    for d in val.size():
                        if isinstance(d, torch.SymInt) and not d in syms:
                            syms.append(d)

        print(f"SYMS: {syms}")
        for s in syms:
            continue
            #pp.nodes.add(holders[str(s)])

    return p.fuse_partitions(parts), parts


###############################################################################
#
# Quantization
#
###############################################################################

# torch._inductor.fx_passes.quantization
def add_quantization(mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # TODO fill this in later
    return mod



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
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> torch.fx.GraphModule:
    mod = add_quantization(mod)
    mod = pointwise_fusion(fgen, mod, example_inputs)
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
        print(f"attempting {backend}")
        backend_compiled = backend(gm, example_inputs)
        if backend_compiled is not None:
            print(f"{backend} COMPILED!")
            return backend_compiled
    except Exception as ex:
        print(f"EX '{ex}'")
        tb = ex.__traceback__
        print(f"EX TRACE")
        traceback.print_tb(tb)
        pass

    return gm.forward


# why doesn't this work?
def graph_print_tabular(g: torch.fx.Graph):
    try:
        from tabulate import tabulate
    except ImportError:
        print("`print_tabular` relies on the library `tabulate`, "
              "which could not be found on this machine. Run `pip "
              "install tabulate` to install the library.")
        raise

    node_specs = [[n.op, n.name, n.target, n.args, n.kwargs, n.meta]
                  for n in g.nodes]
    print(tabulate(node_specs,
                headers=['opcode', 'name', 'target', 'args', 'kwargs', 'meta']))


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

        print(f"ORIGINAL {gm.graph}")

        # hack to get around https://github.com/pytorch/pytorch/issues/108446
        # probably not a good long term solution.
        print(f"inputs: {[type(inp) for inp in example_inputs]}")
        for node in gm.graph.nodes:
            if node.op == 'placeholder' and 'example_value' in node.meta:
                val = node.meta['example_value']
                if isinstance(val, FakeTensor) and any([isinstance(d, torch.SymInt) for d in val.size()]):
                    print(f"FAKE_TENSOR {val.size()}{any([isinstance(d, torch.SymInt) for d in val.size()])}")
                    return gm

        part_gm, parts = partition_graph(gm, example_inputs)

        print(f"BEFORE forward: {part_gm.forward}")

        print(f"part_gm: {part_gm.print_readable(False)}")
        print(f"parts: {parts}")
        newline = "\n"
        print(f"children: {newline.join([f'{cname}: {cm.print_readable(False)}' for cname, cm in part_gm.named_children()])}")
        print(f"end children")

        # get the current FakeTensorMode (there should be one since we are in a backend)
        fake_mode = torch._guards.detect_fake_mode()

        # Is this ok?  probably should save/restore at least
        fake_mode.allow_non_fake_inputs = True

        #example_inputs = [fake_mode.from_tensor(input) for input in example_inputs]

        # There should be an existing fake_mode but double check
        if not fake_mode:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        print(f"fake mode = {fake_mode}")

        # use FakeTensorProp-like class to get example inputs for submodules
        # static_shapes can be applied here
        mig = ModuleInputGenerator(part_gm, fake_mode)
        mig.propagate(*example_inputs)

        print(f"mod args = {mig.module_args}")

        # TODO: store this in the root module state dictionary so that code for
        # all sub-modules is shared?
        fgen = FusedOpGenerator()

        mods_to_compile = []

        for name, m in part_gm.named_modules():
            if module_in_partitions(parts, m):
                assert name in mig.module_args
                module_inputs = mig.module_args[name][0]

                # TODO: make this smarter
                if not module_inputs:
                    print(f"SKIPPING {name} FOR NOW (multiple callers): {m.print_readable(False)}")
                    continue

                print(f"OPTIMIZE! {name}: {m.print_readable(False)}")
                m = optimize(fgen, m, module_inputs)
                setattr(part_gm, name, m)

                print(f"POST OPTIMIZE! {name}: {m.print_readable(False)}")

                # TODO: don't really need to recompile if nothing happened.
                m.recompile()

                print(f"mod inputs {module_inputs}")
                #print(f"fake mode={torch._guards.detect_fake_mode(module_inputs)}")
                if self.final != None:
                    m.forward = backend_compile(m, module_inputs, backend=self.final)

        #part_gm.recompile()
        #part_gm = inline_submodules(part_gm)

        part_gm.recompile()

        print(f"FULL FINAL GRAPH: {part_gm.print_readable(False)}")
        # Add option for backend for this graph?
        #return backend_compile(part_gm, example_inputs)
        return part_gm.forward


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    return backend_class()(gm, example_inputs)


def make_backend(final: str = 'inductor') -> backend_class:
    return backend_class(final)

