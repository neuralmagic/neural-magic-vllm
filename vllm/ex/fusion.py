###############################################################################
#
# Operator fusion pass
#
###############################################################################

import torch

from .code_cache import CodeCache
from .fused_op_generator import FusedOpGenerator, FusionFail
from .register import FUSABLE
from .utils import extract_node_type, ModuleInputGenerator, FlowGraph, node_function_target, print_tabular_to_string

from torch.fx.passes.split_module import split_module
from torch.fx.passes.shape_prop import ShapeProp
from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set
from vllm.logger import init_logger

logger = init_logger(__name__)

"""
Fuse all the nodes in the given module into a single function call.
"""
def fuse_graph_nodes(
    cc: CodeCache,
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule
) -> torch.fx.GraphModule:
    outputs = [n for n in mod.graph.nodes if n.op == 'output']
    inputs = [n for n in mod.graph.nodes if n.op == 'placeholder']

    if len(outputs) != 1:
        raise FusionFail("only single output supported currently.")

    # Collect all kwargs for fused ops and all the nodes that
    # will need to be fused (and erased) later.
    first = None
    nodes_to_fuse = []
    kwargs = dict()
    for n in mod.graph.nodes:
        # Find insertion point for new function call.
        if n.op == 'placeholder':
            first = n

        if n.op != 'call_function':
            continue

        if n.kwargs is not None and len(n.kwargs) > 0:
            kwargs[n.name] = n.kwargs

        nodes_to_fuse.append(n)

    if kwargs is not None and len(kwargs) > 0:
        raise FusionFail(f"kwargs for fused ops not supported. {kwargs}")

    # Lookup or create the fused operation.
    try:
        fn_key = fgen.make_fused_op(inputs, outputs, nodes_to_fuse, kwargs)

        def generate() -> Optional[Callable]:
            fn_dict = fgen.build_ops()
            assert fn_key in fn_dict
            return fn_dict[fn_key]

        fn = cc.lookup_or_create(fn_key, generate)

    except FusionFail as ff:
        logger.info(f"fusion failed '{ff}' for module: {mod}")
        return mod

    if fn is None:
        logger.info(f"fusion failed previously '{ff}' for module: {mod}")
        return mod

    logger.info(f"fused fn = {fn}")


    #
    # Update the graph
    # 1. insert the call_function for the fused op
    # 2. insert new output node(s)
    # 3. delete old call_function and output nodes.
    #

    mod.graph.inserting_after(first)

    # Note: we do not update the meta info for cf here.  It should
    # not be required after transformation anyway.
    cf = mod.graph.call_function(fn, args=tuple(inputs), kwargs=kwargs)
    logger.info(f"fused op: {cf.format_node()}")

    mod.graph.inserting_after(cf)
    mod.graph.output(cf, type_expr=torch.Tensor)

    for o in outputs:
        logger.info(f"ERASE {o}")
        mod.graph.erase_node(o)

    for n in reversed(nodes_to_fuse):
        logger.info(f"ERASE {n}")
        mod.graph.erase_node(n)

    logger.info(f"fuse mod {mod.print_readable(False)}")

    return mod


"""
Determine whether or not node is a fusable operations.
TODO: Smarter filter for 'getitem'.
"""
def is_fusable(node: torch.fx.Node) -> bool:
    if node.op != 'call_function':
        return False

    op_name = node_function_target(node)
    return op_name in FUSABLE and not FUSABLE[op_name]


"""
Determine whether or not node is a fusable compute operation, e.g. gemm.
"""
def is_compute(node: torch.fx.Node) -> bool:
    if node.op != 'call_function':
        return False

    op_name = node_function_target(node)
    return op_name in FUSABLE and FUSABLE[op_name]


def is_getitem(a: torch.fx.Node) -> bool:
    if a.op != 'call_function':
        return False
    return node_function_target(a) == '_operator.getitem'


"""
Are nodes a and b fusable together?
This function assumes 'b' is a direct successor of 'a'.
"""
def is_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return is_fusable(a) and is_fusable(b)


"""
Are nodes 'a' and 'b' fusable together and is 'a' optionally a compute op?
This function assumes 'b' is a direct successor of 'a'.
"""
def is_compute_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return (is_fusable(a) or is_compute(a)) and is_fusable(b)


"""
Determine if the given module is the result of the fusion pass or just
an pre-existing sub-module.
"""
def is_fused_subgraph(mod: torch.fx.GraphModule) -> bool:
    fg = FlowGraph(mod)
    saw_call = False

    if len([n for n in mod.graph.nodes if n.op == 'call_function']) <= 1:
        return False

    for n in mod.graph.nodes:
        if n.op == 'call_module' or n.op == 'call_method':
            return False

        if n.op != 'call_function':
            continue

        pred = is_fusable_pair if saw_call else is_compute_fusable_pair
        saw_call = True

        if not all([pred(n, s) for s in fg.successors(n) if s.op == 'call_function']):
            return False

    return True


"""
Determine if any kwargs associated with 'node' are supported.
"""
def supported_kwargs(node: torch.fx.Node, allow_const_kwargs: bool = False) -> bool:
    if allow_const_kwargs:
        for arg in node.kwargs.values():
            if not isinstance(arg, torch.fx.node.BaseArgumentTypes):
                return False
        return True
    else:
        return node.kwargs is None or len(node.kwargs) == 0


"""
1. create Partition objects from sequences of fusable nodes
2. use fuse_partitions to recreate the graph torch._inductor.fx_passes.group_batch_fusion
"""
def pointwise_fusion(
    cc: CodeCache,
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fuse_inputs: bool = False,
    fuse_with_compute=True
) -> torch.fx.GraphModule:
    # find all groups of nodes that can be fused and assign to
    # unique partition id, i.e. map_node

    fg = FlowGraph(mod)

    node_map = {}
    partition = 0

    def map_node(n: torch.fx.Node) -> int:
        return node_map[n]

    # assumption, graph.nodes are in topo order
    mod.graph.lint()

    logger.info("start fusion")

    # create partition groups
    # run in reverse order so predecesors of non-unary ops will appear
    # in the same partition.
    for n in reversed(mod.graph.nodes):
        logger.info(f"CONSIDER {n}")

        if n.op != 'call_function':
            logger.info(f"  REJECT {n} not call")
            node_map[n] = 0
            continue

        # TODO: handle get_attr ops
        # should probably be lifted/put in partition 0 but not prevent fusion

        pred = is_fusable_pair if not fuse_with_compute else is_compute_fusable_pair

        fusable = [pred(s, n) for s in fg.predecessors(n)]
        if not all(fusable):
            if not n in node_map:
                logger.info(f"  REJECT {n} no fusable preds and not in map: {fusable}, {fg.predecessors(n)}")
                node_map[n] = 0
            continue

        # don't support anything with kwargs for now
        if not supported_kwargs(n):
            logger.info(f"  REJECT {n} unsupported kwargs")
            node_map[n] = 0
            continue

        if n not in node_map:
            partition = partition + 1
            node_map[n] = partition

        for s in fg.predecessors(n):
            node_map[s] = node_map[n]

    logger.info(f"node_map = {node_map}")

    def same_partition(nodes: Set[torch.fx.Node]) -> bool:
        if len(nodes) > 0:
            part = node_map[next(iter(nodes))]
            #logger.info(f"{part}: {[node_map[n] for n in nodes]}")
            return all([node_map[n] == part for n in nodes])
        return False


    def only_pointwise(partition: int) -> bool:
        nodes = [n for n, p in node_map.items() if p == partition]
        return all([is_fusable(n) and not is_compute(n) for n in nodes])


    if fuse_with_compute:
        for n in mod.graph.nodes:
            if n.op != 'call_function':
                continue

            if fuse_inputs:
                nodes = fg.predecessors(n)
            else:
                nodes = fg.successors(n)

            if not is_compute(n):
                continue

            if not same_partition(nodes):
                #logger.info(f"REJECT {n} not all neighbors in same partition {nodes}")
                continue

            fuse_part = next(iter(nodes))

            if only_pointwise(fuse_part):
                node_map[n] = node_map[fuse_part]

    logger.info(f"final node_map = {node_map}")

    assert(all([n in node_map for n in mod.graph.nodes]))

    qualname_map=dict()

    logger.info(f"pre-fusion split mod {print_tabular_to_string(mod.graph)}")

    # create submodules for each fusable set of nodes
    new_mod = split_module(
        mod,
        mod,
        map_node,
        qualname_map,
        keep_original_order=False,
    )

    mig = ModuleInputGenerator(new_mod)
    mig.propagate(*example_inputs)

    # replace the fused submodules with new modules
    for cname, cm in new_mod.named_children():
        if is_fused_subgraph(cm):
            module_inputs = mig.module_args[cname][0]
            ShapeProp(cm).propagate(*module_inputs)
            logger.info(f"Fusing sub-module {cname}:\n{print_tabular_to_string(cm.graph)}")
            cm = fuse_graph_nodes(cc, fgen, cm)
            logger.info(f"Post fusion sub-module {cname}:\n{print_tabular_to_string(cm.graph)}")
            cm.recompile()

    logger.info(f"Post fusion module:\n{print_tabular_to_string(new_mod.graph)}")

    return new_mod
