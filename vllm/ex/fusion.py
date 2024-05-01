import torch

from .code_cache import CodeCache
from .fused_op_generator import FusedOpGenerator, FusionFail
from .utils import extract_node_type, extract_node_tensor_meta, ModuleInputGenerator, FlowGraph

from torch.fx.passes.split_module import split_module
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

###############################################################################
#
# Fusion
#
###############################################################################

#
# Fuse all the nodes in a subgraph into a single node
#
def fuse_graph_nodes(
    cc: CodeCache,
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule
) -> torch.fx.GraphModule:
    first = None

    outputs = [n for n in mod.graph.nodes if n.op == 'output']
    inputs = [n for n in mod.graph.nodes if n.op == 'placeholder']

    newline = "\n"
    print(f"input_meta:\n{newline.join([f'{n}: {extract_node_tensor_meta(n)}' for n in inputs])}")

    # for now
    assert len(outputs) == 1

    nodes_to_erase = []

    kwargs = None
    for n in mod.graph.nodes:
        if n.op == 'placeholder':
            first = n

        if n.op != 'call_function':
            continue

        if n.kwargs:
            if not kwargs:
                kwargs = n.kwargs
            else:
                # TODO: assert no duplicates
                kwargs = {**kwargs, **n.kwargs}

        nodes_to_erase.append(n)

    # TODO: wrap CodeCache around this bit (fn_key is the mangled name)
    try:
        fn_key = fgen.make_fused_op(inputs, outputs, nodes_to_erase, kwargs)

        def generate() -> Optional[Callable]:
            fn_dict = fgen.build_ops()
            assert fn_key in fn_dict
            fn, _, _ = fn_dict[fn_key]
            return fn

        fn = cc.lookup_or_create(fn_key, generate)

    except FusionFail as ff:
        print(f"fusion failed '{ff}' for module: {mod}")
        return mod

    if fn is None:
        print(f"fusion failed previously '{ff}' for module: {mod}")
        return mod

    #print(f"fused fn = {fn}, {type(fn)}, {isinstance(fn, torch.nn.Module)}, {str(fn)}")

    mod.graph.inserting_after(first)

    # TODO: no kwargs for now
    assert kwargs == None or len(kwargs) == 0

    cf = mod.graph.call_function(fn, args=tuple(inputs), kwargs=kwargs)

    # Note: we do not update the meta info for cf here.  It should
    # not be required after transformation anyway.

    # which way is best?  the 'else' seems more general
    # see also eliminate_dead_code()
    if False:
        outputs[0].prev.replace_all_uses_with(cf)
    else:
        mod.graph.inserting_after(cf)
        mod.graph.output(cf, type_expr=torch.Tensor)

        for o in outputs:
            print(f"ERASE {o}")
            mod.graph.erase_node(o)

    print(f"fuse mod {mod.print_readable(False)}")
    print(f"cf {cf.name} {cf.format_node()}")

    nodes_to_erase.reverse()
    for n in nodes_to_erase:
        print(f"ERASE {n}")
        mod.graph.erase_node(n)

    # TODO: see node.replace_all_uses_with(new_node)

    # Do this here or in caller?
    #mod.recompile()

    return mod


# TODO: add more stuff
# Smarter filter for getitem
def is_fusable(a: torch.fx.Node) -> bool:
    pointwise = ['_operator.add', '_operator.mul', '_operator.getitem', 'torch.relu', 'torch.nn.functional.silu']
    if a.op == 'call_function':
        submodules = dict(a.graph.owning_module.named_modules())
        target = get_node_target(submodules, a)
        return target in pointwise
    return False


# TODO: add more stuff
def is_compute(a: torch.fx.Node) -> bool:
    if a.op != 'call_function':
        return False
    submodules = dict(a.graph.owning_module.named_modules())
    return (get_node_target(submodules, a) == 'torch.matmul' or
            get_node_target(submodules, a) == 'torch._C._nn.linear')


def is_getitem(a: torch.fx.Node) -> bool:
    if a.op != 'call_function':
        return False
    submodules = dict(a.graph.owning_module.named_modules())
    return get_node_target(submodules, a) == '_operator.getitem'


def is_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return is_fusable(a) and is_fusable(b)


def is_compute_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return (is_fusable(a) or is_compute(a)) and is_fusable(b)


# TODO: reject singleton/nop partitions
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



# 1. create Partition objects from sequences of fusable nodes
# 2. use fuse_partitions to recreate the graph
# torch._inductor.fx_passes.group_batch_fusion
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

    print("start fusion")

    # create partition groups
    # run in reverse order so predecesors of non-unary ops will appear
    # in the same partition.
    for n in reversed(mod.graph.nodes):
        #print(f"CONSIDER {n}")

        if n.op != 'call_function':
            #print(f"  REJECT {n} not call")
            node_map[n] = 0
            continue

        # TODO: handle get_attr ops
        # should probably be lifted/put in partition 0 but not prevent fusion

        if not all([is_fusable_pair(n, s) for s in fg.predecessors(n)]):
            if not n in node_map:
                #print(f"  REJECT {n} no fusable preds and not in map")
                node_map[n] = 0
            continue

        # don't support anything with kwargs for now
        if n.kwargs and len(n.kwargs) > 0:
            #print(f"  REJECT {n} kwargs")
            node_map[n] = 0
            continue

        if n not in node_map:
            partition = partition + 1
            node_map[n] = partition

        for s in fg.predecessors(n):
            node_map[s] = node_map[n]

    print(f"node_map = {node_map}")

    def same_partition(nodes: Set[torch.fx.Node]) -> bool:
        if len(nodes) > 0:
            part = node_map[next(iter(nodes))]
            #print(f"{part}: {[node_map[n] for n in nodes]}")
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
                #print(f"REJECT {n} not all neighbors in same partition {nodes}")
                continue

            fuse_part = next(iter(nodes))

            if only_pointwise(fuse_part):
                node_map[n] = node_map[fuse_part]

    print(f"final node_map = {node_map}")

    assert(all([n in node_map for n in mod.graph.nodes]))

    qualname_map=dict()

    print(f"mod {mod.print_readable(False)}")
    mod.graph.print_tabular()

    # create submodules for each fusable set of nodes
    new_mod = split_module(
        mod,
        mod,
        map_node,
        qualname_map,
        keep_original_order=False, #True
    )

    mig = ModuleInputGenerator(new_mod)
    mig.propagate(*example_inputs)

    # replace the fused submodules with new modules
    for cname, cm in new_mod.named_children():
        if is_fused_subgraph(cm):
            module_inputs = mig.module_args[cname][0]
            ShapeProp(cm).propagate(*module_inputs)

            print(f"FUSING GRAPH NODES {cname}")
            cm.graph.print_tabular()
            print(cm.graph.python_code(cm).src)
            #graph_print_tabular(cm.graph)
            cm = fuse_graph_nodes(cc, fgen, cm)
            print(f"CM {cname}: {cm}")
            cm.recompile()

    print(f"new_mod {new_mod.print_readable(False)}")
    print(f"new mod {new_mod.graph.print_tabular()}")

    # Do this here or in caller?
    #new_mod.recompile()

    return new_mod

