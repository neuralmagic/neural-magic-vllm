###############################################################################
#
# Utils
#
###############################################################################

import functools
import torch
import torch.utils.cpp_extension
import types

from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.tools_common import get_node_target
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set


def node_function_target(node: torch.fx.Node) -> str:
    return get_node_target(None, node)

# Make this always return a string?
def argument_type(arg: torch.fx.node.Argument):
    if isinstance(arg, torch.fx.Node):
        return extract_node_type(arg)
    elif isinstance(arg, torch.Tensor):
        return arg.dtype
    elif isinstance(arg, torch.dtype):
        return arg
    elif (isinstance(arg, str) or
          isinstance(arg, int) or
          isinstance(arg, float) or
          isinstance(arg, bool)):
        return type(arg)
    elif (isinstance(arg, types.EllipsisType) or
          isinstance(arg, types.NoneType)):
        return arg
    elif isinstance(arg, tuple):
        # TODO: needs some work
        return "t_" + "_".join([str(argument_type(a)) for a in arg])
    else:
        return None # raise Exception(f"unsupported argument type {arg}")


def extract_node_type(n: torch.fx.Node):
    if 'tensor_meta' in n.meta:
        return n.meta['tensor_meta'].dtype
    else:
        return None


"""
Compose two functions.
"""
def compose2(f: Callable, g: Callable) -> Callable:
    return lambda *a, **kw: g(f(*a, **kw))


"""
Compose a list of functions.
"""
def compose(*fs: List[Callable]) -> Callable:
    return functools.reduce(compose2, fs)


"""
Generate a mangled name from a list of call_function nodes.  The mangled
name includes the names of all the operators and their types.
"""
def mangle_name(nodes: List[torch.fx.Node], rep: str = "_P_") -> str:
    name = ""
    sep = ""
    for n in nodes:
        fn = node_function_target(n)
        types = [str(argument_type(arg)).replace("torch.","") for arg in n.args]
        name = name + sep + f"{fn}_{'_'.join(types)}"
        sep = "_"

    return name.replace(".", rep)


###############################################################################
#
# ModuleInputGenerator
#
###############################################################################


# TODO: combine with ShapeProp somehow?
class ModuleInputGenerator(torch.fx.passes.fake_tensor_prop.FakeTensorProp):
    """
    Generate example inputs for all submodules in the given GraphModule.
    """

    def __init__(
            self,
            module: torch.fx.GraphModule,
            mode: Optional[FakeTensorMode] = None,
    ):
        super().__init__(module, mode)
        self.module_args = {}

    def call_module(
            self,
            target: torch.fx.node.Target,
            args: Tuple[torch.fx.node.Argument, ...],
            kwargs: Dict[str, Any]
    ) -> Any:
        # TODO: problem here with multiple call sites and different args,
        # for now set to None if there are multiple callers.
        # Could check for "compatible" inputs and allow.
        if target in self.module_args:
            self.module_args[target] = (None, None)
        else:
            self.module_args[target] = (args, kwargs)

        return super().call_module(target, args, kwargs)


###############################################################################
#
# dataflow graph
#
###############################################################################

class FlowGraph:
    def __init__(self, gm: torch.fx.GraphModule):
        self.module = gm
        self.build()

    def add_edge(self, src: torch.fx.GraphModule, dst: torch.fx.GraphModule):
        if not src in self.succs:
            self.succs[src] = set()
        if not dst in self.preds:
            self.preds[dst] = set()

        self.succs[src].add(dst)
        self.preds[dst].add(src)

    # TODO: turn getitems into "reader views"?
    def build(self):
        self.succs = dict()
        self.preds = dict()
        self.outputs = [n for n in self.module.graph.nodes if n.op == 'output']
        self.inputs = [n for n in self.module.graph.nodes if n.op == 'placeholder']
        visited = set()
        q = self.outputs

        while len(q) > 0:
            n = q.pop()
            if n in visited:
                continue

            visited.add(n)
            for input in n.all_input_nodes:
                self.add_edge(input, n)
                q.append(input)

    def inputs(self) -> List[torch.fx.Node]:
        return self.inputs

    def outputs(self) -> List[torch.fx.Node]:
        return self.outputs

    def successors(self, n: torch.fx.Node) -> Set[torch.fx.Node]:
        return self.succs[n] if n in self.succs else set()

    def predecessors(self, n: torch.fx.Node) -> Set[torch.fx.Node]:
        return self.preds[n] if n in self.preds else set()

    def visit(self, fn: Callable):
        q = self.inputs
        visited = set()
        while len(q) > 0:
            n = q.pop()
            if n in visited:
                continue
            visited.add(n)
            fn(n)
            q = list(self.successors(n)) + q


def build_extension(lib_name: str, sources: List[str]):
    torch.utils.cpp_extension.load(
        name=lib_name,
        sources=sources,
        #extra_cflags=['-O2',f'-DLIBRARY_NAME={lib_name}'],
        extra_cflags=['-g',f'-DLIBRARY_NAME={lib_name}'],
        verbose=True,
        is_python_module=False,
    )

