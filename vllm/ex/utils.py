import functools
import torch
import torch.utils.cpp_extension

from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.tools_common import get_node_target
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

###############################################################################
#
# Utils
#
###############################################################################

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
    elif arg is None:  # an optional arg?
        return None
    else:
        return None # raise Exception(f"unsupported argument type {arg}")


def extract_node_tensor_meta(n: torch.fx.Node):
    if 'tensor_meta' in n.meta:
        return n.meta['tensor_meta']
    return None


def extract_node_type(n: torch.fx.Node):
    if 'tensor_meta' in n.meta:
        return n.meta['tensor_meta'].dtype
    else:
        return None


# compose two functions
def compose2(f: Callable, g: Callable) -> Callable:
    return lambda *a, **kw: g(f(*a, **kw))


# compose a list of functions
def compose(*fs: List[Callable]) -> Callable:
    return functools.reduce(compose2, fs)


def mangle_name(nodes: List[torch.fx.Node], rep: str = "_P_") -> str:
    submodules = dict(nodes[0].graph.owning_module.named_modules())
    name = ""
    sep = ""
    for n in nodes:
        fn = get_node_target(submodules, n)
        types = [str(argument_type(arg)).replace("torch.","") for arg in n.args]
        name = name + sep + f"{fn}_{'_'.join(types)}"
        sep = "_"

    return name.replace(".", rep)


###############################################################################
#
# ModuleInputGenerator
#
###############################################################################


# Combine with ShapeProp somehow?
class ModuleInputGenerator(torch.fx.passes.fake_tensor_prop.FakeTensorProp):
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
        extra_cflags=['-O2',f'-DLIBRARY_NAME={lib_name}'],
        #extra_cflags=['-g',f'-DLIBRARY_NAME={lib_name}'],
        verbose=True,
        is_python_module=False,
    )

