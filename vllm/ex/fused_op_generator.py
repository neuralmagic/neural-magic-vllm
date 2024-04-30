import tempfile
import torch

from .utils import extract_node_type, extract_node_tensor_meta, compose, build_extension

from torch.fx.passes.tools_common import get_node_target
from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

class FusionFail(Exception):
    pass

class FusedOpGenerator:
    N = 0

    def __init__(self):
        self.filename = "fused_"
        self.callables = dict()
        self.reset_fused_op()
        self.N = FusedOpGenerator.N

    def reset_fused_op(self):
        self.fused_op = []
        self.fused_op.append(f'#include <torch/extension.h>')
        self.fused_op.append(f'#include <iostream>')
        self.fused_op.append('#define _torch_add(a, b) ((a) + (b))')
        self.fused_op.append('#define _torch_mul(a, b) ((a) * (b))')
        self.fused_op.append('#define TORCH_LIBRARY_EXPAND(name, mod) TORCH_LIBRARY(name, mod)')
        self.fused_op.append('#define TORCH_LIBRARY_IMPL_EXPAND(name, k, mod) TORCH_LIBRARY_IMPL(name, k, mod)')

    # This should take types into account. (what else?)
    def mangle(self, s: str, rep: str = '_P_') -> str:
        s = s.replace('.', rep)
        return s

    def rename(self, s: str) -> str:
        if s == '_operator.add':
            return '_torch_add'
        elif s == '_operator.mul':
            return '_torch_mul'
        else:
            return s

    #
    # Generate some (dumb) C++/CUDA code for a stack of fused ops.
    #
    # TODO:
    # - use cutlass
    # - include types in mangled names
    # - manage generated code (no duplicates)
    # - handle kwargs
    #
    # See https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?pli=1#heading=h.rmcmku6fe6ug
    #
    # Note: node.meta['tensor_meta'] will have shape and dtype fields
    #
    def make_fused_op(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        nodes: List[torch.fx.Node],
        # make this a list of Dict?
        kwargs: Dict[str, torch.fx.node.Argument]
    ) -> torch.fx.node.Target:
        fns = [n.target for n in nodes]
        print(f"MAKE_FUSED_OP {fns}")
        # See functools.partial for applying args/kwargs

        # assume unary output for now
        assert len(outputs) == 1

        submodules = dict(nodes[0].graph.owning_module.named_modules())

        fn_names = [self.rename(get_node_target(submodules, n)) for n in nodes]

        op = self.mangle("_".join(fn_names)) + '_fused'
        cxx_arg_sig = ''
        arg_sig = ''
        sep = ''
        input_type_map = {}
        for i, n in enumerate(inputs):
            cxx_arg_sig = cxx_arg_sig + sep + f"torch::Tensor const& {n}"
            arg_sig = arg_sig + sep + f"Tensor {n}"
            sep = ", "
            input_type_map[n.name] = extract_node_type(n)

        for n in nodes:
            input_type_map[n.name] = extract_node_type(n)

        def input_type(arg: torch.fx.node.Argument):
            if isinstance(arg, torch.fx.Node):
                return input_type_map[arg.name]
            else:
                return None

        oc = '{'
        cc = '}'

        self.fused_op.append(f'torch::Tensor {op}({cxx_arg_sig})')
        self.fused_op.append('{')
        self.fused_op.append('std::cout << "GOT HERE" << std::endl;')

        for n, fn in zip(nodes, fn_names):
            com_str = f"  // ({', '.join([str(input_type(inp)) for inp in n.args])}) -> {str(extract_node_type(n))}"
            call_str = f"  auto const& {self.mangle(n.name, '_')} = {self.mangle(fn, '::')}("
            sep =''
            for inp in n.args:
                call_str = call_str + sep + self.mangle(str(inp), '_')
                sep = ', '
            call_str = call_str + ');'
            self.fused_op.append(com_str)
            self.fused_op.append(call_str)
        self.fused_op.append(f"  // {str(extract_node_type(outputs[0]))}")
        self.fused_op.append(f"  return {self.mangle(outputs[0].args[0].name, '_')};")

        self.fused_op.append('}')
        #self.fused_op.append(f'TORCH_LIBRARY_EXPAND(fused_ops{self.N}, m) {oc} m.def("{op}({arg_sig}) -> Tensor"); {cc}')
        self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, CPU, m) {oc} m.impl("{op}", &{op}); {cc}')
        self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, CUDA, m) {oc} m.impl("{op}", &{op}); {cc}')
        # TODO: make sure this does the "right thing"
        #self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, Meta, m) {oc} m.impl("{op}", &{op}); {cc}')

        self.callables[op] = (
            f"torch.ops.fused_ops{self.N}.{op}",
            self.generate_op_signature(inputs, outputs, nodes, kwargs),
            self.generate_meta_function(inputs, outputs, nodes, kwargs)
        )

        return op

    # should be derivable from types on input/output nodes
    def generate_op_signature(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        nodes: List[torch.fx.Node],
        # make this a list of Dict?
        kwargs: Dict[str, torch.fx.node.Argument]
    ):
        #submodules = dict(nodes[0].graph.owning_module.named_modules())
        #fn_names = [self.rename(get_node_target(submodules, n)) for n in nodes]
        #op = self.mangle("_".join(fn_names)) + '_fused'

        #print(f"inputs = {[n.format_node() for n in inputs]}")

        sep = f"("
        arg_sig = ""
        for i, n in enumerate(inputs):
            # TODO: the default here is sketchy
            arg_type = self.mangle(n.type.__name__ if n.type is not None else "Tensor", '::')
            arg_name = self.mangle(n.name, '_')
            arg_sig = arg_sig + sep + f"{arg_type} {arg_name}"
            sep = ", "
        arg_sig = arg_sig + ") -> "

        sep = "(" if len(outputs) != 1 else ""

        for i, n in enumerate(outputs):
            # TODO: the default here is sketchy
            arg_type = self.mangle(n.type.__name__ if n.type is not None else "Tensor", '::')
            arg_sig = arg_sig + sep + arg_type
            sep = ", "

        if len(outputs) != 1:
            arg_sig = arg_sig + ")"

        return arg_sig

    def generate_meta_function(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        nodes: List[torch.fx.Node],
        # make this a list of Dict?
        kwargs: Dict[str, torch.fx.node.Argument]
    ) -> Callable:
        submodules = dict(nodes[0].graph.owning_module.named_modules())
        fn_names = [self.rename(get_node_target(submodules, n)) for n in nodes]
        op = self.mangle("_".join(fn_names)) + '_fused'

        # TODO: need to lift inputs/outputs
        fns = [n.target for n in nodes]

        # See functools.partial
        return compose(*fns)


    def register_op_sig(self, lib: str, op: str, sig: str):
        # TODO: registration
        #lib = torch.library.Library(f"fused_ops{self.N}", "DEF") ?
        op = self.mangle(op, '::').replace("torch::ops::", "")
        print(f"ARG_SIG = {op}, {sig}")
        torch.library.define(f"{op}", sig)


    def register_meta_function(self, lib: str, op: str, meta_fn: Callable):
        # torch.library.impl(qualname, types, func=None, *, lib=None)
        # torch.library.impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1)
        #torch.library.impl(lib, f"torch.ops.fused_ops{self.N}.{op}", "Meta")
        op = self.mangle(op, '::').replace("torch::ops::", "")
        print(f"META_FN = {op}, {str(meta_fn)}")
        torch.library.impl(f"{op}", "Meta", func=meta_fn)


    def build_ops(self) -> Dict[torch.fx.node.Target, Tuple[Callable, str, Callable]]:
        # prevent multiple libraries with the same name
        FusedOpGenerator.N = FusedOpGenerator.N + 1

        try:
            op_lib = f"fused_ops{self.N}"

            # TODO: no way to unregister if build goes wrong?
            for k, v in self.callables.items():
                self.register_op_sig(op_lib, v[0], v[1])
                self.register_meta_function(op_lib, v[0], v[2])

            with tempfile.NamedTemporaryFile(
                    prefix=self.filename,
                    suffix=".cpp",
                    mode='w',
                    delete=False, # TODO: True
            ) as out:
                print(f"generating code to: {out.name}")
                for l in self.fused_op:
                    out.write(l)
                    out.write('\n')
                out.close()
                build_extension(op_lib, str(out.name))

            self.N = FusedOpGenerator.N

            for k, v in self.callables.items():
                # there has to be a better way than eval?
                fn = eval(v[0])
                print(f'{self.callables[k]} = {fn}')
                self.callables[k] = (fn, v[1], v[2])
                self.register_meta_function(op_lib, v[0], v[2])

            print(f"CALLABLES {self.callables}")

            callables = self.callables

            self.reset_fused_op()
            self.callables = dict()

            return callables

        except Exception as ex:
            raise FusionFail(ex)
