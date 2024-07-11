import functools
import logging
import torch
import torch.utils.cpp_extension

from .utils import get_function_schema

from typing import List, Callable, Dict
from vllm.logger import init_logger

logger = init_logger(__name__)


def compose2(f: Callable, g: Callable) -> Callable:
    """
    Compose two functions.
    """
    return lambda *a, **kw: g(f(*a, **kw))


def compose(*fs: List[Callable]) -> Callable:
    """
    Compose a list of functions.
    """
    return functools.reduce(compose2, fs)


def is_optional_arg(n: torch.fx.Node, arg_num: int) -> bool:
    """
    Determine if the arg_num-th argument of n is optional or not based on
    the associated function schema.
    """
    s = get_function_schema(n)
    if not s or arg_num >= len(s.arguments):
        return False
    return isinstance(s.arguments[arg_num].real_type, torch._C.OptionalType)


def build_extension(lib_name: str,
                    sources: List[str],
                    opt: str = '-O2',
                    extra_cflags: List[str] = [],
                    extra_ldflags: List[str] = [],
                    verbose: bool = False):
    """
    Given a list of cpp and cuda source files, build and load a pytorch extension
    module with the given name.  Loaded ops will appear in the torch.ops.{lib_name}
    namespace.
    """
    torch.utils.cpp_extension.load(
        name=lib_name,
        sources=sources,
        extra_cflags=[
            opt, f'-DLIBRARY_NAME={lib_name}', *extra_cflags
        ],
        extra_ldflags=extra_ldflags,
        verbose=verbose,
        is_python_module=False,
    )


def arg_schema_type(n: torch.fx.Node, add_prefix: bool = False) -> str:
    """
    Get the schema or C++ type for a fused op argument.
    """
    if n.type is not None:
        ty = n.type.__name__
    elif n.meta.get('type') and n.meta.get('type').__name__ != 'FakeTensor':
        ty = n.meta.get('type').__name__
        if ty == 'Size':
            return 'std::vector<int64_t> const' if add_prefix else 'int[]'
    else:
        # this default is a bit sketchy
        ty = "Tensor"

    builtin_types = {"int":"int64_t", "float":"double"}

    if add_prefix and ty in builtin_types:
        return builtin_types[ty]

    return ty if not add_prefix else f"torch::{ty}"


def generate_op_schema(
    inputs: List[torch.fx.Node],
    outputs: List[torch.fx.Node],
    nodes: List[torch.fx.Node],
    kwargs: Dict[str, Dict[str, torch.fx.node.Argument]]
) -> str:
    sep = f"("
    arg_sig = ""
    for i, n in enumerate(inputs):
        arg_type = arg_schema_type(n).replace(".", "::")
        arg_name = n.name.replace(".", "_")
        arg_sig = arg_sig + sep + f"{arg_type} {arg_name}"
        sep = ", "

    # TODO support kwargs
    assert len(kwargs) == 0

    arg_sig = arg_sig + ") -> "

    sep = "(" if len(outputs) != 1 else ""

    for i, n in enumerate(outputs):
        arg_type = arg_schema_type(n).replace(".", "::")
        arg_sig = arg_sig + sep + arg_type
        sep = ", "

    if len(outputs) != 1:
        arg_sig = arg_sig + ")"

    return arg_sig


def generate_meta_function(
    inputs: List[torch.fx.Node],
    outputs: List[torch.fx.Node],
    nodes: List[torch.fx.Node],
    kwargs: Dict[str, Dict[str, torch.fx.node.Argument]]
) -> Callable:
    """
    Generate a meta function for a fused op by composing the individual
    operations.
    TODO: this only works when the fused op is a nice "funnel", i.e. the first
    op takes all the inputs and chains the rest to subsequent ops.
    See functools.partial and inspect.signature().parameters
    """
    fns = [n.target for n in nodes]
    return compose(*fns)


def register_op_schema(library: str, op: str, sig: str):
    """
    Register schema for the given 'op' in the given 'lib'.
    """
    op = op.mangle(".", "::").replace("torch::ops::", "")
    logger.debug(f"Registering schema for {op}: {sig}")
    torch.library.define(f"{op}", sig)


def register_meta_function(library: str, op: str, meta_fn: Callable):
    """
    Register meta function the given 'op' in the given 'lib'.
    See also: torch.library.impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1)
    """
    op = op.replace(".", "::").replace("torch::ops::", "")
    logger.debug(f"Registering meta function for {op}: {str(meta_fn)}")
    torch.library.impl(f"{op}", "Meta", func=meta_fn)
