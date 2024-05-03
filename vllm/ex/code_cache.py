from typing import Callable, Optional
from vllm.logger import init_logger

logger = init_logger(__name__)

# Note: this can be pre-populated with pre-compiled kernels if needed
class CodeCache:
    """
    The CodeCache is a simple map from mangled function names to Callables.

    The CodeCache can be used to store the results of compiled code so that the
    same Callable can be resued rather than needing to be recompiled.

    Mangled function names should be generated with (or be compatible with) the
    'utils.mangle_name' function.

    Note: the CodeCache can be initialized with pre-compiled functions.
    """

    def __init__(self):
        self.cache = dict()

    def lookup_or_create(
        self, mangled_name: str,
        generator: Callable
    ) -> Optional[Callable]:
        """
        Lookup a Callable for a function based on the 'mangled_name'.  If the name
        is not present in the cache, call the supplied 'generator' to create
        the Callable to be associated with the 'mangled_name'.  If the
        generator fails for any reason a None will be stored in the map and
        returned instead of a Callable.  This will prevent any failed generators
        from being called repeatedly.
        """
        if not mangled_name in self.cache:
            try:
                self.cache[mangled_name] = generator()
            except Exception as ex:
                self.cache[mangled_name] = None
                raise ex
        return self.cache[mangled_name]
