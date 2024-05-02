from typing import Callable, Optional
from vllm.logger import init_logger

logger = init_logger(__name__)

# Note: this can be pre-populated with pre-compiled kernels if needed
class CodeCache:
    def __init__(self):
        self.cache = dict()

    def lookup_or_create(self, mangled_name:str, generator: Callable) -> Optional[Callable]:
        if not mangled_name in self.cache:
            try:
                self.cache[mangled_name] = generator()
            except Exception as ex:
                self.cache[mangled_name] = None
                raise ex
        return self.cache[mangled_name]
