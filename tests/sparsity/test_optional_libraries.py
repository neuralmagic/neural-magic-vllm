import sys

from vllm import LLM

optional_libraries = ["magic_wand"]


def maybe_clear_libraries(libraries_to_remove):
    # Explicitly remove optional libs if they were previously imported
    # to ensure a clean test environment.
    for lib in libraries_to_remove:
        if lib in sys.modules:
            del sys.modules[lib]


def test_magic_wand_not_imported():
    maybe_clear_libraries(optional_libraries)

    # This line should not require importing magic_wand
    _ = LLM("facebook/opt-125m")

    for lib in optional_libraries:
        assert lib not in sys.modules


def test_magic_wand_imported():
    maybe_clear_libraries(optional_libraries)

    # This line should require importing magic_wand
    _ = LLM("facebook/opt-125m", sparsity="sparse_w16a16")

    for lib in optional_libraries:
        assert lib in sys.modules
