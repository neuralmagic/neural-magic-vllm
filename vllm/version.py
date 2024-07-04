# UPSTREAM SYNC: take downstream
import warnings

try:
    import vllm.commit_id
    __commit__ = vllm.commit_id.__commit__
except Exception as e:
    warnings.warn(f"failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)
    __commit__ = "COMMIT_HASH_PLACEHOLDER"

__version__ = "0.5.1"
