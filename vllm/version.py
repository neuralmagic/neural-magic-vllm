# UPSTREAM SYNC: take downstream
try:
    import vllm.commit_id
    __commit__ = vllm.commit_id.__commit__
except:
    __commit__ = "COMMIT_HASH_PLACEHOLDER"
    
__version__ = "0.5.1"
