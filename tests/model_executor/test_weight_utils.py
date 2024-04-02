import pytest
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from vllm.model_executor.weight_utils import hf_model_weights_iterator
from transformers import AutoModelForCausalLM, AutoConfig
from sparseml.transformers import SparseAutoModelForCausalLM
import shutil

@pytest.fixture
def hf_folder_dense(tmp_path):
    model_id = "stas/tiny-random-llama-2"
    cache_dir = tmp_path / "cache"
    yield snapshot_download(model_id, cache_dir=cache_dir)
    shutil.rmtree(cache_dir)

@pytest.fixture
def hf_folder_sparse(tmp_path):
    model_id = "mgoin/TinyLlama-1.1B-Chat-v1.0-pruned2.4-compressed"
    cache_dir = tmp_path / "cache"
    yield snapshot_download(model_id, cache_dir=cache_dir)
    shutil.rmtree(cache_dir)
     


def test_hf_model_weights_iterator_dense_compressor(hf_folder_dense):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(hf_folder_dense)
        model = AutoModelForCausalLM.from_config(config) 
    assert not hasattr(config, "sparsity_config")
    model_weight_names = set([name for name, _ in model.named_parameters()])
    loaded_weight_names = set(name for name, _ in hf_model_weights_iterator(hf_folder_dense))
    assert model_weight_names == loaded_weight_names
    
def test_hf_model_weights_iterator_sparse_compressor(tmp_path, hf_folder_sparse):
    # save the sparse model using the compressed safetensor format
    model = SparseAutoModelForCausalLM.from_pretrained(hf_folder_sparse)
    save_path = tmp_path / "save_path"
    model.save_pretrained(save_path, save_compressed=True, skip_compression_stats=True)
    
    # test the weight loading logic
    with init_empty_weights():
        config = AutoConfig.from_pretrained(save_path)
        model = AutoModelForCausalLM.from_config(config) 
    assert hasattr(config, "sparsity_config")
    model_weight_names = set([name for name, _ in model.named_parameters()])
    loaded_weight_names = set(name for name, _ in hf_model_weights_iterator(save_path))
    assert model_weight_names == loaded_weight_names
    
    