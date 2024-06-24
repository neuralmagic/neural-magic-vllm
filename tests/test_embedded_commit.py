import vllm


def test_embedded_commit_exists():
    assert vllm.__commit__
