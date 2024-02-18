test_dir=$1
xml_format=$2

pytest -v -s --junitxml=$xml_format $test_dir/test_regression.py
pytest -v -s --junitxml=$xml_format $test_dir/async_engine
pytest -v -s --junitxml=$xml_format --forked $test_dir/basic_correctness
pytest -v -s --junitxml=$xml_format --forked $test_dir/distributed/test_comm_ops.py
pytest -v -s --junitxml=$xml_format --forked $test_dir/distributed/test_basic_distributed_correctness.py
pytest -v -s --junitxml=$xml_format $test_dir/engine
pytest -v -s --junitxml=$xml_format $test_dir/entrypoints
pytest -v -s --junitxml=$xml_format $test_dir/kernels
pytest -v -s --junitxml=$xml_format --forked $test_dir/models
pytest -v -s --junitxml=$xml_format $test_dir/prefix_caching
pytest -v -s --junitxml=$xml_format --forked $test_dir/samplers
pytest -v -s --junitxml=$xml_format $test_dir/worker
pytest -v -s --junitxml=$xml_format $test_dir/lora
