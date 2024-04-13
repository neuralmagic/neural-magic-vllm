from vllm import LLM, SamplingParams
import torch

hf_path="nm-testing/Nous-Hermes-Llama2-13b-smoothquant"
model_path=hf_path

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_k = 1,max_tokens=20)

# Create an LLM.
llm = LLM(
    model="nm-testing/Nous-Hermes-Llama2-13b-smoothquant",
    gpu_memory_utilization=0.9,
    max_model_len=2048,
    quantization="smoothquant",
    dtype=torch.float,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_num_batched_tokens=7000)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
