import argparse
from vllm import LLM, SamplingParams

MODELS = {
    "tinyllama-fp16": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama-marlin": "neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin",
    "tinyllama-gptq": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    "tinyllama-awq": "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--tensor-parallel-size", type=int, default=1)
args = parser.parse_args()

if args.model not in MODELS:
    print(f"Got model id of {args.model}; Must be in {list(MODELS.keys())}")
    raise ValueError
else:
    model_id = MODELS[args.model]
    print(f"Using model_id = {model_id}")

messages=[{
    "role": "system",
    "content": "You are a helpful assistant."
}, {
    "role": "user",
    "content": "What is deep learning?"
}]

model = LLM(model_id, enforce_eager=True, max_model_len=2048, tensor_parallel_size=args.tensor_parallel_size, dtype="float16")
prompt = model.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(prompt, SamplingParams(max_tokens=50))
print(f"\n-----prompt\n{prompt}")
print(f"\n-----generation\n{out[0].outputs[0].text}")
