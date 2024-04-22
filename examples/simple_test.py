import argparse
from vllm import LLM, SamplingParams

MODELS = {
    "tinyllama-fp16": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama-marlin": "neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin",
    "tinyllama-gptq": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    "tinyllama-awq": "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
    "gemma-fp16": "google/gemma-1.1-2b-it",
    "gemma-awq": "TechxGenus/gemma-1.1-2b-it-AWQ",
    "gemma-gptq": "TechxGenus/gemma-1.1-2b-it-GPTQ",
    "phi-2-fp16": "abacaj/phi-2-super",
    "phi-2-marlin": "neuralmagic/phi-2-super-marlin",
    "deepseek-fp16": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-gptq": "TheBloke/deepseek-coder-1.3b-instruct-GPTQ",
    "deepseek-awq": "TheBloke/deepseek-coder-1.3b-instruct-AWQ",
    "deepseek-moe-fp16": "deepseek-ai/deepseek-moe-16b-chat",
    "baichuan-fp16": "baichuan-inc/Baichuan2-7B-Chat",
    "baichuan-gptq": "csdc-atl/Baichuan2-7B-Chat-GPTQ-Int4",
    "qwen-fp16": "Qwen/Qwen1.5-1.8B",
    "qwen-gptq": "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",
    "qwen-awq": "Qwen/Qwen1.5-1.8B-Chat-AWQ",
    "gpt2-fp16": "openai-community/gpt2",
    "gpt2-gptq": "etyacke/GPT2-GPTQ-int4",
    "starcoder2-fp16": "bigcode/starcoder2-3b",
    "starcoder2-gptq": "TechxGenus/starcoder2-3b-GPTQ",
    "starcoder2-awq": "TechxGenus/starcoder2-3b-AWQ",
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
    "role": "user",
    "content": "What is deep learning?"
}]

model = LLM(model_id, enforce_eager=True, max_model_len=1024, tensor_parallel_size=args.tensor_parallel_size, dtype="float16", trust_remote_code=True)
prompt = model.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(prompt, SamplingParams(max_tokens=50))
print(f"\n-----prompt\n{prompt}")
print(f"\n-----generation\n{out[0].outputs[0].text}")
