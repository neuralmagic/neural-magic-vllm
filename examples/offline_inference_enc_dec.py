'''
Affirm T5 model outputs match between vLLM and native PyTorch

Scenarios:
* t5-small, t5-large
* float16, float32, bfloat16, bfloat32
* Custom prompts & num. prompts

Output: for several prompts, compare native PyTorch & vLLM prompt completions
'''
import warnings
import torch
from vllm import LLM, SamplingParams
from transformers import T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="transformers.generation.utils.*")

hf_model_id = "t5-large"
dtype = "float32"
prompts = [
    #"Who are you?",
    #"Who are you?",
    #"How do",
    "Who are you? Write a very long response.",
]

dtype_obj = getattr(torch, dtype)

# Native PyTorch test

# - Model and tokenizer initialization
tokenizer = T5Tokenizer.from_pretrained(hf_model_id, legacy=False)
model:T5ForConditionalGeneration  = T5ForConditionalGeneration.from_pretrained(hf_model_id).to(
    dtype=dtype_obj)

# - Assume 'dtype' is already defined, e.g., dtype=torch.float32
# - Tokenizing the prompts list with specified data type
input_ids = tokenizer(prompts,
                      return_tensors="pt",
                      padding=True,
                      truncation=True).input_ids

# - If using GPU, also send input_ids to the same device as the model
if torch.cuda.is_available():
    model = model.cuda()  # Move model to GPU
    input_ids = input_ids.cuda()  # Move input_ids to GPU

# - Max token count for both native and vLLM test
max_tokens = 512

# - Generating outputs for all tokenized prompts
native_outputs = model.generate(input_ids,max_length = max_tokens).cpu()

# vLLM test
model: LLM = LLM(hf_model_id,
            enforce_eager=True,
            dtype=dtype,
            gpu_memory_utilization=0.25)

sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)

vllm_outputs = model.generate(
    prompts,
    sampling_params=sampling_params
)

print(native_outputs)
print(vllm_outputs)

# Print native & vLLM outputs
i = 0
for native_output, vllm_output in zip(native_outputs, vllm_outputs):
    print(f"Prompt {i}:")

    prompt = prompts[i]  # Get the corresponding prompt for this output
    native_generated_text = tokenizer.decode(
        native_output, skip_special_tokens=True)  # Decode the generated text
    vllm_generated_text = vllm_output.outputs[0].text
    print(
        f"- Prompt: {prompt!r}, Native PyTorch generated text: " \
        f"{native_generated_text!r}, " \
        f"vLLM generated text: {vllm_generated_text!r}"
    )

    print("- Asserting textual match")
    assert native_generated_text == vllm_generated_text
    print("- Asserting token match")
    assert native_output[1:-1].tolist() == vllm_output.outputs[0].token_ids[:-1]

    i += 1