# Based on this tutorial
# https://huggingface.co/docs/transformers/model_doc/whisper

# TODO: install huggingface 'transformers', 'datasets', 'soundfile', 'librosa' in requirements.txt or work around it
# to obtain audio dataset
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from vllm import LLM, SamplingParams

# TODO: vLLM audio frontend performs audio tokenization
# This code is a standin for a new audio frontend
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
audio_sample = ds[0]["audio"]
waveform = audio_sample["array"] # (93680,) floating-point values
processor_params = {
    "sampling_rate": audio_sample["sampling_rate"] # 16KHz
}

# (1, 80, 3000)
input_features = processor(
    waveform, sampling_rate=processor_params["sampling_rate"], return_tensors="pt"
).input_features

# Transcription decoder text token sampling params object.
sampling_params = SamplingParams(temperature=0.0)

# Create an LLM.
#
# Equivalent to:
#
#   processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
#   model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
#
# The HuggingFace model identifier is used to pull in (1) the model pretrained weights & config and
# (2) the WhisperProcessor trained weights associated with the model
#

#
llm = LLM(model="openai/whisper-tiny")



# Encode audio
#
# This is a change from how decoder-only LLM works *and* how HF transformers whisper workflow operates:
# - For encoder/decoder (E/D) models LLM.generate() is equivalent to
#
# input_features = processor(
#    waveform, sampling_rate=processor_params["sampling_rate"], return_tensors="pt"
# ).input_features
#
# predicted_ids = model.generate(input_features)
#
# i.e. LLM.generate() facilitates encoding.
#
# vLLM convention appears to be to abstract tokenization/preprocessing behind .generate()
#
predicted_ids = llm.generate(prompt_token_ids=input_features)

# Decoder token ids to transcription
#
# .batch_decode() is not yet a method of LLM. Its proposed function signature matches LLM.generate(tokens, sampling params)
# but with added kwargs to support typical use-cases, i.e. "skip_special_tokens".
#
# Equivalent to
#
#   transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
#
# The HuggingFace transcriptions output is a list of transcription strings.
#
# vLLM typical decoder-only behavior is that each element of 'output' list is a data structure with
# output.prompt, and output.outputs[...].text. So most likely we would wrap decoder outputs to respect
# this data structure.
#
transcriptions = processor.batch_decode(predicted_ids, sampling_params, skip_special_tokens=True)

# Print the outputs.
for transcription in transcriptions:
    # transcription_predicted_ids = transcription.predicted_ids
    transcription_text = transcription.outputs[0].text
    print(f"Transcription: {transcription_text!r}")
