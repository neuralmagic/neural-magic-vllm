# Based on this tutorial
# https://huggingface.co/docs/transformers/model_doc/whisper

# TODO: install huggingface 'datasets' in requirements.txt or work around it
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Select an audio file and read it:
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# Audio encoder processor params
audio_sample = ds[0]["audio"]
waveform = audio_sample["array"]
processor_params = {
    "sampling_rate": audio_sample["sampling_rate"]
}

# Transcription decoder sampling params object.
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
predicted_ids = llm.generate(waveform, processor_params=processor_params)

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
transcriptions = llm.batch_decode(predicted_ids, sampling_params, skip_special_tokens=True)

# Print the outputs.
for transcription in transcriptions:
    # transcription_predicted_ids = transcription.predicted_ids
    transcription_text = transcription.outputs[0].text
    print(f"Transcription: {transcription_text!r}")
