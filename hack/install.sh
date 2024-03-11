#!/bin/bash

# setup env
python3 -m venv .hack
source .hack/bin/activate

# takes a while
pip3 install -e .
pip install nm-magic-wand

# install packages to use peft in vllm, takes about 3-5 mins
VLLM_INSTALL_PUNICA_KERNELS=1 python3 setup.py build_ext --inplace



# run the server
python -m vllm.entrypoints.openai.api_server \
    --model /network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/training \
    --enable-lora \
    --lora-modules \
        arc-easy=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_arc_easy \
        arc-hard=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_arc_hard \
        law=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_law \
        mc=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_mc \
        obqa=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_obqa \
        sci-ele=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_science_elementary \
        sci-middle=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_science_middle \
        race=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/PEFT-mmlu_race 
    --max-loras=8 \
    --max-cpu-loras=9


# In the client run the following code (non-stream)
# select model from 
# arc-easy, arc-hard, ..., race from above
:<<'COMMENT'
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "arc-easy",
        "messages": [
            {"role": "user", "content": "Name me cold-blooded animals"}
        ],
        "max_tokens": 256
    }'

COMMENT

# Streaming
:<<'COMMENT'
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "arc-easy",
        "messages": [
            {"role": "user", "content": "Name me cold-blooded animals"}
        ],
        "stream: true,
        "max_tokens": 256
    }'

COMMENT