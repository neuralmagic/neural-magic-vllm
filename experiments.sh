#! /bin/bash

set -e
set -u
#set -x

# global args
fp8_model_path=neuralmagic/Meta-Llama-3-8B-Instruct-FP8
fp16_model_path=NousResearch/Meta-Llama-3-8B-Instruct

# model generation args
enforce_eager=False
max_seq_len=2048
max_num_batched_tokens=7000
tensor_parallel_size=1

# experiment args
prefill_prompt_len=512
decode_batch_sizes=(1 2 8 16 32 64 128)

run_prefill() {
  model=$1
  desc=$2
  dtype=$3
  output_directory=$4

  now=`date +"%Y-%m-%d-%I-%M-%S"`
  out_base=${output_directory}/prefill_${prefill_prompt_len}_llama13_${desc}-${now}

  echo "Running prefill ${prefill_prompt_len} model ${model} desc ${desc} dtype ${dtype} store at ${out_base}"
  python3 examples/offline_profile.py --model $model \
                                      --batch-size 1 \
                                      --prompt-len $prefill_prompt_len \
                                      --max-seq-len $max_seq_len \
                                      --dtype $dtype \
                                      --max-num-batched-tokens $max_num_batched_tokens \
                                      --tensor-parallel-size $tensor_parallel_size  > ${out_base}_stdout.txt 2>&1
                                      #--allow-cuda-graphs > ${out_base}_stdout.txt 2>&1
                                      #--json $out_base \
                                      #--csv  $out_base > ${out_base}_stdout.txt 2>&1
}

run_decode() {
  model=$1
  desc=$2
  dtype=$3
  output_directory=$4

  for bs in "${decode_batch_sizes[@]}"
  do
    now=`date +"%Y-%m-%d-%I-%M-%S"`
    out_base=${output_directory}/decode_bs_${bs}_llama13_${desc}-${now}

    echo "Running decode bs ${bs} model ${model} desc ${desc} dtype ${dtype} store at ${out_base}" 
    python3 examples/offline_profile.py --model $model \
                                        --batch-size $bs \
                                        --prompt-len 16 \
                                        --max-seq-len $max_seq_len \
                                        --dtype $dtype \
                                        --max-num-batched-tokens $max_num_batched_tokens \
                                        --tensor-parallel-size $tensor_parallel_size > ${out_base}_stdout.txt 2>&1
                                        #--allow-cuda-graphs > ${out_base}_stdout.txt 2>&1
                                        #--json $out_base \
                                        #--csv  $out_base > ${out_base}_stdout.txt 2>&1 
  done
}

#run_prefill ${fp8_model_path} "fp8-prefill" auto ./
#run_prefill ${fp16_model_path} "fp16-prefill" auto ./profiles
run_decode ${fp8_model_path} "fp8-decode" auto ./
run_decode ${fp16_model_path} "fp16-decode" auto ./
