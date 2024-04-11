#! /bin/bash

set -e
set -u
set -x

# global args
model_path=/home/varun/code/vllm/llama-models/Nous-Hermes-Llama2-13b
tokenizer=/home/varun/code/vllm/llama-models/Nous-Hermes-Llama2-13b/
max_seq_len=2048
max_num_batched_tokens=7000
tensor_parallel_size=1

# experiment args
prefill_prompt_len=512
decode_batch_sizes=(1 2 8 16 32 64 128)

# quantization specific args
quant_model_path=$model_path/quantized_model/llama-13b/Nous-Hermes-Llama2-13b-smoothquant/

# kv quant specific args
kv_cache_dtype=int8
kv_quant_params_path=/home/varun/code/vllm/act_quant_data/exported_kv/

run_quantized_prefill() {
  output_directory=$1

  now=`date +"%Y-%m-%d-%I-%M-%S"`
  out_base=${output_directory}/prefill_${prefill_prompt_len}_llama13_quantized-${now}

  echo "Running prefill ${prefill_prompt_len} store at ${out_base}"
  python3 examples/offline_profile.py --model $quant_model_path \
                                      --tokenizer $tokenizer \
                                      --batch-size 1 \
                                      --prompt-len $prefill_prompt_len \
                                      --quantization smoothquant \
                                      --max-seq-len $max_seq_len \
                                      --max-num-batched-tokens $max_num_batched_tokens \
                                      --tensor-parallel-size $tensor_parallel_size \
                                      --json $out_base \
                                      --csv  $out_base > ${out_base}_stdout.txt 2>&1
}

run_quantized_decode() {
  output_directory=$1

  for bs in "${decode_batch_sizes[@]}"
  do
    now=`date +"%Y-%m-%d-%I-%M-%S"`
    out_base=${output_directory}/decode_bs_${bs}_llama13_quantized-${now}

    echo "Running decode bs ${bs} store at ${out_base}" 
    python3 examples/offline_profile.py --model $quant_model_path \
                                        --tokenizer $tokenizer \
                                        --batch-size $bs \
                                        --prompt-len 1 \
                                        --quantization smoothquant \
                                        --max-seq-len $max_seq_len \
                                        --max-num-batched-tokens $max_num_batched_tokens \
                                        --tensor-parallel-size $tensor_parallel_size \
                                        --json $out_base \
                                        --csv  $out_base > ${out_base}_stdout.txt 2>&1 
  done
}

run_kv_quant_prefill() {

  output_directory=$1
  now=`date +"%Y-%m-%d-%I-%M-%S"`
  out_base=${output_directory}/prefill_${prefill_prompt_len}_llama13_kv_quant-${now}

  echo "Running prefill ${prefill_prompt_len} store at ${out_base}"

  python3 examples/offline_profile.py --model $model_path \
                                      --tokenizer $tokenizer \
                                      --batch-size 1  \
                                      --prompt-len $prefill_prompt_len \
                                      --kv-cache-dtype $kv_cache_dtype \
                                      --kv-quant-params-path $kv_quant_params_path  \
                                      --max-seq-len $max_seq_len \
                                      --max-num-batched-tokens $max_num_batched_tokens \
                                      --tensor-parallel-size $tensor_parallel_size \
                                      --json $out_base \
                                      --csv  $out_base > ${out_base}_stdout.txt 2>&1
}

run_kv_quant_decode() {
  output_directory=$1

  for bs in "${decode_batch_sizes[@]}"
  do
    now=`date +"%Y-%m-%d-%I-%M-%S"`
    out_base=${output_directory}/decode_bs_${bs}_llama13_kv_quant-${now}

    echo "Running decode bs ${bs} store at ${out_base}" 
    python3 examples/offline_profile.py --model $model_path \
                                        --tokenizer $tokenizer \
                                        --batch-size $bs  \
                                        --prompt-len 1 \
                                        --kv-cache-dtype $kv_cache_dtype \
                                        --kv-quant-params-path $kv_quant_params_path  \
                                        --max-seq-len $max_seq_len \
                                        --max-num-batched-tokens $max_num_batched_tokens \
                                        --tensor-parallel-size $tensor_parallel_size \
                                        --json $out_base \
                                        --csv  $out_base > ${out_base}_stdout.txt 2>&1
  done
}

run_fp16_prefill() {

  output_directory=$1
  now=`date +"%Y-%m-%d-%I-%M-%S"`
  out_base=${output_directory}/prefill_${prefill_prompt_len}_llama13_fp16-${now}

  echo "Running prefill ${prefill_prompt_len} store at ${out_base}"

  python3 examples/offline_profile.py --model $model_path \
                                      --tokenizer $tokenizer \
                                      --batch-size 1  \
                                      --prompt-len $prefill_prompt_len \
                                      --max-seq-len $max_seq_len \
                                      --max-num-batched-tokens $max_num_batched_tokens \
                                      --tensor-parallel-size $tensor_parallel_size \
                                      --json $out_base \
                                      --csv  $out_base > ${out_base}_stdout.txt 2>&1
}

run_fp16_decode() {
  output_directory=$1

  for bs in "${decode_batch_sizes[@]}"
  do
    now=`date +"%Y-%m-%d-%I-%M-%S"`
    out_base=${output_directory}/decode_bs_${bs}_llama13_fp16-${now}

    echo "Running decode bs ${bs} store at ${out_base}" 
    python3 examples/offline_profile.py --model $model_path \
                                        --tokenizer $tokenizer \
                                        --batch-size $bs  \
                                        --prompt-len 1 \
                                        --max-seq-len $max_seq_len \
                                        --max-num-batched-tokens $max_num_batched_tokens \
                                        --tensor-parallel-size $tensor_parallel_size \
                                        --json $out_base \
                                        --csv  $out_base > ${out_base}_stdout.txt 2>&1
  done
}

## Arg parser and invocation

usage() {
    echo``
    echo "Run profiler"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -t    - pass in w8a8 or kv_quant"
    echo "  -n    - pass in num_benchmark_iterations"
    echo "  -o    - out directory"
    echo
}

exp_type="" # should be either w8a8 or kv_quant
num_benchmark_iterations=1
output_directory="./"

while getopts ':t:n:o:h:' OPT; do
    case "${OPT}" in
        t)
            exp_type="${OPTARG}"
            ;;
        n)
            num_benchmark_iterations=${OPTARG}
            ;;
        o)
            output_directory="${OPTARG}"
            ;;
        h)
            usage
            exit 1
            ;;
    esac
done

if [ "$exp_type" != "w8a8" -a "$exp_type" != "kv_quant" -a "$exp_type" != "fp16" ];
then
  echo "Invalid arg $exp_type"
  usage
  exit 1
fi

for i in $(seq 1 $num_benchmark_iterations);
do
  echo "Running benchmark iteration ${i} ..."
  if [[ "${exp_type}" == "w8a8" ]];
  then
    run_quantized_prefill $output_directory
    run_quantized_decode $output_directory
  fi
  if [[ "${exp_type}" == "kv_quant" ]];
  then
    run_kv_quant_prefill $output_directory
    run_kv_quant_decode $output_directory
  fi
  if [[ "${exp_type}" == "fp16" ]];
  then
    run_fp16_prefill $output_directory
    run_fp16_decode $output_directory
  fi
done
