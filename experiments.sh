#! /bin/bash

set -e
set -u
#set -x

# global args
model_path=NousResearch/Nous-Hermes-Llama2-13b
quant_model_path=nm-testing/Nous-Hermes-Llama2-13b-smoothquant

# model generation args
enforce_eager=True
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
                                      --quantization smoothquant \
                                      --max-seq-len $max_seq_len \
                                      --dtype $dtype \
                                      --max-num-batched-tokens $max_num_batched_tokens \
                                      --tensor-parallel-size $tensor_parallel_size \
                                      --json $out_base \
                                      --csv  $out_base > ${out_base}_stdout.txt 2>&1
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
                                        --prompt-len 1 \
                                        --quantization smoothquant \
                                        --max-seq-len $max_seq_len \
                                        --dtype $dtype
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
    echo "  -t    - pass in quant or base"
    echo "  -d    - description"
    echo "  -m    - model data type"
    echo "  -n    - pass in num_benchmark_iterations"
    echo "  -o    - out directory"
    echo
}

exp_type="" # should be either quant or base
num_benchmark_iterations=1
output_directory="./"
desc=""
dtype=""

while getopts ':t:n:o:d:m:h:' OPT; do
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
        d)
            desc="${OPTARG}"
            ;;
        m)
            dtype="${OPTARG}"
            ;;
        h)
            usage
            exit 1
            ;;
    esac
done

if [ "$exp_type" != "quant" -a "$exp_type" != "base" ];
then
  echo "Invalid arg $exp_type"
  usage
  exit 1
fi

if [[ "${output_directory}" == "" || "${desc}" == "" || "${dtype}" == "" ]];
then
  echo "Either output_directory or desc is not set"
  usage
  exit 1
fi


for i in $(seq 1 $num_benchmark_iterations);
do
  echo "Running benchmark iteration ${i} ..."
  if [[ "${exp_type}" == "quant" ]];
  then
    run_prefill $quant_model_path "${exp_type}-${desc}" $dtype $output_directory
    #run_decode $quant_model_path "${exp_type}-${desc}" $dtype $output_directory
  fi
  if [[ "${exp_type}" == "base" ]];
  then
    run_prefill $model_path "${exp_type}-${desc}" $dtype $output_directory
    #run_decode $model_path "${exp_type}-${desc}" $dtype $output_directory
  fi
done
