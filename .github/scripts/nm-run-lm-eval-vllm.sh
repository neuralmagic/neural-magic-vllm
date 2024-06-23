#!/bin/bash
# We can use this script to compute baseline accuracy on GSM for transformers.
#
# Make sure you have lm-eval-harness installed:
#   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@9516087b81a61d0e220b22cc1b75be76de23bc10

usage() {
    echo``
    echo "Runs lm eval harness on GSM8k using vllm server and compares to "
    echo "precomputed baseline (measured by HF transformers.)"
    echo
    echo "This script should be run from the /nm-vllm directory" 
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -c    - path to the test data config (e.g. .github/lm-eval-configs/small-models-smoke.txt)"
    echo "  -t    - tensor parallel size to run the test at"
    echo
}

while getopts "c:t:" OPT; do
  case ${OPT} in
    c ) 
        CONFIG="$OPTARG"
        ;;
    t )
        TP_SIZE="$OPTARG"
        ;;
    \? ) 
        usage
        exit 1
        ;;
  esac
done

SUCCESS=0
cat $CONFIG | while read MODEL_CONFIG 
do
    LOCAL_SUCCESS=0
    
    echo "=== RUNNING MODEL: $MODEL_CONFIG =="
    
    MODEL_CONFIG_PATH=$PWD/.github/lm-eval-configs/models/${MODEL_CONFIG}
    LM_EVAL_TEST_DATA_FILE=$MODEL_CONFIG_PATH LM_EVAL_TP_SIZE=$TP_SIZE pytest -s tests/accuracy/test_lm_eval_correctness.py || LOCAL_SUCCESS=$?

    if [[ $LOCAL_SUCCESS == 0 ]]; then
        echo "=== PASSED MODEL: ${MODEL_CONFIG} ==="
    else
        echo "=== FAILED MODEL: ${MODEL_CONFIG} ==="
        SUCCESS=$((SUCCESS + LOCAL_SUCCESS))
    fi
done 

if [ "${SUCCESS}" -eq "0" ]; then
    exit 0
else
    exit 1
fi
