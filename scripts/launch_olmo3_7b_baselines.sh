#!/bin/bash
# Submit baseline evaluations for OLMo 3 7B across gsm8k, math, triviaqa.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH="${MODEL_PATH:-$HOME/scratch/olmo3_7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/scratch/forgetting-llms/runs}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/scratch/forgetting-llms/manual_runs}"
DATASETS=(gsm8k math triviaqa)

for dataset in "${DATASETS[@]}"; do
    RUN_ROOT="$RUNS_ROOT/olmo3_7b_baseline_${dataset}"
    sbatch "$SCRIPT_DIR/run_vllm_and_runner_sbatch.sh" \
        --model "$MODEL_PATH" \
        --run-root "$RUN_ROOT" \
        --max-model-len 8192 \
        -- \
        python "$SCRIPT_DIR/prime_rl_runner.py" baseline \
            --model "$MODEL_PATH" \
            --dataset "$dataset" \
            --output-root "$OUTPUT_ROOT" \
            --hf-home "${HF_HOME:-$HOME/scratch/huggingface}" \
            --rollouts-per-prompt 8 \
            --temperature 1.0 \
            --top-p 1.0 \
            --max-model-len 8192 \
            --max-tokens 1024 \
            --wandb-mode online
done
