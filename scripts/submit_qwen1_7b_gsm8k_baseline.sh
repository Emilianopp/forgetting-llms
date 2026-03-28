#!/usr/bin/env bash
# Turnkey batch submission for the same baseline experiment as
# launch_qwen1_7b_gsm8k_baseline_interactive.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL_PATH="${MODEL_PATH:-$HOME/scratch/qwen1.7b}"
DATASET="${DATASET:-gsm8k}"
RUN_ROOT="${RUN_ROOT:-$HOME/scratch/forgetting-llms/manual_runs/baseline_qwen1_7b_gsm8k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/scratch/forgetting-llms/runs}"
PORT="${PORT:-8000}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
WANDB_MODE="${WANDB_MODE:-online}"

sbatch "$SCRIPT_DIR/run_vllm_and_runner_sbatch.sh" \
  --model "$MODEL_PATH" \
  --run-root "$RUN_ROOT" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  -- \
  python "$SCRIPT_DIR/prime_rl_runner.py" baseline \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-root "$OUTPUT_ROOT" \
    --hf-home "${HF_HOME:-$HOME/scratch/huggingface}" \
    --rollouts-per-prompt "$ROLLOUTS_PER_PROMPT" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --wandb-mode "$WANDB_MODE"
