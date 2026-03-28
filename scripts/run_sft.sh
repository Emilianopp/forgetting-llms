#!/usr/bin/env bash
# Native parquet-backed SFT training wrapper.
#
# Usage:
#   bash scripts/run_sft.sh <dataset> <model> <experiment_name> <data_variant>
#
# Arguments:
#   $1 = dataset name (gsm8k, math, triviaqa, synthetic2_sft_verified, ...)
#   $2 = model path (HF model ID or local path)
#   $3 = experiment name
#   $4 = data variant (gt, sf, cf)
#
# Optional checkpoint mirroring:
#   CHECKPOINT_MIRROR_ROOT=/path/to/mounted/google-drive/root or gdrive:forgetting-llms-backups
#   CHECKPOINT_MIRROR_SOURCE_BASE=~/scratch/forgetting-llms
#   CHECKPOINT_MIRROR_POLL_SECS=300
#   CHECKPOINT_MIRROR_PRUNE_DEST=0

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"

DATASET=${1:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}
MODEL=${2:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}
EXPERIMENT_NAME=${3:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}
DATA_VARIANT=${4:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}

module load python/3.10 >/dev/null 2>&1 || true
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    :
elif [[ -n "${VENV_DIR:-}" && -x "$VENV_DIR/bin/python" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
elif [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
else
    # shellcheck disable=SC1091
    source "$HOME/envs/forgetting/bin/activate"
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/forgetting-llms/wandb/${EXPERIMENT_NAME}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/forgetting-llms/wandb_cache/${EXPERIMENT_NAME}}"
unset ROCR_VISIBLE_DEVICES

if [[ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"
fi

case "$DATA_VARIANT" in
    gt) DATA_DIR="${DATA_DIR:-$HOME/scratch/forgetting-llms/data/${DATASET}_sft}" ;;
    sf) DATA_DIR="${DATA_DIR:-$HOME/scratch/forgetting-llms/data/${DATASET}_sf_sft}" ;;
    cf) DATA_DIR="${DATA_DIR:-$HOME/scratch/forgetting-llms/data/${DATASET}_cf_sft}" ;;
    *)  echo "ERROR: Unknown data variant '$DATA_VARIANT'. Use gt, sf, or cf." >&2; exit 1 ;;
esac

TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
EVAL_FILE="${EVAL_FILE:-$DATA_DIR/test.parquet}"
SAVE_DIR="${SAVE_DIR:-$HOME/scratch/forgetting-llms/checkpoints/${EXPERIMENT_NAME}}"

case "$DATASET" in
    gsm8k) DEFAULT_MAX_LENGTH=2304 ;;
    math|polaris_math|openr1_math) DEFAULT_MAX_LENGTH=3200 ;;
    triviaqa) DEFAULT_MAX_LENGTH=2176 ;;
    synthetic2_sft_verified|dolci_think_sft_7b|dolci_think_sft_32b|tau2bench) DEFAULT_MAX_LENGTH=8192 ;;
    olmo_rl_zero_math|dolci_rl_zero_math) DEFAULT_MAX_LENGTH=3072 ;;
    *) DEFAULT_MAX_LENGTH=4096 ;;
esac
if [[ "$DATA_VARIANT" == "gt" ]]; then
    case "$DATASET" in
        gsm8k) DEFAULT_MAX_LENGTH=2048 ;;
        math|polaris_math|openr1_math|olmo_rl_zero_math|dolci_rl_zero_math) DEFAULT_MAX_LENGTH=3072 ;;
        triviaqa) DEFAULT_MAX_LENGTH=512 ;;
    esac
fi

MAX_LENGTH="${MAX_LENGTH:-$DEFAULT_MAX_LENGTH}"
MAX_STEPS="${MAX_STEPS:--1}"
EPOCHS="${EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-forgetting-llms}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
QUESTION_FIELD="${QUESTION_FIELD:-extra_info.question}"
ANSWER_FIELD="${ANSWER_FIELD:-extra_info.answer}"
CHECKPOINT_MIRROR_ROOT="${CHECKPOINT_MIRROR_ROOT:-}"
CHECKPOINT_MIRROR_SOURCE_BASE="${CHECKPOINT_MIRROR_SOURCE_BASE:-$HOME/scratch/forgetting-llms}"
CHECKPOINT_MIRROR_POLL_SECS="${CHECKPOINT_MIRROR_POLL_SECS:-300}"
CHECKPOINT_MIRROR_PRUNE_DEST="${CHECKPOINT_MIRROR_PRUNE_DEST:-0}"
CHECKPOINT_MIRROR_STOP_FILE=""
CHECKPOINT_MIRROR_PID=""

cleanup() {
    local exit_code="$1"
    if [[ -n "$CHECKPOINT_MIRROR_PID" ]]; then
        touch "$CHECKPOINT_MIRROR_STOP_FILE" 2>/dev/null || true
        wait "$CHECKPOINT_MIRROR_PID" || true
    fi
    if [[ -n "$CHECKPOINT_MIRROR_STOP_FILE" ]]; then
        rm -f "$CHECKPOINT_MIRROR_STOP_FILE"
    fi
    return 0
}
trap 'cleanup "$?"' EXIT

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: Training data not found at $TRAIN_FILE" >&2
    exit 1
fi
mkdir -p "$SAVE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

if [[ -n "$CHECKPOINT_MIRROR_ROOT" ]]; then
    CHECKPOINT_MIRROR_STOP_FILE="${TMPDIR:-/tmp}/forgetting_llms_${EXPERIMENT_NAME}_$$.mirror.stop"
    rm -f "$CHECKPOINT_MIRROR_STOP_FILE"
    env \
        MIRROR_STOP_FILE="$CHECKPOINT_MIRROR_STOP_FILE" \
        MIRROR_POLL_SECS="$CHECKPOINT_MIRROR_POLL_SECS" \
        MIRROR_SOURCE_BASE="$CHECKPOINT_MIRROR_SOURCE_BASE" \
        MIRROR_PRUNE_DEST="$CHECKPOINT_MIRROR_PRUNE_DEST" \
        bash "$SCRIPT_DIR/watch_directory_mirror.sh" \
          "$SAVE_DIR" \
          "$CHECKPOINT_MIRROR_ROOT" \
          "sft:$EXPERIMENT_NAME" &
    CHECKPOINT_MIRROR_PID=$!
fi

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    WORLD_SIZE="$NPROC_PER_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    WORLD_SIZE=$(python - <<'PY'
import os
value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
parts = [item for item in value.split(",") if item.strip()]
print(len(parts) if parts else 1)
PY
)
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    WORLD_SIZE="$SLURM_GPUS_ON_NODE"
else
    WORLD_SIZE=1
fi

echo "========================================="
echo "  Native SFT Training — $EXPERIMENT_NAME"
echo "========================================="
echo "Model:      $MODEL"
echo "Dataset:    $DATASET (variant: $DATA_VARIANT)"
echo "Train file: $TRAIN_FILE"
echo "Eval file:  $EVAL_FILE"
echo "Save dir:   $SAVE_DIR"
echo "CUDA vis:   ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "World size: $WORLD_SIZE"
echo "Max length: $MAX_LENGTH"
echo "Max steps:  $MAX_STEPS"
echo "LoRA rank:  $LORA_RANK"
if [[ -n "$CHECKPOINT_MIRROR_ROOT" ]]; then
    echo "Mirror root: $CHECKPOINT_MIRROR_ROOT"
fi
echo "========================================="

TORCHRUN_ARGS=(--standalone --nnodes=1 --nproc_per_node="$WORLD_SIZE")

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    GC_FLAG="--gradient-checkpointing"
else
    GC_FLAG="--no-gradient-checkpointing"
fi

CMD=(
    torchrun "${TORCHRUN_ARGS[@]}"
    "$REPO_DIR/src/training/plain_sft.py"
    --train-file "$TRAIN_FILE"
    --model "$MODEL"
    --output-dir "$SAVE_DIR"
    --run-name "$EXPERIMENT_NAME"
    --question-field "$QUESTION_FIELD"
    --answer-field "$ANSWER_FIELD"
    --max-length "$MAX_LENGTH"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --epochs "$EPOCHS"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --eval-steps "$EVAL_STEPS"
    --logging-steps "$LOGGING_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --mixed-precision "$MIXED_PRECISION"
    "$GC_FLAG"
    --lora-rank "$LORA_RANK"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --wandb-project "$WANDB_PROJECT"
    --wandb-mode "$WANDB_MODE"
)

if [[ -f "$EVAL_FILE" ]]; then
    CMD+=(--eval-file "$EVAL_FILE")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
fi

"${CMD[@]}"

echo "========================================="
echo "  Native SFT Training Complete"
echo "========================================="
echo "Checkpoints: $SAVE_DIR"
echo "========================================="
