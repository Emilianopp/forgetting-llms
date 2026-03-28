#!/bin/bash
# Self-distillation training launcher using the native trainer in src/training/self_distill.py.
#
# Usage:
#   sbatch scripts/run_self_distill.sh <train_parquet> <student_model> <teacher_model> <experiment_name>
#
# Environment overrides:
#   QUESTION_FIELD                Default: extra_info.question
#   ANSWER_FIELD                  Default: extra_info.answer (legacy existing-trace mode / privileged fallback)
#   GROUND_TRUTH_FIELD            Default: extra_info.ground_truth
#   DATA_SOURCE_FIELD             Default: data_source
#   TRACE_SOURCE                  Default: generate
#   PRIVILEGED_SOURCE             Default: auto
#   STUDENT_TEMPLATE              Default: internal tagged-answer template
#   TEACHER_TEMPLATE              Default: internal privileged tagged-answer template
#   STUDENT_MAX_NEW_TOKENS        Default: 1024
#   TEACHER_MAX_NEW_TOKENS        Default: 1024
#   STUDENT_TEMPERATURE           Default: 1.0
#   TEACHER_TEMPERATURE           Default: 1.0
#   STUDENT_TOP_P                 Default: 1.0
#   TEACHER_TOP_P                 Default: 1.0
#   TEACHER_MAX_ATTEMPTS          Default: 4
#   STUDENT_DEVICE                Default: cuda:0
#   TEACHER_DEVICE                Default: cuda:1
#   TEACHER_SYNC_MODE             Default: step
#   TEACHER_SYNC_EVERY            Default: 1
#   PARALLEL_TRACE_GENERATION     Default: 1
#   KL_MODE                       Default: forward
#   KL_INTERP_ALPHA               Default: 0.5
#   KL_TOKEN_CHUNK_SIZE           Default: 128
#   CE_WEIGHT                     Default: 1.0
#   DISTILL_WEIGHT                Default: 1.0
#   DISTILL_TEMPERATURE           Default: 1.0
#   EPOCHS                        Default: 1
#   MAX_STEPS                     Default: empty (use full epochs)
#   PER_DEVICE_BATCH_SIZE         Default: 1
#   GRAD_ACCUM_STEPS              Default: 1
#   LEARNING_RATE                 Default: 1e-5
#   WEIGHT_DECAY                  Default: 0.01
#   WARMUP_RATIO                  Default: 0.05
#   LR_SCHEDULER_TYPE             Default: cosine
#   MAX_LENGTH                    Default: 8192
#   GRADIENT_CHECKPOINTING        Default: 1
#   SAVE_STEPS                    Default: 100
#   LOGGING_STEPS                 Default: 10
#   LORA_RANK                     Default: 0
#   LORA_ALPHA                    Default: 16
#   LORA_DROPOUT                  Default: 0.0
#   LORA_TARGET_MODULES           Default: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
#   MIXED_PRECISION               Default: bf16
#   WANDB_MODE                    Default: online
#   WANDB_PROJECT                 Default: forgetting-llms
#   CHECKPOINT_MIRROR_ROOT        Optional backup root: local path or rclone remote like gdrive:forgetting-llms-backups
#   CHECKPOINT_MIRROR_SOURCE_BASE Default: ~/scratch/forgetting-llms
#   CHECKPOINT_MIRROR_POLL_SECS   Default: 300
#   CHECKPOINT_MIRROR_PRUNE_DEST  Default: 0

#SBATCH --job-name=self-distill
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euo pipefail

TRAIN_FILE=${1:?Usage: run_self_distill.sh <train_parquet> <student_model> <teacher_model> <experiment_name>}
STUDENT_MODEL=${2:?Usage: run_self_distill.sh <train_parquet> <student_model> <teacher_model> <experiment_name>}
TEACHER_MODEL=${3:?Usage: run_self_distill.sh <train_parquet> <student_model> <teacher_model> <experiment_name>}
EXPERIMENT_NAME=${4:?Usage: run_self_distill.sh <train_parquet> <student_model> <teacher_model> <experiment_name>}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/scratch/forgetting-llms/checkpoints/$EXPERIMENT_NAME}"

QUESTION_FIELD="${QUESTION_FIELD:-extra_info.question}"
ANSWER_FIELD="${ANSWER_FIELD:-extra_info.answer}"
GROUND_TRUTH_FIELD="${GROUND_TRUTH_FIELD:-extra_info.ground_truth}"
DATA_SOURCE_FIELD="${DATA_SOURCE_FIELD:-data_source}"
TRACE_SOURCE="${TRACE_SOURCE:-generate}"
PRIVILEGED_SOURCE="${PRIVILEGED_SOURCE:-auto}"
STUDENT_TEMPLATE="${STUDENT_TEMPLATE:-}"
TEACHER_TEMPLATE="${TEACHER_TEMPLATE:-}"
STUDENT_MAX_NEW_TOKENS="${STUDENT_MAX_NEW_TOKENS:-1024}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-1.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
STUDENT_TOP_P="${STUDENT_TOP_P:-1.0}"
TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
TEACHER_MAX_ATTEMPTS="${TEACHER_MAX_ATTEMPTS:-4}"
STUDENT_DEVICE="${STUDENT_DEVICE:-cuda:0}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:1}"
TEACHER_SYNC_MODE="${TEACHER_SYNC_MODE:-step}"
TEACHER_SYNC_EVERY="${TEACHER_SYNC_EVERY:-1}"
PARALLEL_TRACE_GENERATION="${PARALLEL_TRACE_GENERATION:-1}"
KL_MODE="${KL_MODE:-forward}"
KL_INTERP_ALPHA="${KL_INTERP_ALPHA:-0.5}"
KL_TOKEN_CHUNK_SIZE="${KL_TOKEN_CHUNK_SIZE:-128}"
CE_WEIGHT="${CE_WEIGHT:-1.0}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-forgetting-llms}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
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

module load python/3.10
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    :
elif [[ -n "${VENV_DIR:-}" && -x "${VENV_DIR/#\~/$HOME}/bin/python" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR/#\~/$HOME}/bin/activate"
elif [ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
else
    echo "No project venv found. Expected \$VIRTUAL_ENV, \$VENV_DIR, $REPO_DIR/.venv, or ~/scratch/forgetting-llms/.venv" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/forgetting-llms/wandb/$EXPERIMENT_NAME}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/forgetting-llms/wandb_cache/$EXPERIMENT_NAME}"
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

mkdir -p slurm_logs "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

if [[ -n "$CHECKPOINT_MIRROR_ROOT" ]]; then
    CHECKPOINT_MIRROR_STOP_FILE="${TMPDIR:-/tmp}/forgetting_llms_${EXPERIMENT_NAME}_$$.mirror.stop"
    rm -f "$CHECKPOINT_MIRROR_STOP_FILE"
    env \
        MIRROR_STOP_FILE="$CHECKPOINT_MIRROR_STOP_FILE" \
        MIRROR_POLL_SECS="$CHECKPOINT_MIRROR_POLL_SECS" \
        MIRROR_SOURCE_BASE="$CHECKPOINT_MIRROR_SOURCE_BASE" \
        MIRROR_PRUNE_DEST="$CHECKPOINT_MIRROR_PRUNE_DEST" \
        bash "$SCRIPT_DIR/watch_directory_mirror.sh" \
          "$OUTPUT_DIR" \
          "$CHECKPOINT_MIRROR_ROOT" \
          "self_distill:$EXPERIMENT_NAME" &
    CHECKPOINT_MIRROR_PID=$!
fi

CMD=(
    python
    "$REPO_DIR/src/training/self_distill.py"
    --train-file "$TRAIN_FILE"
    --student-model "$STUDENT_MODEL"
    --teacher-model "$TEACHER_MODEL"
    --output-dir "$OUTPUT_DIR"
    --run-name "$EXPERIMENT_NAME"
    --question-field "$QUESTION_FIELD"
    --answer-field "$ANSWER_FIELD"
    --ground-truth-field "$GROUND_TRUTH_FIELD"
    --data-source-field "$DATA_SOURCE_FIELD"
    --trace-source "$TRACE_SOURCE"
    --privileged-source "$PRIVILEGED_SOURCE"
    --student-max-new-tokens "$STUDENT_MAX_NEW_TOKENS"
    --teacher-max-new-tokens "$TEACHER_MAX_NEW_TOKENS"
    --student-temperature "$STUDENT_TEMPERATURE"
    --teacher-temperature "$TEACHER_TEMPERATURE"
    --student-top-p "$STUDENT_TOP_P"
    --teacher-top-p "$TEACHER_TOP_P"
    --teacher-max-attempts "$TEACHER_MAX_ATTEMPTS"
    --student-device "$STUDENT_DEVICE"
    --teacher-device "$TEACHER_DEVICE"
    --teacher-sync-mode "$TEACHER_SYNC_MODE"
    --teacher-sync-every "$TEACHER_SYNC_EVERY"
    --epochs "$EPOCHS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --lr-scheduler-type "$LR_SCHEDULER_TYPE"
    --max-length "$MAX_LENGTH"
    --save-steps "$SAVE_STEPS"
    --logging-steps "$LOGGING_STEPS"
    --ce-weight "$CE_WEIGHT"
    --distill-weight "$DISTILL_WEIGHT"
    --distill-temperature "$DISTILL_TEMPERATURE"
    --kl-token-chunk-size "$KL_TOKEN_CHUNK_SIZE"
    --kl-mode "$KL_MODE"
    --kl-interp-alpha "$KL_INTERP_ALPHA"
    --lora-rank "$LORA_RANK"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --lora-target-modules "$LORA_TARGET_MODULES"
    --mixed-precision "$MIXED_PRECISION"
    --wandb-project "$WANDB_PROJECT"
    --wandb-mode "$WANDB_MODE"
)

if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "$STUDENT_TEMPLATE" ]]; then
    CMD+=(--student-template "$STUDENT_TEMPLATE")
fi
if [[ -n "$TEACHER_TEMPLATE" ]]; then
    CMD+=(--teacher-template "$TEACHER_TEMPLATE")
fi
if [[ -n "$MAX_STEPS" ]]; then
    CMD+=(--max-steps "$MAX_STEPS")
fi
if [[ "$PARALLEL_TRACE_GENERATION" == "1" ]]; then
    CMD+=(--parallel-trace-generation)
else
    CMD+=(--no-parallel-trace-generation)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    CMD+=(--gradient-checkpointing)
fi

echo "========================================="
echo "  Self-Distillation Training — $EXPERIMENT_NAME"
echo "========================================="
echo "Train file:    $TRAIN_FILE"
echo "Student model: $STUDENT_MODEL"
echo "Teacher model: $TEACHER_MODEL"
echo "Output dir:    $OUTPUT_DIR"
echo "Trace source:  $TRACE_SOURCE"
echo "Student dev:   $STUDENT_DEVICE"
echo "Teacher dev:   $TEACHER_DEVICE"
echo "KL mode:       $KL_MODE"
echo "LoRA rank:     $LORA_RANK"
if [[ -n "$CHECKPOINT_MIRROR_ROOT" ]]; then
    echo "Mirror root:   $CHECKPOINT_MIRROR_ROOT"
fi
echo "========================================="

"${CMD[@]}"
