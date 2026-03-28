#!/bin/bash
# Build two IID shards from a matched pooled pair and run staged GRPO:
# shard1 then shard2.
#
# Usage:
#   bash scripts/run_grpo_iid_control.sh <dataset_a> <dataset_b> <model_path> <experiment_prefix>

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/require_prime_only.sh"

DATASET_A="${1:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_prefix>}"
DATASET_B="${2:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_prefix>}"
MODEL_PATH="${3:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_prefix>}"
EXPERIMENT_PREFIX="${4:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_prefix>}"
RUN_DIRECT="${RUN_DIRECT:-0}"

REPO_DIR=$HOME/forgetting-llms
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
elif [ -f "$HOME/envs/forgetting/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$HOME/envs/forgetting/bin/activate"
fi
DATA_ROOT=~/scratch/forgetting-llms/data
PAIR_NAME="${DATASET_A}_${DATASET_B}"
IID_ROOT="$DATA_ROOT/${PAIR_NAME}_iid"

python "$REPO_DIR/scripts/build_iid_control_dataset.py" \
    --dataset-a "$DATASET_A" \
    --dataset-b "$DATASET_B" \
    --output-root "$IID_ROOT"

STAGE1_NAME="${EXPERIMENT_PREFIX}_iid_stage1"
STAGE2_NAME="${EXPERIMENT_PREFIX}_iid_stage2"
if [ "$RUN_DIRECT" = "1" ]; then
    CKPT_ROOT=~/scratch/forgetting-llms/checkpoints/$STAGE1_NAME
    if compgen -G "$CKPT_ROOT/global_step_*" >/dev/null; then
        echo "Reusing existing stage1 checkpoints under $CKPT_ROOT"
    else
        bash "$REPO_DIR/scripts/run_grpo_dataset_dir.sh" \
            "${PAIR_NAME}_iid_shard1" "$IID_ROOT/shard1" "$MODEL_PATH" "$STAGE1_NAME"
    fi

    LATEST_CKPT=$(find "$CKPT_ROOT" -maxdepth 1 -name "global_step_*" -type d | sort -t_ -k3 -n | tail -n 1)
    if [ -z "${LATEST_CKPT:-}" ]; then
        echo "No checkpoint found under $CKPT_ROOT" >&2
        exit 1
    fi

    bash "$REPO_DIR/scripts/run_grpo_dataset_dir.sh" \
        "${PAIR_NAME}_iid_shard2" "$IID_ROOT/shard2" "$LATEST_CKPT" "$STAGE2_NAME"
else
    STAGE1_JOB=$(sbatch --parsable "$REPO_DIR/scripts/run_grpo_dataset_dir.sh" \
        "${PAIR_NAME}_iid_shard1" "$IID_ROOT/shard1" "$MODEL_PATH" "$STAGE1_NAME")

    echo "Submitted stage1 job: $STAGE1_JOB"
    echo "When stage1 finishes, submit stage2 from the latest checkpoint:"
    echo "  bash $REPO_DIR/scripts/submit_grpo_continue_latest.sh $STAGE1_NAME data_dir $IID_ROOT/shard2 $STAGE2_NAME"
fi
