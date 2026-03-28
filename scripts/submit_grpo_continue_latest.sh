#!/bin/bash
# Submit a GRPO continuation job from the latest checkpoint of a previous run.
#
# Usage:
#   bash scripts/submit_grpo_continue_latest.sh <previous_experiment> <mode> <target> <next_experiment>
#
# Modes:
#   dataset   target is a dataset name for run_grpo_sequential.sh
#   data_dir  target is an explicit dataset directory for run_grpo_dataset_dir.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/require_prime_only.sh"

PREV_EXPERIMENT="${1:?Usage: $0 <previous_experiment> <mode> <target> <next_experiment>}"
MODE="${2:?Usage: $0 <previous_experiment> <mode> <target> <next_experiment>}"
TARGET="${3:?Usage: $0 <previous_experiment> <mode> <target> <next_experiment>}"
NEXT_EXPERIMENT="${4:?Usage: $0 <previous_experiment> <mode> <target> <next_experiment>}"

REPO_DIR=$HOME/forgetting-llms
CKPT_ROOT=~/scratch/forgetting-llms/checkpoints/$PREV_EXPERIMENT
LATEST_CKPT=$(find "$CKPT_ROOT" -maxdepth 1 -name "global_step_*" -type d | sort -t_ -k3 -n | tail -n 1)

if [ -z "${LATEST_CKPT:-}" ]; then
    echo "No checkpoint found under $CKPT_ROOT" >&2
    exit 1
fi

case "$MODE" in
    dataset)
        sbatch "$REPO_DIR/scripts/run_grpo_sequential.sh" "$TARGET" "$LATEST_CKPT" "$NEXT_EXPERIMENT"
        ;;
    data_dir)
        DATA_LABEL=$(basename "$TARGET")
        MAX_PROMPT=${MAX_PROMPT:-512} MAX_RESPONSE=${MAX_RESPONSE:-1024} TOTAL_EPOCHS=${TOTAL_EPOCHS:-15} \
        sbatch "$REPO_DIR/scripts/run_grpo_dataset_dir.sh" "$DATA_LABEL" "$TARGET" "$LATEST_CKPT" "$NEXT_EXPERIMENT"
        ;;
    *)
        echo "Unknown mode '$MODE'. Use dataset or data_dir." >&2
        exit 1
        ;;
esac
