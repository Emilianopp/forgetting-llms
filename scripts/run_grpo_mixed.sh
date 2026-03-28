#!/bin/bash
# Build a balanced mixed dataset for a pair and run GRPO on it.
#
# Usage:
#   sbatch scripts/run_grpo_mixed.sh <dataset_a> <dataset_b> <model_path> <experiment_name>

#SBATCH --job-name=grpo-mixed
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/require_prime_only.sh"

DATASET_A="${1:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_name>}"
DATASET_B="${2:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_name>}"
MODEL_PATH="${3:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_name>}"
EXPERIMENT_NAME="${4:?Usage: $0 <dataset_a> <dataset_b> <model_path> <experiment_name>}"

module load python/3.10
REPO_DIR=$HOME/forgetting-llms
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
else
    # shellcheck disable=SC1091
    source $HOME/envs/forgetting/bin/activate
fi
export HF_HOME=~/scratch/huggingface

PAIR_NAME="${DATASET_A}_${DATASET_B}"
DATA_DIR=~/scratch/forgetting-llms/data/${PAIR_NAME}_mixed

python "$REPO_DIR/scripts/build_mixed_dataset.py" \
    --dataset-a "$DATASET_A" \
    --dataset-b "$DATASET_B" \
    --output-dir "$DATA_DIR"

bash "$REPO_DIR/scripts/run_grpo_dataset_dir.sh" \
    "${PAIR_NAME}_mixed" "$DATA_DIR" "$MODEL_PATH" "$EXPERIMENT_NAME"
