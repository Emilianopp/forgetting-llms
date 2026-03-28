#!/bin/bash
# Evaluate training checkpoints on the IN-DISTRIBUTION task they were trained on.
# Uses vLLM for fast generation + unified_reward.py for grading.
#
# Usage:
#   sbatch scripts/eval_task_accuracy.sh <checkpoint_dir> <dataset> [base_model] [results_tag]
#
# Arguments:
#   $1 = checkpoint directory (e.g., ~/scratch/forgetting-llms/checkpoints/gt_sft_qwen3_1.7b_gsm8k)
#   $2 = dataset name: gsm8k, math, triviaqa
#   $3 = base model HF ID (default: Qwen/Qwen3-1.7B)
#   $4 = optional results tag appended to the output filename
#
# Output: ~/scratch/forgetting-llms/eval_results/<basename>/task_accuracy[_{tag}].json

#SBATCH --job-name=task-acc
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -uxo pipefail

# --- Environment ---
module load python/3.10
if [ -f "$HOME/forgetting-llms/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$HOME/forgetting-llms/.venv/bin/activate"
else
    # shellcheck disable=SC1091
    source "$HOME/envs/forgetting/bin/activate"
fi
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Configuration ---
CKPT_DIR=${1:?Usage: eval_task_accuracy.sh <checkpoint_dir> <dataset> [base_model] [results_tag]}
DATASET=${2:?Usage: eval_task_accuracy.sh <checkpoint_dir> <dataset> [base_model] [results_tag]}
BASE_MODEL=${3:-"Qwen/Qwen3-1.7B"}
RESULTS_TAG=${4:-""}
RESULTS_NAME=$(basename "$CKPT_DIR")
RESULTS_DIR=~/scratch/forgetting-llms/eval_results/$RESULTS_NAME
if [ -n "$RESULTS_TAG" ]; then
    OUTPUT_PATH="$RESULTS_DIR/task_accuracy_${RESULTS_TAG}.json"
else
    OUTPUT_PATH="$RESULTS_DIR/task_accuracy.json"
fi
REPO_DIR=$HOME/forgetting-llms

mkdir -p slurm_logs
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  Task Accuracy Evaluation"
echo "========================================="
echo "Checkpoint dir: $CKPT_DIR"
echo "Dataset:        $DATASET"
echo "Base model:     $BASE_MODEL"
echo "Results path:   $OUTPUT_PATH"
echo "========================================="

# --- Run evaluation ---
python "$REPO_DIR/scripts/eval_task_accuracy.py" \
    --checkpoint_dir "$CKPT_DIR" \
    --dataset "$DATASET" \
    --base_model "$BASE_MODEL" \
    --output_path "$OUTPUT_PATH"

echo ""
echo "========================================="
echo "  Task Accuracy Evaluation Complete"
echo "========================================="
echo "Results: $OUTPUT_PATH"
echo "========================================="
