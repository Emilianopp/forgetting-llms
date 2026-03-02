#!/bin/bash
# Evaluate training checkpoints on the IN-DISTRIBUTION task they were trained on.
# Uses vLLM for fast generation + unified_reward.py for grading.
#
# Usage:
#   sbatch scripts/eval_task_accuracy.sh <checkpoint_dir> <dataset> [base_model]
#
# Arguments:
#   $1 = checkpoint directory (e.g., ~/scratch/forgetting-llms/checkpoints/gt_sft_qwen3_1.7b_gsm8k)
#   $2 = dataset name: gsm8k, math, triviaqa
#   $3 = base model HF ID (default: Qwen/Qwen3-1.7B)
#
# Output: ~/scratch/forgetting-llms/eval_results/<basename>/task_accuracy.json

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
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Configuration ---
CKPT_DIR=${1:?Usage: eval_task_accuracy.sh <checkpoint_dir> <dataset> [base_model]}
DATASET=${2:?Usage: eval_task_accuracy.sh <checkpoint_dir> <dataset> [base_model]}
BASE_MODEL=${3:-"Qwen/Qwen3-1.7B"}
RESULTS_NAME=$(basename "$CKPT_DIR")
RESULTS_DIR=~/scratch/forgetting-llms/eval_results/$RESULTS_NAME
REPO_DIR=$HOME/forgetting-llms

mkdir -p slurm_logs
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  Task Accuracy Evaluation"
echo "========================================="
echo "Checkpoint dir: $CKPT_DIR"
echo "Dataset:        $DATASET"
echo "Base model:     $BASE_MODEL"
echo "Results dir:    $RESULTS_DIR"
echo "========================================="

# --- Run evaluation ---
python "$REPO_DIR/scripts/eval_task_accuracy.py" \
    --checkpoint_dir "$CKPT_DIR" \
    --dataset "$DATASET" \
    --base_model "$BASE_MODEL" \
    --output_path "$RESULTS_DIR/task_accuracy.json"

echo ""
echo "========================================="
echo "  Task Accuracy Evaluation Complete"
echo "========================================="
echo "Results: $RESULTS_DIR/task_accuracy.json"
echo "========================================="
