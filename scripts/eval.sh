#!/bin/bash
# Slurm evaluation launcher
# Usage: sbatch scripts/eval.sh <run_name> [base_model]
# Example: sbatch scripts/eval.sh gt_sft_qwen3_1.7b_gsm8k Qwen/Qwen3-1.7B

#SBATCH --job-name=eval
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

RUN_NAME=${1:?Usage: eval.sh <run_name> [base_model]}
BASE_MODEL=${2:-"Qwen/Qwen3-1.7B"}

module load python/3.10
if [ -f "$HOME/forgetting-llms/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$HOME/forgetting-llms/.venv/bin/activate"
else
    # shellcheck disable=SC1091
    source "$HOME/envs/forgetting/bin/activate"
fi
export HF_HOME=~/scratch/huggingface

CHECKPOINT_DIR=~/scratch/forgetting-llms/checkpoints/${RUN_NAME}
RESULTS_DIR=~/scratch/forgetting-llms/eval_results/${RUN_NAME}

mkdir -p ${RESULTS_DIR}

echo "=== Running evaluation ==="
echo "Run: ${RUN_NAME}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Base model: ${BASE_MODEL}"
echo "Results: ${RESULTS_DIR}"
echo "=========================="

python src/evaluation/run_eval.py \
    --model_path ${CHECKPOINT_DIR} \
    --suite all \
    --output_dir ${RESULTS_DIR} \
    --run_name ${RUN_NAME} \
    --base_model ${BASE_MODEL}
