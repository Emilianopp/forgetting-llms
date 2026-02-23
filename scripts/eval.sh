#!/bin/bash
# Slurm evaluation launcher
# Usage: sbatch scripts/eval.sh <run_name>
# Example: sbatch scripts/eval.sh gt_sft_math_base_qwen_3b

#SBATCH --job-name=eval
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

RUN_NAME=${1:?Usage: eval.sh <run_name>}

module load python/3.11 2>/dev/null || true
source $HOME/envs/forgetting/bin/activate

CHECKPOINT_DIR="${SCRATCH}/forgetting-llms/checkpoints/${RUN_NAME}"
RESULTS_DIR="results/${RUN_NAME}"

mkdir -p ${RESULTS_DIR}

echo "=== Running evaluation ==="
echo "Run: ${RUN_NAME}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "=========================="

python src/evaluation/run_eval.py \
    --model_path ${CHECKPOINT_DIR} \
    --suite all \
    --output_dir ${RESULTS_DIR} \
    --run_name ${RUN_NAME}
