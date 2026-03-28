#!/bin/bash
# Cross-family teacher trajectory generation for CF-SFT.
#
# Usage:
#   sbatch scripts/generate_cross_family_trajectories.sh [model] [dataset] [n_samples]

#SBATCH --job-name=gen-cf-trajectories
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

MODEL=${1:-"meta-llama/Llama-3.1-70B-Instruct"}
DATASET=${2:-"gsm8k"}
N_SAMPLES=${3:-4}

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
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

OUTPUT_DIR=~/scratch/forgetting-llms/data/${DATASET}_cf_sft
mkdir -p slurm_logs

python "$REPO_DIR/src/data/generate_teacher_solutions.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --samples_per_round "$N_SAMPLES" \
    --max_total_samples "${MAX_TOTAL_SAMPLES:-16}" \
    --target_correct_per_question "${TARGET_CORRECT_PER_QUESTION:-2}" \
    --min_correct_per_question "${MIN_CORRECT_PER_QUESTION:-2}" \
    --solutions_per_question "${SOLUTIONS_PER_QUESTION:-2}" \
    --tensor_parallel_size 2 \
    --max_tokens 2048 \
    --chunk_size 500 \
    --temperature "${TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.9}" \
    --output_dir "$OUTPUT_DIR"
