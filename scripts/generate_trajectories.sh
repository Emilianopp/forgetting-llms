#!/bin/bash
# Teacher Trajectory Generation: vLLM batch inference on 2x A100 80GB
#
# Generates teacher solutions from Qwen3-32B for SF-SFT training.
# For each GSM8K question, generates N=4 candidate solutions, keeps the
# first correct one as the training target.
#
# Usage:
#   sbatch scripts/generate_trajectories.sh [model] [dataset] [n_samples]
#
# Arguments (all optional):
#   $1 = teacher model (default: Qwen/Qwen3-32B)
#   $2 = dataset (default: gsm8k)
#   $3 = n_samples per question (default: 4)
#
# Examples:
#   sbatch scripts/generate_trajectories.sh
#   sbatch scripts/generate_trajectories.sh Qwen/Qwen3-32B gsm8k 8

#SBATCH --job-name=gen-trajectories
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

# --- Arguments ---
MODEL=${1:-"Qwen/Qwen3-32B"}
DATASET=${2:-"gsm8k"}
N_SAMPLES=${3:-4}

# --- Environment ---
module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Paths ---
REPO_DIR=$HOME/forgetting-llms
OUTPUT_DIR=~/scratch/forgetting-llms/data/${DATASET}_sf_sft

mkdir -p slurm_logs

echo "========================================="
echo "  Teacher Trajectory Generation"
echo "========================================="
echo "Teacher:   $MODEL"
echo "Dataset:   $DATASET"
echo "N samples: $N_SAMPLES"
echo "Output:    $OUTPUT_DIR"
echo "GPUs:      2x A100 80GB (TP=2)"
echo "========================================="

python $REPO_DIR/src/data/generate_teacher_solutions.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_samples "$N_SAMPLES" \
    --tensor_parallel_size 2 \
    --max_tokens 2048 \
    --chunk_size 500 \
    --output_dir "$OUTPUT_DIR"

echo "========================================="
echo "  Trajectory Generation Complete"
echo "========================================="
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next: submit SF-SFT training:"
echo "  sbatch scripts/run_sft.sh $DATASET Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_${DATASET} sf"
echo "========================================="
