#!/bin/bash
# Generic Slurm training launcher
# Usage: sbatch scripts/train.sh <method> <domain> <starting_point> <model_scale>
# Example: sbatch scripts/train.sh gt_sft math base qwen_3b

#SBATCH --job-name=forgetting
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

METHOD=${1:?Usage: train.sh <method> <domain> <starting_point> <model_scale>}
DOMAIN=${2:?Missing domain}
STARTING_POINT=${3:?Missing starting_point (base|safe)}
MODEL_SCALE=${4:?Missing model_scale (qwen_1.5b|qwen_3b|qwen_7b)}

# Update job name
export SLURM_JOB_NAME="forgetting-${METHOD}-${DOMAIN}-${STARTING_POINT}-${MODEL_SCALE}"

module load python/3.11 2>/dev/null || true
source $HOME/envs/forgetting/bin/activate

RUN_NAME="${METHOD}_${DOMAIN}_${STARTING_POINT}_${MODEL_SCALE}"
CHECKPOINT_DIR="${SCRATCH}/forgetting-llms/checkpoints/${RUN_NAME}"

echo "=== Starting training ==="
echo "Method: ${METHOD}"
echo "Domain: ${DOMAIN}"
echo "Starting point: ${STARTING_POINT}"
echo "Model scale: ${MODEL_SCALE}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "========================="

srun torchrun --nproc_per_node=2 src/training/${METHOD}.py \
    --config configs/methods/${METHOD}.yaml \
    --domain configs/domains/${DOMAIN}.yaml \
    --model configs/models/${MODEL_SCALE}.yaml \
    --starting_point ${STARTING_POINT} \
    --output_dir ${CHECKPOINT_DIR} \
    --wandb_project forgetting-llms \
    --wandb_run_name ${RUN_NAME}
