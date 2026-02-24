#!/bin/bash
# SFT Training: VeRL fsdp_sft_trainer on 2x A100 80GB
#
# Usage:
#   sbatch scripts/run_sft.sh <dataset> <model> <experiment_name> <data_variant>
#
# Arguments:
#   $1 = dataset name (gsm8k, math, triviaqa)
#   $2 = model path (HF model ID or local path, e.g. Qwen/Qwen3-1.7B)
#   $3 = experiment name (e.g. gt_sft_qwen3_1.7b_gsm8k)
#   $4 = data variant (gt, sf, cf — determines which data dir to use)
#
# Examples:
#   sbatch scripts/run_sft.sh gsm8k Qwen/Qwen3-1.7B gt_sft_qwen3_1.7b_gsm8k gt
#   sbatch scripts/run_sft.sh gsm8k Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_gsm8k sf

#SBATCH --job-name=sft-train
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

# --- Validate arguments ---
DATASET=${1:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}
MODEL=${2:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}
EXPERIMENT_NAME=${3:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}
DATA_VARIANT=${4:?Usage: run_sft.sh <dataset> <model> <experiment_name> <data_variant>}

# --- Environment ---
module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Paths ---
# Data variant determines directory suffix: gt -> _sft, sf -> _sf_sft, cf -> _cf_sft
case "$DATA_VARIANT" in
    gt) DATA_DIR=~/scratch/forgetting-llms/data/${DATASET}_sft ;;
    sf) DATA_DIR=~/scratch/forgetting-llms/data/${DATASET}_sf_sft ;;
    cf) DATA_DIR=~/scratch/forgetting-llms/data/${DATASET}_cf_sft ;;
    *)  echo "ERROR: Unknown data variant '$DATA_VARIANT'. Use gt, sf, or cf."; exit 1 ;;
esac

# --- Per-dataset config ---
case "$DATASET" in
    gsm8k)       MAX_LENGTH=2048; TOTAL_EPOCHS=3 ;;
    math)        MAX_LENGTH=3072; TOTAL_EPOCHS=3 ;;
    triviaqa)    MAX_LENGTH=512;  TOTAL_EPOCHS=3 ;;
    *)           MAX_LENGTH=2048; TOTAL_EPOCHS=3 ;;
esac

SAVE_DIR=~/scratch/forgetting-llms/checkpoints/${EXPERIMENT_NAME}
REPO_DIR=$HOME/forgetting-llms

mkdir -p slurm_logs
mkdir -p "$SAVE_DIR"

# --- Verify data exists ---
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/train.parquet"
    echo "For GT-SFT, run: python scripts/preprocess_data.py --dataset $DATASET --format sft"
    echo "For SF-SFT, run: sbatch scripts/generate_trajectories.sh"
    exit 1
fi

echo "========================================="
echo "  SFT Training — $EXPERIMENT_NAME"
echo "========================================="
echo "Model:    $MODEL"
echo "Dataset:  $DATASET (variant: $DATA_VARIANT)"
echo "Data:     $DATA_DIR"
echo "Save:     $SAVE_DIR"
echo "GPUs:     2x A100 80GB"
echo "========================================="

# --- Launch SFT training ---
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys="['question']" \
    +data.response_dict_keys="['answer']" \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=$MAX_LENGTH \
    model.partial_pretrain=$MODEL \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.model_dtype=bf16 \
    model.lora_rank=0 \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.lr_scheduler=cosine \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=forgetting-llms \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$SAVE_DIR

echo "========================================="
echo "  SFT Training Complete"
echo "========================================="
echo "Checkpoints: $SAVE_DIR"
echo "========================================="
