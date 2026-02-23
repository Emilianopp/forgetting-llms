#!/bin/bash
# GRPO Smoke Test: Qwen3-1.7B + GSM8K on 2x A100 80GB
#
# This verifies the VeRL + GRPO training pipeline works end-to-end.
# Run from the repo root on Mila:
#   sbatch scripts/run_grpo_smoke_test.sh
#
# Prerequisites:
#   1. Environment set up: bash scripts/setup_env.sh
#   2. Data preprocessed: python scripts/preprocess_data.py --dataset gsm8k

#SBATCH --job-name=grpo-smoke-test
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

# --- Environment ---
module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
# Mila sets ROCR_VISIBLE_DEVICES (AMD ROCm) even on NVIDIA nodes,
# which conflicts with CUDA_VISIBLE_DEVICES in VeRL's worker init
unset ROCR_VISIBLE_DEVICES

# --- Paths ---
DATA_DIR=~/scratch/forgetting-llms/data/gsm8k
SAVE_DIR=~/scratch/forgetting-llms/checkpoints/grpo_smoke_test
REPO_DIR=$HOME/forgetting-llms

# --- Preprocess data if needed ---
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Preprocessing GSM8K data..."
    python $REPO_DIR/scripts/preprocess_data.py \
        --dataset gsm8k \
        --output_dir $DATA_DIR
fi

mkdir -p slurm_logs
mkdir -p $SAVE_DIR

echo "=== GRPO Smoke Test ==="
echo "Model: Qwen/Qwen3-1.7B"
echo "Data:  GSM8K ($DATA_DIR)"
echo "Save:  $SAVE_DIR"
echo "GPUs:  2x A100 80GB"
echo "======================="

# --- Launch GRPO training ---
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$REPO_DIR/src/rewards/math_reward.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=forgetting-llms \
    trainer.experiment_name=grpo_smoke_test_qwen3_1.7b_gsm8k \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=$SAVE_DIR
