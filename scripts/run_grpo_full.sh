#!/bin/bash
# GRPO Full Run: Qwen3-1.7B + GSM8K on 2x A100 80GB (8 hours)
#
# Full overnight training run. No KL regularization.
# Run from the repo root on Mila:
#   sbatch scripts/run_grpo_full.sh

#SBATCH --job-name=grpo-full
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

# --- Environment ---
module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Paths ---
DATA_DIR=~/scratch/forgetting-llms/data/gsm8k
SAVE_DIR=~/scratch/forgetting-llms/checkpoints/grpo_full_qwen3_1.7b_gsm8k
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

echo "=== GRPO Full Run ==="
echo "Model: Qwen/Qwen3-1.7B"
echo "Data:  GSM8K ($DATA_DIR)"
echo "Save:  $SAVE_DIR"
echo "GPUs:  2x A100 80GB"
echo "Time:  8 hours"
echo "====================="

# --- Launch GRPO training ---
# Key change from smoke test: optimizer_offload=False
# Smoke test OOM'd at 48G because optimizer offload pushes ~14GB to CPU.
# GPU has 30GB headroom (50.5/80 in smoke test), so optimizer fits on GPU.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$REPO_DIR/src/rewards/math_reward.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=forgetting-llms \
    trainer.experiment_name=grpo_full_qwen3_1.7b_gsm8k \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=$SAVE_DIR
