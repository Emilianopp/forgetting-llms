#!/bin/bash
# Sequential GRPO: Parameterized script for chaining RL runs across datasets.
#
# Usage:
#   sbatch scripts/run_grpo_sequential.sh <dataset> <model_path> <experiment_name>
#
# Arguments:
#   $1 = dataset name: gsm8k, math, codecontest, naturalquestions
#   $2 = model path: HF model ID (e.g. Qwen/Qwen3-1.7B) or local checkpoint dir
#   $3 = experiment name for WandB (e.g. grpo_gsm8k_then_math)
#
# Chaining example:
#   JOB1=$(sbatch --parsable scripts/run_grpo_sequential.sh gsm8k Qwen/Qwen3-1.7B grpo_gsm8k)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_grpo_sequential.sh math \
#          ~/scratch/forgetting-llms/checkpoints/grpo_gsm8k/best_merged grpo_gsm8k_then_math)

#SBATCH --job-name=grpo-seq
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

# --- Validate arguments ---
DATASET="${1:?Usage: $0 <dataset> <model_path> <experiment_name>}"
MODEL_PATH="${2:?Usage: $0 <dataset> <model_path> <experiment_name>}"
EXPERIMENT_NAME="${3:?Usage: $0 <dataset> <model_path> <experiment_name>}"

# --- Environment ---
module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Paths ---
DATA_DIR=~/scratch/forgetting-llms/data/$DATASET
SAVE_DIR=~/scratch/forgetting-llms/checkpoints/$EXPERIMENT_NAME
REPO_DIR=$HOME/forgetting-llms

mkdir -p slurm_logs
mkdir -p "$SAVE_DIR"

# --- Detect and handle FSDP checkpoints ---
# If MODEL_PATH points to an FSDP checkpoint (has actor/ subdir), merge it first
RESOLVED_MODEL="$MODEL_PATH"
if [ -d "$MODEL_PATH/actor" ]; then
    MERGED_DIR="${MODEL_PATH}/actor_merged"
    if [ ! -d "$MERGED_DIR" ] || [ -z "$(ls -A "$MERGED_DIR" 2>/dev/null)" ]; then
        echo "Detected FSDP checkpoint. Merging to HF format..."
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$MODEL_PATH/actor" \
            --target_dir "$MERGED_DIR"
    fi
    RESOLVED_MODEL="$MERGED_DIR"
    echo "Using merged model: $RESOLVED_MODEL"
fi

# --- Preprocess data if needed ---
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Preprocessing $DATASET data..."
    python "$REPO_DIR/scripts/preprocess_data.py" \
        --dataset "$DATASET" \
        --output_dir "$DATA_DIR"
fi

# --- Select reward function based on dataset ---
case "$DATASET" in
    gsm8k|math)
        REWARD_PATH="$REPO_DIR/src/rewards/math_reward.py"
        ;;
    *)
        REWARD_PATH="$REPO_DIR/src/rewards/unified_reward.py"
        ;;
esac

echo "========================================="
echo "  Sequential GRPO Run"
echo "========================================="
echo "Dataset:    $DATASET"
echo "Model:      $RESOLVED_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "Data:       $DATA_DIR"
echo "Save:       $SAVE_DIR"
echo "Reward:     $REWARD_PATH"
echo "GPUs:       2x A100 80GB"
echo "========================================="

# --- Launch GRPO training ---
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$RESOLVED_MODEL" \
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
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=forgetting-llms \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$SAVE_DIR"

echo "========================================="
echo "  GRPO Sequential Run Complete"
echo "========================================="
echo "Checkpoints: $SAVE_DIR"
