#!/bin/bash
# Generic GRPO launcher for an arbitrary dataset directory containing
# train.parquet and test.parquet in VeRL RL format.
#
# Usage:
#   sbatch scripts/run_grpo_dataset_dir.sh <data_label> <data_dir> <model_path> <experiment_name>

#SBATCH --job-name=grpo-data-dir
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/require_prime_only.sh"

DATA_LABEL="${1:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"
DATA_DIR="${2:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"
MODEL_PATH="${3:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"
EXPERIMENT_NAME="${4:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"
ALLOW_UNSUPPORTED_OLMO_RL="${ALLOW_UNSUPPORTED_OLMO_RL:-0}"

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
export WANDB_DIR=~/scratch/forgetting-llms/wandb/${EXPERIMENT_NAME}
export WANDB_CACHE_DIR=~/scratch/forgetting-llms/wandb_cache/${EXPERIMENT_NAME}
unset ROCR_VISIBLE_DEVICES

mkdir -p slurm_logs "$WANDB_DIR" "$WANDB_CACHE_DIR"

SAVE_DIR=~/scratch/forgetting-llms/checkpoints/$EXPERIMENT_NAME
REWARD_PATH="$REPO_DIR/src/rewards/unified_reward.py"
MAX_PROMPT=${MAX_PROMPT:-512}
MAX_RESPONSE=${MAX_RESPONSE:-1024}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUTS_PER_PROMPT=${ROLLOUTS_PER_PROMPT:-4}
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}
ACTOR_LR=${ACTOR_LR:-1e-6}

mkdir -p "$SAVE_DIR"

if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "ERROR: Missing RL train parquet: $DATA_DIR/train.parquet" >&2
    exit 1
fi

if [[ -f "$DATA_DIR/metadata.json" ]]; then
    REWARD_SUPPORT=$(python3 - "$DATA_DIR/metadata.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("reward_support", "unknown"))
PY
)
    if [[ "$REWARD_SUPPORT" != "full" && "$ALLOW_UNSUPPORTED_OLMO_RL" != "1" ]]; then
        echo "Refusing to launch GRPO on dataset dir with reward_support=$REWARD_SUPPORT: $DATA_DIR" >&2
        echo "This repo's current reward path only safely supports imported OLMo RL math." >&2
        echo "Override with ALLOW_UNSUPPORTED_OLMO_RL=1 only if you have added the correct reward implementation." >&2
        exit 1
    fi
fi

RESOLVED_MODEL="$MODEL_PATH"
if [ -d "$MODEL_PATH/actor" ]; then
    MERGED_DIR="${MODEL_PATH}/actor_merged"
    if [ ! -d "$MERGED_DIR" ] || [ -z "$(ls -A "$MERGED_DIR" 2>/dev/null)" ]; then
        python -m verl.model_merger merge --backend fsdp --local_dir "$MODEL_PATH/actor" --target_dir "$MERGED_DIR"
    fi
    RESOLVED_MODEL="$MERGED_DIR"
elif ls "$MODEL_PATH"/model_world_size_*_rank_*.pt &>/dev/null; then
    MERGED_DIR="${MODEL_PATH}/merged_hf"
    if [ ! -d "$MERGED_DIR" ] || [ -z "$(ls -A "$MERGED_DIR" 2>/dev/null)" ]; then
        python -m verl.model_merger merge --backend fsdp --local_dir "$MODEL_PATH" --target_dir "$MERGED_DIR"
    fi
    RESOLVED_MODEL="$MERGED_DIR"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESPONSE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$RESOLVED_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=$ROLLOUTS_PER_PROMPT \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
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
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$SAVE_DIR"
