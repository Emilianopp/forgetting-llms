#!/usr/bin/env bash
# Local GRPO launcher for an arbitrary dataset directory with train/test parquet files.
#
# Usage:
#   bash scripts/run_grpo_dataset_dir_local.sh <data_label> <data_dir> <model_path> <experiment_name>

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/require_prime_only.sh"

DATA_LABEL="${1:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"
DATA_DIR="${2:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"
MODEL_PATH="${3:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"
EXPERIMENT_NAME="${4:?Usage: $0 <data_label> <data_dir> <model_path> <experiment_name>}"

module load python/3.10 >/dev/null 2>&1 || true
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    :
elif [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$XDG_CACHE_HOME/pycache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export TMPDIR="${TMPDIR:-$HOME/scratch/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/forgetting-llms/wandb/${EXPERIMENT_NAME}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/forgetting-llms/wandb_cache/${EXPERIMENT_NAME}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
unset ROCR_VISIBLE_DEVICES

mkdir -p \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$HF_HUB_CACHE" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$XDG_CACHE_HOME" \
    "$PIP_CACHE_DIR" \
    "$UV_CACHE_DIR" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$PYTHONPYCACHEPREFIX" \
    "$MPLCONFIGDIR" \
    "$TMPDIR" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR"

if [[ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    N_GPUS_PER_NODE=$(python - <<'PY'
import os
value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
parts = [item for item in value.split(",") if item.strip()]
print(len(parts) if parts else 1)
PY
)
else
    N_GPUS_PER_NODE=1
fi

SAVE_DIR="${SAVE_DIR:-$HOME/scratch/forgetting-llms/checkpoints/${EXPERIMENT_NAME}}"
REWARD_PATH="${REWARD_PATH:-$REPO_DIR/src/rewards/unified_reward.py}"
ALLOW_UNSUPPORTED_DATASET_RL="${ALLOW_UNSUPPORTED_DATASET_RL:-0}"

MAX_PROMPT="${MAX_PROMPT:-512}"
MAX_RESPONSE="${MAX_RESPONSE:-1024}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-4}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
SAVE_FREQ="${SAVE_FREQ:-200}"
TEST_FREQ="${TEST_FREQ:-50}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"

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
    if [[ "$REWARD_SUPPORT" == "missing" && "$ALLOW_UNSUPPORTED_DATASET_RL" != "1" ]]; then
        echo "ERROR: Refusing dataset-dir RL launch with reward_support=missing: $DATA_DIR" >&2
        echo "Set ALLOW_UNSUPPORTED_DATASET_RL=1 only if you added the correct reward implementation." >&2
        exit 1
    fi
fi

RESOLVED_MODEL="$MODEL_PATH"
if [[ -d "$MODEL_PATH/actor" ]]; then
    MERGED_DIR="${MODEL_PATH}/actor_merged"
    if [[ ! -d "$MERGED_DIR" || -z "$(ls -A "$MERGED_DIR" 2>/dev/null)" ]]; then
        python -m verl.model_merger merge --backend fsdp --local_dir "$MODEL_PATH/actor" --target_dir "$MERGED_DIR"
    fi
    RESOLVED_MODEL="$MERGED_DIR"
elif ls "$MODEL_PATH"/model_world_size_*_rank_*.pt &>/dev/null; then
    MERGED_DIR="${MODEL_PATH}/merged_hf"
    if [[ ! -d "$MERGED_DIR" || -z "$(ls -A "$MERGED_DIR" 2>/dev/null)" ]]; then
        python -m verl.model_merger merge --backend fsdp --local_dir "$MODEL_PATH" --target_dir "$MERGED_DIR"
    fi
    RESOLVED_MODEL="$MERGED_DIR"
fi

echo "========================================="
echo "  Dataset-Dir GRPO Run"
echo "========================================="
echo "Data label:  $DATA_LABEL"
echo "Data dir:    $DATA_DIR"
echo "Model:       $RESOLVED_MODEL"
echo "Save dir:    $SAVE_DIR"
echo "CUDA vis:    ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "GPUs:        $N_GPUS_PER_NODE"
echo "Epochs:      $TOTAL_EPOCHS"
echo "Save freq:   $SAVE_FREQ"
echo "Test freq:   $TEST_FREQ"
echo "========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESPONSE" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$RESOLVED_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.n="$ROLLOUTS_PER_PROMPT" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=forgetting-llms \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_local_dir="$SAVE_DIR"

touch "$SAVE_DIR/completed.marker"
