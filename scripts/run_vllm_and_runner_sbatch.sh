#!/usr/bin/env bash
#SBATCH --job-name=vllm-runner
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#
# Example:
#   sbatch scripts/run_vllm_and_runner_sbatch.sh \
#     --model ~/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B \
#     --run-root ~/scratch/forgetting-llms/manual_runs/gsm8k_baseline \
#     -- \
#     python scripts/prime_rl_runner.py baseline \
#       --model ~/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B \
#       --dataset gsm8k \
#       --output-root ~/scratch/forgetting-llms/runs \
#       --wandb-mode offline

set -euo pipefail

module load python/3.10
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
if [[ -f "${UV_ENV_PATH:-$REPO_DIR/.venv}/bin/activate" ]]; then
    # Prefer a uv-managed project environment when present.
    # shellcheck disable=SC1090
    source "${UV_ENV_PATH:-$REPO_DIR/.venv}/bin/activate"
elif [[ -f "$HOME/envs/forgetting/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/envs/forgetting/bin/activate"
else
    echo "No virtual environment found. Expected \$REPO_DIR/.venv or \$HOME/envs/forgetting." >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

mkdir -p slurm_logs

# shellcheck source=/dev/null
source "$SCRIPT_DIR/vllm_runner_common.sh"

parse_vllm_runner_args "$@"

trap stop_vllm_server EXIT INT TERM

{
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "HOSTNAME=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "RUN_ROOT=$RUN_ROOT"
} >"$RUN_ROOT/job_env.txt"

start_vllm_server
wait_for_vllm_server
export_vllm_client_env
run_runner_command
