#!/usr/bin/env bash
# Run inside an existing interactive Slurm allocation (for example after salloc).
#
# Example:
#   bash scripts/run_vllm_and_runner_interactive.sh \
#     --model ~/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B \
#     --run-root ~/scratch/forgetting-llms/manual_runs/gsm8k_baseline \
#     --port 8000 \
#     -- \
#     python scripts/prime_rl_runner.py baseline \
#       --model ~/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B \
#       --dataset gsm8k \
#       --output-root ~/scratch/forgetting-llms/runs \
#       --wandb-mode offline

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/vllm_runner_common.sh"

parse_vllm_runner_args "$@"

trap stop_vllm_server EXIT INT TERM

start_vllm_server
wait_for_vllm_server
export_vllm_client_env
run_runner_command
