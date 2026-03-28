#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export WANDB_DIR="/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/wandb"
export WANDB_CACHE_DIR="/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/wandb-cache"
export VLLM_CONFIG_ROOT="/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/vllm"
export PRIME_RUN_DIR="/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq"
export PRIME_ENV_NAME="gsm8k"
export PRIME_CHECKPOINT_DIR="/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/checkpoints"
export PRIME_METRICS_DIR="/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/metrics"
export WANDB_PROJECT="forgetting-llms"
export WANDB_MODE="online"
export PYTHONUNBUFFERED="1"
mkdir -p "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/logs" "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/metrics" "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/checkpoints" "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/wandb" "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/wandb-cache"

uv run rl --trainer @ "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/configs/trainer.toml" --orchestrator @ "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/configs/orchestrator.toml" --inference @ "/Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq/configs/inference.toml" --output-dir /Users/danemalenfant/PycharmProjects/forgetting-llms/.tmp_prime_smoke/smoke_prime_seq --max-steps 1000 --wandb.project forgetting-llms --wandb.name smoke_prime_seq --trainer.monitor.wandb.id c4d021bfcd9b514c8b355ffac5570d8c --orchestrator.monitor.wandb.id 68cb7a488dfe53749d0e10ccebd159b2 --ckpt --ckpt.interval 5 --ckpt.keep-last 3 --ckpt.keep-interval 50 --ckpt.resume-step 25 "$@"
