#!/bin/bash
# Setup environment on Mila cluster
# Usage: bash scripts/setup_env.sh

set -e

echo "=== Setting up forgetting-llms environment ==="

# Load modules (Mila)
module load python/3.11 2>/dev/null || true

# Create virtual environment
python -m venv $HOME/envs/forgetting
source $HOME/envs/forgetting/bin/activate

# Core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate
pip install wandb
pip install peft  # LoRA if needed

# VeRL — primary training framework (SFT + online RL: GRPO, PPO, REINFORCE++, SPIN)
pip install verl

# vLLM — fast inference for data generation and VeRL rollouts
pip install vllm

# TRL — fallback for offline DPO (OFF-RL method only, VeRL doesn't support offline DPO)
pip install trl

# GEM environment suite
pip install gem-llm

# Ray — required by VeRL for distributed RL training
pip install ray[default]

# Evaluation
pip install lm-eval  # lm-evaluation-harness

# Analysis
pip install matplotlib seaborn pandas scipy

echo "=== Environment setup complete ==="
echo "Activate with: source \$HOME/envs/forgetting/bin/activate"
