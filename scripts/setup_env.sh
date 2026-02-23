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
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install trl  # SFTTrainer, DPOTrainer
pip install vllm  # Fast inference for data generation
pip install wandb
pip install peft  # LoRA if needed

# GEM environment suite
pip install gem-llm

# Evaluation
pip install lm-eval  # lm-evaluation-harness

# Analysis
pip install matplotlib seaborn pandas scipy

echo "=== Environment setup complete ==="
echo "Activate with: source \$HOME/envs/forgetting/bin/activate"
