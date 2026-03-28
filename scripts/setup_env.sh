#!/bin/bash
# Setup environment on Mila cluster
# Usage:
#   bash scripts/setup_env.sh
#   USE_UV=1 bash scripts/setup_env.sh
#   REBUILD=1 USE_UV=1 bash scripts/setup_env.sh
#   REBUILD=1 USE_UV=0 VENV_COPIES=1 bash scripts/setup_env.sh
#   REBUILD=1 USE_UV=0 VENV_COPIES=1 FORCE_REINSTALL_CORE=1 bash scripts/setup_env.sh
#   VENV_DIR=~/scratch/forgetting-llms/.venv bash scripts/setup_env.sh

set -euo pipefail

echo "=== Setting up forgetting-llms environment ==="

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"

# Load modules (Mila)
module load python/3.10

VENV_DIR="${VENV_DIR:-$HOME/scratch/forgetting-llms/.venv}"
USE_UV="${USE_UV:-auto}"
REBUILD="${REBUILD:-0}"
VENV_COPIES="${VENV_COPIES:-0}"
FORCE_REINSTALL_CORE="${FORCE_REINSTALL_CORE:-0}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ "$VENV_DIR" != "$SCRATCH_ROOT"/* ]]; then
    echo "Refusing to create a virtual environment outside scratch: $VENV_DIR" >&2
    echo "Set VENV_DIR under $SCRATCH_ROOT or override this script explicitly." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/scratch/.cache/triton}"
export TMPDIR="${TMPDIR:-$HOME/scratch/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/forgetting-llms/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/forgetting-llms/wandb_cache}"

mkdir -p \
    "$(dirname "$VENV_DIR")" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$PIP_CACHE_DIR" \
    "$UV_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$TMPDIR" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR"

echo "Scratch root: $SCRATCH_ROOT"
echo "Virtual environment: $VENV_DIR"
echo "Python binary: $PYTHON_BIN"
echo "Python version: $("$PYTHON_BIN" --version)"
echo "TMPDIR: $TMPDIR"
echo "PIP cache: $PIP_CACHE_DIR"
echo "UV cache: $UV_CACHE_DIR"

if [[ -d "$VENV_DIR" && "$REBUILD" != "1" ]]; then
    if [[ -e "$VENV_DIR/bin/python" ]] && ! "$VENV_DIR/bin/python" --version >/dev/null 2>&1; then
        echo "Detected broken virtual environment at $VENV_DIR; rebuilding with python -m venv --copies"
        REBUILD=1
        USE_UV=0
        VENV_COPIES=1
    fi
fi

if [[ "$REBUILD" == "1" && -d "$VENV_DIR" ]]; then
    echo "Rebuilding scratch virtual environment: removing $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ "$USE_UV" == "1" ]] || [[ "$USE_UV" == "auto" && -x "$(command -v uv)" ]]; then
    echo "Using uv to create and populate $VENV_DIR"
    uv venv --python "$PYTHON_BIN" "$VENV_DIR"
    INSTALL_CMD=(uv pip install --python "$VENV_DIR/bin/python")
else
    echo "Using python -m venv to create $VENV_DIR"
    VENV_ARGS=()
    if [[ "$VENV_COPIES" == "1" ]]; then
        VENV_ARGS+=(--copies)
    fi
    "$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$VENV_DIR"
    INSTALL_CMD=("$VENV_DIR/bin/pip" install)
fi

source "$VENV_DIR/bin/activate"

REINSTALL_FLAGS=()
if [[ "$FORCE_REINSTALL_CORE" == "1" ]]; then
    REINSTALL_FLAGS+=(--force-reinstall --no-cache-dir)
    echo "Force-reinstalling core ML packages in $VENV_DIR"
    "$VENV_DIR/bin/pip" uninstall -y \
        transformers vllm tokenizers huggingface_hub accelerate datasets pandas pyarrow safetensors \
        >/dev/null 2>&1 || true

    SITE_PACKAGES=$("$VENV_DIR/bin/python" - <<'PY'
import site
paths = [p for p in site.getsitepackages() if "site-packages" in p]
print(paths[0] if paths else "")
PY
)
    if [[ -n "$SITE_PACKAGES" && -d "$SITE_PACKAGES" ]]; then
        find "$SITE_PACKAGES" -maxdepth 1 \
            \( -name 'transformers' -o -name 'transformers-*' \
            -o -name 'vllm' -o -name 'vllm-*' \
            -o -name 'tokenizers' -o -name 'tokenizers-*' \
            -o -name 'huggingface_hub' -o -name 'huggingface_hub-*' \
            -o -name 'accelerate' -o -name 'accelerate-*' \
            -o -name 'datasets' -o -name 'datasets-*' \
            -o -name 'pandas' -o -name 'pandas-*' \
            -o -name 'pyarrow' -o -name 'pyarrow-*' \
            -o -name 'safetensors' -o -name 'safetensors-*' \) \
            -exec rm -rf {} +
    fi
fi

# Core dependencies
"${INSTALL_CMD[@]}" --upgrade pip setuptools wheel
"${INSTALL_CMD[@]}" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
"${INSTALL_CMD[@]}" "${REINSTALL_FLAGS[@]}" transformers datasets accelerate pandas pyarrow huggingface_hub
"${INSTALL_CMD[@]}" wandb
"${INSTALL_CMD[@]}" peft  # LoRA if needed

# vLLM — fast inference for data generation and PRIME-RL rollouts
"${INSTALL_CMD[@]}" "${REINSTALL_FLAGS[@]}" vllm

# GEM environment suite
"${INSTALL_CMD[@]}" gem-llm

# Ray — required by PRIME / distributed workloads
"${INSTALL_CMD[@]}" "ray[default]"

# Evaluation
"${INSTALL_CMD[@]}" lm-eval lighteval evalplus openai labbench verifiers  # benchmark runners + PRIME setup

# Analysis
"${INSTALL_CMD[@]}" matplotlib seaborn pandas scipy

"$VENV_DIR/bin/python" - <<'PY'
from pathlib import Path
import transformers

root = Path(transformers.__file__).resolve().parent
required = [
    root / "models" / "llava" / "configuration_llava.py",
    root / "models" / "roformer" / "modeling_tf_roformer.py",
    root / "models" / "dinov2" / "modeling_flax_dinov2.py",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(
        "Incomplete transformers install; missing expected files:\n" + "\n".join(missing)
    )
print("Verified transformers package layout for vLLM/Qwen3.5 imports.")
PY

echo "=== Environment setup complete ==="
echo "Virtual environment: $VENV_DIR"
echo "Activate with: source \"$VENV_DIR/bin/activate\""
echo "Then install PRIME-RL with: bash scripts/setup_prime_rl.sh"
