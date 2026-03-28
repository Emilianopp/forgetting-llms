#!/usr/bin/env bash
# Import an official OLMo RL dataset variant into scratch-local parquet files.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/prepare_olmo_rl_data.sh <variant> [extra import_olmo_rl_data.py args...]

Variants:
  math
  code
  if
  general
  mix
  instruct

Example:
  bash scripts/prepare_olmo_rl_data.sh math
  bash scripts/prepare_olmo_rl_data.sh instruct --test-fraction 0.01
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
    usage
    exit 0
fi

VARIANT="$1"
shift

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

module load python/3.10 >/dev/null 2>&1 || true
if [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/../.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/../.venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
export TMPDIR="${TMPDIR:-$HOME/scratch/tmp}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

python3 "$SCRIPT_DIR/import_olmo_rl_data.py" --variant "$VARIANT" "$@"
