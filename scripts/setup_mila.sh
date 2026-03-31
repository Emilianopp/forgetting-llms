#!/usr/bin/env bash
# One-shot Mila bootstrap: main env first, then PRIME-RL.

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export SCRATCH="${SCRATCH:-$HOME/scratch}"
export SCRATCH_HOME="${SCRATCH_HOME:-$SCRATCH}"
export APP_ROOT="${APP_ROOT:-$SCRATCH/forgetting-llms}"
export VENV_DIR="${VENV_DIR:-$APP_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export HF_AUTH_ENV_FILE="${HF_AUTH_ENV_FILE:-$APP_ROOT/hf_auth.sh}"
export PRIME_RUNTIME_ENV_FILE="${PRIME_RUNTIME_ENV_FILE:-$APP_ROOT/prime_rl_env.sh}"

module purge >/dev/null 2>&1 || true
module load python/3.10

bash "$SCRIPT_DIR/setup_env.sh"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
bash "$SCRIPT_DIR/setup_prime_rl.sh"

echo "Mila setup complete."
echo "Main env: source $VENV_DIR/bin/activate"
echo "PRIME env: source $PRIME_RUNTIME_ENV_FILE"
