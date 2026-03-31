#!/usr/bin/env bash
# One-shot Alliance / Compute Canada bootstrap: main env first, then PRIME-RL.

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH is not set. Log in to an Alliance cluster shell or export SCRATCH manually." >&2
    exit 1
fi

export SCRATCH_HOME="${SCRATCH_HOME:-$SCRATCH}"
export APP_ROOT="${APP_ROOT:-$SCRATCH/forgetting-llms}"
export VENV_DIR="${VENV_DIR:-$APP_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export HF_AUTH_ENV_FILE="${HF_AUTH_ENV_FILE:-$APP_ROOT/hf_auth.sh}"
export PRIME_RUNTIME_ENV_FILE="${PRIME_RUNTIME_ENV_FILE:-$APP_ROOT/prime_rl_env.sh}"

module load CCEnv StdEnv >/dev/null 2>&1 || true
module load python/3.10

bash "$SCRIPT_DIR/setup_env.sh"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
bash "$SCRIPT_DIR/setup_prime_rl.sh"

echo "Alliance setup complete."
echo "Main env: source $VENV_DIR/bin/activate"
echo "PRIME env: source $PRIME_RUNTIME_ENV_FILE"
