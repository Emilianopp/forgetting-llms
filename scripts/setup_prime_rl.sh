#!/usr/bin/env bash
# Bootstrap PRIME-RL into a scratch-local project environment without any
# coding-agent integration prompts.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PRIME_RL_TARBALL_URL="${PRIME_RL_TARBALL_URL:-https://github.com/PrimeIntellect-ai/prime-rl/archive/refs/heads/main.tar.gz}"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"

module load python/3.10 >/dev/null 2>&1 || true
SCRATCH_HOME="${SCRATCH_HOME:-${SCRATCH:-$HOME/scratch}}"
APP_ROOT="${APP_ROOT:-${SCRATCH_ROOT:-$SCRATCH_HOME/forgetting-llms}}"
DEFAULT_VENV_DIR="${VENV_DIR:-$APP_ROOT/.venv}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    if [[ "$VIRTUAL_ENV" != "$SCRATCH_HOME"/* ]]; then
        echo "Active virtual environment is outside scratch: $VIRTUAL_ENV" >&2
        echo "Activate a scratch-local environment first, e.g. source $APP_ROOT/.venv/bin/activate" >&2
        exit 1
    fi
elif [[ -f "${DEFAULT_VENV_DIR/#\~/$HOME}/bin/activate" ]]; then
    if [[ "${DEFAULT_VENV_DIR/#\~/$HOME}" != "$SCRATCH_HOME"/* ]]; then
        echo "VENV_DIR is outside scratch: ${DEFAULT_VENV_DIR/#\~/$HOME}" >&2
        echo "Point VENV_DIR under $SCRATCH_HOME." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${DEFAULT_VENV_DIR/#\~/$HOME}/bin/activate"
else
    echo "No active venv found. Activate $APP_ROOT/.venv first." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for PRIME-RL setup but was not found on PATH." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$SCRATCH_HOME/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_HOME/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_HOME/.cache}"
export TMPDIR="${TMPDIR:-$SCRATCH_HOME/tmp}"
PRIME_RL_ROOT="${PRIME_RL_ROOT:-$APP_ROOT/vendor/prime-rl}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRATCH_HOME/.local/share/uv/python}"
export UV_PYTHON_CACHE_DIR="${UV_PYTHON_CACHE_DIR:-$SCRATCH_HOME/.cache/uv/python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$APP_ROOT/prime-rl-env}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$APP_ROOT" "$(dirname "$PRIME_RL_ROOT")" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT"

cd "$REPO_DIR"

download_prime_rl_tarball() {
    local parent_dir archive_path extract_dir
    parent_dir=$(dirname "$PRIME_RL_ROOT")
    archive_path="$TMPDIR/prime-rl-main.tar.gz"
    extract_dir="$TMPDIR/prime-rl-main-extract"

    rm -rf "$extract_dir" "$archive_path" "$PRIME_RL_ROOT"
    mkdir -p "$extract_dir" "$parent_dir"

    echo "Downloading PRIME-RL tarball into scratch..."
    if command -v curl >/dev/null 2>&1; then
        curl -L "$PRIME_RL_TARBALL_URL" -o "$archive_path"
    else
        python - <<PY
import urllib.request
urllib.request.urlretrieve("${PRIME_RL_TARBALL_URL}", "${archive_path}")
PY
    fi

    tar -xzf "$archive_path" -C "$extract_dir"
    mv "$extract_dir"/prime-rl-main "$PRIME_RL_ROOT"
}

if command -v git >/dev/null 2>&1; then
    if [[ -d "$PRIME_RL_ROOT/.git" ]]; then
        echo "Updating PRIME-RL checkout at $PRIME_RL_ROOT..."
        git -C "$PRIME_RL_ROOT" pull --ff-only
    else
        rm -rf "$PRIME_RL_ROOT"
        echo "Cloning PRIME-RL into $PRIME_RL_ROOT..."
        git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL_ROOT"
    fi
else
    download_prime_rl_tarball
fi

echo "Installing managed Python 3.12 for PRIME-RL..."
uv python install 3.12

echo "Syncing PRIME-RL environment..."
uv sync --project "$PRIME_RL_ROOT" --all-extras

echo "Verifying PRIME-RL entrypoint..."
uv --project "$PRIME_RL_ROOT" run rl --help >/dev/null

PRIME_RUNTIME_ENV_FILE="${PRIME_RUNTIME_ENV_FILE:-$APP_ROOT/prime_rl_env.sh}"

cat > "$PRIME_RUNTIME_ENV_FILE" <<EOF
export PRIME_RL_ROOT="$PRIME_RL_ROOT"
export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export UV_PYTHON_CACHE_DIR="$UV_PYTHON_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_LINK_MODE="$UV_LINK_MODE"
export PRIME_COMMAND='uv --project $PRIME_RL_ROOT run rl'
EOF

echo "PRIME-RL setup complete."
echo "Source: source $PRIME_RUNTIME_ENV_FILE"
