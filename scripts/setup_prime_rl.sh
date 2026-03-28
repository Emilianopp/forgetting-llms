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
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    if [[ "$VIRTUAL_ENV" != "$SCRATCH_ROOT"/* ]]; then
        echo "Active virtual environment is outside scratch: $VIRTUAL_ENV" >&2
        echo "Activate a scratch-local environment first, e.g. source $HOME/scratch/forgetting-llms/.venv/bin/activate" >&2
        exit 1
    fi
elif [[ -n "${VENV_DIR:-}" && -f "${VENV_DIR/#\~/$HOME}/bin/activate" ]]; then
    if [[ "${VENV_DIR/#\~/$HOME}" != "$SCRATCH_ROOT"/* ]]; then
        echo "VENV_DIR is outside scratch: ${VENV_DIR/#\~/$HOME}" >&2
        echo "Point VENV_DIR under $SCRATCH_ROOT." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${VENV_DIR/#\~/$HOME}/bin/activate"
elif [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
else
    echo "No active venv found. Activate ~/scratch/forgetting-llms/.venv first." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for PRIME-RL setup but was not found on PATH." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
export TMPDIR="${TMPDIR:-$HOME/scratch/tmp}"
PRIME_RL_ROOT="${PRIME_RL_ROOT:-$HOME/scratch/forgetting-llms/vendor/prime-rl}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/scratch/.local/share/uv/python}"
export UV_PYTHON_CACHE_DIR="${UV_PYTHON_CACHE_DIR:-$HOME/scratch/.cache/uv/python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/scratch/forgetting-llms/prime-rl-env}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$(dirname "$PRIME_RL_ROOT")" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT"

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

cat > "$HOME/scratch/forgetting-llms/prime_rl_env.sh" <<EOF
export PRIME_RL_ROOT="$PRIME_RL_ROOT"
export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export UV_PYTHON_CACHE_DIR="$UV_PYTHON_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_LINK_MODE="$UV_LINK_MODE"
export PRIME_COMMAND='uv --project $PRIME_RL_ROOT run rl'
EOF

echo "PRIME-RL setup complete."
echo "Source: source $HOME/scratch/forgetting-llms/prime_rl_env.sh"
