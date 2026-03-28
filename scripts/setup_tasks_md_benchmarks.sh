#!/usr/bin/env bash
# Install external benchmark repos needed by the tasks_md suite.

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup_tasks_md_benchmarks.sh

This script installs the external benchmark repos used by tasks_md:
  - BFCL (Berkeley Function Calling Leaderboard)
  - safety-eval
  - LAB-Bench
  - rg-mix-env (if a local checkout is available)

For conflict-free Python dependencies, prefer pairing this with:
  bash scripts/setup_runner_venvs.sh

This script still installs benchmark Python packages into the active env for
convenience, but isolated per-runner envs are the recommended path.

Default install locations:
  BFCL repo root:      ~/scratch/gorilla
  BFCL benchmark root: ~/scratch/gorilla/berkeley-function-call-leaderboard
  safety-eval root:    ~/scratch/safety-eval
  LAB-Bench root:      ~/scratch/LAB-Bench
  RG-mix root:         <repo>/environments/rg_mix_env (if present)

It also writes/updates:
  ~/scratch/forgetting-llms/benchmark_env.sh

Environment knobs:
  SCRATCH_HOME=~/scratch
  SCRATCH_ROOT=~/scratch/forgetting-llms
  BENCHMARK_ENV_FILE=~/scratch/forgetting-llms/benchmark_env.sh
  BFCL_REPO_ROOT=~/scratch/gorilla
  BFCL_ROOT=~/scratch/gorilla/berkeley-function-call-leaderboard
  SAFETY_EVAL_ROOT=~/scratch/safety-eval
  LABBENCH_REPO_ROOT=~/scratch/LAB-Bench
  NLTK_DATA_DIR=~/scratch/forgetting-llms/nltk_data
  RG_MIX_ROOT=...                 Optional local rg_mix_env checkout path
  UPDATE_EXISTING=0              Set to 1 to git pull existing repos
  INSTALL_PYTHON_PACKAGES=0      Set to 1 to install benchmark Python deps into the active env
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

module load python/3.10 >/dev/null 2>&1 || true
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    :
elif [[ -n "${VENV_DIR:-}" && -x "${VENV_DIR/#\~/$HOME}/bin/python" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR/#\~/$HOME}/bin/activate"
elif [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
fi

SCRATCH_HOME="${SCRATCH_HOME:-$HOME/scratch}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/forgetting-llms}"
if [[ -z "${BENCHMARK_ENV_FILE:-}" ]]; then
    if [[ -f "$REPO_DIR/benchmark_env.sh" ]]; then
        BENCHMARK_ENV_FILE="$REPO_DIR/benchmark_env.sh"
    else
        BENCHMARK_ENV_FILE="$SCRATCH_ROOT/benchmark_env.sh"
    fi
fi
BFCL_REPO_ROOT="${BFCL_REPO_ROOT:-$SCRATCH_HOME/gorilla}"
BFCL_ROOT="${BFCL_ROOT:-$BFCL_REPO_ROOT/berkeley-function-call-leaderboard}"
SAFETY_EVAL_ROOT="${SAFETY_EVAL_ROOT:-$SCRATCH_HOME/safety-eval}"
LABBENCH_REPO_ROOT="${LABBENCH_REPO_ROOT:-$SCRATCH_HOME/LAB-Bench}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-$SCRATCH_ROOT/nltk_data}"
DEFAULT_RG_MIX_ROOT=""
if [[ -f "$REPO_DIR/environments/rg_mix_env/rg_mix_env.py" ]]; then
    DEFAULT_RG_MIX_ROOT="$REPO_DIR/environments/rg_mix_env"
fi
RG_MIX_ROOT="${RG_MIX_ROOT:-$DEFAULT_RG_MIX_ROOT}"
UPDATE_EXISTING="${UPDATE_EXISTING:-0}"
INSTALL_PYTHON_PACKAGES="${INSTALL_PYTHON_PACKAGES:-0}"

mkdir -p "$SCRATCH_HOME" "$SCRATCH_ROOT"

clone_or_update() {
    local repo_url="$1"
    local repo_root="$2"
    if [[ ! -d "$repo_root/.git" ]]; then
        git clone "$repo_url" "$repo_root"
        return 0
    fi
    if [[ "$UPDATE_EXISTING" == "1" ]]; then
        git -C "$repo_root" pull --ff-only
    fi
}

has_nltk_resource() {
    local resource="$1"
    python3 - "$resource" <<'PY' >/dev/null 2>&1
import sys
import nltk

try:
    nltk.data.find(sys.argv[1])
except LookupError:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

download_nltk_package_direct() {
    local package="$1"
    local subdir="$2"
    python3 - "$package" "$subdir" "$NLTK_DATA_DIR" <<'PY'
import io
import pathlib
import ssl
import sys
import urllib.request
import zipfile

package, subdir, root = sys.argv[1:]
root_path = pathlib.Path(root).expanduser()
target_dir = root_path / subdir
target_dir.mkdir(parents=True, exist_ok=True)
url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/{subdir}/{package}.zip"
try:
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    context = ssl.create_default_context()
with urllib.request.urlopen(url, timeout=60, context=context) as response:
    payload = response.read()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(target_dir)
PY
}

ensure_nltk_tokenizers() {
    export NLTK_DATA="$NLTK_DATA_DIR"
    if has_nltk_resource "tokenizers/punkt" && has_nltk_resource "tokenizers/punkt_tab"; then
        return 0
    fi

    mkdir -p "$NLTK_DATA_DIR"
    echo "Fetching punkt/punkt_tab directly"
    download_nltk_package_direct punkt tokenizers >/dev/null 2>&1 || true
    download_nltk_package_direct punkt_tab tokenizers >/dev/null 2>&1 || true
    if has_nltk_resource "tokenizers/punkt" && has_nltk_resource "tokenizers/punkt_tab"; then
        return 0
    fi

    echo "Direct fetch did not finish; falling back to nltk.downloader"
    python3 -m nltk.downloader -d "$NLTK_DATA_DIR" punkt punkt_tab >/dev/null 2>&1 || true

    if ! has_nltk_resource "tokenizers/punkt" || ! has_nltk_resource "tokenizers/punkt_tab"; then
        echo "ERROR: Could not provision NLTK punkt/punkt_tab under $NLTK_DATA_DIR" >&2
        exit 1
    fi
}

resolve_bfcl_root() {
    local preferred="$1"
    local repo_root="$2"
    local candidate
    for candidate in \
        "$preferred" \
        "$repo_root/berkeley-function-call-leaderboard" \
        "$repo_root"; do
        [[ -d "$candidate" ]] || continue
        if [[ -f "$candidate/pyproject.toml" || -f "$candidate/setup.py" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

upsert_export() {
    local file="$1"
    local var_name="$2"
    local var_value="$3"
    local tmp
    tmp=$(mktemp)
    if [[ -f "$file" ]]; then
        awk -v key="$var_name" -v value="$var_value" '
            BEGIN { done = 0 }
            $0 ~ "^export " key "=" {
                print "export " key "=\"" value "\""
                done = 1
                next
            }
            { print }
            END {
                if (!done) {
                    print "export " key "=\"" value "\""
                }
            }
        ' "$file" > "$tmp"
    else
        {
            print "# Auto-generated by scripts/setup_tasks_md_benchmarks.sh"
            print "export " var_name "=\"" var_value "\""
        } > "$tmp"
    fi
    mv "$tmp" "$file"
}

echo "Installing BFCL into: $BFCL_REPO_ROOT"
clone_or_update "https://github.com/ShishirPatil/gorilla.git" "$BFCL_REPO_ROOT"
RESOLVED_BFCL_ROOT="$(resolve_bfcl_root "$BFCL_ROOT" "$BFCL_REPO_ROOT")" || {
    echo "ERROR: Could not resolve BFCL root after clone under: $BFCL_REPO_ROOT" >&2
    exit 1
}
BFCL_ROOT="$RESOLVED_BFCL_ROOT"
if [[ ! -d "$BFCL_ROOT" ]]; then
    echo "ERROR: Resolved BFCL root missing after clone: $BFCL_ROOT" >&2
    exit 1
fi

echo "Installing safety-eval into: $SAFETY_EVAL_ROOT"
clone_or_update "https://github.com/allenai/safety-eval.git" "$SAFETY_EVAL_ROOT"
if [[ ! -f "$SAFETY_EVAL_ROOT/evaluation/eval.py" ]]; then
    echo "ERROR: Expected safety-eval entrypoint missing: $SAFETY_EVAL_ROOT/evaluation/eval.py" >&2
    exit 1
fi

echo "Installing LAB-Bench into: $LABBENCH_REPO_ROOT"
clone_or_update "https://github.com/Future-House/LAB-Bench.git" "$LABBENCH_REPO_ROOT"
if [[ -f "$LABBENCH_REPO_ROOT/labbench/pyproject.toml" ]]; then
    LABBENCH_INSTALL_ROOT="$LABBENCH_REPO_ROOT/labbench"
elif [[ -f "$LABBENCH_REPO_ROOT/pyproject.toml" ]]; then
    LABBENCH_INSTALL_ROOT="$LABBENCH_REPO_ROOT"
else
    echo "ERROR: Could not find a LAB-Bench Python package under: $LABBENCH_REPO_ROOT" >&2
    exit 1
fi

if [[ "$INSTALL_PYTHON_PACKAGES" == "1" ]]; then
    echo "Installing BFCL Python package into the active env"
    python3 -m pip install -e "${BFCL_ROOT}[oss_eval_vllm]"

    echo "Installing safety-eval Python package into the active env"
    python3 -m pip install -r "$SAFETY_EVAL_ROOT/requirements.txt"
    python3 -m pip install -e "$SAFETY_EVAL_ROOT"

    echo "Installing EvalPlus and parser dependencies into the active env"
    python3 -m pip install -U "evalplus" "tree-sitter>=0.23,<0.24" "tree-sitter-python>=0.23,<0.24"

    echo "Installing LAB-Bench Python package into the active env"
    python3 -m pip uninstall -y labbench >/dev/null 2>&1 || true
    python3 -m pip install -e "$LABBENCH_INSTALL_ROOT"

    echo "Installing LightEval endpoint dependency into the active env"
    python3 -m pip install "litellm[caching]>=1.66.0"
    python3 -m pip install nltk

    echo "Installing RG-mix benchmark dependencies into the active env"
    python3 -m pip install "reasoning-gym>=0.1.10"
    if [[ -n "$RG_MIX_ROOT" && -d "$RG_MIX_ROOT" && -f "$RG_MIX_ROOT/pyproject.toml" ]]; then
        echo "Installing rg-mix-env from: $RG_MIX_ROOT"
        python3 -m pip install -e "$RG_MIX_ROOT"
    else
        echo "Skipping rg-mix-env install: RG_MIX_ROOT not set or path missing"
    fi
else
    echo "Skipping benchmark Python package installs into the active env."
    echo "Use isolated runner envs instead:"
    echo "  bash scripts/setup_runner_venvs.sh lighteval evalplus labbench rg_mix safety bfcl"
fi

echo "Installing NLTK tokenizers for LiveCodeBench"
ensure_nltk_tokenizers

mkdir -p "$(dirname "$BENCHMARK_ENV_FILE")"
if [[ ! -f "$BENCHMARK_ENV_FILE" && -f "$SCRIPT_DIR/benchmark_env.sh.example" ]]; then
    cp "$SCRIPT_DIR/benchmark_env.sh.example" "$BENCHMARK_ENV_FILE"
fi

upsert_export "$BENCHMARK_ENV_FILE" "BFCL_ROOT" "$BFCL_ROOT"
upsert_export "$BENCHMARK_ENV_FILE" "SAFETY_EVAL_ROOT" "$SAFETY_EVAL_ROOT"
upsert_export "$BENCHMARK_ENV_FILE" "NLTK_DATA" "$NLTK_DATA_DIR"
if [[ -n "$RG_MIX_ROOT" && -d "$RG_MIX_ROOT" ]]; then
    upsert_export "$BENCHMARK_ENV_FILE" "RG_MIX_ROOT" "$RG_MIX_ROOT"
fi

echo
echo "Installed tasks_md external benchmarks."
echo "BFCL_ROOT=$BFCL_ROOT"
echo "SAFETY_EVAL_ROOT=$SAFETY_EVAL_ROOT"
echo "NLTK_DATA=$NLTK_DATA_DIR"
if [[ -n "$RG_MIX_ROOT" && -d "$RG_MIX_ROOT" ]]; then
    echo "RG_MIX_ROOT=$RG_MIX_ROOT"
fi
echo "Updated env file: $BENCHMARK_ENV_FILE"
