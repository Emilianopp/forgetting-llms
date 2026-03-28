#!/usr/bin/env bash
# Export pinned requirements snapshots for the repo-managed virtualenvs.

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/export_venv_requirements.sh

Exports one requirements snapshot per detected environment into:
  ./requirements-locks

Environment knobs:
  SCRATCH_ROOT=~/scratch/forgetting-llms
  OUTPUT_DIR=./requirements-locks
  INCLUDE_BENCHMARK_VENVS=1
  INCLUDE_PRIME_RL=1

Notes:
  - The main repo env is exported from $SCRATCH_ROOT/.venv when present.
  - Benchmark runner envs are exported from $SCRATCH_ROOT/.venvs/* when present.
  - PRIME-RL is exported from $SCRATCH_ROOT/prime-rl-env when present.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/forgetting-llms}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/requirements-locks}"
INCLUDE_BENCHMARK_VENVS="${INCLUDE_BENCHMARK_VENVS:-1}"
INCLUDE_PRIME_RL="${INCLUDE_PRIME_RL:-1}"
MAIN_VENV="${MAIN_VENV:-$SCRATCH_ROOT/.venv}"
RUNNER_VENV_ROOT="${RUNNER_VENV_ROOT:-$SCRATCH_ROOT/.venvs}"
PRIME_RL_ENV="${PRIME_RL_ENV:-$SCRATCH_ROOT/prime-rl-env}"

mkdir -p "$OUTPUT_DIR"

export_one() {
    local label="$1"
    local venv_dir="$2"
    local out_file="$OUTPUT_DIR/${label}.txt"
    local python_bin="$venv_dir/bin/python"

    if [[ ! -x "$python_bin" ]]; then
        echo "Skipping $label: no python at $python_bin"
        return 0
    fi

    echo "Exporting $label -> $out_file"
    "$python_bin" -m pip freeze | LC_ALL=C sort > "$out_file"
}

export_one "main" "$MAIN_VENV"

if [[ "$INCLUDE_BENCHMARK_VENVS" == "1" && -d "$RUNNER_VENV_ROOT" ]]; then
    while IFS= read -r -d '' venv_dir; do
        runner_name=$(basename "$venv_dir")
        export_one "runner_${runner_name}" "$venv_dir"
    done < <(find "$RUNNER_VENV_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [[ "$INCLUDE_PRIME_RL" == "1" ]]; then
    export_one "prime_rl" "$PRIME_RL_ENV"
fi

echo "Wrote requirements snapshots under: $OUTPUT_DIR"
