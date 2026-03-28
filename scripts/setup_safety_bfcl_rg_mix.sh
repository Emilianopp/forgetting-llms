#!/usr/bin/env bash
# Convenience wrapper for the remaining heavy tasks_md benchmarks:
#   safety, bfcl, rg_mix

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup_safety_bfcl_rg_mix.sh

This wrapper:
1. installs / updates the external benchmark repos under scratch
2. creates isolated runner virtualenvs for:
   - safety
   - bfcl
   - rg_mix

It delegates to:
  bash scripts/setup_tasks_md_benchmarks.sh
  bash scripts/setup_runner_venvs.sh safety bfcl rg_mix

Important environment knobs:
  SCRATCH_HOME=~/scratch
  SCRATCH_ROOT=~/scratch/forgetting-llms
  BENCHMARK_ENV_FILE=~/scratch/forgetting-llms/benchmark_env.sh
  BFCL_REPO_ROOT=~/scratch/gorilla
  BFCL_ROOT=~/scratch/gorilla/berkeley-function-call-leaderboard
  SAFETY_EVAL_ROOT=~/scratch/safety-eval
  RG_MIX_ROOT=...                  Optional rg_mix_env checkout path
  UPDATE_EXISTING=0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "==> Installing / updating external benchmark repos"
bash "$SCRIPT_DIR/setup_tasks_md_benchmarks.sh"

echo "==> Creating isolated runner virtualenvs"
bash "$SCRIPT_DIR/setup_runner_venvs.sh" safety bfcl rg_mix

echo
echo "Setup complete for: safety, bfcl, rg_mix"
echo "Benchmark env file: ${BENCHMARK_ENV_FILE:-$HOME/scratch/forgetting-llms/benchmark_env.sh}"
echo "Next step: source that file before running evals."
