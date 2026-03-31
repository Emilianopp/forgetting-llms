#!/usr/bin/env bash
# Watch a PRIME-RL run bundle and evaluate newly exported checkpoints on a separate GPU.

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/watch_prime_run_eval.sh <prime_runs_root> <run_name> <output_root>

This watcher polls a PRIME-RL run under <prime_runs_root>/<run_name> and reruns
eval_prime_checkpoint_sweep.py so that newly exported `step_*` checkpoints are
evaluated while training continues on other GPUs.

Environment knobs:
  EVAL_GPU=3                           CUDA_VISIBLE_DEVICES used for eval subprocesses
  EVAL_POLL_SECS=60                    Poll interval
  EVAL_SUITE=tasks_md                  Benchmark suite
  TASK_PASS_K=512                      Local task eval pass@k
  TASK_EVAL_MAX_SAMPLES=...            Optional cap on local task eval prompts
  TASK_EVAL_MAX_MODEL_LEN=...           Optional; defaults to the model-config max
  TASK_EVAL_MAX_TOKENS=8192
  TASK_EVAL_TP=1
  TASK_EVAL_GPU_MEMORY_UTILIZATION=0.9
  STOP_FILE=...                        Optional file path; watcher exits after one stable poll once it exists
  TRAIN_COMPLETED_MARKER=...           Optional file path; same behavior as STOP_FILE
  SKIP_TASK_EVALS=0                    Set to 1 to skip local task evals
  SKIP_BENCHMARK_EVALS=0               Set to 1 to skip benchmark evals
  CONTINUE_ON_ERROR=0                  Forwarded to eval_prime_checkpoint_sweep.py
  AUTO_START_EVAL_SERVER=1             Serve checkpoints locally on the eval GPU for benchmark evals
  BENCHMARK_ENV_FILE=...               Optional env file sourced by the local-server wrapper
  SWEEP_EXTRA_ARGS="..."               Extra raw args appended to eval_prime_checkpoint_sweep.py
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

PRIME_RUNS_ROOT=${1:?Usage: watch_prime_run_eval.sh <prime_runs_root> <run_name> <output_root>}
RUN_NAME=${2:?Usage: watch_prime_run_eval.sh <prime_runs_root> <run_name> <output_root>}
OUTPUT_ROOT=${3:?Usage: watch_prime_run_eval.sh <prime_runs_root> <run_name> <output_root>}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

module load python/3.10 >/dev/null 2>&1 || true
DEFAULT_SCRATCH_HOME="${SCRATCH:-$HOME/scratch}"
DEFAULT_SCRATCH_ROOT="${DEFAULT_SCRATCH_HOME}/forgetting-llms"
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    :
elif [[ -n "${VENV_DIR:-}" && -x "$VENV_DIR/bin/python" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
elif [[ -f "$DEFAULT_SCRATCH_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$DEFAULT_SCRATCH_ROOT/.venv/bin/activate"
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
fi

EVAL_GPU="${EVAL_GPU:-0}"
EVAL_POLL_SECS="${EVAL_POLL_SECS:-60}"
EVAL_SUITE="${EVAL_SUITE:-tasks_md}"
TASK_PASS_K="${TASK_PASS_K:-512}"
TASK_EVAL_MAX_SAMPLES="${TASK_EVAL_MAX_SAMPLES:-}"
TASK_EVAL_MAX_MODEL_LEN="${TASK_EVAL_MAX_MODEL_LEN:-}"
TASK_EVAL_MAX_TOKENS="${TASK_EVAL_MAX_TOKENS:-8192}"
TASK_EVAL_TP="${TASK_EVAL_TP:-1}"
TASK_EVAL_GPU_MEMORY_UTILIZATION="${TASK_EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
STOP_FILE="${STOP_FILE:-}"
TRAIN_COMPLETED_MARKER="${TRAIN_COMPLETED_MARKER:-}"
SKIP_TASK_EVALS="${SKIP_TASK_EVALS:-0}"
SKIP_BENCHMARK_EVALS="${SKIP_BENCHMARK_EVALS:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
AUTO_START_EVAL_SERVER="${AUTO_START_EVAL_SERVER:-1}"
SWEEP_EXTRA_ARGS="${SWEEP_EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_ROOT"

count_ready_targets() {
    python3 - "$PRIME_RUNS_ROOT" "$RUN_NAME" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
run_name = sys.argv[2]
run_dir = root / run_name
search_root = run_dir / "checkpoints"
if not search_root.exists():
    search_root = run_dir

count = 0
for step_dir in sorted(search_root.glob("step_*")):
    if not step_dir.is_dir():
        continue
    ready = False
    for config in step_dir.rglob("config.json"):
        model_dir = config.parent
        if (
            any(model_dir.glob("*.safetensors"))
            or any(model_dir.glob("pytorch_model*.bin"))
            or (model_dir / "model.safetensors.index.json").exists()
            or (model_dir / "pytorch_model.bin.index.json").exists()
        ):
            ready = True
            break
    if ready:
        count += 1
print(count)
PY
}

run_sweep() {
    local cmd=(
        python3 "$REPO_DIR/scripts/eval_prime_checkpoint_sweep.py"
        --prime-runs-root "$PRIME_RUNS_ROOT"
        --output-root "$OUTPUT_ROOT"
        --run "$RUN_NAME"
        --suite "$EVAL_SUITE"
        --task-pass-k "$TASK_PASS_K"
        --task-max-tokens "$TASK_EVAL_MAX_TOKENS"
        --task-tensor-parallel-size "$TASK_EVAL_TP"
        --task-gpu-memory-utilization "$TASK_EVAL_GPU_MEMORY_UTILIZATION"
    )
    if [[ -n "$TASK_EVAL_MAX_SAMPLES" ]]; then
        cmd+=(--task-max-samples "$TASK_EVAL_MAX_SAMPLES")
    fi
    if [[ -n "$TASK_EVAL_MAX_MODEL_LEN" ]]; then
        cmd+=(--task-max-model-len "$TASK_EVAL_MAX_MODEL_LEN")
    fi
    if [[ "$SKIP_TASK_EVALS" == "1" ]]; then
        cmd+=(--skip-task-evals)
    fi
    if [[ "$SKIP_BENCHMARK_EVALS" == "1" ]]; then
        cmd+=(--skip-benchmark-evals)
    fi
    if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
        cmd+=(--continue-on-error)
    fi
    if [[ "$AUTO_START_EVAL_SERVER" == "1" ]]; then
        cmd+=(--auto-start-eval-server)
    fi
    if [[ -n "$SWEEP_EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        local extra_args=( $SWEEP_EXTRA_ARGS )
        cmd+=("${extra_args[@]}")
    fi
    env \
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
        AUTO_START_EVAL_SERVER="$AUTO_START_EVAL_SERVER" \
        BENCHMARK_ENV_FILE="${BENCHMARK_ENV_FILE:-}" \
        "${cmd[@]}"
}

LAST_READY_COUNT=-1

while true; do
    READY_COUNT=$(count_ready_targets)
    run_sweep

    if [[ -n "$STOP_FILE" && -f "$STOP_FILE" ]]; then
        if [[ "$READY_COUNT" -eq "$LAST_READY_COUNT" ]]; then
            break
        fi
    fi
    if [[ -n "$TRAIN_COMPLETED_MARKER" && -f "$TRAIN_COMPLETED_MARKER" ]]; then
        if [[ "$READY_COUNT" -eq "$LAST_READY_COUNT" ]]; then
            break
        fi
    fi

    LAST_READY_COUNT="$READY_COUNT"
    sleep "$EVAL_POLL_SECS"
done
