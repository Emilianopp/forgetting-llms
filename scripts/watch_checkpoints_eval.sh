#!/usr/bin/env bash
# Watch a checkpoint directory and evaluate new HF checkpoints on a separate GPU.

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/watch_checkpoints_eval.sh <checkpoints_dir> <eval_root> <run_name_prefix>

This watcher polls for new `checkpoint-*` directories containing `config.json`
and runs `src/evaluation/run_eval.py` on each new checkpoint. It is intended to
run alongside native SFT training on a dedicated evaluation GPU.

Environment knobs:
  EVAL_GPU=3                      CUDA_VISIBLE_DEVICES used for eval subprocesses
  EVAL_POLL_SECS=60               Poll interval
  EVAL_SUITE=tasks_md             Benchmark suite
  EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"
  AUTO_START_EVAL_SERVER=1        Serve the checkpoint locally on the eval GPU
  STOP_FILE=...                   Optional file path; watcher exits once it exists
  TRAIN_COMPLETED_MARKER=...      Optional file path; watcher exits after it exists and no new checkpoints remain
  SKIP_EXISTING=1                 Skip eval dirs with eval_summary.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CHECKPOINTS_DIR=${1:?Usage: watch_checkpoints_eval.sh <checkpoints_dir> <eval_root> <run_name_prefix>}
EVAL_ROOT=${2:?Usage: watch_checkpoints_eval.sh <checkpoints_dir> <eval_root> <run_name_prefix>}
RUN_NAME_PREFIX=${3:?Usage: watch_checkpoints_eval.sh <checkpoints_dir> <eval_root> <run_name_prefix>}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

module load python/3.10 >/dev/null 2>&1 || true
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    :
elif [[ -n "${VENV_DIR:-}" && -x "$VENV_DIR/bin/python" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
elif [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
fi

EVAL_GPU="${EVAL_GPU:-0}"
EVAL_POLL_SECS="${EVAL_POLL_SECS:-60}"
EVAL_SUITE="${EVAL_SUITE:-tasks_md}"
DEFAULT_EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-$DEFAULT_EVAL_EXTRA_ARGS}"
STOP_FILE="${STOP_FILE:-}"
TRAIN_COMPLETED_MARKER="${TRAIN_COMPLETED_MARKER:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
AUTO_START_EVAL_SERVER="${AUTO_START_EVAL_SERVER:-1}"

sanitize_eval_extra_args() {
    local raw="$1"
    [[ -n "$raw" ]] || return 0
    # shellcheck disable=SC2206
    local parsed=( $raw )
    local cleaned=()
    local removed=0
    local arg
    for arg in "${parsed[@]}"; do
        if [[ "$arg" == "--no-lighteval-chat-template" ]]; then
            removed=1
            continue
        fi
        cleaned+=("$arg")
    done
    if [[ "$removed" == "1" ]]; then
        echo "Ignoring stale eval arg: --no-lighteval-chat-template" >&2
    fi
    printf '%s' "${cleaned[*]}"
}

EVAL_EXTRA_ARGS="$(sanitize_eval_extra_args "$EVAL_EXTRA_ARGS")"

mkdir -p "$EVAL_ROOT"

checkpoint_ready() {
    local checkpoint_dir="$1"
    [[ -d "$checkpoint_dir" ]] || return 1
    [[ -f "$checkpoint_dir/config.json" ]] || return 1
    compgen -G "$checkpoint_dir/*.safetensors" >/dev/null || \
        compgen -G "$checkpoint_dir/pytorch_model*.bin" >/dev/null || \
        [[ -f "$checkpoint_dir/model.safetensors.index.json" ]] || \
        [[ -f "$checkpoint_dir/pytorch_model.bin.index.json" ]] || return 1
}

run_eval_checkpoint() {
    local checkpoint_dir="$1"
    local label
    label=$(basename "$checkpoint_dir")
    local output_dir="$EVAL_ROOT/$label"
    if [[ "$SKIP_EXISTING" == "1" && -f "$output_dir/eval_summary.json" ]]; then
        return 0
    fi

    if [[ "$AUTO_START_EVAL_SERVER" == "1" ]]; then
        env \
            CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            EVAL_SUITE="$EVAL_SUITE" \
            EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS" \
            CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR" \
            bash "$SCRIPT_DIR/run_eval_with_local_server.sh" \
              "$checkpoint_dir" \
              "$output_dir" \
              "${RUN_NAME_PREFIX}_${label}"
    else
        local cmd=(
            python3 "$REPO_DIR/src/evaluation/run_eval.py"
            --model_path "$checkpoint_dir"
            --suite "$EVAL_SUITE"
            --output_dir "$output_dir"
            --run_name "${RUN_NAME_PREFIX}_${label}"
        )
        if [[ "$SKIP_EXISTING" != "1" ]]; then
            cmd+=(--force-rerun)
        fi
        if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
            cmd+=(--continue_on_error)
        fi
        if [[ -n "$EVAL_EXTRA_ARGS" ]]; then
            # shellcheck disable=SC2206
            local extra_args=( $EVAL_EXTRA_ARGS )
            cmd+=("${extra_args[@]}")
        fi
        env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${cmd[@]}"
    fi
}

have_pending_checkpoint() {
    local checkpoint_dir
    for checkpoint_dir in "$CHECKPOINTS_DIR"/checkpoint-*; do
        [[ -e "$checkpoint_dir" ]] || continue
        checkpoint_ready "$checkpoint_dir" || continue
        local label output_dir
        label=$(basename "$checkpoint_dir")
        output_dir="$EVAL_ROOT/$label"
        if [[ ! -f "$output_dir/eval_summary.json" ]]; then
            return 0
        fi
    done
    return 1
}

while true; do
    for checkpoint_dir in "$CHECKPOINTS_DIR"/checkpoint-*; do
        [[ -e "$checkpoint_dir" ]] || continue
        checkpoint_ready "$checkpoint_dir" || continue
        run_eval_checkpoint "$checkpoint_dir"
    done

    if [[ -n "$STOP_FILE" && -f "$STOP_FILE" ]]; then
        if ! have_pending_checkpoint; then
            break
        fi
    fi
    if [[ -n "$TRAIN_COMPLETED_MARKER" && -f "$TRAIN_COMPLETED_MARKER" ]]; then
        if ! have_pending_checkpoint; then
            break
        fi
    fi
    sleep "$EVAL_POLL_SECS"
done
