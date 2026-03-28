#!/usr/bin/env bash
# Run a sequential SFT stage chain and benchmark-plan eval after each stage.
#
# Example:
#   bash scripts/run_sft_stage_chain.sh \
#     "$HOME/scratch/forgetting-llms/models/allenai__Olmo-3-7B-Instruct" \
#     olmo_seq_synth2_triviaqa_gsm8k \
#     synthetic2_sft_verified triviaqa gsm8k

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_sft_stage_chain.sh <base_model> <run_prefix> <stage1> [stage2 ...]

Runs each stage sequentially:
  1. trains on the stage dataset
  2. evaluates the stage checkpoint on the benchmark-plan suite
  3. uses that stage output dir as the next stage's model

Environment knobs:
  DATA_VARIANT=sf                 Data variant for run_sft.sh (default: sf)
  STAGE_VARIANT_<DATASET>=...     Optional per-stage variant override, e.g. STAGE_VARIANT_TRIVIAQA=gt
  STAGE_DATA_DIR_<DATASET>=...    Optional per-stage data dir override, e.g. STAGE_DATA_DIR_TRIVIAQA=/scratch/.../triviaqa_sft
  MAX_STEPS=300                   Stage training steps
  SAVE_STEPS=100                  Checkpoint frequency inside each stage
  EVAL_STEPS=100                  Eval frequency inside each stage
  RUN_EVALS=1                     Run benchmark-plan eval after each stage
  EVAL_SUITE=tasks_md             Benchmark suite passed to run_eval.py
  EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"
  SKIP_EXISTING=1                 Skip completed stage/eval outputs
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

BASE_MODEL=${1:?Usage: run_sft_stage_chain.sh <base_model> <run_prefix> <stage1> [stage2 ...]}
RUN_PREFIX=${2:?Usage: run_sft_stage_chain.sh <base_model> <run_prefix> <stage1> [stage2 ...]}
shift 2
if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi
STAGES=("$@")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

module load python/3.10 >/dev/null 2>&1 || true
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    :
elif [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/forgetting-llms}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH_ROOT/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-$SCRATCH_ROOT/benchmark_plan_evals}"
CHAIN_ROOT="${CHAIN_ROOT:-$SCRATCH_ROOT/orchestration/${RUN_PREFIX}_sft_chain}"
DATA_VARIANT="${DATA_VARIANT:-sf}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
RUN_EVALS="${RUN_EVALS:-1}"
EVAL_SUITE="${EVAL_SUITE:-tasks_md}"
DEFAULT_EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-$DEFAULT_EVAL_EXTRA_ARGS}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

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

mkdir -p "$CHECKPOINT_ROOT" "$EVAL_ROOT" "$CHAIN_ROOT"

STATE_FILE="$CHAIN_ROOT/state.json"
MANIFEST_FILE="$CHAIN_ROOT/manifest.json"

stage_env_key() {
    echo "$1" | tr '[:lower:]-./' '[:upper:]____'
}

python3 - <<PY
import json
from pathlib import Path
payload = {
    "base_model": ${BASE_MODEL@Q},
    "run_prefix": ${RUN_PREFIX@Q},
    "stages": ${STAGES[*]@Q}.split(),
    "data_variant": ${DATA_VARIANT@Q},
    "max_steps": int(${MAX_STEPS@Q}),
}
Path(${MANIFEST_FILE@Q}).write_text(json.dumps(payload, indent=2) + "\\n")
PY

CURRENT_MODEL="$BASE_MODEL"

for idx in "${!STAGES[@]}"; do
    STAGE_DATASET="${STAGES[$idx]}"
    STAGE_KEY=$(stage_env_key "$STAGE_DATASET")
    STAGE_VARIANT_VAR="STAGE_VARIANT_${STAGE_KEY}"
    STAGE_DATA_DIR_VAR="STAGE_DATA_DIR_${STAGE_KEY}"
    STAGE_VARIANT="${!STAGE_VARIANT_VAR:-$DATA_VARIANT}"
    STAGE_DATA_DIR="${!STAGE_DATA_DIR_VAR:-}"
    STAGE_NUM=$((idx + 1))
    STAGE_NAME="${RUN_PREFIX}_stage$(printf '%02d' "$STAGE_NUM")_${STAGE_DATASET}"
    STAGE_SAVE_DIR="$CHECKPOINT_ROOT/$STAGE_NAME"
    STAGE_EVAL_DIR="$EVAL_ROOT/$STAGE_NAME"

    echo "========================================="
    echo "Stage $STAGE_NUM/${#STAGES[@]}: $STAGE_DATASET"
    echo "Source model: $CURRENT_MODEL"
    echo "Stage output: $STAGE_SAVE_DIR"
    echo "Stage eval:   $STAGE_EVAL_DIR"
    echo "Variant:      $STAGE_VARIANT"
    if [[ -n "$STAGE_DATA_DIR" ]]; then
        echo "Data dir:      $STAGE_DATA_DIR"
    fi
    echo "========================================="

    if [[ "$SKIP_EXISTING" == "1" && -f "$STAGE_SAVE_DIR/completed.marker" ]]; then
        echo "Skipping completed stage training: $STAGE_NAME"
    else
        MAX_STEPS="$MAX_STEPS" \
        SAVE_STEPS="$SAVE_STEPS" \
        EVAL_STEPS="$EVAL_STEPS" \
        SAVE_DIR="$STAGE_SAVE_DIR" \
        DATA_DIR="$STAGE_DATA_DIR" \
        bash "$SCRIPT_DIR/run_sft.sh" "$STAGE_DATASET" "$CURRENT_MODEL" "$STAGE_NAME" "$STAGE_VARIANT"
    fi

    CURRENT_MODEL="$STAGE_SAVE_DIR"

    if [[ "$RUN_EVALS" == "1" ]]; then
        if [[ "$SKIP_EXISTING" == "1" && -f "$STAGE_EVAL_DIR/eval_summary.json" ]]; then
            echo "Skipping completed benchmark eval: $STAGE_NAME"
        else
            EVAL_CMD=(
                python3 "$REPO_DIR/src/evaluation/run_eval.py"
                --model_path "$CURRENT_MODEL"
                --suite "$EVAL_SUITE"
                --output_dir "$STAGE_EVAL_DIR"
                --run_name "$STAGE_NAME"
                --continue_on_error
            )
            if [[ -n "$EVAL_EXTRA_ARGS" ]]; then
                # shellcheck disable=SC2206
                EXTRA_ARGS=( $EVAL_EXTRA_ARGS )
                EVAL_CMD+=("${EXTRA_ARGS[@]}")
            fi
            "${EVAL_CMD[@]}"
        fi
    fi

    python3 - <<PY
import json
from pathlib import Path
state = {
    "current_stage": ${STAGE_DATASET@Q},
    "current_model": ${CURRENT_MODEL@Q},
    "completed_stages": ${STAGES[*]@Q}.split()[:${STAGE_NUM}],
}
Path(${STATE_FILE@Q}).write_text(json.dumps(state, indent=2) + "\\n")
PY
done

echo "========================================="
echo "Sequential SFT chain complete"
echo "Final model: $CURRENT_MODEL"
echo "State file:  $STATE_FILE"
echo "========================================="
