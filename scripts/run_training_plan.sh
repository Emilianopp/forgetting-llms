#!/usr/bin/env bash
# Unified stage launcher for native SFT, PRIME-RL, or SFT+RL.
#
# Supports:
#   - training_mode: sft | rl | sft_rl
#   - schedule:      individual | sequential
#   - benchmark-plan eval after each stage
#   - backward-task pass@k evals after each stage
#
# Example:
#   bash scripts/run_training_plan.sh \
#     sft_rl sequential \
#     "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-0.6B" \
#     qwen06_smoke \
#     gsm8k math

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_training_plan.sh <training_mode> <schedule> <base_model> <run_prefix> <stage1> [stage2 ...]

Modes:
  training_mode:
    sft      Run native parquet SFT on each stage
    rl       Run PRIME-RL on each stage
    sft_rl   Run SFT, then PRIME-RL on the same stage before moving on

  schedule:
    individual  Each stage starts from the original base model
    sequential  Each stage starts from the previous stage's final model

Environment knobs:
  DATA_VARIANT=sf                 Default SFT data variant used by run_sft.sh
  SFT_MAX_STEPS=300               SFT stage steps
  RL_MAX_STEPS=300                RL stage steps
  RL_TOTAL_EPOCHS=1               Dataset-dir GRPO epochs for custom RL stages
  RL_CKPT_INTERVAL=300            PRIME checkpoint interval for stage-end evals
  PRIME_NO_AUTO_RESUME=1          Disable PRIME auto-resume from an existing run dir
  TAU2BENCH_SMOKE_BOOTSTRAP=1     Create a tiny tau2bench smoke parquet if no tau2bench data exists
  TRAIN_CUDA_VISIBLE_DEVICES=0,1,2  Optional GPU list reserved for training
  ASYNC_EVAL_GPU=3                Optional dedicated GPU for async checkpoint eval
  ASYNC_EVAL_POLL_SECS=60         Poll interval for async eval watcher
  BENCHMARK_ENV_FILE=...          Optional env file sourced before evals (roots, commands, endpoint)
  AUTO_START_EVAL_SERVER=1        Start a local vLLM server for benchmark evals on the eval GPU
  RUN_BENCHMARK_EVALS=1           Run tasks_md benchmark suite after each stage
  RUN_TASK_EVALS=1                Run local task/backward pass@k evals after each stage
  EVAL_CONTINUE_ON_ERROR=0        Fail fast on benchmark/task eval errors
  CHECKPOINT_MIRROR_ROOT=...      Optional backup root: local path, rclone remote like gdrive:forgetting-llms-backups, or hf:model:<user>/<repo>
  CHECKPOINT_MIRROR_SOURCE_BASE=~/scratch/forgetting-llms
  CHECKPOINT_MIRROR_POLL_SECS=300
  CHECKPOINT_MIRROR_PRUNE_DEST=0
  TASK_PASS_K=512                 Rollouts per prompt for local task evals
  TASK_EVAL_MAX_SAMPLES=...       Optional cap on local task eval prompts for smoke tests
  TASK_EVAL_MAX_MODEL_LEN=...     Optional; defaults to the model-config max
  TASK_EVAL_MAX_TOKENS=8192       Local task eval max new tokens
  TASK_EVAL_TP=1                  Local task eval tensor parallel size
  EVAL_SUITE=tasks_md             Benchmark suite for run_eval.py
  EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"

Per-stage overrides:
  STAGE_VARIANT_<DATASET>=...     Per-stage SFT data variant override
  STAGE_DATA_DIR_<DATASET>=...    Per-stage SFT data dir override
  STAGE_ENV_<DATASET>=...         PRIME environment name override
  STAGE_RL_BACKEND_<DATASET>=...  Per-stage RL backend override: prime | dataset_dir
  STAGE_RL_DATA_DIR_<DATASET>=... Per-stage RL dataset-dir parquet root override
  STAGE_COMBINED_CONFIG_<DATASET>=...
  STAGE_TRAINER_CONFIG_<DATASET>=...
  STAGE_ORCHESTRATOR_CONFIG_<DATASET>=...
  STAGE_INFERENCE_CONFIG_<DATASET>=...
  TASK_EVAL_DATASET_PATH_<DATASET>=...
  TASK_EVAL_DATA_SOURCE_<DATASET>=...
  STAGE_TASK_EVAL_DATASETS_<DATASET>=...

Global task-eval policy:
  TASK_EVAL_REVERSE=0            Reverse local task eval order

Small Qwen RL smoke test:
  export SMALL_QWEN_MODEL="$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-0.6B"
  bash scripts/run_training_plan.sh rl individual "$SMALL_QWEN_MODEL" qwen06_rl_smoke gsm8k
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

TRAINING_MODE=${1:?Usage: run_training_plan.sh <training_mode> <schedule> <base_model> <run_prefix> <stage1> [stage2 ...]}
SCHEDULE=${2:?Usage: run_training_plan.sh <training_mode> <schedule> <base_model> <run_prefix> <stage1> [stage2 ...]}
BASE_MODEL=${3:?Usage: run_training_plan.sh <training_mode> <schedule> <base_model> <run_prefix> <stage1> [stage2 ...]}
RUN_PREFIX=${4:?Usage: run_training_plan.sh <training_mode> <schedule> <base_model> <run_prefix> <stage1> [stage2 ...]}
shift 4
if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi
STAGES=("$@")

case "$TRAINING_MODE" in
    sft|rl|sft_rl) ;;
    *) echo "Unknown training_mode: $TRAINING_MODE" >&2; usage; exit 1 ;;
esac
case "$SCHEDULE" in
    individual|sequential) ;;
    *) echo "Unknown schedule: $SCHEDULE" >&2; usage; exit 1 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"

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
if [[ -d "$HOME/scratch/forgetting-llms/bin" ]]; then
    export PATH="$HOME/scratch/forgetting-llms/bin:$PATH"
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/forgetting-llms}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH_ROOT/checkpoints}"
PRIME_RUNS_ROOT="${PRIME_RUNS_ROOT:-$SCRATCH_ROOT/prime_runs}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH_ROOT/runs}"
BENCHMARK_EVAL_ROOT="${BENCHMARK_EVAL_ROOT:-$SCRATCH_ROOT/benchmark_plan_evals}"
PLAN_ROOT="${PLAN_ROOT:-$SCRATCH_ROOT/orchestration/${RUN_PREFIX}_training_plan}"
if [[ -z "${BENCHMARK_ENV_FILE:-}" ]]; then
    if [[ -f "$REPO_DIR/benchmark_env.sh" ]]; then
        BENCHMARK_ENV_FILE="$REPO_DIR/benchmark_env.sh"
    else
        BENCHMARK_ENV_FILE="$SCRATCH_ROOT/benchmark_env.sh"
    fi
fi
PRIME_RUNTIME_ENV_FILE="${PRIME_RUNTIME_ENV_FILE:-$SCRATCH_ROOT/prime_rl_env.sh}"
if [[ -f "$PRIME_RUNTIME_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$PRIME_RUNTIME_ENV_FILE"
fi
PRIME_RL_ROOT="${PRIME_RL_ROOT:-$SCRATCH_ROOT/vendor/prime-rl}"
PRIME_COMMAND="${PRIME_COMMAND:-uv --project $PRIME_RL_ROOT run rl}"
HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"

if [[ -f "$BENCHMARK_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$BENCHMARK_ENV_FILE"
fi

DATA_VARIANT="${DATA_VARIANT:-sf}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-300}"
RL_MAX_STEPS="${RL_MAX_STEPS:-300}"
RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-1}"
RL_CKPT_INTERVAL="${RL_CKPT_INTERVAL:-300}"
PRIME_NO_AUTO_RESUME="${PRIME_NO_AUTO_RESUME:-0}"
RUN_BENCHMARK_EVALS="${RUN_BENCHMARK_EVALS:-1}"
RUN_TASK_EVALS="${RUN_TASK_EVALS:-1}"
TASK_PASS_K="${TASK_PASS_K:-512}"
TASK_EVAL_MAX_SAMPLES="${TASK_EVAL_MAX_SAMPLES:-}"
TASK_EVAL_MAX_MODEL_LEN="${TASK_EVAL_MAX_MODEL_LEN:-}"
TASK_EVAL_MAX_TOKENS="${TASK_EVAL_MAX_TOKENS:-8192}"
TASK_EVAL_TP="${TASK_EVAL_TP:-1}"
TASK_EVAL_GPU_MEMORY_UTILIZATION="${TASK_EVAL_GPU_MEMORY_UTILIZATION:-0.90}"
TASK_EVAL_TEMPERATURE="${TASK_EVAL_TEMPERATURE:-1.0}"
TASK_EVAL_TOP_P="${TASK_EVAL_TOP_P:-1.0}"
TASK_EVAL_REVERSE="${TASK_EVAL_REVERSE:-0}"
EVAL_SUITE="${EVAL_SUITE:-tasks_md}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"
ASYNC_EVAL_GPU="${ASYNC_EVAL_GPU:-}"
ASYNC_EVAL_POLL_SECS="${ASYNC_EVAL_POLL_SECS:-60}"
EVAL_CONTINUE_ON_ERROR="${EVAL_CONTINUE_ON_ERROR:-0}"
AUTO_START_EVAL_SERVER="${AUTO_START_EVAL_SERVER:-1}"
PRIME_BATCH_SIZE="${PRIME_BATCH_SIZE:-1}"
PRIME_SEQ_LEN="${PRIME_SEQ_LEN:-4096}"
PRIME_MAX_TOKENS="${PRIME_MAX_TOKENS:-512}"
PRIME_ROLLOUTS_PER_PROMPT="${PRIME_ROLLOUTS_PER_PROMPT:-1}"
PRIME_MAX_ASYNC_LEVEL="${PRIME_MAX_ASYNC_LEVEL:-1}"
PRIME_ENFORCE_EAGER="${PRIME_ENFORCE_EAGER:-1}"
PRIME_WANDB_MODE="${PRIME_WANDB_MODE:-disabled}"
PRIME_EXTRA_ARGS="${PRIME_EXTRA_ARGS:-}"
DEFAULT_EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-$DEFAULT_EVAL_EXTRA_ARGS}"
WANDB_MODE="${WANDB_MODE:-disabled}"
CHECKPOINT_MIRROR_ROOT="${CHECKPOINT_MIRROR_ROOT:-}"
CHECKPOINT_MIRROR_SOURCE_BASE="${CHECKPOINT_MIRROR_SOURCE_BASE:-$SCRATCH_ROOT}"
CHECKPOINT_MIRROR_POLL_SECS="${CHECKPOINT_MIRROR_POLL_SECS:-300}"
CHECKPOINT_MIRROR_PRUNE_DEST="${CHECKPOINT_MIRROR_PRUNE_DEST:-0}"
CHECKPOINT_MIRROR_PID=""
CHECKPOINT_MIRROR_STOP_FILE=""

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

start_checkpoint_mirror() {
    local source_dir="$1"
    local label="$2"
    if [[ -z "$CHECKPOINT_MIRROR_ROOT" ]]; then
        return 0
    fi
    stop_checkpoint_mirror
    mkdir -p "$source_dir"
    CHECKPOINT_MIRROR_STOP_FILE="$PLAN_ROOT/${label//[^A-Za-z0-9_.-]/_}.mirror.stop"
    rm -f "$CHECKPOINT_MIRROR_STOP_FILE"
    env \
        MIRROR_STOP_FILE="$CHECKPOINT_MIRROR_STOP_FILE" \
        MIRROR_POLL_SECS="$CHECKPOINT_MIRROR_POLL_SECS" \
        MIRROR_SOURCE_BASE="$CHECKPOINT_MIRROR_SOURCE_BASE" \
        MIRROR_PRUNE_DEST="$CHECKPOINT_MIRROR_PRUNE_DEST" \
        bash "$SCRIPT_DIR/watch_directory_mirror.sh" \
          "$source_dir" \
          "$CHECKPOINT_MIRROR_ROOT" \
          "$label" &
    CHECKPOINT_MIRROR_PID=$!
}

stop_checkpoint_mirror() {
    if [[ -n "$CHECKPOINT_MIRROR_PID" ]]; then
        touch "$CHECKPOINT_MIRROR_STOP_FILE" 2>/dev/null || true
        wait "$CHECKPOINT_MIRROR_PID" || true
    fi
    if [[ -n "$CHECKPOINT_MIRROR_STOP_FILE" ]]; then
        rm -f "$CHECKPOINT_MIRROR_STOP_FILE"
    fi
    CHECKPOINT_MIRROR_PID=""
    CHECKPOINT_MIRROR_STOP_FILE=""
}

cleanup() {
    local exit_code="$1"
    stop_checkpoint_mirror
    return 0
}
trap 'cleanup "$?"' EXIT

mkdir -p "$CHECKPOINT_ROOT" "$PRIME_RUNS_ROOT" "$RUNS_ROOT" "$BENCHMARK_EVAL_ROOT" "$PLAN_ROOT"

STATE_FILE="$PLAN_ROOT/state.json"
MANIFEST_FILE="$PLAN_ROOT/manifest.json"

stage_key() {
    echo "$1" | tr '[:lower:]-./' '[:upper:]____'
}

ensure_prime_runtime() {
    if ! bash -lc "$PRIME_COMMAND --help >/dev/null 2>&1"; then
        echo "ERROR: PRIME RL runtime is not installed or not reachable via PRIME_COMMAND: $PRIME_COMMAND" >&2
        echo "Run: bash scripts/setup_prime_rl.sh" >&2
        exit 1
    fi
}

is_olmo_import_stage() {
    case "$1" in
        olmo_rl_zero_math|dolci_rl_zero_math|\
        olmo_rl_zero_code|dolci_rl_zero_code|\
        olmo_rl_zero_if|dolci_rl_zero_if|\
        olmo_rl_zero_general|dolci_rl_zero_general|\
        olmo_rl_zero_mix|dolci_rl_zero_mix|\
        olmo_instruct_rl|dolci_instruct_rl)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

canonical_olmo_import_stage() {
    case "$1" in
        dolci_rl_zero_math) echo "olmo_rl_zero_math" ;;
        dolci_rl_zero_code) echo "olmo_rl_zero_code" ;;
        dolci_rl_zero_if) echo "olmo_rl_zero_if" ;;
        dolci_rl_zero_general) echo "olmo_rl_zero_general" ;;
        dolci_rl_zero_mix) echo "olmo_rl_zero_mix" ;;
        dolci_instruct_rl) echo "olmo_instruct_rl" ;;
        *) echo "$1" ;;
    esac
}

olmo_import_variant_name() {
    local canonical
    canonical=$(canonical_olmo_import_stage "$1")
    case "$canonical" in
        olmo_rl_zero_math) echo "math" ;;
        olmo_rl_zero_code) echo "code" ;;
        olmo_rl_zero_if) echo "if" ;;
        olmo_rl_zero_general) echo "general" ;;
        olmo_rl_zero_mix) echo "mix" ;;
        olmo_instruct_rl) echo "instruct" ;;
        *) return 1 ;;
    esac
}

olmo_import_data_source_for_name() {
    local canonical
    canonical=$(canonical_olmo_import_stage "$1")
    case "$canonical" in
        olmo_rl_zero_math) echo "math" ;;
        olmo_rl_zero_code) echo "olmo_rl_zero_code" ;;
        olmo_rl_zero_if) echo "olmo_rl_zero_if" ;;
        olmo_rl_zero_general) echo "olmo_rl_zero_general" ;;
        olmo_rl_zero_mix) echo "olmo_rl_zero_mix" ;;
        olmo_instruct_rl) echo "olmo_instruct_rl" ;;
        *) return 1 ;;
    esac
}

default_stage_variant_for_name() {
    if is_olmo_import_stage "$1"; then
        echo "gt"
        return 0
    fi
    case "$1" in
        synthetic2_sft_verified|dolci_think_sft_7b|dolci_think_sft_32b) echo "sf" ;;
        tau2bench) echo "gt" ;;
        *) echo "$DATA_VARIANT" ;;
    esac
}

default_sft_data_dir_for_name() {
    if is_olmo_import_stage "$1"; then
        local canonical
        canonical=$(canonical_olmo_import_stage "$1")
        echo "$SCRATCH_ROOT/data/${canonical}_sft"
        return 0
    fi
    case "$1" in
        synthetic2_sft_verified) echo "$SCRATCH_ROOT/data/synthetic2_sft_verified_sf_sft" ;;
        dolci_think_sft_7b) echo "$SCRATCH_ROOT/data/dolci_think_sft_7b_sf_sft" ;;
        dolci_think_sft_32b) echo "$SCRATCH_ROOT/data/dolci_think_sft_32b_sf_sft" ;;
        tau2bench) echo "$SCRATCH_ROOT/data/tau2bench_sft" ;;
        *) echo "" ;;
    esac
}

default_rl_backend_for_name() {
    if is_olmo_import_stage "$1"; then
        echo "dataset_dir"
        return 0
    fi
    case "$1" in
        synthetic2_sft_verified|dolci_think_sft_7b|dolci_think_sft_32b|tau2bench)
            echo "dataset_dir"
            ;;
        *)
            echo "prime"
            ;;
    esac
}

default_rl_data_dir_for_name() {
    if is_olmo_import_stage "$1"; then
        local variant
        variant=$(olmo_import_variant_name "$1") || return 1
        echo "$SCRATCH_ROOT/data/olmo_rl_${variant}"
        return 0
    fi
    case "$1" in
        synthetic2_sft_verified) echo "$SCRATCH_ROOT/data/synthetic2_sft_verified_rl" ;;
        dolci_think_sft_7b) echo "$SCRATCH_ROOT/data/dolci_think_sft_7b_rl" ;;
        dolci_think_sft_32b) echo "$SCRATCH_ROOT/data/dolci_think_sft_32b_rl" ;;
        tau2bench) echo "$SCRATCH_ROOT/data/tau2bench_rl" ;;
        *) echo "" ;;
    esac
}

default_rl_data_source_for_name() {
    if is_olmo_import_stage "$1"; then
        olmo_import_data_source_for_name "$1"
        return 0
    fi
    case "$1" in
        tau2bench) echo "tau2bench" ;;
        *) echo "$1" ;;
    esac
}

maybe_prepare_stage_datasets() {
    local stage_name="$1"
    local sft_dir="$2"
    local rl_dir="$3"
    local rl_data_source
    rl_data_source=$(default_rl_data_source_for_name "$stage_name")

    if [[ "$stage_name" == "tau2bench" && "${TAU2BENCH_SMOKE_BOOTSTRAP:-0}" == "1" ]]; then
        if [[ -n "$sft_dir" && ! -f "$sft_dir/train.parquet" && ( -z "$rl_dir" || ! -f "$rl_dir/train.parquet" ) ]]; then
            python3 "$REPO_DIR/scripts/bootstrap_tau2bench_smoke_data.py" \
              --output-dir "$sft_dir" >/dev/null
        fi
    fi

    case "$stage_name" in
        synthetic2_sft_verified|dolci_think_sft_7b|dolci_think_sft_32b|tau2bench)
            if [[ -n "$sft_dir" && -f "$sft_dir/train.parquet" && -n "$rl_dir" && ! -f "$rl_dir/train.parquet" ]]; then
                python3 "$REPO_DIR/scripts/convert_sft_dataset_to_rl.py" \
                  --input-dir "$sft_dir" \
                  --output-dir "$rl_dir" \
                  --data-source "$rl_data_source" >/dev/null
            fi
            if [[ "$stage_name" == "tau2bench" && -n "$rl_dir" && -f "$rl_dir/train.parquet" && -n "$sft_dir" && ! -f "$sft_dir/train.parquet" ]]; then
                python3 "$REPO_DIR/scripts/convert_rl_dataset_to_sft.py" \
                  --input-dir "$rl_dir" \
                  --output-dir "$sft_dir" \
                  --data-source "$rl_data_source" >/dev/null
            fi
            ;;
        *)
            if is_olmo_import_stage "$stage_name"; then
                if [[ -n "$rl_dir" && -f "$rl_dir/train.parquet" && -n "$sft_dir" && ! -f "$sft_dir/train.parquet" ]]; then
                    python3 "$REPO_DIR/scripts/convert_rl_dataset_to_sft.py" \
                      --input-dir "$rl_dir" \
                      --output-dir "$sft_dir" \
                      --data-source "$rl_data_source" >/dev/null
                fi
            fi
            ;;
    esac
}

fail_missing_stage_dataset() {
    local stage_name="$1"
    local mode="$2"
    local sft_dir="$3"
    local rl_dir="$4"
    local stage_key="$5"
    local sft_var="STAGE_DATA_DIR_${stage_key}"
    local rl_var="STAGE_RL_DATA_DIR_${stage_key}"

    echo "ERROR: Stage '$stage_name' is missing required parquet data for $mode." >&2
    if [[ -n "$sft_dir" ]]; then
        echo "Expected SFT parquet: $sft_dir/train.parquet" >&2
    fi
    if [[ -n "$rl_dir" ]]; then
        echo "Expected RL parquet:  $rl_dir/train.parquet" >&2
    fi
    echo "Set $sft_var or $rl_var to an existing parquet root, or prepare one of the default dataset dirs first." >&2
    if [[ "$stage_name" == "tau2bench" ]]; then
        echo "tau2bench is a parquet-backed local stage in this repo, not the native Sierra tau2-bench environment." >&2
        echo "At least one of these must exist before launch:" >&2
        echo "  $SCRATCH_ROOT/data/tau2bench_sft/train.parquet" >&2
        echo "  $SCRATCH_ROOT/data/tau2bench_rl/train.parquet" >&2
        echo "For a launcher-only smoke test, set TAU2BENCH_SMOKE_BOOTSTRAP=1 to generate a tiny local tau2bench parquet." >&2
    fi
    exit 1
}

validate_stage_dataset_inputs() {
    local stage_name="$1"
    local training_mode="$2"
    local rl_backend="$3"
    local sft_dir="$4"
    local rl_dir="$5"
    local stage_key="$6"

    if [[ "$training_mode" == "sft" || "$training_mode" == "sft_rl" ]]; then
        if [[ -z "$sft_dir" || ! -f "$sft_dir/train.parquet" ]]; then
            fail_missing_stage_dataset "$stage_name" "SFT" "$sft_dir" "$rl_dir" "$stage_key"
        fi
    fi

    if [[ "$rl_backend" == "dataset_dir" && ( "$training_mode" == "rl" || "$training_mode" == "sft_rl" ) ]]; then
        if [[ -z "$rl_dir" || ! -f "$rl_dir/train.parquet" ]]; then
            fail_missing_stage_dataset "$stage_name" "dataset-dir RL" "$sft_dir" "$rl_dir" "$stage_key"
        fi
    fi
}

infer_model_max_len() {
    python3 "$REPO_DIR/scripts/infer_model_max_len.py" --model "$1" 2>/dev/null || true
}

resolve_prime_model_path() {
    local run_dir="$1"
    local search_root="$run_dir/checkpoints"
    [[ -d "$search_root" ]] || search_root="$run_dir"

    local step_dir
    local step_dirs=()
    while IFS= read -r step_dir; do
        step_dirs+=("$step_dir")
    done < <(find "$search_root" -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort -t_ -k2 -n)

    if [[ ${#step_dirs[@]} -gt 0 ]]; then
        local idx
        for (( idx=${#step_dirs[@]}-1; idx>=0; idx-- )); do
            local candidate_config
            candidate_config=$(find "${step_dirs[$idx]}" -type f -name 'config.json' | head -n 1)
            if [[ -n "$candidate_config" ]]; then
                dirname "$candidate_config"
                return 0
            fi
        done
    fi

    local candidates=()
    while IFS= read -r candidate; do
        candidates+=("$candidate")
    done < <(find "$search_root" -type f -name 'config.json' -exec dirname {} \; 2>/dev/null | sort -u)

    if [[ ${#candidates[@]} -eq 0 ]]; then
        return 1
    fi
    printf '%s\n' "${candidates[0]}"
}

resolve_dataset_dir_rl_model_path() {
    local run_dir="$1"
    local checkpoint_dir
    local checkpoint_dirs=()
    while IFS= read -r checkpoint_dir; do
        checkpoint_dirs+=("$checkpoint_dir")
    done < <(find "$run_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -t_ -k3 -n)

    if [[ ${#checkpoint_dirs[@]} -eq 0 ]]; then
        [[ -f "$run_dir/config.json" ]] && printf '%s\n' "$run_dir" && return 0
        return 1
    fi

    local latest="${checkpoint_dirs[$((${#checkpoint_dirs[@]} - 1))]}"
    if [[ -d "$latest/actor" ]]; then
        local merged_dir="$latest/actor_merged"
        if [[ ! -f "$merged_dir/config.json" ]]; then
            python3 -m verl.model_merger merge \
              --backend fsdp \
              --local_dir "$latest/actor" \
              --target_dir "$merged_dir" >/dev/null
        fi
        printf '%s\n' "$merged_dir"
        return 0
    fi
    if ls "$latest"/model_world_size_*_rank_*.pt &>/dev/null; then
        local merged_dir="$latest/merged_hf"
        if [[ ! -f "$merged_dir/config.json" ]]; then
            python3 -m verl.model_merger merge \
              --backend fsdp \
              --local_dir "$latest" \
              --target_dir "$merged_dir" >/dev/null
        fi
        printf '%s\n' "$merged_dir"
        return 0
    fi
    if [[ -f "$latest/config.json" ]]; then
        printf '%s\n' "$latest"
        return 0
    fi
    return 1
}

resolve_sft_model_path() {
    local run_dir="$1"
    local merged_dir="$run_dir/merged_hf"
    if [[ -f "$merged_dir/config.json" ]]; then
        printf '%s\n' "$merged_dir"
        return 0
    fi
    if [[ -f "$run_dir/adapter_config.json" ]]; then
        python3 "$REPO_DIR/scripts/merge_lora_checkpoint.py" \
          --adapter-dir "$run_dir" >/dev/null
        if [[ -f "$merged_dir/config.json" ]]; then
            printf '%s\n' "$merged_dir"
            return 0
        fi
    fi
    printf '%s\n' "$run_dir"
}

run_benchmark_eval() {
    local model_path="$1"
    local eval_name="$2"
    local output_dir="$BENCHMARK_EVAL_ROOT/$eval_name"
    if [[ "$RUN_BENCHMARK_EVALS" != "1" ]]; then
        return 0
    fi
    if [[ "$SKIP_EXISTING" == "1" && -f "$output_dir/eval_summary.json" ]]; then
        echo "Skipping benchmark eval: $eval_name"
        return 0
    fi
    local cmd=(
        python3 "$REPO_DIR/src/evaluation/run_eval.py"
        --model_path "$model_path"
        --suite "$EVAL_SUITE"
        --output_dir "$output_dir"
        --run_name "$eval_name"
    )
    if [[ "$SKIP_EXISTING" != "1" ]]; then
        cmd+=(--force-rerun)
    fi
    local eval_gpu="${BENCHMARK_EVAL_GPU:-$ASYNC_EVAL_GPU}"
    if [[ "$AUTO_START_EVAL_SERVER" == "1" && -n "$eval_gpu" ]]; then
        env \
            CUDA_VISIBLE_DEVICES="$eval_gpu" \
            EVAL_SUITE="$EVAL_SUITE" \
            EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS" \
            CONTINUE_ON_ERROR="$EVAL_CONTINUE_ON_ERROR" \
            SKIP_EXISTING="$SKIP_EXISTING" \
            BENCHMARK_ENV_FILE="$BENCHMARK_ENV_FILE" \
            bash "$SCRIPT_DIR/run_eval_with_local_server.sh" \
              "$model_path" \
              "$output_dir" \
              "$eval_name"
    else
        if [[ "$EVAL_CONTINUE_ON_ERROR" == "1" ]]; then
            cmd+=(--continue_on_error)
        fi
        if [[ -n "$EVAL_EXTRA_ARGS" ]]; then
            # shellcheck disable=SC2206
            local extra_args=( $EVAL_EXTRA_ARGS )
            cmd+=("${extra_args[@]}")
        fi
        "${cmd[@]}"
    fi
}

run_task_eval_one() {
    local model_path="$1"
    local eval_name="$2"
    local dataset="$3"
    local built_in=0
    case "$dataset" in
        gsm8k|math|triviaqa|polaris_math|openr1_math) built_in=1 ;;
    esac
    local dataset_key
    dataset_key=$(stage_key "$dataset")
    local dataset_path_var="TASK_EVAL_DATASET_PATH_${dataset_key}"
    local data_source_var="TASK_EVAL_DATA_SOURCE_${dataset_key}"
    local dataset_path="${!dataset_path_var:-}"
    local data_source="${!data_source_var:-}"
    if [[ "$built_in" != "1" && -z "$dataset_path" ]]; then
        echo "Skipping local task eval for $dataset: set $dataset_path_var to a parquet path if you want backward-task eval on this stage."
        return 0
    fi
    local run_name="${eval_name}_task_${dataset}"
    local output_root="$RUNS_ROOT/task_evals"
    local effective_task_eval_max_model_len="${TASK_EVAL_MAX_MODEL_LEN:-}"
    if [[ -z "$effective_task_eval_max_model_len" ]]; then
        effective_task_eval_max_model_len=$(infer_model_max_len "$model_path")
    fi
    local cmd=(
        python3 "$REPO_DIR/scripts/prime_rl_runner.py" baseline
        --model "$model_path"
        --dataset "$dataset"
        --output-root "$output_root"
        --run-name "$run_name"
        --max-tokens "$TASK_EVAL_MAX_TOKENS"
        --temperature "$TASK_EVAL_TEMPERATURE"
        --top-p "$TASK_EVAL_TOP_P"
        --rollouts-per-prompt "$TASK_PASS_K"
        --tensor-parallel-size "$TASK_EVAL_TP"
        --gpu-memory-utilization "$TASK_EVAL_GPU_MEMORY_UTILIZATION"
        --hf-home "$HF_HOME"
        --wandb-mode disabled
    )
    if [[ -n "$effective_task_eval_max_model_len" ]]; then
        cmd+=(--max-model-len "$effective_task_eval_max_model_len")
    fi
    if [[ -n "$dataset_path" ]]; then
        cmd+=(--dataset-path "$dataset_path")
    fi
    if [[ -n "$data_source" ]]; then
        cmd+=(--data-source "$data_source")
    fi
    if [[ -n "$TASK_EVAL_MAX_SAMPLES" ]]; then
        cmd+=(--max-samples "$TASK_EVAL_MAX_SAMPLES")
    fi
    "${cmd[@]}"
}

resolve_task_eval_datasets() {
    local stage_name="$1"
    shift
    local datasets=("$@")

    local stage_key_name
    stage_key_name=$(stage_key "$stage_name")
    local explicit_var="STAGE_TASK_EVAL_DATASETS_${stage_key_name}"
    local explicit_datasets_raw="${!explicit_var:-}"
    if [[ -n "$explicit_datasets_raw" ]]; then
        # shellcheck disable=SC2206
        datasets=( $explicit_datasets_raw )
    fi

    if [[ "$TASK_EVAL_REVERSE" == "1" && ${#datasets[@]} -gt 1 ]]; then
        local reversed=()
        local idx
        for (( idx=${#datasets[@]}-1; idx>=0; idx-- )); do
            reversed+=("${datasets[$idx]}")
        done
        datasets=("${reversed[@]}")
    fi

    printf '%s\n' "${datasets[@]}"
}

run_task_eval_chain() {
    local model_path="$1"
    local eval_name="$2"
    local stage_name="$3"
    shift 3
    if [[ "$RUN_TASK_EVALS" != "1" ]]; then
        return 0
    fi
    local datasets=()
    while IFS= read -r dataset; do
        [[ -n "$dataset" ]] && datasets+=("$dataset")
    done < <(resolve_task_eval_datasets "$stage_name" "$@")
    local dataset
    for dataset in "${datasets[@]}"; do
        run_task_eval_one "$model_path" "$eval_name" "$dataset"
    done
}

run_prime_stage() {
    local model_path="$1"
    local dataset="$2"
    local run_name="$3"
    local dataset_key
    dataset_key=$(stage_key "$dataset")

    local env_var="STAGE_ENV_${dataset_key}"
    local combined_var="STAGE_COMBINED_CONFIG_${dataset_key}"
    local trainer_var="STAGE_TRAINER_CONFIG_${dataset_key}"
    local orchestrator_var="STAGE_ORCHESTRATOR_CONFIG_${dataset_key}"
    local inference_var="STAGE_INFERENCE_CONFIG_${dataset_key}"

    local environment_name="${!env_var:-$dataset}"
    local combined_cfg="${!combined_var:-}"
    local trainer_cfg="${!trainer_var:-}"
    local orchestrator_cfg="${!orchestrator_var:-}"
    local inference_cfg="${!inference_var:-}"

    local run_dir="$PRIME_RUNS_ROOT/$run_name"
    if [[ "$SKIP_EXISTING" == "1" && -f "$run_dir/completed.marker" ]]; then
        echo "Skipping completed PRIME stage: $run_name"
        return 0
    fi

    local cmd=(
        python3 "$REPO_DIR/scripts/prime_rl_runner.py" prime
        --model "$model_path"
        --environment-name "$environment_name"
        --output-root "$PRIME_RUNS_ROOT"
        --hf-home "$HF_HOME"
        --run-name "$run_name"
        --max-steps "$RL_MAX_STEPS"
        --max-async-level "$PRIME_MAX_ASYNC_LEVEL"
        --batch-size "$PRIME_BATCH_SIZE"
        --seq-len "$PRIME_SEQ_LEN"
        --max-tokens "$PRIME_MAX_TOKENS"
        --temperature 1.0
        --top-p 1.0
        --rollouts-per-prompt "$PRIME_ROLLOUTS_PER_PROMPT"
        --fake-data-batch-size "$PRIME_BATCH_SIZE"
        --ckpt-interval "$RL_CKPT_INTERVAL"
        --ckpt-keep-last 2
        --ckpt-keep-interval "$RL_CKPT_INTERVAL"
        --prime-command "$PRIME_COMMAND"
        --prime-extra-args "$PRIME_EXTRA_ARGS"
        --wandb-project forgetting-llms
        --wandb-mode "$PRIME_WANDB_MODE"
        --execute
    )
    if [[ "$PRIME_NO_AUTO_RESUME" == "1" ]]; then
        cmd+=(--no-auto-resume)
    fi
    if [[ "$PRIME_ENFORCE_EAGER" == "1" ]]; then
        cmd+=(--enforce-eager)
    fi
    if [[ -n "$combined_cfg" ]]; then
        cmd+=(--combined-config "$combined_cfg")
    else
        if [[ -n "$trainer_cfg" ]]; then
            cmd+=(--trainer-config "$trainer_cfg")
        fi
        if [[ -n "$orchestrator_cfg" ]]; then
            cmd+=(--orchestrator-config "$orchestrator_cfg")
        fi
        if [[ -n "$inference_cfg" ]]; then
            cmd+=(--inference-config "$inference_cfg")
        fi
    fi
    "${cmd[@]}"
}

run_dataset_dir_rl_stage() {
    local model_path="$1"
    local dataset="$2"
    local run_name="$3"
    local dataset_key
    dataset_key=$(stage_key "$dataset")

    local rl_data_dir_var="STAGE_RL_DATA_DIR_${dataset_key}"
    local rl_data_dir="${!rl_data_dir_var:-}"
    if [[ -z "$rl_data_dir" ]]; then
        rl_data_dir=$(default_rl_data_dir_for_name "$dataset")
    fi
    if [[ ! -f "$rl_data_dir/train.parquet" ]]; then
        echo "ERROR: Dataset-dir RL parquet not found for stage $dataset: $rl_data_dir/train.parquet" >&2
        exit 1
    fi

    local run_dir="$CHECKPOINT_ROOT/$run_name"
    local max_prompt_tokens
    max_prompt_tokens=$(( PRIME_SEQ_LEN > PRIME_MAX_TOKENS ? PRIME_SEQ_LEN - PRIME_MAX_TOKENS : PRIME_SEQ_LEN ))
    if [[ "$SKIP_EXISTING" == "1" && -f "$run_dir/completed.marker" ]]; then
        echo "Skipping completed dataset-dir RL stage: $run_name"
        return 0
    fi

    env \
        SAVE_DIR="$run_dir" \
        TRAIN_CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
        TOTAL_EPOCHS="$RL_TOTAL_EPOCHS" \
        MAX_PROMPT="$max_prompt_tokens" \
        MAX_RESPONSE="$PRIME_MAX_TOKENS" \
        SAVE_FREQ="$RL_CKPT_INTERVAL" \
        TEST_FREQ="$RL_CKPT_INTERVAL" \
        TRAIN_BATCH_SIZE="$PRIME_BATCH_SIZE" \
        ROLLOUTS_PER_PROMPT="$PRIME_ROLLOUTS_PER_PROMPT" \
        WANDB_MODE="$PRIME_WANDB_MODE" \
        bash "$REPO_DIR/scripts/run_grpo_dataset_dir_local.sh" \
          "$dataset" \
          "$rl_data_dir" \
          "$model_path" \
          "$run_name"
}

python3 - <<PY
import json
from pathlib import Path
payload = {
    "training_mode": ${TRAINING_MODE@Q},
    "schedule": ${SCHEDULE@Q},
    "base_model": ${BASE_MODEL@Q},
    "run_prefix": ${RUN_PREFIX@Q},
    "stages": ${STAGES[*]@Q}.split(),
    "data_variant": ${DATA_VARIANT@Q},
    "sft_max_steps": int(${SFT_MAX_STEPS@Q}),
    "rl_max_steps": int(${RL_MAX_STEPS@Q}),
}
Path(${MANIFEST_FILE@Q}).write_text(json.dumps(payload, indent=2) + "\\n")
PY

CURRENT_MODEL="$BASE_MODEL"
COMPLETED_STAGES=()

for idx in "${!STAGES[@]}"; do
    STAGE="${STAGES[$idx]}"
    STAGE_NUM=$((idx + 1))
    STAGE_KEY=$(stage_key "$STAGE")
    STAGE_VARIANT_VAR="STAGE_VARIANT_${STAGE_KEY}"
    STAGE_DATA_DIR_VAR="STAGE_DATA_DIR_${STAGE_KEY}"
    STAGE_RL_BACKEND_VAR="STAGE_RL_BACKEND_${STAGE_KEY}"
    STAGE_RL_DATA_DIR_VAR="STAGE_RL_DATA_DIR_${STAGE_KEY}"
    STAGE_VARIANT="${!STAGE_VARIANT_VAR:-$(default_stage_variant_for_name "$STAGE")}"
    STAGE_DATA_DIR="${!STAGE_DATA_DIR_VAR:-$(default_sft_data_dir_for_name "$STAGE")}"
    STAGE_RL_BACKEND="${!STAGE_RL_BACKEND_VAR:-$(default_rl_backend_for_name "$STAGE")}"
    STAGE_RL_DATA_DIR="${!STAGE_RL_DATA_DIR_VAR:-$(default_rl_data_dir_for_name "$STAGE")}"
    maybe_prepare_stage_datasets "$STAGE" "$STAGE_DATA_DIR" "$STAGE_RL_DATA_DIR"
    validate_stage_dataset_inputs "$STAGE" "$TRAINING_MODE" "$STAGE_RL_BACKEND" "$STAGE_DATA_DIR" "$STAGE_RL_DATA_DIR" "$STAGE_KEY"

    if [[ "$SCHEDULE" == "individual" ]]; then
        STAGE_BASE_MODEL="$BASE_MODEL"
    else
        STAGE_BASE_MODEL="$CURRENT_MODEL"
    fi

    echo "========================================="
    echo "Stage $STAGE_NUM/${#STAGES[@]}: $STAGE"
    echo "Training mode: $TRAINING_MODE"
    echo "Schedule:      $SCHEDULE"
    echo "Base model:    $STAGE_BASE_MODEL"
    echo "RL backend:    $STAGE_RL_BACKEND"
    echo "========================================="

    STAGE_FINAL_MODEL="$STAGE_BASE_MODEL"

    if [[ "$TRAINING_MODE" == "sft" || "$TRAINING_MODE" == "sft_rl" ]]; then
        SFT_NAME="${RUN_PREFIX}_stage$(printf '%02d' "$STAGE_NUM")_${STAGE}_sft"
        SFT_SAVE_DIR="$CHECKPOINT_ROOT/$SFT_NAME"
        ASYNC_EVAL_STOP_FILE="$PLAN_ROOT/${SFT_NAME}.async_eval.stop"
        ASYNC_EVAL_PID=""

        if [[ -n "$ASYNC_EVAL_GPU" && "$RUN_BENCHMARK_EVALS" == "1" ]]; then
            rm -f "$ASYNC_EVAL_STOP_FILE"
            EVAL_GPU="$ASYNC_EVAL_GPU" \
            EVAL_POLL_SECS="$ASYNC_EVAL_POLL_SECS" \
            EVAL_SUITE="$EVAL_SUITE" \
            EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS" \
            AUTO_START_EVAL_SERVER="$AUTO_START_EVAL_SERVER" \
            BENCHMARK_ENV_FILE="$BENCHMARK_ENV_FILE" \
            STOP_FILE="$ASYNC_EVAL_STOP_FILE" \
            TRAIN_COMPLETED_MARKER="$SFT_SAVE_DIR/completed.marker" \
            SKIP_EXISTING="$SKIP_EXISTING" \
            CONTINUE_ON_ERROR="$EVAL_CONTINUE_ON_ERROR" \
            bash "$SCRIPT_DIR/watch_checkpoints_eval.sh" \
              "$SFT_SAVE_DIR" \
              "$BENCHMARK_EVAL_ROOT/${SFT_NAME}_async" \
              "$SFT_NAME" &
            ASYNC_EVAL_PID=$!
        fi

        if [[ "$SKIP_EXISTING" == "1" && -f "$SFT_SAVE_DIR/completed.marker" ]]; then
            echo "Skipping completed SFT stage: $SFT_NAME"
        else
            MAX_STEPS="$SFT_MAX_STEPS" \
            SAVE_DIR="$SFT_SAVE_DIR" \
            DATA_DIR="$STAGE_DATA_DIR" \
            TRAIN_CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
            WANDB_MODE="$WANDB_MODE" \
            bash "$SCRIPT_DIR/run_sft.sh" "$STAGE" "$STAGE_BASE_MODEL" "$SFT_NAME" "$STAGE_VARIANT"
        fi

        if [[ -n "$ASYNC_EVAL_PID" ]]; then
            touch "$ASYNC_EVAL_STOP_FILE"
            wait "$ASYNC_EVAL_PID"
        fi

        STAGE_FINAL_MODEL=$(resolve_sft_model_path "$SFT_SAVE_DIR")
        run_benchmark_eval "$STAGE_FINAL_MODEL" "$SFT_NAME"
        if [[ "$SCHEDULE" == "sequential" ]]; then
            run_task_eval_chain "$STAGE_FINAL_MODEL" "$SFT_NAME" "$STAGE" "${COMPLETED_STAGES[@]}" "$STAGE"
        else
            run_task_eval_chain "$STAGE_FINAL_MODEL" "$SFT_NAME" "$STAGE" "$STAGE"
        fi
    fi

    if [[ "$TRAINING_MODE" == "rl" || "$TRAINING_MODE" == "sft_rl" ]]; then
        RL_NAME="${RUN_PREFIX}_stage$(printf '%02d' "$STAGE_NUM")_${STAGE}_rl"
        if [[ "$STAGE_RL_BACKEND" == "dataset_dir" ]]; then
            RL_RUN_DIR="$CHECKPOINT_ROOT/$RL_NAME"
        else
            RL_RUN_DIR="$PRIME_RUNS_ROOT/$RL_NAME"
        fi
        RL_ASYNC_EVAL_STOP_FILE="$PLAN_ROOT/${RL_NAME}.async_eval.stop"
        RL_ASYNC_EVAL_PID=""

        if [[ "$STAGE_RL_BACKEND" == "prime" && -n "$ASYNC_EVAL_GPU" && ( "$RUN_BENCHMARK_EVALS" == "1" || "$RUN_TASK_EVALS" == "1" ) ]]; then
            echo "Starting async PRIME checkpoint eval watcher for $RL_NAME on GPU $ASYNC_EVAL_GPU"
            rl_task_eval_datasets=()
            rl_task_eval_extra_args=()
            while IFS= read -r dataset; do
                [[ -n "$dataset" ]] && rl_task_eval_datasets+=("$dataset")
            done < <(resolve_task_eval_datasets "$STAGE" "$STAGE")
            if [[ "$RUN_TASK_EVALS" == "1" ]]; then
                task_eval_dataset=""
                for task_eval_dataset in "${rl_task_eval_datasets[@]}"; do
                    rl_task_eval_extra_args+=(--task-eval-dataset "$task_eval_dataset")
                done
            fi
            rl_sweep_extra_args=""
            if [[ ${#rl_task_eval_extra_args[@]} -gt 0 ]]; then
                rl_sweep_extra_args=$(printf '%q ' "${rl_task_eval_extra_args[@]}")
                rl_sweep_extra_args="${rl_sweep_extra_args% }"
            fi
            rm -f "$RL_ASYNC_EVAL_STOP_FILE"
            EVAL_GPU="$ASYNC_EVAL_GPU" \
            EVAL_POLL_SECS="$ASYNC_EVAL_POLL_SECS" \
            EVAL_SUITE="$EVAL_SUITE" \
            TASK_PASS_K="$TASK_PASS_K" \
            TASK_EVAL_MAX_SAMPLES="$TASK_EVAL_MAX_SAMPLES" \
            TASK_EVAL_MAX_MODEL_LEN="$TASK_EVAL_MAX_MODEL_LEN" \
            TASK_EVAL_MAX_TOKENS="$TASK_EVAL_MAX_TOKENS" \
            TASK_EVAL_TP="$TASK_EVAL_TP" \
            TASK_EVAL_GPU_MEMORY_UTILIZATION="$TASK_EVAL_GPU_MEMORY_UTILIZATION" \
            SKIP_BENCHMARK_EVALS="$([[ "$RUN_BENCHMARK_EVALS" == "1" ]] && echo 0 || echo 1)" \
            SKIP_TASK_EVALS="$([[ "$RUN_TASK_EVALS" == "1" ]] && echo 0 || echo 1)" \
            CONTINUE_ON_ERROR="$EVAL_CONTINUE_ON_ERROR" \
            AUTO_START_EVAL_SERVER="$AUTO_START_EVAL_SERVER" \
            BENCHMARK_ENV_FILE="$BENCHMARK_ENV_FILE" \
            SWEEP_EXTRA_ARGS="$rl_sweep_extra_args" \
            STOP_FILE="$RL_ASYNC_EVAL_STOP_FILE" \
            TRAIN_COMPLETED_MARKER="$RL_RUN_DIR/completed.marker" \
            bash "$SCRIPT_DIR/watch_prime_run_eval.sh" \
              "$PRIME_RUNS_ROOT" \
              "$RL_NAME" \
              "$BENCHMARK_EVAL_ROOT" &
            RL_ASYNC_EVAL_PID=$!
        elif [[ "$STAGE_RL_BACKEND" == "prime" ]]; then
            if [[ -z "$ASYNC_EVAL_GPU" ]]; then
                echo "Async PRIME checkpoint eval disabled for $RL_NAME: ASYNC_EVAL_GPU is unset."
            fi
            if [[ "$RUN_BENCHMARK_EVALS" != "1" && "$RUN_TASK_EVALS" != "1" ]]; then
                echo "Async PRIME checkpoint eval disabled for $RL_NAME: RUN_BENCHMARK_EVALS=0 and RUN_TASK_EVALS=0."
            fi
        fi

        start_checkpoint_mirror "$RL_RUN_DIR" "rl:$RL_NAME"
        if [[ "$STAGE_RL_BACKEND" == "dataset_dir" ]]; then
            run_dataset_dir_rl_stage "$STAGE_FINAL_MODEL" "$STAGE" "$RL_NAME"
        else
            ensure_prime_runtime
            run_prime_stage "$STAGE_FINAL_MODEL" "$STAGE" "$RL_NAME"
        fi
        stop_checkpoint_mirror
        if [[ -n "$RL_ASYNC_EVAL_PID" ]]; then
            touch "$RL_ASYNC_EVAL_STOP_FILE"
            wait "$RL_ASYNC_EVAL_PID"
        fi
        if [[ "$STAGE_RL_BACKEND" == "dataset_dir" ]]; then
            RL_MODEL_PATH=$(resolve_dataset_dir_rl_model_path "$RL_RUN_DIR") || {
                echo "ERROR: Could not resolve dataset-dir RL model path for $RL_NAME" >&2
                exit 1
            }
        else
            RL_MODEL_PATH=$(resolve_prime_model_path "$RL_RUN_DIR") || {
                echo "ERROR: Could not resolve PRIME model path for $RL_NAME" >&2
                exit 1
            }
        fi
        STAGE_FINAL_MODEL="$RL_MODEL_PATH"
        if [[ "$STAGE_RL_BACKEND" == "dataset_dir" ]]; then
            run_benchmark_eval "$STAGE_FINAL_MODEL" "$RL_NAME"
            if [[ "$SCHEDULE" == "sequential" ]]; then
                run_task_eval_chain "$STAGE_FINAL_MODEL" "$RL_NAME" "$STAGE" "${COMPLETED_STAGES[@]}" "$STAGE"
            else
                run_task_eval_chain "$STAGE_FINAL_MODEL" "$RL_NAME" "$STAGE" "$STAGE"
            fi
        else
            if [[ "$RUN_BENCHMARK_EVALS" == "1" || "$RUN_TASK_EVALS" == "1" ]]; then
                RL_EVAL_GPU="${BENCHMARK_EVAL_GPU:-$ASYNC_EVAL_GPU}"
                RL_EVAL_CMD=(
                    python3 "$REPO_DIR/scripts/eval_prime_checkpoint_sweep.py"
                    --prime-runs-root "$PRIME_RUNS_ROOT"
                    --output-root "$BENCHMARK_EVAL_ROOT"
                    --run "$RL_NAME"
                    --only-latest
                    --suite "$EVAL_SUITE"
                    --task-pass-k "$TASK_PASS_K"
                    --task-max-tokens "$TASK_EVAL_MAX_TOKENS"
                    --task-tensor-parallel-size "$TASK_EVAL_TP"
                    --task-gpu-memory-utilization "$TASK_EVAL_GPU_MEMORY_UTILIZATION"
                    --benchmark-env-file "$BENCHMARK_ENV_FILE"
                )
                if [[ -n "$TASK_EVAL_MAX_SAMPLES" ]]; then
                    RL_EVAL_CMD+=(--task-max-samples "$TASK_EVAL_MAX_SAMPLES")
                fi
                if [[ "$RUN_BENCHMARK_EVALS" != "1" ]]; then
                    RL_EVAL_CMD+=(--skip-benchmark-evals)
                fi
                if [[ "$RUN_TASK_EVALS" != "1" ]]; then
                    RL_EVAL_CMD+=(--skip-task-evals)
                fi
                rl_task_eval_dataset=""
                while IFS= read -r rl_task_eval_dataset; do
                    [[ -n "$rl_task_eval_dataset" ]] || continue
                    RL_EVAL_CMD+=(--task-eval-dataset "$rl_task_eval_dataset")
                done < <(resolve_task_eval_datasets "$STAGE" "$STAGE")
                if [[ -n "$TASK_EVAL_MAX_MODEL_LEN" ]]; then
                    RL_EVAL_CMD+=(--task-max-model-len "$TASK_EVAL_MAX_MODEL_LEN")
                fi
                if [[ "$AUTO_START_EVAL_SERVER" == "1" ]]; then
                    RL_EVAL_CMD+=(--auto-start-eval-server)
                fi
                if [[ "$EVAL_CONTINUE_ON_ERROR" == "1" ]]; then
                    RL_EVAL_CMD+=(--continue-on-error)
                    env \
                        CUDA_VISIBLE_DEVICES="${RL_EVAL_GPU:-${CUDA_VISIBLE_DEVICES:-}}" \
                        AUTO_START_EVAL_SERVER="$AUTO_START_EVAL_SERVER" \
                        BENCHMARK_ENV_FILE="$BENCHMARK_ENV_FILE" \
                        "${RL_EVAL_CMD[@]}"
                else
                    env \
                        CUDA_VISIBLE_DEVICES="${RL_EVAL_GPU:-${CUDA_VISIBLE_DEVICES:-}}" \
                        AUTO_START_EVAL_SERVER="$AUTO_START_EVAL_SERVER" \
                        BENCHMARK_ENV_FILE="$BENCHMARK_ENV_FILE" \
                        "${RL_EVAL_CMD[@]}"
                fi
            else
                echo "Skipping final PRIME eval sweep for $RL_NAME: RUN_BENCHMARK_EVALS=0 and RUN_TASK_EVALS=0."
            fi
        fi
    fi

    if [[ "$SCHEDULE" == "sequential" ]]; then
        CURRENT_MODEL="$STAGE_FINAL_MODEL"
        COMPLETED_STAGES+=("$STAGE")
    fi

    python3 - <<PY
import json
from pathlib import Path
state = {
    "current_stage": ${STAGE@Q},
    "current_model": ${STAGE_FINAL_MODEL@Q},
    "completed_stages": ${COMPLETED_STAGES[*]@Q}.split(),
}
Path(${STATE_FILE@Q}).write_text(json.dumps(state, indent=2) + "\\n")
PY
done

echo "========================================="
echo "Training plan complete"
echo "Final model: ${CURRENT_MODEL:-$BASE_MODEL}"
echo "Plan root:   $PLAN_ROOT"
echo "========================================="
