#!/usr/bin/env bash
# Run the currently wired experiment matrix sequentially inside an existing
# interactive Slurm allocation.
#
# This wrapper is resumable: if a run already has checkpoints or metrics, it is
# skipped by default.
#
# Usage:
#   bash scripts/run_all_tasks_interactive.sh
#   RUN_PREFIX=mar21_full RUN_OLMO_BASELINES=0 bash scripts/run_all_tasks_interactive.sh
#   RUN_EVALS=0 RUN_CF_SFT=0 bash scripts/run_all_tasks_interactive.sh

set -euo pipefail
shopt -s nullglob

usage() {
    cat <<'EOF'
Run the repo's currently wired experiment matrix sequentially inside an existing interactive allocation.

Environment knobs:
  RUN_PREFIX                      Prefix added to every experiment name
  SKIP_EXISTING=1                 Skip runs with existing checkpoints or metrics
  ORCH_ROOT                       Scratch directory for orchestration state
  RUN_RL_PHASES=1                 Master switch for all RL phases
  DATASETS_CSV                    Default: gsm8k,math,triviaqa (add polaris_math as needed)
  PAIR_MATRIX_CSV                 Default: gsm8k:math,gsm8k:triviaqa,math:triviaqa
  RUN_EVALS=1                     Run task + OOD evals where supported
  RUN_SF_SFT=0                    Generate same-family positive-only SFT datasets
  RUN_QWEN_SEQUENTIAL=1           Run 6 Qwen sequential GRPO order experiments
  RUN_MIX=1                       Run 3 Qwen mixed GRPO experiments
  RUN_IID=1                       Run 3 Qwen IID staged controls
  RUN_CF_SFT=0                    Run 3 CF-SFT experiments after trajectory generation
  RUN_OLMO_BASELINES=1            Run 3 OLMo baseline evaluations
  RUN_OLMO_SEQUENTIAL=1           Run 6 OLMo sequential GRPO order experiments
  RUN_LORA_ABLATION=0             Run GT/SF LoRA vs full SFT ablations

Model knobs:
  QWEN_MODEL                      Default: Qwen/Qwen3-1.7B
  QWEN_SF_TEACHER                 Default: Qwen/Qwen3-32B
  CF_TEACHER                      Default: meta-llama/Llama-3.1-70B-Instruct
  OLMO_MODEL                      Default: $HOME/scratch/olmo3_7B-Instruct
  LORA_RANK                       Default: 64

PRIME-RL knobs:
  PRIME_RUNS_ROOT                 Default: ~/scratch/forgetting-llms/prime_runs
  PRIME_COMMAND                   Default: uv --project \$PRIME_RL_ROOT run rl
  PRIME_WANDB_MODE                Default: online
  PRIME_MAX_STEPS                 Default: 1000
  SEQUENTIAL_STAGE_STEPS          Default: 300
  PRIME_CKPT_INTERVAL_STEPS       Default: 5
  PRIME_CKPT_KEEP_LAST            Default: 3
  PRIME_CKPT_KEEP_INTERVAL        Default: 50
  PRIME_BATCH_SIZE                Default: 256
  PRIME_SEQ_LEN                   Default: 8192
  PRIME_MAX_TOKENS                Default: 1024
  PRIME_ROLLOUTS_PER_PROMPT       Default: 8
  PRIME_MAX_ASYNC_LEVEL           Default: 1
  PRIME_ENFORCE_EAGER             Default: 0
  PRIME_EXTRA_ARGS                Extra raw PRIME CLI args, shell-quoted as needed

PRIME config mapping:
  RL phases require PRIME-RL wiring. Missing env/config entries fail fast.
  Single-dataset labels use:
    PRIME_ENV_GSM8K / PRIME_COMBINED_CONFIG_GSM8K
    PRIME_ENV_MATH / PRIME_COMBINED_CONFIG_MATH
    PRIME_ENV_TRIVIAQA / PRIME_COMBINED_CONFIG_TRIVIAQA
    PRIME_ENV_POLARIS_MATH / PRIME_COMBINED_CONFIG_POLARIS_MATH
  Pair labels use:
    PRIME_ENV_MIX_GSM8K_MATH / PRIME_COMBINED_CONFIG_MIX_GSM8K_MATH
    PRIME_ENV_IID_GSM8K_MATH / PRIME_COMBINED_CONFIG_IID_GSM8K_MATH
  Split configs are also supported via PRIME_TRAINER_CONFIG_*,
  PRIME_ORCHESTRATOR_CONFIG_*, PRIME_INFERENCE_CONFIG_*.

Sampling knobs for synthetic SFT data:
  SAMPLES_PER_ROUND               Default: 4
  MAX_TOTAL_SAMPLES               Default: 16
  TARGET_CORRECT_PER_QUESTION     Default: 2
  MIN_CORRECT_PER_QUESTION        Default: 2
  SOLUTIONS_PER_QUESTION          Default: 2
  TEMPERATURE                     Default: 1.0
  TOP_P                           Default: 1.0

GRPO knobs:
  GRPO_MAX_PROMPT                 Default: 512
  GRPO_MAX_RESPONSE               Default: 1024
  GRPO_TOTAL_EPOCHS               Default: 15
  GRPO_TRAIN_BATCH_SIZE           Default: 16
  GRPO_PPO_MINI_BATCH_SIZE        Default: 16
  GRPO_PPO_MICRO_BATCH_SIZE       Default: 2
  GRPO_ROLLOUTS_PER_PROMPT        Default: 8
  GRPO_ROLLOUT_LOGPROB_BATCH      Default: 4
  GRPO_ACTOR_LR                   Default: 1e-6

Example:
  RUN_PREFIX=mar21_full bash scripts/run_all_tasks_interactive.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/forgetting-llms}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_ROOT/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH_ROOT/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-$SCRATCH_ROOT/eval_results}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH_ROOT/runs}"
MANUAL_RUNS_ROOT="${MANUAL_RUNS_ROOT:-$SCRATCH_ROOT/manual_runs}"
PRIME_RUNS_ROOT="${PRIME_RUNS_ROOT:-$SCRATCH_ROOT/prime_runs}"
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export PYTHONUNBUFFERED=1

module load python/3.10 >/dev/null 2>&1 || true
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    # Respect the caller's active environment, which is typically scratch-local on cluster.
    :
elif [[ -n "${VENV_DIR:-}" && -f "${VENV_DIR/#\~/$HOME}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR/#\~/$HOME}/bin/activate"
elif [[ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
elif [[ -f "$HOME/envs/forgetting/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/envs/forgetting/bin/activate"
else
    echo "No virtual environment found. Expected an active VIRTUAL_ENV, VENV_DIR, $HOME/scratch/forgetting-llms/.venv, $REPO_DIR/.venv, or $HOME/envs/forgetting." >&2
    exit 1
fi

mkdir -p "$SCRATCH_ROOT" "$DATA_ROOT" "$CHECKPOINT_ROOT" "$EVAL_ROOT" "$RUNS_ROOT" "$MANUAL_RUNS_ROOT"

RUN_PREFIX="${RUN_PREFIX:-full_tasks_$(date +%Y%m%d)}"
ORCH_ROOT="${ORCH_ROOT:-$SCRATCH_ROOT/orchestration/$RUN_PREFIX}"
ORCH_EXPERIMENTS_DIR="$ORCH_ROOT/experiments"
ORCH_DATA_DIR="$ORCH_ROOT/data"
ORCH_STATE_FILE="$ORCH_ROOT/current_state.env"
ORCH_EVENTS_FILE="$ORCH_ROOT/events.tsv"
ORCH_PLAN_FILE="$ORCH_ROOT/plan.txt"
ORCH_ENV_FILE="$ORCH_ROOT/run_env.env"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_RL_PHASES="${RUN_RL_PHASES:-1}"
DATASETS_CSV="${DATASETS_CSV:-gsm8k,math,triviaqa}"
PAIR_MATRIX_CSV="${PAIR_MATRIX_CSV:-gsm8k:math,gsm8k:triviaqa,math:triviaqa}"
RUN_EVALS="${RUN_EVALS:-1}"
RUN_SF_SFT="${RUN_SF_SFT:-0}"
RUN_QWEN_SEQUENTIAL="${RUN_QWEN_SEQUENTIAL:-1}"
RUN_MIX="${RUN_MIX:-1}"
RUN_IID="${RUN_IID:-1}"
RUN_CF_SFT="${RUN_CF_SFT:-0}"
RUN_OLMO_BASELINES="${RUN_OLMO_BASELINES:-1}"
RUN_OLMO_SEQUENTIAL="${RUN_OLMO_SEQUENTIAL:-1}"
RUN_LORA_ABLATION="${RUN_LORA_ABLATION:-0}"

if [[ "$RUN_RL_PHASES" == "0" ]]; then
    RUN_QWEN_SEQUENTIAL=0
    RUN_MIX=0
    RUN_IID=0
    RUN_OLMO_SEQUENTIAL=0
fi

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-1.7B}"
QWEN_SF_TEACHER="${QWEN_SF_TEACHER:-Qwen/Qwen3-32B}"
CF_TEACHER="${CF_TEACHER:-meta-llama/Llama-3.1-70B-Instruct}"
OLMO_MODEL="${OLMO_MODEL:-$HOME/scratch/olmo3_7B-Instruct}"
LORA_RANK="${LORA_RANK:-64}"

PRIME_RL_ROOT="${PRIME_RL_ROOT:-$SCRATCH_ROOT/vendor/prime-rl}"
PRIME_COMMAND="${PRIME_COMMAND:-uv --project $PRIME_RL_ROOT run rl}"
PRIME_WANDB_MODE="${PRIME_WANDB_MODE:-online}"
PRIME_MAX_STEPS="${PRIME_MAX_STEPS:-1000}"
SEQUENTIAL_STAGE_STEPS="${SEQUENTIAL_STAGE_STEPS:-300}"
PRIME_CKPT_INTERVAL_STEPS="${PRIME_CKPT_INTERVAL_STEPS:-5}"
PRIME_CKPT_KEEP_LAST="${PRIME_CKPT_KEEP_LAST:-3}"
PRIME_CKPT_KEEP_INTERVAL="${PRIME_CKPT_KEEP_INTERVAL:-50}"
PRIME_BATCH_SIZE="${PRIME_BATCH_SIZE:-256}"
PRIME_SEQ_LEN="${PRIME_SEQ_LEN:-8192}"
PRIME_MAX_TOKENS="${PRIME_MAX_TOKENS:-1024}"
PRIME_ROLLOUTS_PER_PROMPT="${PRIME_ROLLOUTS_PER_PROMPT:-8}"
PRIME_MAX_ASYNC_LEVEL="${PRIME_MAX_ASYNC_LEVEL:-1}"
PRIME_ENFORCE_EAGER="${PRIME_ENFORCE_EAGER:-0}"
PRIME_EXTRA_ARGS="${PRIME_EXTRA_ARGS:-}"

SAMPLES_PER_ROUND="${SAMPLES_PER_ROUND:-4}"
MAX_TOTAL_SAMPLES="${MAX_TOTAL_SAMPLES:-16}"
TARGET_CORRECT_PER_QUESTION="${TARGET_CORRECT_PER_QUESTION:-2}"
MIN_CORRECT_PER_QUESTION="${MIN_CORRECT_PER_QUESTION:-2}"
SOLUTIONS_PER_QUESTION="${SOLUTIONS_PER_QUESTION:-2}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

GRPO_MAX_PROMPT="${GRPO_MAX_PROMPT:-512}"
GRPO_MAX_RESPONSE="${GRPO_MAX_RESPONSE:-1024}"
GRPO_TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS:-15}"
GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-16}"
GRPO_PPO_MINI_BATCH_SIZE="${GRPO_PPO_MINI_BATCH_SIZE:-16}"
GRPO_PPO_MICRO_BATCH_SIZE="${GRPO_PPO_MICRO_BATCH_SIZE:-2}"
GRPO_ROLLOUTS_PER_PROMPT="${GRPO_ROLLOUTS_PER_PROMPT:-8}"
GRPO_ROLLOUT_LOGPROB_BATCH="${GRPO_ROLLOUT_LOGPROB_BATCH:-4}"
GRPO_ACTOR_LR="${GRPO_ACTOR_LR:-1e-6}"

export MAX_TOTAL_SAMPLES TARGET_CORRECT_PER_QUESTION MIN_CORRECT_PER_QUESTION
export SOLUTIONS_PER_QUESTION TEMPERATURE TOP_P
export MAX_PROMPT="$GRPO_MAX_PROMPT"
export MAX_RESPONSE="$GRPO_MAX_RESPONSE"
export TOTAL_EPOCHS="$GRPO_TOTAL_EPOCHS"
export TRAIN_BATCH_SIZE="$GRPO_TRAIN_BATCH_SIZE"
export PPO_MINI_BATCH_SIZE="$GRPO_PPO_MINI_BATCH_SIZE"
export PPO_MICRO_BATCH_SIZE_PER_GPU="$GRPO_PPO_MICRO_BATCH_SIZE"
export ROLLOUTS_PER_PROMPT="$GRPO_ROLLOUTS_PER_PROMPT"
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="$GRPO_ROLLOUT_LOGPROB_BATCH"
export ACTOR_LR="$GRPO_ACTOR_LR"

IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
IFS=',' read -r -a RAW_PAIR_SPECS <<< "$PAIR_MATRIX_CSV"
PAIRS=()
for pair_spec in "${RAW_PAIR_SPECS[@]}"; do
    pair_spec="${pair_spec//:/ }"
    [[ -n "${pair_spec// }" ]] || continue
    PAIRS+=("$pair_spec")
done

mkdir -p "$ORCH_ROOT" "$ORCH_EXPERIMENTS_DIR" "$ORCH_DATA_DIR" "$PRIME_RUNS_ROOT"

CURRENT_PHASE=""
CURRENT_UNIT=""
CURRENT_EXPERIMENT=""
CURRENT_KIND=""
CURRENT_OUTPUT_PATH=""
CURRENT_START_TS=""

log() {
    printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

qualify_name() {
    local name="$1"
    printf '%s_%s\n' "$RUN_PREFIX" "$name"
}

timestamp_utc() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

safe_name() {
    echo "$1" | tr '/ ' '__'
}

to_env_key() {
    echo "$1" | tr '[:lower:]-:/. ' '[:upper:]______'
}

get_env_value() {
    local var_name="$1"
    printf '%s' "${!var_name:-}"
}

default_prime_env_value() {
    local key="$1"
    printf '%s' "${key,,}"
}

default_prime_config_path() {
    local key="$1"
    local suffix="$2"
    if [[ -z "${PRIME_CONFIG_ROOT:-}" ]]; then
        return 0
    fi
    printf '%s' "$PRIME_CONFIG_ROOT/${key,,}.${suffix}.toml"
}

path_exists() {
    local path="$1"
    [[ -f "${path/#\~/$HOME}" ]]
}

prime_label_key() {
    local label="$1"
    to_env_key "$label"
}

prime_env_for_label() {
    local key
    key=$(prime_label_key "$1")
    local value
    value=$(get_env_value "PRIME_ENV_${key}")
    if [[ -n "$value" ]]; then
        printf '%s' "$value"
    else
        default_prime_env_value "$key"
    fi
}

prime_combined_config_for_label() {
    local key
    key=$(prime_label_key "$1")
    get_env_value "PRIME_COMBINED_CONFIG_${key}"
}

prime_trainer_config_for_label() {
    local key
    key=$(prime_label_key "$1")
    local value
    value=$(get_env_value "PRIME_TRAINER_CONFIG_${key}")
    if [[ -n "$value" ]]; then
        printf '%s' "$value"
    else
        default_prime_config_path "$key" "trainer"
    fi
}

prime_orchestrator_config_for_label() {
    local key
    key=$(prime_label_key "$1")
    local value
    value=$(get_env_value "PRIME_ORCHESTRATOR_CONFIG_${key}")
    if [[ -n "$value" ]]; then
        printf '%s' "$value"
    else
        default_prime_config_path "$key" "orchestrator"
    fi
}

prime_inference_config_for_label() {
    local key
    key=$(prime_label_key "$1")
    local value
    value=$(get_env_value "PRIME_INFERENCE_CONFIG_${key}")
    if [[ -n "$value" ]]; then
        printf '%s' "$value"
    else
        default_prime_config_path "$key" "inference"
    fi
}

pair_label() {
    local mode="$1"
    local dataset_a="$2"
    local dataset_b="$3"
    printf '%s_%s_%s\n' "$mode" "$dataset_a" "$dataset_b"
}

prime_label_ready() {
    local label="$1"
    local env_name combined trainer_cfg orchestrator_cfg inference_cfg
    env_name=$(prime_env_for_label "$label")
    [[ -n "$env_name" ]] || return 1

    combined=$(prime_combined_config_for_label "$label")
    if [[ -n "$combined" ]] && path_exists "$combined"; then
        return 0
    fi

    trainer_cfg=$(prime_trainer_config_for_label "$label")
    orchestrator_cfg=$(prime_orchestrator_config_for_label "$label")
    inference_cfg=$(prime_inference_config_for_label "$label")
    if [[ -n "$trainer_cfg" && -n "$orchestrator_cfg" && -n "$inference_cfg" ]] \
        && path_exists "$trainer_cfg" \
        && path_exists "$orchestrator_cfg" \
        && path_exists "$inference_cfg"; then
        return 0
    fi

    # PRIME bundle generation can fall back to generated stub configs when no
    # checked-in or scratch-local TOMLs exist yet.
    return 0
}

should_skip_prime_experiment() {
    local experiment="$1"
    [[ "$SKIP_EXISTING" == "1" && -f "$PRIME_RUNS_ROOT/$experiment/completed.marker" ]]
}

resolve_prime_model_path() {
    local experiment="$1"
    local run_dir="$PRIME_RUNS_ROOT/$experiment"
    local explicit_var
    local explicit_path
    explicit_var="PRIME_RESUME_MODEL_PATH_$(prime_label_key "$experiment")"
    explicit_path=$(get_env_value "$explicit_var")
    if [[ -n "$explicit_path" ]]; then
        printf '%s\n' "$explicit_path"
        return 0
    fi

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
                printf '%s\n' "$(dirname "$candidate_config")"
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

    local preferred
    for preferred in final latest merged hf export; do
        local candidate
        for candidate in "${candidates[@]}"; do
            if [[ "$candidate" == *"$preferred"* ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    done

    printf '%s\n' "${candidates[0]}"
}

require_prime_label() {
    local label="$1"
    prime_label_ready "$label" || die "Missing PRIME-RL env/config wiring for label $label"
}

run_prime_experiment() {
    local model="$1"
    local label="$2"
    local experiment="$3"
    local phase="$4"
    local max_steps="${5:-$PRIME_MAX_STEPS}"
    local env_name combined_cfg trainer_cfg orchestrator_cfg inference_cfg
    local run_dir="$PRIME_RUNS_ROOT/$experiment"
    local cmd

    env_name=$(prime_env_for_label "$label")
    combined_cfg=$(prime_combined_config_for_label "$label")
    trainer_cfg=$(prime_trainer_config_for_label "$label")
    orchestrator_cfg=$(prime_orchestrator_config_for_label "$label")
    inference_cfg=$(prime_inference_config_for_label "$label")

    begin_unit "$phase" "$experiment" "$experiment" "prime" "$run_dir"
    if should_skip_prime_experiment "$experiment"; then
        log "Skipping PRIME-RL experiment $experiment"
        snapshot_experiment_state "$experiment" "prime" "$phase" "skipped"
        finish_unit "skipped" ""
        return 0
    fi

    cmd=(
        python "$SCRIPT_DIR/prime_rl_runner.py" prime
        --model "$model"
        --environment-name "$env_name"
        --output-root "$PRIME_RUNS_ROOT"
        --hf-home "$HF_HOME"
        --run-name "$experiment"
        --max-steps "$max_steps"
        --ckpt-interval "$PRIME_CKPT_INTERVAL_STEPS"
        --ckpt-keep-last "$PRIME_CKPT_KEEP_LAST"
        --ckpt-keep-interval "$PRIME_CKPT_KEEP_INTERVAL"
        --max-async-level "$PRIME_MAX_ASYNC_LEVEL"
        --batch-size "$PRIME_BATCH_SIZE"
        --seq-len "$PRIME_SEQ_LEN"
        --max-tokens "$PRIME_MAX_TOKENS"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --rollouts-per-prompt "$PRIME_ROLLOUTS_PER_PROMPT"
        --prime-command "$PRIME_COMMAND"
        --wandb-mode "$PRIME_WANDB_MODE"
        --wandb-project forgetting-llms
        --execute
    )
    if [[ -n "$PRIME_EXTRA_ARGS" ]]; then
        cmd+=(--prime-extra-args "$PRIME_EXTRA_ARGS")
    fi
    if [[ "$PRIME_ENFORCE_EAGER" == "1" ]]; then
        cmd+=(--enforce-eager)
    fi
    if [[ -n "$combined_cfg" && -f "${combined_cfg/#\~/$HOME}" ]]; then
        cmd+=(--combined-config "$combined_cfg")
    elif [[ -n "$trainer_cfg" && -n "$orchestrator_cfg" && -n "$inference_cfg" \
        && -f "${trainer_cfg/#\~/$HOME}" && -f "${orchestrator_cfg/#\~/$HOME}" && -f "${inference_cfg/#\~/$HOME}" ]]; then
        cmd+=(--trainer-config "$trainer_cfg" --orchestrator-config "$orchestrator_cfg" --inference-config "$inference_cfg")
    else
        log "No PRIME config files found for label $label; using generated PRIME stub configs"
    fi

    log "Running PRIME-RL experiment $experiment via label $label"
    "${cmd[@]}"
    touch "$run_dir/completed.marker"
    snapshot_experiment_state "$experiment" "prime" "$phase" "completed"
    finish_unit "completed" ""
}

run_prime_eval_bundle() {
    local experiment="$1"
    local model_path="$2"
    local phase_name="$3"
    shift 3
    local dataset task_path task_run_name

    begin_unit "$phase_name" "${experiment}_prime_eval" "$experiment" "prime_eval" "$EVAL_ROOT/$experiment"

    mkdir -p "$EVAL_ROOT/$experiment"
    for dataset in "$@"; do
        task_path="$EVAL_ROOT/$experiment/task_accuracy_${dataset}.json"
        if ! should_skip_file "$task_path"; then
            task_run_name="${experiment}_taskeval_${dataset}"
            python "$SCRIPT_DIR/prime_rl_runner.py" baseline \
                --model "$model_path" \
                --dataset "$dataset" \
                --output-root "$RUNS_ROOT" \
                --run-name "$task_run_name" \
                --hf-home "$HF_HOME" \
                --rollouts-per-prompt 1 \
                --temperature 0.0 \
                --top-p 1.0 \
                --max-model-len 8192 \
                --max-tokens 1024 \
                --wandb-mode disabled
            cp "$RUNS_ROOT/$task_run_name/metrics/metrics.json" "$task_path"
        fi
    done

    if [[ ! -f "$EVAL_ROOT/$experiment/eval_summary.json" ]]; then
        python "$REPO_DIR/src/evaluation/run_eval.py" \
            --model_path "$model_path" \
            --suite forgetting \
            --output_dir "$EVAL_ROOT/$experiment" \
            --run_name "$experiment"
    fi

    snapshot_experiment_state "$experiment" "prime_eval" "$phase_name" "completed"
    finish_unit "completed" ""
}

write_orch_plan() {
    cat > "$ORCH_PLAN_FILE" <<'EOF'
Execution order:
1. Same-family positive SFT data generation
2. LoRA vs full SFT ablations
3. Cross-family SFT data generation + CF-SFT
4. OLMo baseline evaluations
5. RL phases last:
   - Qwen sequential
   - Qwen mixed
   - Qwen IID
   - OLMo sequential
EOF
}

write_orch_env() {
    cat > "$ORCH_ENV_FILE" <<EOF
RUN_PREFIX=$RUN_PREFIX
SCRATCH_ROOT=$SCRATCH_ROOT
DATA_ROOT=$DATA_ROOT
CHECKPOINT_ROOT=$CHECKPOINT_ROOT
EVAL_ROOT=$EVAL_ROOT
RUNS_ROOT=$RUNS_ROOT
MANUAL_RUNS_ROOT=$MANUAL_RUNS_ROOT
ORCH_ROOT=$ORCH_ROOT
SKIP_EXISTING=$SKIP_EXISTING
RUN_RL_PHASES=$RUN_RL_PHASES
RUN_EVALS=$RUN_EVALS
RUN_SF_SFT=$RUN_SF_SFT
RUN_QWEN_SEQUENTIAL=$RUN_QWEN_SEQUENTIAL
RUN_MIX=$RUN_MIX
RUN_IID=$RUN_IID
RUN_CF_SFT=$RUN_CF_SFT
RUN_OLMO_BASELINES=$RUN_OLMO_BASELINES
RUN_OLMO_SEQUENTIAL=$RUN_OLMO_SEQUENTIAL
RUN_LORA_ABLATION=$RUN_LORA_ABLATION
QWEN_MODEL=$QWEN_MODEL
QWEN_SF_TEACHER=$QWEN_SF_TEACHER
CF_TEACHER=$CF_TEACHER
OLMO_MODEL=$OLMO_MODEL
LORA_RANK=$LORA_RANK
PRIME_RUNS_ROOT=$PRIME_RUNS_ROOT
PRIME_COMMAND=$PRIME_COMMAND
PRIME_WANDB_MODE=$PRIME_WANDB_MODE
PRIME_MAX_STEPS=$PRIME_MAX_STEPS
SEQUENTIAL_STAGE_STEPS=$SEQUENTIAL_STAGE_STEPS
PRIME_CKPT_INTERVAL_STEPS=$PRIME_CKPT_INTERVAL_STEPS
PRIME_CKPT_KEEP_LAST=$PRIME_CKPT_KEEP_LAST
PRIME_CKPT_KEEP_INTERVAL=$PRIME_CKPT_KEEP_INTERVAL
PRIME_BATCH_SIZE=$PRIME_BATCH_SIZE
PRIME_SEQ_LEN=$PRIME_SEQ_LEN
PRIME_MAX_TOKENS=$PRIME_MAX_TOKENS
PRIME_ROLLOUTS_PER_PROMPT=$PRIME_ROLLOUTS_PER_PROMPT
PRIME_MAX_ASYNC_LEVEL=$PRIME_MAX_ASYNC_LEVEL
PRIME_ENFORCE_EAGER=$PRIME_ENFORCE_EAGER
DATASETS_CSV=$DATASETS_CSV
PAIR_MATRIX_CSV=$PAIR_MATRIX_CSV
SAMPLES_PER_ROUND=$SAMPLES_PER_ROUND
MAX_TOTAL_SAMPLES=$MAX_TOTAL_SAMPLES
TARGET_CORRECT_PER_QUESTION=$TARGET_CORRECT_PER_QUESTION
MIN_CORRECT_PER_QUESTION=$MIN_CORRECT_PER_QUESTION
SOLUTIONS_PER_QUESTION=$SOLUTIONS_PER_QUESTION
TEMPERATURE=$TEMPERATURE
TOP_P=$TOP_P
GRPO_MAX_PROMPT=$GRPO_MAX_PROMPT
GRPO_MAX_RESPONSE=$GRPO_MAX_RESPONSE
GRPO_TOTAL_EPOCHS=$GRPO_TOTAL_EPOCHS
GRPO_TRAIN_BATCH_SIZE=$GRPO_TRAIN_BATCH_SIZE
GRPO_PPO_MINI_BATCH_SIZE=$GRPO_PPO_MINI_BATCH_SIZE
GRPO_PPO_MICRO_BATCH_SIZE=$GRPO_PPO_MICRO_BATCH_SIZE
GRPO_ROLLOUTS_PER_PROMPT=$GRPO_ROLLOUTS_PER_PROMPT
GRPO_ROLLOUT_LOGPROB_BATCH=$GRPO_ROLLOUT_LOGPROB_BATCH
GRPO_ACTOR_LR=$GRPO_ACTOR_LR
SLURM_JOB_ID=${SLURM_JOB_ID:-}
HOSTNAME=$(hostname)
STARTED_AT=$(timestamp_utc)
EOF
}

write_current_state() {
    local status="$1"
    cat > "$ORCH_STATE_FILE" <<EOF
STATUS=$status
UPDATED_AT=$(timestamp_utc)
SLURM_JOB_ID=${SLURM_JOB_ID:-}
RUN_PREFIX=$RUN_PREFIX
CURRENT_PHASE=${CURRENT_PHASE:-}
CURRENT_UNIT=${CURRENT_UNIT:-}
CURRENT_EXPERIMENT=${CURRENT_EXPERIMENT:-}
CURRENT_KIND=${CURRENT_KIND:-}
CURRENT_OUTPUT_PATH=${CURRENT_OUTPUT_PATH:-}
CURRENT_START_TS=${CURRENT_START_TS:-}
EOF
}

append_event() {
    local status="$1"
    local note="${2:-}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp_utc)" \
        "$status" \
        "${CURRENT_PHASE:-}" \
        "${CURRENT_UNIT:-}" \
        "${CURRENT_EXPERIMENT:-}" \
        "${CURRENT_KIND:-}" \
        "$note" >> "$ORCH_EVENTS_FILE"
}

snapshot_experiment_state() {
    local experiment="$1"
    local kind="$2"
    local phase="$3"
    local state="$4"
    local snapshot="$ORCH_EXPERIMENTS_DIR/$(safe_name "$experiment").env"
    local checkpoint_dir="$CHECKPOINT_ROOT/$experiment"
    local eval_dir="$EVAL_ROOT/$experiment"
    local run_dir="$RUNS_ROOT/$experiment"
    local prime_run_dir="$PRIME_RUNS_ROOT/$experiment"
    local manual_run_dir="$MANUAL_RUNS_ROOT/$experiment"
    local checkpoint_count="0"
    local latest_step=""
    local latest_checkpoint=""
    local task_eval_count="0"
    local ood_eval_count="0"
    local plots_dir=""
    local metrics_path=""
    local backend_hint="standard"

    if [[ ! -d "$checkpoint_dir" && -d "$prime_run_dir/checkpoints" ]]; then
        checkpoint_dir="$prime_run_dir/checkpoints"
        backend_hint="prime"
    fi
    if [[ ! -d "$run_dir" && -d "$prime_run_dir" ]]; then
        run_dir="$prime_run_dir"
        backend_hint="prime"
    fi

    if [[ -d "$checkpoint_dir" ]]; then
        if compgen -G "$checkpoint_dir/global_step_*" >/dev/null; then
            checkpoint_count=$(find "$checkpoint_dir" -maxdepth 1 -name 'global_step_*' -type d | wc -l | tr -d ' ')
            latest_checkpoint=$(find "$checkpoint_dir" -maxdepth 1 -name 'global_step_*' -type d | sort -t_ -k3 -n | tail -n 1)
            if [[ -n "$latest_checkpoint" ]]; then
                latest_step=$(basename "$latest_checkpoint" | awk -F_ '{print $3}')
            fi
        elif compgen -G "$checkpoint_dir/step_*" >/dev/null; then
            checkpoint_count=$(find "$checkpoint_dir" -maxdepth 1 -name 'step_*' -type d | wc -l | tr -d ' ')
            latest_checkpoint=$(find "$checkpoint_dir" -maxdepth 1 -name 'step_*' -type d | sort -t_ -k2 -n | tail -n 1)
            if [[ -n "$latest_checkpoint" ]]; then
                latest_step=$(basename "$latest_checkpoint" | awk -F_ '{print $2}')
            fi
        else
            checkpoint_count=$(find "$checkpoint_dir" -mindepth 1 | wc -l | tr -d ' ')
            latest_checkpoint=$(find "$checkpoint_dir" -maxdepth 3 -type f -name 'config.json' | head -n 1)
            if [[ -n "$latest_checkpoint" ]]; then
                latest_checkpoint=$(dirname "$latest_checkpoint")
            fi
        fi
    fi

    if [[ -d "$eval_dir" ]]; then
        task_eval_count=$(find "$eval_dir" -maxdepth 1 -name 'task_accuracy*.json' -type f | wc -l | tr -d ' ')
        ood_eval_count=$(find "$eval_dir" -maxdepth 1 -mindepth 1 -type d ! -name 'plots' | wc -l | tr -d ' ')
        if [[ -d "$eval_dir/plots" ]]; then
            plots_dir="$eval_dir/plots"
        fi
    fi

    if [[ -f "$run_dir/metrics/metrics.json" ]]; then
        metrics_path="$run_dir/metrics/metrics.json"
    fi

    local wandb_dir="$SCRATCH_ROOT/wandb/$experiment"
    local wandb_cache_dir="$SCRATCH_ROOT/wandb_cache/$experiment"
    if [[ -d "$run_dir/wandb" ]]; then
        wandb_dir="$run_dir/wandb"
    fi
    if [[ -d "$run_dir/wandb-cache" ]]; then
        wandb_cache_dir="$run_dir/wandb-cache"
    fi

    cat > "$snapshot" <<EOF
EXPERIMENT=$experiment
KIND=$kind
PHASE=$phase
STATE=$state
BACKEND_HINT=$backend_hint
UPDATED_AT=$(timestamp_utc)
CHECKPOINT_DIR=$checkpoint_dir
CHECKPOINT_COUNT=$checkpoint_count
LATEST_STEP=${latest_step:-}
LATEST_CHECKPOINT=${latest_checkpoint:-}
EVAL_DIR=$eval_dir
TASK_EVAL_COUNT=$task_eval_count
OOD_EVAL_COUNT=$ood_eval_count
PLOTS_DIR=${plots_dir:-}
RUN_DIR=$run_dir
METRICS_PATH=${metrics_path:-}
MANUAL_RUN_DIR=$manual_run_dir
WANDB_DIR=$wandb_dir
WANDB_CACHE_DIR=$wandb_cache_dir
EOF
}

snapshot_data_state() {
    local unit_name="$1"
    local phase="$2"
    local state="$3"
    local output_dir="$4"
    local snapshot="$ORCH_DATA_DIR/$(safe_name "$unit_name").env"

    cat > "$snapshot" <<EOF
UNIT=$unit_name
PHASE=$phase
STATE=$state
UPDATED_AT=$(timestamp_utc)
OUTPUT_DIR=$output_dir
TRAIN_EXISTS=$([[ -f "$output_dir/train.parquet" ]] && echo 1 || echo 0)
TEST_EXISTS=$([[ -f "$output_dir/test.parquet" ]] && echo 1 || echo 0)
CHECKPOINT_EXISTS=$([[ -f "$output_dir/checkpoint.parquet" ]] && echo 1 || echo 0)
STATUS_EXISTS=$([[ -f "$output_dir/status.parquet" ]] && echo 1 || echo 0)
SUMMARY_EXISTS=$([[ -f "$output_dir/summary.json" ]] && echo 1 || echo 0)
MANIFEST_EXISTS=$([[ -f "$output_dir/manifest.json" ]] && echo 1 || echo 0)
EOF
}

begin_unit() {
    CURRENT_PHASE="$1"
    CURRENT_UNIT="$2"
    CURRENT_EXPERIMENT="${3:-}"
    CURRENT_KIND="${4:-generic}"
    CURRENT_OUTPUT_PATH="${5:-}"
    CURRENT_START_TS="$(timestamp_utc)"
    write_current_state "running"
    append_event "running" ""
}

finish_unit() {
    local status="$1"
    local note="${2:-}"
    write_current_state "$status"
    append_event "$status" "$note"
    CURRENT_PHASE=""
    CURRENT_UNIT=""
    CURRENT_EXPERIMENT=""
    CURRENT_KIND=""
    CURRENT_OUTPUT_PATH=""
    CURRENT_START_TS=""
}

handle_exit() {
    local exit_code="$1"
    if [[ -n "${CURRENT_UNIT:-}" ]]; then
        if [[ -n "${CURRENT_EXPERIMENT:-}" ]]; then
            snapshot_experiment_state "$CURRENT_EXPERIMENT" "$CURRENT_KIND" "$CURRENT_PHASE" "interrupted"
        elif [[ -n "${CURRENT_OUTPUT_PATH:-}" ]]; then
            snapshot_data_state "$CURRENT_UNIT" "$CURRENT_PHASE" "interrupted" "$CURRENT_OUTPUT_PATH"
        fi
        write_current_state "interrupted"
        append_event "interrupted" "exit_code=$exit_code"
    fi
}

trap 'handle_exit $?' EXIT INT TERM

write_orch_plan
write_orch_env
if [[ ! -f "$ORCH_EVENTS_FILE" ]]; then
    printf 'timestamp_utc\tstatus\tphase\tunit\texperiment\tkind\tnote\n' > "$ORCH_EVENTS_FILE"
fi
write_current_state "initialized"

has_checkpoints() {
    local experiment="$1"
    compgen -G "$CHECKPOINT_ROOT/$experiment/global_step_*" >/dev/null
}

latest_checkpoint_dir() {
    local experiment="$1"
    local ckpt
    ckpt=$(find "$CHECKPOINT_ROOT/$experiment" -maxdepth 1 -name "global_step_*" -type d | sort -t_ -k3 -n | tail -n 1)
    [[ -n "${ckpt:-}" ]] || die "No checkpoint found for $experiment under $CHECKPOINT_ROOT/$experiment"
    printf '%s\n' "$ckpt"
}

should_skip_experiment() {
    local experiment="$1"
    [[ "$SKIP_EXISTING" == "1" ]] && has_checkpoints "$experiment"
}

should_skip_file() {
    local path="$1"
    [[ "$SKIP_EXISTING" == "1" && -f "$path" ]]
}

verify_file() {
    local path="$1"
    [[ -f "$path" ]] || die "Expected file missing: $path"
}

verify_dir_nonempty() {
    local path="$1"
    [[ -d "$path" ]] || die "Expected directory missing: $path"
    compgen -G "$path/*" >/dev/null || die "Directory is empty: $path"
}

ensure_grpo_data() {
    local dataset="$1"
    local dir="$DATA_ROOT/$dataset"
    if [[ -f "$dir/train.parquet" && -f "$dir/test.parquet" ]]; then
        return 0
    fi
    log "Preprocessing GRPO data for $dataset"
    python "$SCRIPT_DIR/preprocess_data.py" --dataset "$dataset" --format grpo --output_dir "$dir"
    verify_file "$dir/train.parquet"
    verify_file "$dir/test.parquet"
}

ensure_gt_sft_data() {
    local dataset="$1"
    local dir="$DATA_ROOT/${dataset}_sft"
    if [[ -f "$dir/train.parquet" && -f "$dir/test.parquet" ]]; then
        return 0
    fi
    log "Preprocessing GT-SFT data for $dataset"
    python "$SCRIPT_DIR/preprocess_data.py" --dataset "$dataset" --format sft --output_dir "$dir"
    verify_file "$dir/train.parquet"
    verify_file "$dir/test.parquet"
}

run_task_eval() {
    local experiment="$1"
    local dataset="$2"
    local base_model="$3"
    local result="$EVAL_ROOT/$experiment/task_accuracy_${dataset}.json"
    if should_skip_file "$result"; then
        log "Skipping task eval for $experiment on $dataset"
        return 0
    fi
    log "Task eval: $experiment on $dataset"
    bash "$SCRIPT_DIR/eval_task_accuracy.sh" "$CHECKPOINT_ROOT/$experiment" "$dataset" "$base_model" "$dataset"
    verify_file "$result"
}

run_ood_eval() {
    local experiment="$1"
    local base_model="$2"
    local plots_dir="$EVAL_ROOT/$experiment/plots"
    if [[ "$SKIP_EXISTING" == "1" && -d "$plots_dir" ]]; then
        log "Skipping OOD eval sweep for $experiment"
        return 0
    fi
    log "OOD eval sweep: $experiment"
    bash "$SCRIPT_DIR/eval_sweep_resumable.sh" "$CHECKPOINT_ROOT/$experiment" "$experiment" "$base_model"
    verify_dir_nonempty "$plots_dir"
}

run_standard_eval_bundle() {
    local experiment="$1"
    local base_model="$2"
    local phase_name="${3:-eval}"
    shift 3
    local dataset
    begin_unit "$phase_name" "${experiment}_eval" "$experiment" "eval" "$EVAL_ROOT/$experiment"
    for dataset in "$@"; do
        run_task_eval "$experiment" "$dataset" "$base_model"
    done
    run_ood_eval "$experiment" "$base_model"
    snapshot_experiment_state "$experiment" "eval" "$phase_name" "completed"
    finish_unit "completed" ""
}

run_same_family_sft_data() {
    local dataset="$1"
    local unit_name="sf_data_${dataset}"
    local output_dir="$DATA_ROOT/${dataset}_sf_sft"
    begin_unit "non_rl_data" "$unit_name" "" "data" "$output_dir"
    if should_skip_file "$output_dir/train.parquet"; then
        log "Skipping same-family trajectories for $dataset"
        snapshot_data_state "$unit_name" "non_rl_data" "skipped" "$output_dir"
        finish_unit "skipped" ""
        return 0
    fi
    log "Generating same-family positive SFT data for $dataset"
    bash "$SCRIPT_DIR/generate_trajectories.sh" "$QWEN_SF_TEACHER" "$dataset" "$SAMPLES_PER_ROUND"
    verify_file "$output_dir/train.parquet"
    verify_file "$output_dir/summary.json"
    snapshot_data_state "$unit_name" "non_rl_data" "completed" "$output_dir"
    finish_unit "completed" ""
}

run_cross_family_sft_data() {
    local dataset="$1"
    local unit_name="cf_data_${dataset}"
    local output_dir="$DATA_ROOT/${dataset}_cf_sft"
    begin_unit "non_rl_data" "$unit_name" "" "data" "$output_dir"
    if should_skip_file "$output_dir/train.parquet"; then
        log "Skipping cross-family trajectories for $dataset"
        snapshot_data_state "$unit_name" "non_rl_data" "skipped" "$output_dir"
        finish_unit "skipped" ""
        return 0
    fi
    log "Generating cross-family positive SFT data for $dataset"
    bash "$SCRIPT_DIR/generate_cross_family_trajectories.sh" "$CF_TEACHER" "$dataset" "$SAMPLES_PER_ROUND"
    verify_file "$output_dir/train.parquet"
    verify_file "$output_dir/summary.json"
    snapshot_data_state "$unit_name" "non_rl_data" "completed" "$output_dir"
    finish_unit "completed" ""
}

run_sft_experiment() {
    local dataset="$1"
    local model="$2"
    local experiment="$3"
    local variant="$4"
    local lora_rank_value="$5"
    begin_unit "non_rl_sft" "$experiment" "$experiment" "sft" "$CHECKPOINT_ROOT/$experiment"
    if should_skip_experiment "$experiment"; then
        log "Skipping SFT experiment $experiment"
        snapshot_experiment_state "$experiment" "sft" "non_rl_sft" "skipped"
        finish_unit "skipped" ""
        return 0
    fi
    log "Running SFT experiment $experiment"
    LORA_RANK="$lora_rank_value" bash "$SCRIPT_DIR/run_sft.sh" "$dataset" "$model" "$experiment" "$variant"
    has_checkpoints "$experiment" || die "No checkpoints produced for $experiment"
    snapshot_experiment_state "$experiment" "sft" "non_rl_sft" "completed"
    finish_unit "completed" ""
    if [[ "$RUN_EVALS" == "1" ]]; then
        run_standard_eval_bundle "$experiment" "$model" "non_rl_eval" "$dataset"
    fi
}

run_sequential_pair() {
    local model="$1"
    local prefix="$2"
    local dataset_a="$3"
    local dataset_b="$4"
    local stage1
    local stage2
    local stage1_model_path
    local stage2_model_path

    stage1=$(qualify_name "${prefix}_${dataset_a}")
    stage2=$(qualify_name "${prefix}_${dataset_a}_then_${dataset_b}")

    ensure_grpo_data "$dataset_a"
    ensure_grpo_data "$dataset_b"

    require_prime_label "$dataset_a"
    run_prime_experiment "$model" "$dataset_a" "$stage1" "rl_sequential" "$SEQUENTIAL_STAGE_STEPS"
    stage1_model_path=$(resolve_prime_model_path "$stage1") || die "Could not resolve PRIME model path for $stage1"

    require_prime_label "$dataset_b"
    run_prime_experiment "$stage1_model_path" "$dataset_b" "$stage2" "rl_sequential" "$SEQUENTIAL_STAGE_STEPS"
    stage2_model_path=$(resolve_prime_model_path "$stage2") || die "Could not resolve PRIME model path for $stage2"

    if [[ "$RUN_EVALS" == "1" ]]; then
        run_prime_eval_bundle "$stage2" "$stage2_model_path" "rl_eval" "$dataset_b" "$dataset_a"
    fi
}

run_mixed_pair() {
    local dataset_a="$1"
    local dataset_b="$2"
    local experiment
    local label
    local final_model_path
    experiment=$(qualify_name "qwen_mixed_${dataset_a}_${dataset_b}")
    label=$(pair_label "mix" "$dataset_a" "$dataset_b")

    ensure_grpo_data "$dataset_a"
    ensure_grpo_data "$dataset_b"

    require_prime_label "$label"
    run_prime_experiment "$QWEN_MODEL" "$label" "$experiment" "rl_mixed"
    final_model_path=$(resolve_prime_model_path "$experiment") || die "Could not resolve PRIME model path for $experiment"

    if [[ "$RUN_EVALS" == "1" ]]; then
        run_prime_eval_bundle "$experiment" "$final_model_path" "rl_eval" "$dataset_a" "$dataset_b"
    fi
}

run_iid_pair() {
    local dataset_a="$1"
    local dataset_b="$2"
    local prefix
    local stage2
    local label
    local final_model_path
    prefix=$(qualify_name "qwen_iid_${dataset_a}_${dataset_b}")
    stage2="${prefix}_iid_stage2"
    label=$(pair_label "iid" "$dataset_a" "$dataset_b")

    ensure_grpo_data "$dataset_a"
    ensure_grpo_data "$dataset_b"

    require_prime_label "$label"
    run_prime_experiment "$QWEN_MODEL" "$label" "$stage2" "rl_iid"
    final_model_path=$(resolve_prime_model_path "$stage2") || die "Could not resolve PRIME model path for $stage2"

    if [[ "$RUN_EVALS" == "1" ]]; then
        run_prime_eval_bundle "$stage2" "$final_model_path" "rl_eval" "$dataset_a" "$dataset_b"
    fi
}

run_baseline_eval() {
    local model="$1"
    local dataset="$2"
    local experiment="$3"
    local run_dir="$RUNS_ROOT/$experiment"
    local metrics_path="$run_dir/metrics/metrics.json"

    ensure_grpo_data "$dataset"

    begin_unit "non_rl_baseline" "$experiment" "$experiment" "baseline" "$run_dir"
    if should_skip_file "$metrics_path"; then
        log "Skipping baseline evaluation $experiment"
        snapshot_experiment_state "$experiment" "baseline" "non_rl_baseline" "skipped"
        finish_unit "skipped" ""
        return 0
    fi

    log "Running baseline evaluation $experiment"
    bash "$SCRIPT_DIR/run_vllm_and_runner_interactive.sh" \
        --model "$model" \
        --run-root "$MANUAL_RUNS_ROOT/$experiment" \
        --max-model-len 8192 \
        -- \
        python "$SCRIPT_DIR/prime_rl_runner.py" baseline \
            --model "$model" \
            --dataset "$dataset" \
            --output-root "$RUNS_ROOT" \
            --run-name "$experiment" \
            --hf-home "$HF_HOME" \
            --rollouts-per-prompt 8 \
            --temperature 1.0 \
            --top-p 1.0 \
            --max-model-len 8192 \
            --max-tokens 1024 \
            --wandb-mode online

    verify_file "$metrics_path"
    snapshot_experiment_state "$experiment" "baseline" "non_rl_baseline" "completed"
    finish_unit "completed" ""
}

update_board() {
    log "Updating experiment board"
    python "$SCRIPT_DIR/update_experiment_board.py" \
        --prime_runs_root "$PRIME_RUNS_ROOT" \
        --output "$REPO_DIR/EXPERIMENT_BOARD.md" \
        --json_output "$ORCH_ROOT/experiment_board.json"
}

log "Run prefix: $RUN_PREFIX"
log "Interactive allocation: ${SLURM_JOB_ID:-not detected}"
log "Qwen model: $QWEN_MODEL"
log "OLMo model: $OLMO_MODEL"
log "RL backend: PRIME-RL only"
log "PRIME runs root: $PRIME_RUNS_ROOT"
log "Orchestration state: $ORCH_ROOT"

for dataset in "${DATASETS[@]}"; do
    ensure_grpo_data "$dataset"
done

if [[ "$RUN_SF_SFT" == "1" || "$RUN_LORA_ABLATION" == "1" ]]; then
    for dataset in "${DATASETS[@]}"; do
        run_same_family_sft_data "$dataset"
    done
    update_board
fi

if [[ "$RUN_LORA_ABLATION" == "1" ]]; then
    for dataset in "${DATASETS[@]}"; do
        ensure_gt_sft_data "$dataset"
        run_sft_experiment "$dataset" "$QWEN_MODEL" "$(qualify_name "gt_sft_full_${dataset}")" gt 0
        run_sft_experiment "$dataset" "$QWEN_MODEL" "$(qualify_name "gt_sft_lora_${dataset}")" gt "$LORA_RANK"
        run_sft_experiment "$dataset" "$QWEN_MODEL" "$(qualify_name "sf_sft_full_${dataset}")" sf 0
        run_sft_experiment "$dataset" "$QWEN_MODEL" "$(qualify_name "sf_sft_lora_${dataset}")" sf "$LORA_RANK"
    done
    update_board
fi

if [[ "$RUN_CF_SFT" == "1" ]]; then
    for dataset in "${DATASETS[@]}"; do
        run_cross_family_sft_data "$dataset"
        run_sft_experiment "$dataset" "$QWEN_MODEL" "$(qualify_name "cf_sft_${dataset}")" cf 0
    done
    update_board
fi

if [[ "$RUN_OLMO_BASELINES" == "1" ]]; then
    for dataset in "${DATASETS[@]}"; do
        run_baseline_eval "$OLMO_MODEL" "$dataset" "$(qualify_name "olmo3_7b_baseline_${dataset}")"
    done
    update_board
fi

log "Starting RL phases last"

if [[ "$RUN_QWEN_SEQUENTIAL" == "1" ]]; then
    for pair in "${PAIRS[@]}"; do
        read -r dataset_a dataset_b <<<"$pair"
        run_sequential_pair "$QWEN_MODEL" "qwen_seq" "$dataset_a" "$dataset_b"
        run_sequential_pair "$QWEN_MODEL" "qwen_seq" "$dataset_b" "$dataset_a"
    done
    update_board
fi

if [[ "$RUN_MIX" == "1" ]]; then
    for pair in "${PAIRS[@]}"; do
        read -r dataset_a dataset_b <<<"$pair"
        run_mixed_pair "$dataset_a" "$dataset_b"
    done
    update_board
fi

if [[ "$RUN_IID" == "1" ]]; then
    for pair in "${PAIRS[@]}"; do
        read -r dataset_a dataset_b <<<"$pair"
        run_iid_pair "$dataset_a" "$dataset_b"
    done
    update_board
fi

if [[ "$RUN_OLMO_SEQUENTIAL" == "1" ]]; then
    for pair in "${PAIRS[@]}"; do
        read -r dataset_a dataset_b <<<"$pair"
        run_sequential_pair "$OLMO_MODEL" "olmo3_7b_seq" "$dataset_a" "$dataset_b"
        run_sequential_pair "$OLMO_MODEL" "olmo3_7b_seq" "$dataset_b" "$dataset_a"
    done
    update_board
fi

log "All requested experiment phases completed"
write_current_state "completed"
update_board
