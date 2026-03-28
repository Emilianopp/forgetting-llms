#!/usr/bin/env bash
# Export a complete interactive-session environment for task completion, then
# launch the PRIME-only experiment orchestrator.
#
# Usage:
#   bash scripts/run_all_tasks_prime_interactive_session.sh
#   bash scripts/run_all_tasks_prime_interactive_session.sh non_rl
#   bash scripts/run_all_tasks_prime_interactive_session.sh rl
#
# Optional:
#   TASK_ENV_FILE=/path/to/overrides.sh bash scripts/run_all_tasks_prime_interactive_session.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Export the full task-completion environment and run the interactive orchestrator.

Modes:
  all      Run non-RL phases first, then RL phases (default)
  non_rl   Run only non-RL phases
  rl       Run only RL phases; skips completed non-RL setup stages

Override file:
  TASK_ENV_FILE                  Optional shell file sourced before defaults.
                                 Use it to override PRIME env names/config paths.

Key defaults exported by this wrapper:
  RUN_PREFIX=mar21_tasks
  DATASETS_CSV=gsm8k,math,triviaqa
  PAIR_MATRIX_CSV=gsm8k:math,gsm8k:triviaqa,math:triviaqa
  PRIME_COMMAND='uv --project $PRIME_RL_ROOT run rl'
  PRIME_CONFIG_ROOT=$HOME/scratch/forgetting-llms/prime-configs
  SEQUENTIAL_STAGE_STEPS=300
  PRIME_CKPT_INTERVAL_STEPS=5

Optional Polaris math support:
  INCLUDE_POLARIS_MATH=1         Adds polaris_math to DATASETS_CSV
  POLARIS_MATH_DATASET_ID=...    Required if polaris_math is enabled

Examples:
  bash scripts/run_all_tasks_prime_interactive_session.sh
  bash scripts/run_all_tasks_prime_interactive_session.sh non_rl
  TASK_ENV_FILE=$HOME/prime_task_env.sh bash scripts/run_all_tasks_prime_interactive_session.sh rl
EOF
}

MODE="${1:-all}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
    usage
    exit 0
fi
case "$MODE" in
    all|non_rl|rl) ;;
    *) echo "Unknown mode: $MODE" >&2; usage; exit 1 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TASK_ENV_FILE="${TASK_ENV_FILE:-$REPO_DIR/scripts/prime_task_env.sh}"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"

if [[ -f "$TASK_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$TASK_ENV_FILE"
fi

export RUN_PREFIX="${RUN_PREFIX:-mar21_tasks}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/forgetting-llms}"
export DATA_ROOT="${DATA_ROOT:-$SCRATCH_ROOT/data}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH_ROOT/checkpoints}"
export EVAL_ROOT="${EVAL_ROOT:-$SCRATCH_ROOT/eval_results}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH_ROOT/runs}"
export MANUAL_RUNS_ROOT="${MANUAL_RUNS_ROOT:-$SCRATCH_ROOT/manual_runs}"
export PRIME_RUNS_ROOT="${PRIME_RUNS_ROOT:-$SCRATCH_ROOT/prime_runs}"
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
PRIME_RUNTIME_ENV_FILE="${PRIME_RUNTIME_ENV_FILE:-$SCRATCH_ROOT/prime_rl_env.sh}"

if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$HOME/scratch/"* ]]; then
    echo "ERROR: Active virtual environment is outside scratch: $VIRTUAL_ENV" >&2
    echo "Activate ~/scratch/forgetting-llms/.venv first." >&2
    exit 1
fi

if [[ -f "$PRIME_RUNTIME_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$PRIME_RUNTIME_ENV_FILE"
fi

export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export RUN_EVALS="${RUN_EVALS:-1}"
export RUN_SF_SFT="${RUN_SF_SFT:-0}"
export RUN_QWEN_SEQUENTIAL="${RUN_QWEN_SEQUENTIAL:-1}"
export RUN_MIX="${RUN_MIX:-1}"
export RUN_IID="${RUN_IID:-1}"
export RUN_CF_SFT="${RUN_CF_SFT:-0}"
export RUN_OLMO_BASELINES="${RUN_OLMO_BASELINES:-1}"
export RUN_OLMO_SEQUENTIAL="${RUN_OLMO_SEQUENTIAL:-1}"
export RUN_LORA_ABLATION="${RUN_LORA_ABLATION:-0}"

export QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-1.7B}"
export QWEN_SF_TEACHER="${QWEN_SF_TEACHER:-Qwen/Qwen3-32B}"
export CF_TEACHER="${CF_TEACHER:-meta-llama/Llama-3.1-70B-Instruct}"
export OLMO_MODEL="${OLMO_MODEL:-$HOME/scratch/olmo3_7B-Instruct}"
export LORA_RANK="${LORA_RANK:-64}"

export PRIME_RL_ROOT="${PRIME_RL_ROOT:-$SCRATCH_ROOT/vendor/prime-rl}"
export PRIME_COMMAND="${PRIME_COMMAND:-uv --project $PRIME_RL_ROOT run rl}"
export PRIME_WANDB_MODE="${PRIME_WANDB_MODE:-online}"
export PRIME_MAX_STEPS="${PRIME_MAX_STEPS:-1000}"
export SEQUENTIAL_STAGE_STEPS="${SEQUENTIAL_STAGE_STEPS:-300}"
export PRIME_CKPT_INTERVAL_STEPS="${PRIME_CKPT_INTERVAL_STEPS:-5}"
export PRIME_CKPT_KEEP_LAST="${PRIME_CKPT_KEEP_LAST:-3}"
export PRIME_CKPT_KEEP_INTERVAL="${PRIME_CKPT_KEEP_INTERVAL:-50}"
export PRIME_BATCH_SIZE="${PRIME_BATCH_SIZE:-256}"
export PRIME_SEQ_LEN="${PRIME_SEQ_LEN:-8192}"
export PRIME_MAX_TOKENS="${PRIME_MAX_TOKENS:-1024}"
export PRIME_ROLLOUTS_PER_PROMPT="${PRIME_ROLLOUTS_PER_PROMPT:-8}"
export PRIME_MAX_ASYNC_LEVEL="${PRIME_MAX_ASYNC_LEVEL:-1}"
export PRIME_ENFORCE_EAGER="${PRIME_ENFORCE_EAGER:-0}"
export PRIME_EXTRA_ARGS="${PRIME_EXTRA_ARGS:-}"
export PRIME_INFERENCE_READY_TIMEOUT_SECS="${PRIME_INFERENCE_READY_TIMEOUT_SECS:-900}"
export AUTO_BOOTSTRAP_PRIME_CONFIGS="${AUTO_BOOTSTRAP_PRIME_CONFIGS:-1}"
export REWRITE_BOOTSTRAPPED_PRIME_CONFIGS="${REWRITE_BOOTSTRAPPED_PRIME_CONFIGS:-0}"

export SAMPLES_PER_ROUND="${SAMPLES_PER_ROUND:-4}"
export MAX_TOTAL_SAMPLES="${MAX_TOTAL_SAMPLES:-16}"
export TARGET_CORRECT_PER_QUESTION="${TARGET_CORRECT_PER_QUESTION:-2}"
export MIN_CORRECT_PER_QUESTION="${MIN_CORRECT_PER_QUESTION:-2}"
export SOLUTIONS_PER_QUESTION="${SOLUTIONS_PER_QUESTION:-2}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"

export GRPO_MAX_PROMPT="${GRPO_MAX_PROMPT:-512}"
export GRPO_MAX_RESPONSE="${GRPO_MAX_RESPONSE:-1024}"
export GRPO_TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS:-15}"
export GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-16}"
export GRPO_PPO_MINI_BATCH_SIZE="${GRPO_PPO_MINI_BATCH_SIZE:-16}"
export GRPO_PPO_MICRO_BATCH_SIZE="${GRPO_PPO_MICRO_BATCH_SIZE:-2}"
export GRPO_ROLLOUTS_PER_PROMPT="${GRPO_ROLLOUTS_PER_PROMPT:-8}"
export GRPO_ROLLOUT_LOGPROB_BATCH="${GRPO_ROLLOUT_LOGPROB_BATCH:-4}"
export GRPO_ACTOR_LR="${GRPO_ACTOR_LR:-1e-6}"

export PRIME_CONFIG_ROOT="${PRIME_CONFIG_ROOT:-$SCRATCH_ROOT/prime-configs}"
export INCLUDE_POLARIS_MATH="${INCLUDE_POLARIS_MATH:-0}"
export POLARIS_MATH_DATASET_ID="${POLARIS_MATH_DATASET_ID:-}"

base_datasets="gsm8k,math,triviaqa"
if [[ "$INCLUDE_POLARIS_MATH" == "1" ]]; then
    export DATASETS_CSV="${DATASETS_CSV:-$base_datasets,polaris_math}"
else
    export DATASETS_CSV="${DATASETS_CSV:-$base_datasets}"
fi
export PAIR_MATRIX_CSV="${PAIR_MATRIX_CSV:-gsm8k:math,gsm8k:triviaqa,math:triviaqa}"

export PRIME_ENV_GSM8K="${PRIME_ENV_GSM8K:-gsm8k}"
export PRIME_COMBINED_CONFIG_GSM8K="${PRIME_COMBINED_CONFIG_GSM8K:-}"
export PRIME_TRAINER_CONFIG_GSM8K="${PRIME_TRAINER_CONFIG_GSM8K:-$PRIME_CONFIG_ROOT/gsm8k.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_GSM8K="${PRIME_ORCHESTRATOR_CONFIG_GSM8K:-$PRIME_CONFIG_ROOT/gsm8k.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_GSM8K="${PRIME_INFERENCE_CONFIG_GSM8K:-$PRIME_CONFIG_ROOT/gsm8k.inference.toml}"

export PRIME_ENV_MATH="${PRIME_ENV_MATH:-math}"
export PRIME_COMBINED_CONFIG_MATH="${PRIME_COMBINED_CONFIG_MATH:-}"
export PRIME_TRAINER_CONFIG_MATH="${PRIME_TRAINER_CONFIG_MATH:-$PRIME_CONFIG_ROOT/math.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_MATH="${PRIME_ORCHESTRATOR_CONFIG_MATH:-$PRIME_CONFIG_ROOT/math.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_MATH="${PRIME_INFERENCE_CONFIG_MATH:-$PRIME_CONFIG_ROOT/math.inference.toml}"

export PRIME_ENV_TRIVIAQA="${PRIME_ENV_TRIVIAQA:-triviaqa}"
export PRIME_COMBINED_CONFIG_TRIVIAQA="${PRIME_COMBINED_CONFIG_TRIVIAQA:-}"
export PRIME_TRAINER_CONFIG_TRIVIAQA="${PRIME_TRAINER_CONFIG_TRIVIAQA:-$PRIME_CONFIG_ROOT/triviaqa.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_TRIVIAQA="${PRIME_ORCHESTRATOR_CONFIG_TRIVIAQA:-$PRIME_CONFIG_ROOT/triviaqa.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_TRIVIAQA="${PRIME_INFERENCE_CONFIG_TRIVIAQA:-$PRIME_CONFIG_ROOT/triviaqa.inference.toml}"

export PRIME_ENV_POLARIS_MATH="${PRIME_ENV_POLARIS_MATH:-polaris_math}"
export PRIME_COMBINED_CONFIG_POLARIS_MATH="${PRIME_COMBINED_CONFIG_POLARIS_MATH:-}"
export PRIME_TRAINER_CONFIG_POLARIS_MATH="${PRIME_TRAINER_CONFIG_POLARIS_MATH:-$PRIME_CONFIG_ROOT/polaris_math.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_POLARIS_MATH="${PRIME_ORCHESTRATOR_CONFIG_POLARIS_MATH:-$PRIME_CONFIG_ROOT/polaris_math.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_POLARIS_MATH="${PRIME_INFERENCE_CONFIG_POLARIS_MATH:-$PRIME_CONFIG_ROOT/polaris_math.inference.toml}"

export PRIME_ENV_MIX_GSM8K_MATH="${PRIME_ENV_MIX_GSM8K_MATH:-mix_gsm8k_math}"
export PRIME_COMBINED_CONFIG_MIX_GSM8K_MATH="${PRIME_COMBINED_CONFIG_MIX_GSM8K_MATH:-}"
export PRIME_TRAINER_CONFIG_MIX_GSM8K_MATH="${PRIME_TRAINER_CONFIG_MIX_GSM8K_MATH:-$PRIME_CONFIG_ROOT/mix_gsm8k_math.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_MIX_GSM8K_MATH="${PRIME_ORCHESTRATOR_CONFIG_MIX_GSM8K_MATH:-$PRIME_CONFIG_ROOT/mix_gsm8k_math.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_MIX_GSM8K_MATH="${PRIME_INFERENCE_CONFIG_MIX_GSM8K_MATH:-$PRIME_CONFIG_ROOT/mix_gsm8k_math.inference.toml}"

export PRIME_ENV_IID_GSM8K_MATH="${PRIME_ENV_IID_GSM8K_MATH:-iid_gsm8k_math}"
export PRIME_COMBINED_CONFIG_IID_GSM8K_MATH="${PRIME_COMBINED_CONFIG_IID_GSM8K_MATH:-}"
export PRIME_TRAINER_CONFIG_IID_GSM8K_MATH="${PRIME_TRAINER_CONFIG_IID_GSM8K_MATH:-$PRIME_CONFIG_ROOT/iid_gsm8k_math.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_IID_GSM8K_MATH="${PRIME_ORCHESTRATOR_CONFIG_IID_GSM8K_MATH:-$PRIME_CONFIG_ROOT/iid_gsm8k_math.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_IID_GSM8K_MATH="${PRIME_INFERENCE_CONFIG_IID_GSM8K_MATH:-$PRIME_CONFIG_ROOT/iid_gsm8k_math.inference.toml}"

export PRIME_ENV_MIX_GSM8K_TRIVIAQA="${PRIME_ENV_MIX_GSM8K_TRIVIAQA:-mix_gsm8k_triviaqa}"
export PRIME_COMBINED_CONFIG_MIX_GSM8K_TRIVIAQA="${PRIME_COMBINED_CONFIG_MIX_GSM8K_TRIVIAQA:-}"
export PRIME_TRAINER_CONFIG_MIX_GSM8K_TRIVIAQA="${PRIME_TRAINER_CONFIG_MIX_GSM8K_TRIVIAQA:-$PRIME_CONFIG_ROOT/mix_gsm8k_triviaqa.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_MIX_GSM8K_TRIVIAQA="${PRIME_ORCHESTRATOR_CONFIG_MIX_GSM8K_TRIVIAQA:-$PRIME_CONFIG_ROOT/mix_gsm8k_triviaqa.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_MIX_GSM8K_TRIVIAQA="${PRIME_INFERENCE_CONFIG_MIX_GSM8K_TRIVIAQA:-$PRIME_CONFIG_ROOT/mix_gsm8k_triviaqa.inference.toml}"

export PRIME_ENV_IID_GSM8K_TRIVIAQA="${PRIME_ENV_IID_GSM8K_TRIVIAQA:-iid_gsm8k_triviaqa}"
export PRIME_COMBINED_CONFIG_IID_GSM8K_TRIVIAQA="${PRIME_COMBINED_CONFIG_IID_GSM8K_TRIVIAQA:-}"
export PRIME_TRAINER_CONFIG_IID_GSM8K_TRIVIAQA="${PRIME_TRAINER_CONFIG_IID_GSM8K_TRIVIAQA:-$PRIME_CONFIG_ROOT/iid_gsm8k_triviaqa.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_IID_GSM8K_TRIVIAQA="${PRIME_ORCHESTRATOR_CONFIG_IID_GSM8K_TRIVIAQA:-$PRIME_CONFIG_ROOT/iid_gsm8k_triviaqa.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_IID_GSM8K_TRIVIAQA="${PRIME_INFERENCE_CONFIG_IID_GSM8K_TRIVIAQA:-$PRIME_CONFIG_ROOT/iid_gsm8k_triviaqa.inference.toml}"

export PRIME_ENV_MIX_MATH_TRIVIAQA="${PRIME_ENV_MIX_MATH_TRIVIAQA:-mix_math_triviaqa}"
export PRIME_COMBINED_CONFIG_MIX_MATH_TRIVIAQA="${PRIME_COMBINED_CONFIG_MIX_MATH_TRIVIAQA:-}"
export PRIME_TRAINER_CONFIG_MIX_MATH_TRIVIAQA="${PRIME_TRAINER_CONFIG_MIX_MATH_TRIVIAQA:-$PRIME_CONFIG_ROOT/mix_math_triviaqa.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_MIX_MATH_TRIVIAQA="${PRIME_ORCHESTRATOR_CONFIG_MIX_MATH_TRIVIAQA:-$PRIME_CONFIG_ROOT/mix_math_triviaqa.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_MIX_MATH_TRIVIAQA="${PRIME_INFERENCE_CONFIG_MIX_MATH_TRIVIAQA:-$PRIME_CONFIG_ROOT/mix_math_triviaqa.inference.toml}"

export PRIME_ENV_IID_MATH_TRIVIAQA="${PRIME_ENV_IID_MATH_TRIVIAQA:-iid_math_triviaqa}"
export PRIME_COMBINED_CONFIG_IID_MATH_TRIVIAQA="${PRIME_COMBINED_CONFIG_IID_MATH_TRIVIAQA:-}"
export PRIME_TRAINER_CONFIG_IID_MATH_TRIVIAQA="${PRIME_TRAINER_CONFIG_IID_MATH_TRIVIAQA:-$PRIME_CONFIG_ROOT/iid_math_triviaqa.trainer.toml}"
export PRIME_ORCHESTRATOR_CONFIG_IID_MATH_TRIVIAQA="${PRIME_ORCHESTRATOR_CONFIG_IID_MATH_TRIVIAQA:-$PRIME_CONFIG_ROOT/iid_math_triviaqa.orchestrator.toml}"
export PRIME_INFERENCE_CONFIG_IID_MATH_TRIVIAQA="${PRIME_INFERENCE_CONFIG_IID_MATH_TRIVIAQA:-$PRIME_CONFIG_ROOT/iid_math_triviaqa.inference.toml}"

case "$MODE" in
    non_rl)
        export RUN_RL_PHASES=0
        ;;
    rl)
        export RUN_RL_PHASES=1
        export RUN_SF_SFT=0
        export RUN_CF_SFT=0
        export RUN_LORA_ABLATION=0
        export RUN_OLMO_BASELINES=0
        ;;
    all)
        export RUN_RL_PHASES=1
        ;;
esac

die() {
    echo "ERROR: $*" >&2
    exit 1
}

scratch_model_dir() {
    local model_ref="$1"
    printf '%s/models/%s\n' "$SCRATCH_ROOT" "${model_ref//\//__}"
}

resolve_local_model_snapshot() {
    local model_ref="$1"
    local resolved=""
    if [[ -z "$model_ref" ]]; then
        die "Model reference is empty"
    fi

    if [[ "$model_ref" == /* || "$model_ref" == ./* || "$model_ref" == ../* || "$model_ref" == ~/* ]]; then
        resolved="${model_ref/#\~/$HOME}"
    else
        resolved="$(scratch_model_dir "$model_ref")"
    fi

    if [[ ! -d "$resolved" ]]; then
        die "Local model snapshot not found: $resolved"
    fi

    printf '%s\n' "$resolved"
}

require_nonempty() {
    local name="$1"
    local value="$2"
    [[ -n "$value" ]] || die "Missing required environment value: $name"
}

to_env_key() {
    echo "$1" | tr '[:lower:]-:/. ' '[:upper:]______'
}

default_prime_env_from_var() {
    local env_var="$1"
    local key="${env_var#PRIME_ENV_}"
    printf '%s' "${key,,}"
}

default_prime_config_from_var() {
    local cfg_var="$1"
    local suffix="$2"
    local prefix="PRIME_${suffix^^}_CONFIG_"
    local key="${cfg_var#$prefix}"
    if [[ -z "${PRIME_CONFIG_ROOT:-}" ]]; then
        return 0
    fi
    printf '%s' "$PRIME_CONFIG_ROOT/${key,,}.${suffix}.toml"
}

validate_prime_mapping() {
    local label="$1"
    local env_var="$2"
    local cfg_var="$3"
    local env_value="${!env_var:-}"
    local cfg_value="${!cfg_var:-}"
    local trainer_var="${4:-}"
    local orchestrator_var="${5:-}"
    local inference_var="${6:-}"
    local trainer_value=""
    local orchestrator_value=""
    local inference_value=""
    if [[ -z "$env_value" ]]; then
        env_value=$(default_prime_env_from_var "$env_var")
    fi
    require_nonempty "$env_var" "$env_value"
    if [[ -n "$cfg_value" && -f "${cfg_value/#\~/$HOME}" ]]; then
        return 0
    fi
    if [[ -n "$trainer_var" && -n "$orchestrator_var" && -n "$inference_var" ]]; then
        trainer_value="${!trainer_var:-}"
        orchestrator_value="${!orchestrator_var:-}"
        inference_value="${!inference_var:-}"
        if [[ -z "$trainer_value" ]]; then
            trainer_value=$(default_prime_config_from_var "$trainer_var" "trainer")
        fi
        if [[ -z "$orchestrator_value" ]]; then
            orchestrator_value=$(default_prime_config_from_var "$orchestrator_var" "orchestrator")
        fi
        if [[ -z "$inference_value" ]]; then
            inference_value=$(default_prime_config_from_var "$inference_var" "inference")
        fi
        if [[ -n "$trainer_value" && -n "$orchestrator_value" && -n "$inference_value" \
            && -f "${trainer_value/#\~/$HOME}" && -f "${orchestrator_value/#\~/$HOME}" && -f "${inference_value/#\~/$HOME}" ]]; then
            return 0
        fi
    fi
    # Missing config files are allowed here. The PRIME launcher will auto-generate
    # scratch-local stub configs if no real TOMLs are provided.
}

bootstrap_prime_configs() {
    local bootstrap_model=""
    local rewrite_flag=()

    if [[ "$AUTO_BOOTSTRAP_PRIME_CONFIGS" != "1" ]]; then
        return 0
    fi

    if [[ "$RUN_QWEN_SEQUENTIAL" == "1" && "$RUN_OLMO_SEQUENTIAL" != "1" ]]; then
        bootstrap_model="$QWEN_MODEL"
    elif [[ "$RUN_OLMO_SEQUENTIAL" == "1" && "$RUN_QWEN_SEQUENTIAL" != "1" ]]; then
        bootstrap_model="$OLMO_MODEL"
    elif [[ "$RUN_MIX" == "1" || "$RUN_IID" == "1" ]]; then
        bootstrap_model="$QWEN_MODEL"
    else
        return 0
    fi

    mkdir -p "$PRIME_CONFIG_ROOT"
    if [[ "$REWRITE_BOOTSTRAPPED_PRIME_CONFIGS" == "1" ]]; then
        rewrite_flag=(--rewrite-existing)
    fi

    python "$SCRIPT_DIR/bootstrap_prime_configs.py" \
        --output-root "$PRIME_CONFIG_ROOT" \
        --model "$bootstrap_model" \
        --datasets-csv "$DATASETS_CSV" \
        --pair-matrix-csv "$PAIR_MATRIX_CSV" \
        --max-steps "$PRIME_MAX_STEPS" \
        --seq-len "$PRIME_SEQ_LEN" \
        --max-tokens "$PRIME_MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --rollouts-per-prompt "$PRIME_ROLLOUTS_PER_PROMPT" \
        "${rewrite_flag[@]}"
}

validate_all() {
    local datasets_string="$DATASETS_CSV"
    local pair_string="$PAIR_MATRIX_CSV"
    IFS=',' read -r -a datasets <<< "$datasets_string"
    IFS=',' read -r -a pairs <<< "$pair_string"

    require_nonempty "RUN_PREFIX" "$RUN_PREFIX"
    require_nonempty "PRIME_COMMAND" "$PRIME_COMMAND"

    if [[ "$MODE" != "non_rl" && ( "$RUN_QWEN_SEQUENTIAL" == "1" || "$RUN_OLMO_SEQUENTIAL" == "1" || "$RUN_MIX" == "1" || "$RUN_IID" == "1" ) ]]; then
        local dataset
        if [[ "$RUN_QWEN_SEQUENTIAL" == "1" || "$RUN_OLMO_SEQUENTIAL" == "1" ]]; then
            for dataset in "${datasets[@]}"; do
                local dataset_key
                dataset_key=$(to_env_key "$dataset")
                if [[ "$dataset" == "polaris_math" ]]; then
                    require_nonempty "POLARIS_MATH_DATASET_ID" "$POLARIS_MATH_DATASET_ID"
                fi
                validate_prime_mapping \
                    "$dataset" \
                    "PRIME_ENV_${dataset_key}" \
                    "PRIME_COMBINED_CONFIG_${dataset_key}" \
                    "PRIME_TRAINER_CONFIG_${dataset_key}" \
                    "PRIME_ORCHESTRATOR_CONFIG_${dataset_key}" \
                    "PRIME_INFERENCE_CONFIG_${dataset_key}"
            done
        fi

        if [[ "$RUN_MIX" == "1" || "$RUN_IID" == "1" ]]; then
            local pair
            for pair in "${pairs[@]}"; do
                local pair_key
                pair_key=$(to_env_key "$pair")
                if [[ "$RUN_MIX" == "1" ]]; then
                    validate_prime_mapping \
                        "$pair" \
                        "PRIME_ENV_MIX_${pair_key}" \
                        "PRIME_COMBINED_CONFIG_MIX_${pair_key}" \
                        "PRIME_TRAINER_CONFIG_MIX_${pair_key}" \
                        "PRIME_ORCHESTRATOR_CONFIG_MIX_${pair_key}" \
                        "PRIME_INFERENCE_CONFIG_MIX_${pair_key}"
                fi
                if [[ "$RUN_IID" == "1" ]]; then
                    validate_prime_mapping \
                        "$pair" \
                        "PRIME_ENV_IID_${pair_key}" \
                        "PRIME_COMBINED_CONFIG_IID_${pair_key}" \
                        "PRIME_TRAINER_CONFIG_IID_${pair_key}" \
                        "PRIME_ORCHESTRATOR_CONFIG_IID_${pair_key}" \
                        "PRIME_INFERENCE_CONFIG_IID_${pair_key}"
                fi
            done
        fi
    fi
}

QWEN_MODEL="$(resolve_local_model_snapshot "$QWEN_MODEL")"
if [[ "$RUN_SF_SFT" == "1" ]]; then
    QWEN_SF_TEACHER="$(resolve_local_model_snapshot "$QWEN_SF_TEACHER")"
fi
if [[ "$RUN_CF_SFT" == "1" ]]; then
    CF_TEACHER="$(resolve_local_model_snapshot "$CF_TEACHER")"
fi
if [[ "$RUN_OLMO_BASELINES" == "1" || "$RUN_OLMO_SEQUENTIAL" == "1" ]]; then
    OLMO_MODEL="$(resolve_local_model_snapshot "$OLMO_MODEL")"
fi
export QWEN_MODEL QWEN_SF_TEACHER CF_TEACHER OLMO_MODEL

check_prime_runtime() {
    if [[ "$MODE" == "non_rl" ]]; then
        return 0
    fi

    if [[ "$RUN_QWEN_SEQUENTIAL" != "1" && "$RUN_OLMO_SEQUENTIAL" != "1" \
        && "$RUN_MIX" != "1" && "$RUN_IID" != "1" ]]; then
        return 0
    fi

    if [[ "$PRIME_COMMAND" == *" run rl"* || "$PRIME_COMMAND" == *"prime-rl"* ]]; then
        if ! bash -lc "$PRIME_COMMAND --help >/dev/null 2>&1"; then
            die "PRIME RL runtime is not installed or not reachable via PRIME_COMMAND. Run: bash scripts/setup_prime_rl.sh"
        fi
    fi
}

check_python_runtime() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        local pyver
        pyver=$("$VIRTUAL_ENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        [[ "$pyver" == "3.10" ]] || die "Active scratch venv uses Python $pyver. Recreate ~/scratch/forgetting-llms/.venv with Python 3.10."
    fi
}

bootstrap_prime_configs
validate_all
check_python_runtime
check_prime_runtime

printf '\n[task-session] mode=%s\n' "$MODE"
printf '[task-session] RUN_PREFIX=%s\n' "$RUN_PREFIX"
printf '[task-session] DATASETS_CSV=%s\n' "$DATASETS_CSV"
printf '[task-session] PAIR_MATRIX_CSV=%s\n' "$PAIR_MATRIX_CSV"
printf '[task-session] PRIME_CONFIG_ROOT=%s\n' "$PRIME_CONFIG_ROOT"
printf '[task-session] PRIME_RUNS_ROOT=%s\n' "$PRIME_RUNS_ROOT"
printf '[task-session] SEQUENTIAL_STAGE_STEPS=%s\n' "$SEQUENTIAL_STAGE_STEPS"
printf '[task-session] PRIME_CKPT_INTERVAL_STEPS=%s\n' "$PRIME_CKPT_INTERVAL_STEPS"
if [[ -f "$TASK_ENV_FILE" ]]; then
    printf '[task-session] overrides=%s\n' "$TASK_ENV_FILE"
fi

exec bash "$SCRIPT_DIR/run_all_tasks_interactive.sh"
