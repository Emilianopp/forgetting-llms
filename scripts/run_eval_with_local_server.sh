#!/usr/bin/env bash
# Serve a checkpoint on the current GPU set, then run run_eval.py against it.
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_eval_with_local_server.sh <model_path> <output_dir> <run_name> [run_eval.py extra args...]

This wrapper starts a local vLLM OpenAI-compatible server for the supplied
checkpoint/model and then runs src/evaluation/run_eval.py against that server.
It is intended for evaluation on a dedicated GPU while training continues on
other GPUs.

Environment knobs:
  EVAL_SUITE=tasks_md
  EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"
  CONTINUE_ON_ERROR=0
  SKIP_EXISTING=1
  FORCE_RERUN=0
  RUN_NON_ENDPOINT_EVAL=1
  RUN_ENDPOINT_EVAL=1
  NON_ENDPOINT_RUNNERS=lm_eval,supergpqa,bfcl,safety_eval
  ENDPOINT_RUNNERS=lighteval,evalplus,labbench,rg_mix
  EVAL_SERVER_PORT=8000
  EVAL_SERVER_TP=1
  EVAL_SERVER_GPU_MEMORY_UTILIZATION=0.85
  EVAL_SERVER_MAX_MODEL_LEN=<unset: use model config max>
  EVAL_SERVER_MAX_NUM_SEQS=1
  EVAL_SERVER_MAX_NUM_BATCHED_TOKENS=4096
  EVAL_SERVER_STARTUP_TIMEOUT=600
  EVAL_SERVER_API_KEY=EMPTY
  NLTK_DATA=~/scratch/forgetting-llms/nltk_data
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODEL_PATH=${1:?Usage: run_eval_with_local_server.sh <model_path> <output_dir> <run_name> [run_eval.py extra args...]}
OUTPUT_DIR=${2:?Usage: run_eval_with_local_server.sh <model_path> <output_dir> <run_name> [run_eval.py extra args...]}
RUN_NAME=${3:?Usage: run_eval_with_local_server.sh <model_path> <output_dir> <run_name> [run_eval.py extra args...]}
shift 3
RUN_EVAL_EXTRA_ARGS=("$@")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"

if [[ -z "${BENCHMARK_ENV_FILE:-}" ]]; then
    if [[ -f "$REPO_DIR/benchmark_env.sh" ]]; then
        BENCHMARK_ENV_FILE="$REPO_DIR/benchmark_env.sh"
    else
        BENCHMARK_ENV_FILE="$HOME/scratch/forgetting-llms/benchmark_env.sh"
    fi
fi
if [[ -f "$BENCHMARK_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$BENCHMARK_ENV_FILE"
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/.cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_ROOT/.cache/triton}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$SCRATCH_ROOT/.cache/pycache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH_ROOT/.cache/matplotlib}"
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/forgetting-llms/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH_ROOT/forgetting-llms/wandb_cache}"
export NLTK_DATA="${NLTK_DATA:-$SCRATCH_ROOT/forgetting-llms/nltk_data}"

mkdir -p \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$HF_HUB_CACHE" \
    "$XDG_CACHE_HOME" \
    "$PIP_CACHE_DIR" \
    "$UV_CACHE_DIR" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$PYTHONPYCACHEPREFIX" \
    "$MPLCONFIGDIR" \
    "$TMPDIR" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" \
    "$NLTK_DATA"

EVAL_SUITE="${EVAL_SUITE:-tasks_md}"
DEFAULT_EVAL_EXTRA_ARGS="--sampling-temperature 1.0 --sampling-top-p 1.0 --lighteval-endpoint-max-concurrent-requests 1 --lighteval-endpoint-timeout 180 --lighteval-endpoint-api-max-retry 2"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-$DEFAULT_EVAL_EXTRA_ARGS}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_NON_ENDPOINT_EVAL="${RUN_NON_ENDPOINT_EVAL:-1}"
RUN_ENDPOINT_EVAL="${RUN_ENDPOINT_EVAL:-1}"
NON_ENDPOINT_RUNNERS="${NON_ENDPOINT_RUNNERS:-lm_eval,supergpqa,bfcl,safety_eval}"
ENDPOINT_RUNNERS="${ENDPOINT_RUNNERS:-lighteval,evalplus,labbench,rg_mix}"
EVAL_SERVER_PORT="${EVAL_SERVER_PORT:-8000}"
EVAL_SERVER_TP="${EVAL_SERVER_TP:-1}"
EVAL_SERVER_GPU_MEMORY_UTILIZATION="${EVAL_SERVER_GPU_MEMORY_UTILIZATION:-0.85}"
EVAL_SERVER_MAX_MODEL_LEN="${EVAL_SERVER_MAX_MODEL_LEN:-}"
EVAL_SERVER_MAX_NUM_SEQS="${EVAL_SERVER_MAX_NUM_SEQS:-1}"
EVAL_SERVER_MAX_NUM_BATCHED_TOKENS="${EVAL_SERVER_MAX_NUM_BATCHED_TOKENS:-4096}"
EVAL_SERVER_STARTUP_TIMEOUT="${EVAL_SERVER_STARTUP_TIMEOUT:-600}"
EVAL_SERVER_API_KEY="${EVAL_SERVER_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
if [[ -z "${VLLM_SERVER_EXTRA_ARGS:-}" ]]; then
    export VLLM_SERVER_EXTRA_ARGS="--generation-config vllm --max-num-seqs ${EVAL_SERVER_MAX_NUM_SEQS} --max-num-batched-tokens ${EVAL_SERVER_MAX_NUM_BATCHED_TOKENS}"
fi

sanitize_eval_extra_args_string() {
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

sanitize_eval_extra_args_array() {
    local cleaned=()
    local removed=0
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--no-lighteval-chat-template" ]]; then
            removed=1
            continue
        fi
        cleaned+=("$arg")
    done
    if [[ "$removed" == "1" ]]; then
        echo "Ignoring stale eval arg: --no-lighteval-chat-template" >&2
    fi
    printf '%s\n' "${cleaned[@]}"
}

normalize_runner_csv() {
    local raw="$1"
    local role="$2"
    local cleaned="${raw// /}"
    local output=()
    local seen=","
    local item
    IFS=',' read -r -a items <<< "$cleaned"
    for item in "${items[@]}"; do
        [[ -n "$item" ]] || continue
        if [[ "$role" == "non_endpoint" && "$item" == "lighteval" ]]; then
            echo "Ignoring stale non-endpoint runner override: lighteval" >&2
            continue
        fi
        if [[ "$role" == "endpoint" && "$item" == "lighteval" ]]; then
            :
        fi
        if [[ "$seen" == *",$item,"* ]]; then
            continue
        fi
        output+=("$item")
        seen+="$item,"
    done
    if [[ "$role" == "endpoint" && "$seen" != *",lighteval,"* ]]; then
        output=("lighteval" "${output[@]}")
    fi
    local joined=""
    for item in "${output[@]}"; do
        if [[ -n "$joined" ]]; then
            joined+=","
        fi
        joined+="$item"
    done
    printf '%s' "$joined"
}

EVAL_EXTRA_ARGS="$(sanitize_eval_extra_args_string "$EVAL_EXTRA_ARGS")"
if [[ ${#RUN_EVAL_EXTRA_ARGS[@]} -gt 0 ]]; then
    mapfile -t RUN_EVAL_EXTRA_ARGS < <(sanitize_eval_extra_args_array "${RUN_EVAL_EXTRA_ARGS[@]}")
fi
NON_ENDPOINT_RUNNERS="$(normalize_runner_csv "$NON_ENDPOINT_RUNNERS" non_endpoint)"
ENDPOINT_RUNNERS="$(normalize_runner_csv "$ENDPOINT_RUNNERS" endpoint)"

csv_to_runner_flags() {
    local raw="$1"
    local cleaned="${raw// /}"
    [[ -n "$cleaned" ]] || return 0
    local flags=()
    local item
    IFS=',' read -r -a items <<< "$cleaned"
    for item in "${items[@]}"; do
        [[ -n "$item" ]] || continue
        flags+=(--include-runner "$item")
    done
    printf '%s\n' "${flags[@]}"
}

mapfile -t NON_ENDPOINT_RUNNER_FLAGS < <(csv_to_runner_flags "$NON_ENDPOINT_RUNNERS")
mapfile -t ENDPOINT_RUNNER_FLAGS < <(csv_to_runner_flags "$ENDPOINT_RUNNERS")

RUN_ROOT="$OUTPUT_DIR/_server"
mkdir -p "$OUTPUT_DIR" "$RUN_ROOT"

echo "Local eval server run root: $RUN_ROOT"
echo "vLLM server log: $RUN_ROOT/vllm_server.log"
echo "Eval runner log: $RUN_ROOT/runner.log"
echo "Scratch cache root: $XDG_CACHE_HOME"
echo "HF cache root: $HF_HOME"
echo "TMPDIR: $TMPDIR"

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
    python3 - "$package" "$subdir" "${NLTK_DATA:-}" <<'PY'
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
    export NLTK_DATA="${NLTK_DATA:-$HOME/scratch/forgetting-llms/nltk_data}"
    if has_nltk_resource "tokenizers/punkt" && has_nltk_resource "tokenizers/punkt_tab"; then
        return 0
    fi

    mkdir -p "$NLTK_DATA"
    echo "Ensuring NLTK tokenizers under: $NLTK_DATA"
    echo "Fetching punkt/punkt_tab directly"
    download_nltk_package_direct punkt tokenizers >/dev/null 2>&1 || true
    download_nltk_package_direct punkt_tab tokenizers >/dev/null 2>&1 || true

    if has_nltk_resource "tokenizers/punkt" && has_nltk_resource "tokenizers/punkt_tab"; then
        return 0
    fi

    echo "Direct fetch did not finish; falling back to nltk.downloader"
    python3 -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab >/dev/null 2>&1 || true

    if has_nltk_resource "tokenizers/punkt" && has_nltk_resource "tokenizers/punkt_tab"; then
        return 0
    fi

    echo "ERROR: Missing NLTK tokenizers punkt/punkt_tab under $NLTK_DATA" >&2
    echo "Run: python3 -m nltk.downloader -d \"$NLTK_DATA\" punkt punkt_tab" >&2
    return 1
}

BASE_RUN_EVAL_CMD=(
    python3 "$REPO_DIR/src/evaluation/run_eval.py"
    --model_path "$MODEL_PATH"
    --suite "$EVAL_SUITE"
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
)
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    BASE_RUN_EVAL_CMD+=(--continue_on_error)
fi
if [[ "$FORCE_RERUN" == "1" || "$SKIP_EXISTING" != "1" ]]; then
    BASE_RUN_EVAL_CMD+=(--force-rerun)
fi
if [[ -n "$EVAL_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=( $EVAL_EXTRA_ARGS )
    BASE_RUN_EVAL_CMD+=("${extra_args[@]}")
fi
if [[ ${#RUN_EVAL_EXTRA_ARGS[@]} -gt 0 ]]; then
    BASE_RUN_EVAL_CMD+=("${RUN_EVAL_EXTRA_ARGS[@]}")
fi

run_non_endpoint_eval() {
    if [[ ${#NON_ENDPOINT_RUNNER_FLAGS[@]} -eq 0 ]]; then
        echo "Skipping non-endpoint benchmark phase: NON_ENDPOINT_RUNNERS is empty"
        return 0
    fi
    local cmd=("${BASE_RUN_EVAL_CMD[@]}")
    if [[ ${#NON_ENDPOINT_RUNNER_FLAGS[@]} -gt 0 ]]; then
        cmd+=("${NON_ENDPOINT_RUNNER_FLAGS[@]}")
    fi
    echo "Running non-endpoint benchmarks without local server"
    printf '  %q' "${cmd[@]}"
    echo
    env -u OPENAI_BASE_URL -u OPENAI_API_BASE -u OPENAI_API_KEY "${cmd[@]}"
}

run_endpoint_eval() {
    if [[ ${#ENDPOINT_RUNNER_FLAGS[@]} -eq 0 ]]; then
        echo "Skipping endpoint benchmark phase: ENDPOINT_RUNNERS is empty"
        return 0
    fi
    ensure_nltk_tokenizers
    local runner_cmd=("${BASE_RUN_EVAL_CMD[@]}")
    if [[ ${#ENDPOINT_RUNNER_FLAGS[@]} -gt 0 ]]; then
        runner_cmd+=("${ENDPOINT_RUNNER_FLAGS[@]}")
    fi
    if [[ ",${ENDPOINT_RUNNERS// /}," == *",evalplus,"* ]]; then
        # Force EvalPlus onto the served endpoint path here. This must be an
        # explicit CLI arg so it wins over stale --evalplus-backend flags in
        # EVAL_EXTRA_ARGS / RUN_EVAL_EXTRA_ARGS.
        runner_cmd+=(--evalplus-backend openai)
    fi
    local server_cmd=(
        bash "$SCRIPT_DIR/run_vllm_and_runner_interactive.sh"
        --model "$MODEL_PATH" \
        --run-root "$RUN_ROOT" \
        --port "$EVAL_SERVER_PORT" \
        --served-model-name "$(basename "$MODEL_PATH")" \
        --tensor-parallel-size "$EVAL_SERVER_TP" \
        --gpu-memory-utilization "$EVAL_SERVER_GPU_MEMORY_UTILIZATION" \
        --api-key "$EVAL_SERVER_API_KEY" \
        --startup-timeout "$EVAL_SERVER_STARTUP_TIMEOUT" \
    )
    if [[ -n "$EVAL_SERVER_MAX_MODEL_LEN" ]]; then
        server_cmd+=(--max-model-len "$EVAL_SERVER_MAX_MODEL_LEN")
    fi
    echo "Running endpoint benchmarks with EvalPlus routed through the local OpenAI-compatible server"
    server_cmd+=(-- "${runner_cmd[@]}")
    env EVALPLUS_BACKEND="${EVALPLUS_BACKEND:-openai}" "${server_cmd[@]}"
}

if [[ "$RUN_NON_ENDPOINT_EVAL" == "1" ]]; then
    run_non_endpoint_eval
fi
if [[ "$RUN_ENDPOINT_EVAL" == "1" ]]; then
    run_endpoint_eval
fi
