#!/usr/bin/env bash

# Shared helpers for scripts that launch a vLLM OpenAI server and then run a
# client/runner process against it.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"

usage_vllm_runner_common() {
    cat <<'EOF'
Common options:
  --model PATH_OR_HF_ID           Model to serve (required)
  --run-root PATH                 Directory for logs / state
  --host HOST                     Server host (default: 127.0.0.1)
  --port PORT                     Server port (default: 8000)
  --served-model-name NAME        Optional served model alias
  --tensor-parallel-size N        vLLM tensor parallel size (default: 1)
  --gpu-memory-utilization FLOAT  vLLM GPU memory utilization (default: 0.90)
  --max-model-len N               Optional max model length
  --api-key TOKEN                 API key exposed to clients (default: EMPTY)
  --startup-timeout SEC           Health-check timeout (default: 600)
  --server-log PATH               Optional explicit server log path
  --runner-log PATH               Optional explicit runner log path
  --no-health-check               Skip polling /health before starting runner
  --
  <runner command...>             Command executed after the server is ready

Environment:
  VLLM_SERVER_EXTRA_ARGS          Extra args appended to the vLLM server command
  SUPPRESS_LITELLM_PROVIDER_LIST Default: 1
  VLLM_SERVER_USE_V1             Default: 1
  VLLM_WORKER_MULTIPROC_METHOD   Default: spawn
  VLLM_USE_STANDALONE_COMPILE    Default: 0
  VLLM_DISABLE_COMPILE_CACHE     Default: 1
  VLLM_SERVER_ENFORCE_EAGER      Default: 1
  VLLM_USE_FLASHINFER_SAMPLER    Default: 0
  VLLM_LOG_TAIL_LINES            Default: 200
EOF
}

parse_vllm_runner_args() {
    MODEL=""
    RUN_ROOT=""
    HOST="127.0.0.1"
    PORT="8000"
    SERVED_MODEL_NAME=""
    TENSOR_PARALLEL_SIZE="1"
    GPU_MEMORY_UTILIZATION="0.90"
    MAX_MODEL_LEN=""
    API_KEY="EMPTY"
    STARTUP_TIMEOUT="600"
    SERVER_LOG=""
    RUNNER_LOG=""
    SKIP_HEALTH_CHECK="0"
    RUNNER_CMD=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --run-root)
                RUN_ROOT="$2"
                shift 2
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --served-model-name)
                SERVED_MODEL_NAME="$2"
                shift 2
                ;;
            --tensor-parallel-size)
                TENSOR_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --gpu-memory-utilization)
                GPU_MEMORY_UTILIZATION="$2"
                shift 2
                ;;
            --max-model-len)
                MAX_MODEL_LEN="$2"
                shift 2
                ;;
            --api-key)
                API_KEY="$2"
                shift 2
                ;;
            --startup-timeout)
                STARTUP_TIMEOUT="$2"
                shift 2
                ;;
            --server-log)
                SERVER_LOG="$2"
                shift 2
                ;;
            --runner-log)
                RUNNER_LOG="$2"
                shift 2
                ;;
            --no-health-check)
                SKIP_HEALTH_CHECK="1"
                shift
                ;;
            --)
                shift
                RUNNER_CMD=("$@")
                break
                ;;
            -h|--help)
                usage_vllm_runner_common
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage_vllm_runner_common >&2
                exit 1
                ;;
        esac
    done

    if [[ -z "$MODEL" ]]; then
        echo "--model is required" >&2
        exit 1
    fi
    if [[ -z "$RUN_ROOT" ]]; then
        echo "--run-root is required" >&2
        exit 1
    fi
    if [[ ${#RUNNER_CMD[@]} -eq 0 ]]; then
        echo "A runner command is required after --" >&2
        exit 1
    fi

    mkdir -p "$RUN_ROOT"
    if [[ -z "$SERVER_LOG" ]]; then
        SERVER_LOG="$RUN_ROOT/vllm_server.log"
    fi
    if [[ -z "$RUNNER_LOG" ]]; then
        RUNNER_LOG="$RUN_ROOT/runner.log"
    fi
    : >"$SERVER_LOG"
    : >"$RUNNER_LOG"
}

print_log_tail() {
    local label="$1"
    local path="$2"
    local lines="${VLLM_LOG_TAIL_LINES:-200}"
    if [[ -f "$path" ]]; then
        echo "===== ${label} (last ${lines} lines) =====" >&2
        tail -n "$lines" "$path" >&2 || true
        echo "===== end ${label} =====" >&2
    else
        echo "Missing ${label}: $path" >&2
    fi
}

start_vllm_server() {
    local server_use_v1="${VLLM_SERVER_USE_V1:-1}"
    if [[ "${VLLM_USE_V1:-}" != "" && "${VLLM_USE_V1}" != "$server_use_v1" ]]; then
        echo "Ignoring inherited VLLM_USE_V1=$VLLM_USE_V1 for API server; using VLLM_SERVER_USE_V1=$server_use_v1"
    fi
    export VLLM_USE_V1="$server_use_v1"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
    export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    local enforce_eager="${VLLM_SERVER_ENFORCE_EAGER:-1}"

    local cmd=(
        python3 -m vllm.entrypoints.openai.api_server
        --model "$MODEL"
        --host "$HOST"
        --port "$PORT"
        --api-key "$API_KEY"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    )

    if [[ -n "$SERVED_MODEL_NAME" ]]; then
        cmd+=(--served-model-name "$SERVED_MODEL_NAME")
    fi
    if [[ -n "$MAX_MODEL_LEN" ]]; then
        cmd+=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [[ "$enforce_eager" == "1" ]]; then
        cmd+=(--enforce-eager)
    fi
    if [[ -n "${VLLM_SERVER_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        extra_args=(${VLLM_SERVER_EXTRA_ARGS})
        cmd+=("${extra_args[@]}")
    fi

    echo "Starting vLLM server:"
    echo "  VLLM_USE_V1=$VLLM_USE_V1"
    echo "  VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD"
    echo "  VLLM_USE_STANDALONE_COMPILE=$VLLM_USE_STANDALONE_COMPILE"
    echo "  VLLM_DISABLE_COMPILE_CACHE=$VLLM_DISABLE_COMPILE_CACHE"
    echo "  VLLM_SERVER_ENFORCE_EAGER=$enforce_eager"
    echo "  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
    printf '  %q' "${cmd[@]}"
    echo
    "${cmd[@]}" >"$SERVER_LOG" 2>&1 &
    VLLM_SERVER_PID=$!
    export VLLM_SERVER_PID
    echo "$VLLM_SERVER_PID" >"$RUN_ROOT/vllm_server.pid"
}

wait_for_vllm_server() {
    if [[ "$SKIP_HEALTH_CHECK" == "1" ]]; then
        return 0
    fi

    local health_url="http://${HOST}:${PORT}/health"
    local start_ts
    start_ts=$(date +%s)

    echo "Waiting for vLLM health endpoint: $health_url"
    while true; do
        if command -v curl >/dev/null 2>&1; then
            health_ok() { curl -fsS "$health_url" >/dev/null 2>&1; }
        else
            health_ok() {
                python3 - "$health_url" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
            }
        fi

        if health_ok; then
            echo "vLLM server is ready"
            return 0
        fi
        if ! kill -0 "$VLLM_SERVER_PID" >/dev/null 2>&1; then
            echo "vLLM server exited before becoming ready. Check $SERVER_LOG" >&2
            print_log_tail "vLLM server log" "$SERVER_LOG"
            return 1
        fi
        local now_ts
        now_ts=$(date +%s)
        if (( now_ts - start_ts >= STARTUP_TIMEOUT )); then
            echo "Timed out waiting for vLLM server after ${STARTUP_TIMEOUT}s" >&2
            print_log_tail "vLLM server log" "$SERVER_LOG"
            return 1
        fi
        sleep 5
    done
}

resolve_served_model_name() {
    if [[ -n "$SERVED_MODEL_NAME" ]]; then
        printf '%s\n' "$SERVED_MODEL_NAME"
        return 0
    fi

    local models_url="http://${HOST}:${PORT}/v1/models"
    python3 - "$models_url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=10) as response:
    payload = json.load(response)

data = payload.get("data") or []
if not data:
    raise SystemExit(1)

model_id = data[0].get("id")
if not model_id:
    raise SystemExit(1)

print(model_id)
PY
}

export_vllm_client_env() {
    local base_url="http://${HOST}:${PORT}/v1"
    local no_proxy_value="127.0.0.1,localhost,::1"
    local resolved_model_name=""
    resolved_model_name="$(resolve_served_model_name || true)"
    export OPENAI_API_BASE="$base_url"
    export OPENAI_BASE_URL="$base_url"
    export VLLM_BASE_URL="$base_url"
    export OPENAI_API_KEY="$API_KEY"
    export VLLM_API_KEY="$API_KEY"
    export VLLM_SERVER_HOST="$HOST"
    export VLLM_SERVER_PORT="$PORT"
    export NO_PROXY="$no_proxy_value"
    export no_proxy="$no_proxy_value"
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    if [[ -n "$resolved_model_name" ]]; then
        export OPENAI_MODEL_NAME="$resolved_model_name"
        export SERVED_MODEL_NAME="$resolved_model_name"
    fi

    cat >"$RUN_ROOT/vllm_client_env.sh" <<EOF
export OPENAI_API_BASE="$base_url"
export OPENAI_BASE_URL="$base_url"
export VLLM_BASE_URL="$base_url"
export OPENAI_API_KEY="$API_KEY"
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVER_HOST="$HOST"
export VLLM_SERVER_PORT="$PORT"
export NO_PROXY="$no_proxy_value"
export no_proxy="$no_proxy_value"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
EOF
    if [[ -n "$resolved_model_name" ]]; then
        {
            printf 'export OPENAI_MODEL_NAME="%s"\n' "$resolved_model_name"
            printf 'export SERVED_MODEL_NAME="%s"\n' "$resolved_model_name"
        } >>"$RUN_ROOT/vllm_client_env.sh"
    fi
}

stop_vllm_server() {
    if [[ -n "${VLLM_SERVER_PID:-}" ]] && kill -0 "$VLLM_SERVER_PID" >/dev/null 2>&1; then
        echo "Stopping vLLM server pid=$VLLM_SERVER_PID"
        kill "$VLLM_SERVER_PID" >/dev/null 2>&1 || true
        wait "$VLLM_SERVER_PID" >/dev/null 2>&1 || true
    fi
}

run_runner_command() {
    echo "Running runner command:"
    printf '  %q' "${RUNNER_CMD[@]}"
    echo
    local status=0
    local suppress_litellm_provider_list="${SUPPRESS_LITELLM_PROVIDER_LIST:-1}"
    if [[ "$suppress_litellm_provider_list" == "1" ]]; then
        "${RUNNER_CMD[@]}" \
            > >(tee "$RUNNER_LOG" | grep -Fv 'docs.litellm.ai/docs/providers') \
            2> >(tee -a "$RUNNER_LOG" | grep -Fv 'docs.litellm.ai/docs/providers' >&2) || status=$?
    else
        "${RUNNER_CMD[@]}" > >(tee "$RUNNER_LOG") 2> >(tee -a "$RUNNER_LOG" >&2) || status=$?
    fi
    if (( status != 0 )); then
        echo "Runner command failed with exit code $status" >&2
        print_log_tail "eval runner log" "$RUNNER_LOG"
        print_log_tail "vLLM server log" "$SERVER_LOG"
        return "$status"
    fi
}
