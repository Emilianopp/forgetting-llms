#!/usr/bin/env bash
# Source scratch-local Hugging Face auth for repo launchers.

if [[ "${_FORGETTING_LLMS_HF_AUTH_LOADED:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
export _FORGETTING_LLMS_HF_AUTH_LOADED=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HF_HOME/token}"
if [[ -z "${HF_AUTH_ENV_FILE:-}" ]]; then
    if [[ -f "$REPO_DIR/.hf_auth.sh" ]]; then
        export HF_AUTH_ENV_FILE="$REPO_DIR/.hf_auth.sh"
    else
        export HF_AUTH_ENV_FILE="$HOME/scratch/forgetting-llms/hf_auth.sh"
    fi
fi

mkdir -p "$HF_HOME"

if [[ -f "$HF_AUTH_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$HF_AUTH_ENV_FILE"
fi

if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    mkdir -p "$(dirname "$HF_TOKEN_PATH")"
    printf '%s' "$HF_TOKEN" > "$HF_TOKEN_PATH"
    chmod 600 "$HF_TOKEN_PATH" 2>/dev/null || true
fi
