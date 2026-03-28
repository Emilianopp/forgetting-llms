#!/usr/bin/env bash
# Periodically mirror a training output directory into a secondary filesystem or remote.
#
# Intended use:
#   - source lives on fast scratch
#   - destination lives on a mounted backup path, an rclone remote such as Google Drive,
#     or a Hugging Face Hub repo
#   - mirroring is best-effort and should not fail the main training job
#
# Usage:
#   bash scripts/watch_directory_mirror.sh <source_dir> <mirror_root> [label]
#
# Environment knobs:
#   MIRROR_STOP_FILE=...     Optional file path; if it exists, do one last sync and exit
#   MIRROR_POLL_SECS=120     Poll interval between sync attempts
#   MIRROR_SOURCE_BASE=...   Optional prefix stripped from source_dir to preserve relative layout
#   MIRROR_RELATIVE_PATH=... Optional explicit relative path under mirror_root
#   MIRROR_PRUNE_DEST=0      If 1, delete files from destination that no longer exist in source
#   MIRROR_RCLONE_TRANSFERS=4
#   MIRROR_RCLONE_CHECKERS=8
#   MIRROR_HF_PRIVATE=0      If 1, create the HF repo as private on first upload
#   MIRROR_HF_PUBLIC=0       If 1, create the HF repo as public on first upload
#   MIRROR_HF_COMMIT_PREFIX=mirror:<label>  Commit message prefix for HF uploads
#
# Mirror root formats:
#   /local/path
#   gdrive:forgetting-llms-backups
#   hf:model:<user_or_org>/<repo>
#   hf:dataset:<user_or_org>/<repo>
#   hf:<user_or_org>/<repo>            (defaults to repo_type=model)

set -euo pipefail

SOURCE_DIR=${1:?Usage: watch_directory_mirror.sh <source_dir> <mirror_root> [label]}
MIRROR_ROOT=${2:?Usage: watch_directory_mirror.sh <source_dir> <mirror_root> [label]}
LABEL=${3:-mirror}

MIRROR_STOP_FILE=${MIRROR_STOP_FILE:-}
MIRROR_POLL_SECS=${MIRROR_POLL_SECS:-120}
MIRROR_SOURCE_BASE=${MIRROR_SOURCE_BASE:-}
MIRROR_RELATIVE_PATH=${MIRROR_RELATIVE_PATH:-}
MIRROR_PRUNE_DEST=${MIRROR_PRUNE_DEST:-0}
MIRROR_RCLONE_TRANSFERS=${MIRROR_RCLONE_TRANSFERS:-4}
MIRROR_RCLONE_CHECKERS=${MIRROR_RCLONE_CHECKERS:-8}
MIRROR_HF_PRIVATE=${MIRROR_HF_PRIVATE:-0}
MIRROR_HF_PUBLIC=${MIRROR_HF_PUBLIC:-0}
MIRROR_HF_COMMIT_PREFIX=${MIRROR_HF_COMMIT_PREFIX:-mirror:$LABEL}
LAST_SYNC_FINGERPRINT=""

is_rclone_remote() {
    local path="$1"
    if is_hf_remote "$path"; then
        return 1
    fi
    [[ "$path" =~ ^[A-Za-z0-9_-]+:.*$ ]]
}

is_hf_remote() {
    local path="$1"
    [[ "$path" =~ ^hf: ]]
}

expand_path() {
    local path="$1"
    if is_rclone_remote "$path" || is_hf_remote "$path"; then
        printf '%s\n' "$path"
        return 0
    fi
    if [[ "$path" == "~"* ]]; then
        path="${HOME}${path:1}"
    fi
    if [[ "$path" == /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$PWD" "$path"
    fi
}

SOURCE_DIR=$(expand_path "$SOURCE_DIR")
MIRROR_ROOT=$(expand_path "$MIRROR_ROOT")
if [[ -n "$MIRROR_SOURCE_BASE" ]]; then
    MIRROR_SOURCE_BASE=$(expand_path "$MIRROR_SOURCE_BASE")
fi

join_dest_path() {
    local root="$1"
    local relative="$2"
    relative="${relative#/}"
    if is_hf_remote "$root"; then
        if [[ -n "$relative" ]]; then
            printf '%s/%s\n' "$root" "$relative"
        else
            printf '%s\n' "$root"
        fi
        return 0
    fi
    if is_rclone_remote "$root"; then
        if [[ "$root" == *: ]]; then
            printf '%s%s\n' "$root" "$relative"
        elif [[ -n "$relative" ]]; then
            printf '%s/%s\n' "$root" "$relative"
        else
            printf '%s\n' "$root"
        fi
    else
        if [[ -n "$relative" ]]; then
            printf '%s/%s\n' "$root" "$relative"
        else
            printf '%s\n' "$root"
        fi
    fi
}

if [[ -n "$MIRROR_RELATIVE_PATH" ]]; then
    DEST_DIR=$(join_dest_path "$MIRROR_ROOT" "$MIRROR_RELATIVE_PATH")
elif [[ -n "$MIRROR_SOURCE_BASE" && "$SOURCE_DIR" == "$MIRROR_SOURCE_BASE"/* ]]; then
    DEST_DIR=$(join_dest_path "$MIRROR_ROOT" "${SOURCE_DIR#"$MIRROR_SOURCE_BASE"/}")
else
    DEST_DIR=$(join_dest_path "$MIRROR_ROOT" "$(basename "$SOURCE_DIR")")
fi

DEST_RELATIVE_PATH=""
if [[ -n "$MIRROR_RELATIVE_PATH" ]]; then
    DEST_RELATIVE_PATH="${MIRROR_RELATIVE_PATH#/}"
elif [[ -n "$MIRROR_SOURCE_BASE" && "$SOURCE_DIR" == "$MIRROR_SOURCE_BASE"/* ]]; then
    DEST_RELATIVE_PATH="${SOURCE_DIR#"$MIRROR_SOURCE_BASE"/}"
else
    DEST_RELATIVE_PATH="$(basename "$SOURCE_DIR")"
fi

parse_hf_remote() {
    local path="$1"
    local spec="${path#hf:}"
    local repo_type="model"
    local repo_id="$spec"
    case "$spec" in
        model:*)
            repo_type="model"
            repo_id="${spec#model:}"
            ;;
        dataset:*)
            repo_type="dataset"
            repo_id="${spec#dataset:}"
            ;;
        space:*)
            repo_type="space"
            repo_id="${spec#space:}"
            ;;
    esac
    if [[ -z "$repo_id" || "$repo_id" != */* ]]; then
        echo "[mirror:$LABEL] error: invalid Hugging Face mirror root: $path" >&2
        exit 1
    fi
    printf '%s\n%s\n' "$repo_type" "$repo_id"
}

tree_fingerprint() {
    python3 - "$SOURCE_DIR" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    print("missing")
    raise SystemExit(0)

entries = []
for path in sorted(root.rglob("*")):
    try:
        stat = path.stat()
    except OSError:
        continue
    rel = path.relative_to(root).as_posix()
    kind = "d" if path.is_dir() else "f"
    size = stat.st_size if path.is_file() else 0
    mtime = int(stat.st_mtime_ns)
    entries.append(f"{kind}\t{rel}\t{size}\t{mtime}")

digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
print(digest)
PY
}

if ! is_rclone_remote "$MIRROR_ROOT" && ! is_hf_remote "$MIRROR_ROOT"; then
    mkdir -p "$MIRROR_ROOT"
elif is_rclone_remote "$MIRROR_ROOT" && ! command -v rclone >/dev/null 2>&1; then
    echo "[mirror:$LABEL] error: MIRROR_ROOT looks like an rclone remote but rclone is not installed: $MIRROR_ROOT" >&2
    exit 1
fi

sync_once() {
    if [[ ! -d "$SOURCE_DIR" ]]; then
        echo "[mirror:$LABEL] source does not exist yet: $SOURCE_DIR"
        return 0
    fi

    local current_fingerprint=""
    current_fingerprint=$(tree_fingerprint)
    if [[ -n "$LAST_SYNC_FINGERPRINT" && "$current_fingerprint" == "$LAST_SYNC_FINGERPRINT" ]]; then
        echo "[mirror:$LABEL] unchanged -> $DEST_DIR"
        return 0
    fi

    if is_hf_remote "$MIRROR_ROOT"; then
        local hf_repo_type=""
        local hf_repo_id=""
        mapfile -t hf_parts < <(parse_hf_remote "$MIRROR_ROOT")
        hf_repo_type="${hf_parts[0]}"
        hf_repo_id="${hf_parts[1]}"
        local hf_args=(
            python3
            "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/upload_hf_folder.py"
            --repo-id "$hf_repo_id"
            --repo-type "$hf_repo_type"
            --folder-path "$SOURCE_DIR"
            --path-in-repo "$DEST_RELATIVE_PATH"
            --commit-message "$MIRROR_HF_COMMIT_PREFIX sync $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        )
        if [[ "$MIRROR_HF_PRIVATE" == "1" ]]; then
            hf_args+=(--private)
        elif [[ "$MIRROR_HF_PUBLIC" == "1" ]]; then
            hf_args+=(--public)
        fi
        if ! "${hf_args[@]}"; then
            echo "[mirror:$LABEL] warning: HF upload failed from $SOURCE_DIR to $DEST_DIR" >&2
            return 1
        fi
    elif is_rclone_remote "$MIRROR_ROOT"; then
        if ! rclone copy \
            "$SOURCE_DIR" \
            "$DEST_DIR" \
            --create-empty-src-dirs \
            --transfers "$MIRROR_RCLONE_TRANSFERS" \
            --checkers "$MIRROR_RCLONE_CHECKERS"; then
            echo "[mirror:$LABEL] warning: rclone copy failed from $SOURCE_DIR to $DEST_DIR" >&2
            return 1
        fi
    else
        mkdir -p "$DEST_DIR" || {
            echo "[mirror:$LABEL] warning: could not create destination: $DEST_DIR" >&2
            return 1
        }

        if command -v rsync >/dev/null 2>&1; then
            local rsync_args=(
                -a
                --partial
                --human-readable
                --exclude=.nfs*
                --exclude=.DS_Store
            )
            if [[ "$MIRROR_PRUNE_DEST" == "1" ]]; then
                rsync_args+=(--delete)
            fi
            if ! rsync "${rsync_args[@]}" "$SOURCE_DIR"/ "$DEST_DIR"/; then
                echo "[mirror:$LABEL] warning: rsync failed from $SOURCE_DIR to $DEST_DIR" >&2
                return 1
            fi
        else
            if ! cp -a "$SOURCE_DIR"/. "$DEST_DIR"/; then
                echo "[mirror:$LABEL] warning: cp fallback failed from $SOURCE_DIR to $DEST_DIR" >&2
                return 1
            fi
        fi
    fi

    local checkpoint_count=0
    checkpoint_count=$(
        find "$SOURCE_DIR" -maxdepth 1 -type d \
            \( -name 'checkpoint-*' -o -name 'global_step_*' -o -name 'step_*' \) \
            2>/dev/null | wc -l | tr -d ' '
    )
    LAST_SYNC_FINGERPRINT="$current_fingerprint"
    echo "[mirror:$LABEL] synced -> $DEST_DIR (checkpoint_dirs=$checkpoint_count)"
    return 0
}

echo "[mirror:$LABEL] source: $SOURCE_DIR"
echo "[mirror:$LABEL] dest:   $DEST_DIR"
echo "[mirror:$LABEL] poll:   ${MIRROR_POLL_SECS}s"

while true; do
    sync_once || true
    if [[ -n "$MIRROR_STOP_FILE" && -f "$MIRROR_STOP_FILE" ]]; then
        echo "[mirror:$LABEL] stop requested; running final sync"
        sync_once || true
        exit 0
    fi
    sleep "$MIRROR_POLL_SECS"
done
