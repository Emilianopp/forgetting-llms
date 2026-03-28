#!/usr/bin/env python3
"""Watch a training run directory and upload ready checkpoints to a gdrive rclone remote.

This script assumes:
- `rclone` is installed and on PATH
- an rclone remote named `gdrive` is already configured

It uploads each ready checkpoint directory once per local state file.
Supported checkpoint layouts:
- SFT: `checkpoint-*` or `global_step_*` directly under the source dir
- PRIME-RL: `checkpoints/step_*` under the source dir
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Local run/checkpoint root on scratch.")
    parser.add_argument(
        "--remote-root",
        default="gdrive:forgetting-llms-backups",
        help="rclone remote destination root (default: gdrive:forgetting-llms-backups).",
    )
    parser.add_argument(
        "--source-base",
        default=None,
        help="Optional local prefix stripped from source-dir to preserve relative layout on Drive.",
    )
    parser.add_argument("--poll-secs", type=int, default=60)
    parser.add_argument("--transfers", type=int, default=4)
    parser.add_argument("--checkers", type=int, default=8)
    parser.add_argument("--stop-file", default=None, help="Optional file; exit after a final stable poll once it exists.")
    parser.add_argument("--state-file", default=None, help="Optional JSON file tracking already-uploaded checkpoints.")
    parser.add_argument("--run-once", action="store_true", help="Scan once, upload ready checkpoints, then exit.")
    return parser.parse_args()


def expand(path: str | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def weights_ready(model_dir: Path) -> bool:
    return (
        any(model_dir.glob("*.safetensors"))
        or any(model_dir.glob("pytorch_model*.bin"))
        or (model_dir / "model.safetensors.index.json").exists()
        or (model_dir / "pytorch_model.bin.index.json").exists()
    )


def checkpoint_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "config.json").exists() and weights_ready(path):
        return True
    for config in path.rglob("config.json"):
        if weights_ready(config.parent):
            return True
    return False


def discover_candidates(source_dir: Path) -> list[Path]:
    patterns = (
        "checkpoint-*",
        "global_step_*",
        "step_*",
        "checkpoints/checkpoint-*",
        "checkpoints/global_step_*",
        "checkpoints/step_*",
    )
    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in sorted(source_dir.glob(pattern)):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if checkpoint_ready(candidate):
                candidates.append(candidate)
    candidates.sort()
    return candidates


def join_remote(remote_root: str, relative: str) -> str:
    relative = relative.strip("/")
    if not relative:
        return remote_root
    if remote_root.endswith(":"):
        return f"{remote_root}{relative}"
    return f"{remote_root.rstrip('/')}/{relative}"


def load_uploaded(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        payload = json.loads(state_file.read_text())
    except Exception:
        return set()
    if not isinstance(payload, list):
        return set()
    return {item for item in payload if isinstance(item, str)}


def save_uploaded(state_file: Path, uploaded: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(sorted(uploaded), indent=2) + "\n")


def upload_checkpoint(
    checkpoint_dir: Path,
    remote_dest: str,
    *,
    transfers: int,
    checkers: int,
) -> None:
    cmd = [
        "rclone",
        "copy",
        str(checkpoint_dir),
        remote_dest,
        "--create-empty-src-dirs",
        "--transfers",
        str(transfers),
        "--checkers",
        str(checkers),
    ]
    print(f"Uploading {checkpoint_dir} -> {remote_dest}")
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found in PATH")

    source_dir = expand(args.source_dir)
    if source_dir is None or not source_dir.is_dir():
        raise SystemExit(f"Source dir not found: {args.source_dir}")

    source_base = expand(args.source_base) if args.source_base else None
    if source_base is not None and source_dir.is_relative_to(source_base):
        base_relative = str(source_dir.relative_to(source_base)).replace("\\", "/")
    else:
        base_relative = source_dir.name

    remote_run_root = join_remote(args.remote_root, base_relative)
    stop_file = expand(args.stop_file) if args.stop_file else None
    state_file = expand(args.state_file) if args.state_file else source_dir / ".gdrive_uploaded_checkpoints.json"
    uploaded = load_uploaded(state_file)

    print(f"Source dir:   {source_dir}")
    print(f"Remote root:  {remote_run_root}")
    print(f"Poll secs:    {args.poll_secs}")
    print(f"State file:   {state_file}")

    while True:
        ready = discover_candidates(source_dir)
        new_uploads = 0
        for checkpoint_dir in ready:
            relative = str(checkpoint_dir.relative_to(source_dir)).replace("\\", "/")
            if relative in uploaded:
                continue
            remote_dest = join_remote(remote_run_root, relative)
            try:
                upload_checkpoint(
                    checkpoint_dir,
                    remote_dest,
                    transfers=args.transfers,
                    checkers=args.checkers,
                )
            except subprocess.CalledProcessError as exc:
                print(f"Upload failed for {checkpoint_dir}: exit={exc.returncode}", file=sys.stderr)
                continue
            uploaded.add(relative)
            save_uploaded(state_file, uploaded)
            new_uploads += 1

        print(f"Ready checkpoints: {len(ready)} | Uploaded so far: {len(uploaded)} | New this poll: {new_uploads}")

        if args.run_once:
            return 0
        if stop_file is not None and stop_file.exists() and new_uploads == 0:
            print("Stop file detected and no pending new checkpoints. Exiting.")
            return 0
        time.sleep(args.poll_secs)


if __name__ == "__main__":
    raise SystemExit(main())
