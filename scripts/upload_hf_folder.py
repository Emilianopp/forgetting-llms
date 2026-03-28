#!/usr/bin/env python3
"""Upload a local folder into a Hugging Face Hub repo path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--folder-path", required=True)
    parser.add_argument("--path-in-repo", default="")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if args.private and args.public:
        parser.error("Choose at most one of --private or --public.")
    return args


def resolve_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path:
        path = Path(token_path).expanduser()
        if path.exists():
            token = path.read_text().strip()
            if token:
                return token
    return None


def main() -> int:
    args = parse_args()
    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    token = resolve_token()
    if not token:
        raise SystemExit("Missing Hugging Face write token. Set HF_TOKEN or HF_TOKEN_PATH.")

    private: bool | None = None
    if args.private:
        private = True
    elif args.public:
        private = False

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=private,
        exist_ok=True,
    )

    commit_message = args.commit_message or f"Upload {folder.name}"
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=folder,
        path_in_repo=args.path_in_repo or None,
        commit_message=commit_message,
        token=token,
    )
    print(commit.commit_url or commit.oid or "upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
