#!/usr/bin/env python3
"""Download Hugging Face model snapshots into scratch-local directories.

Examples:
    python scripts/download_models.py --model Qwen/Qwen3-1.7B
    python scripts/download_models.py \
        --config /path/to/config.yaml \
        --config-key teacher_model \
        --config-key student_model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


REPO_MODEL_BUNDLES: dict[str, list[str]] = {
    "sequential_normal": [
        "Qwen/Qwen3-1.7B",
        "allenai/Olmo-3-7B-Instruct",
    ],
    "repo_active": [
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-32B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "allenai/Olmo-3-7B-Instruct",
    ],
    "repo_all_referenced": [
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "allenai/Olmo-3-7B-Instruct",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Hugging Face repo id to download. Repeatable.",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        choices=sorted(REPO_MODEL_BUNDLES),
        help=(
            "Named bundle of repo-referenced models to download. Repeatable. "
            "Use --list-bundles to inspect contents."
        ),
    )
    parser.add_argument(
        "--list-bundles",
        action="store_true",
        help="Print available download bundles and exit.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file containing model ids.",
    )
    parser.add_argument(
        "--config-key",
        action="append",
        default=[],
        help=(
            "YAML key to read from --config for model ids. Repeatable, for example "
            "--config-key teacher_model --config-key student_model."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="~/scratch/forgetting-llms/models",
        help="Root directory under which model snapshots are materialized.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default="~/scratch/huggingface",
        help="HF_HOME path. Defaults to scratch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the destination directory already exists and is non-empty.",
    )
    return parser.parse_args()


def load_models_from_config(config_path: str, keys: list[str]) -> list[str]:
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    models: list[str] = []
    for key in keys:
        value = config.get(key)
        if not value:
            raise ValueError(f"Config key '{key}' not found or empty in {config_path}")
        if isinstance(value, str):
            models.append(value)
        elif isinstance(value, list):
            models.extend(str(item) for item in value if item)
        else:
            raise ValueError(f"Config key '{key}' must be a string or list of strings")
    return models


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def main() -> int:
    args = parse_args()

    if args.list_bundles:
        for bundle_name, models in REPO_MODEL_BUNDLES.items():
            print(f"{bundle_name}:")
            for repo_id in models:
                print(f"  - {repo_id}")
        return 0

    requested_models = list(args.model)
    for bundle_name in args.bundle:
        requested_models.extend(REPO_MODEL_BUNDLES[bundle_name])
    if args.config:
        requested_models.extend(load_models_from_config(args.config, args.config_key))

    unique_models: list[str] = []
    seen = set()
    for repo_id in requested_models:
        if repo_id not in seen:
            unique_models.append(repo_id)
            seen.add(repo_id)

    if not unique_models:
        raise SystemExit("No models requested. Pass --model or --config with --config-key.")

    hf_home = Path(args.hf_home).expanduser()
    output_root = Path(args.output_root).expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    print(f"HF_HOME={hf_home}")
    print(f"Output root={output_root}")

    for repo_id in unique_models:
        destination = output_root / sanitize_repo_id(repo_id)
        if destination.exists() and any(destination.iterdir()) and not args.force:
            print(f"Skipping {repo_id} -> {destination} (already exists)")
            continue

        print(f"Downloading {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
