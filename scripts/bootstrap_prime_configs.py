#!/usr/bin/env python3
"""Bootstrap scratch-local PRIME split config TOMLs.

These configs are intentionally minimal and match the documented PRIME-RL
split-config entrypoint shape:

- trainer config
- orchestrator config
- inference config

The active Verifiers environment name is supplied by the launcher, not
hardcoded into the config files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets-csv", default="gsm8k,math,triviaqa")
    parser.add_argument("--pair-matrix-csv", default="gsm8k:math,gsm8k:triviaqa,math:triviaqa")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--rewrite-existing", action="store_true")
    return parser.parse_args()


def trainer_text(args: argparse.Namespace, label: str) -> str:
    return (
        "# Auto-generated PRIME-RL trainer config.\n"
        f'# label = "{label}"\n\n'
        f"max_steps = {args.max_steps}\n"
        "max_async_level = 1\n\n"
        "[data.fake]\n"
        f"batch_size = {max(1, args.rollouts_per_prompt * 2)}\n\n"
        "[model]\n"
        f'name = "{args.model}"\n'
        f"seq_len = {args.seq_len}\n"
        'optimization_dtype = "bfloat16"\n'
        'reduce_dtype = "bfloat16"\n'
    )


def orchestrator_text(args: argparse.Namespace, label: str) -> str:
    return (
        "# Auto-generated PRIME-RL orchestrator config.\n"
        f'# label = "{label}"\n\n'
        f"max_steps = {args.max_steps}\n"
        "max_async_level = 1\n"
        f"batch_size = {max(1, args.rollouts_per_prompt * 2)}\n\n"
        f"rollouts_per_example = {args.rollouts_per_prompt}\n\n"
        "[model]\n"
        f'name = "{args.model}"\n\n'
        "[sampling]\n"
        f"max_tokens = {args.max_tokens}\n"
        f"temperature = {args.temperature}\n"
    )


def inference_text(args: argparse.Namespace, label: str) -> str:
    return (
        "# Auto-generated PRIME-RL inference config.\n"
        f'# label = "{label}"\n\n'
        "[model]\n"
        f'name = "{args.model}"\n'
        "enforce_eager = false\n"
    )


def write_if_needed(path: Path, content: str, rewrite_existing: bool) -> None:
    if path.exists() and not rewrite_existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()

    datasets = [item.strip() for item in args.datasets_csv.split(",") if item.strip()]
    pairs = [item.strip() for item in args.pair_matrix_csv.split(",") if item.strip()]

    for dataset in datasets:
        trainer_path = output_root / f"{dataset}.trainer.toml"
        orchestrator_path = output_root / f"{dataset}.orchestrator.toml"
        inference_path = output_root / f"{dataset}.inference.toml"
        write_if_needed(trainer_path, trainer_text(args, dataset), args.rewrite_existing)
        write_if_needed(orchestrator_path, orchestrator_text(args, dataset), args.rewrite_existing)
        write_if_needed(inference_path, inference_text(args, dataset), args.rewrite_existing)
        print(trainer_path)
        print(orchestrator_path)
        print(inference_path)

    for pair in pairs:
        normalized = pair.replace(":", "_")
        for prefix in ("mix", "iid"):
            label = f"{prefix}_{normalized}"
            trainer_path = output_root / f"{label}.trainer.toml"
            orchestrator_path = output_root / f"{label}.orchestrator.toml"
            inference_path = output_root / f"{label}.inference.toml"
            write_if_needed(trainer_path, trainer_text(args, label), args.rewrite_existing)
            write_if_needed(orchestrator_path, orchestrator_text(args, label), args.rewrite_existing)
            write_if_needed(inference_path, inference_text(args, label), args.rewrite_existing)
            print(trainer_path)
            print(orchestrator_path)
            print(inference_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
