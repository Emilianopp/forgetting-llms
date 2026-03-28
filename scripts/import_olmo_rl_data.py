#!/usr/bin/env python3
"""Import official OLMo 3 RL datasets into scratch-local parquet files.

This script is intentionally conservative:
- it preserves the raw Hugging Face rows
- it writes a normalized parquet shape compatible with this repo's utilities
- it records whether the imported variant is compatible with the current
  `src/rewards/unified_reward.py` path

The current repo reward path only fully supports math-style exact / boxed
answer supervision. Other OLMo RL variants can be imported and inspected, but
should not be launched through `run_grpo_dataset_dir.sh` without adding the
correct reward implementation first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


VARIANT_SPECS: dict[str, dict[str, str]] = {
    "math": {
        "dataset_id": "allenai/Dolci-RL-Zero-Math-7B",
        "notes": "Official OLMo RL-Zero math dataset.",
        "reward_support": "full",
    },
    "code": {
        "dataset_id": "allenai/Dolci-RL-Zero-Code-7B",
        "notes": "Official OLMo RL-Zero code dataset. Current repo reward path does not execute tests.",
        "reward_support": "missing",
    },
    "if": {
        "dataset_id": "allenai/Dolci-RL-Zero-IF-7B",
        "notes": "Official OLMo RL-Zero instruction-following dataset. Current repo lacks the constraint verifier.",
        "reward_support": "missing",
    },
    "general": {
        "dataset_id": "allenai/Dolci-RL-Zero-General-7B",
        "notes": "Official OLMo RL-Zero general-chat dataset. Current repo lacks the LM-judge reward.",
        "reward_support": "missing",
    },
    "mix": {
        "dataset_id": "allenai/Dolci-RL-Zero-Mix-7B",
        "notes": "Official OLMo RL-Zero mixed dataset spanning math/code/IF/general.",
        "reward_support": "missing",
    },
    "instruct": {
        "dataset_id": "allenai/Dolci-Instruct-RL",
        "notes": "Official OLMo Instruct RL dataset used for Olmo-3-7B-Instruct.",
        "reward_support": "missing",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANT_SPECS),
        help="Official OLMo RL dataset variant to import.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to ~/scratch/forgetting-llms/data/olmo_rl_<variant>",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Source split to load from HF. Defaults to train.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.02,
        help="Fraction held out into test.parquet. Defaults to 0.02.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_messages(prompt_value: Any) -> list[dict[str, str]]:
    if isinstance(prompt_value, list):
        normalized: list[dict[str, str]] = []
        for item in prompt_value:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                normalized.append({"role": role, "content": content})
        if normalized:
            return normalized
    if isinstance(prompt_value, str):
        return [{"role": "user", "content": prompt_value}]
    return [{"role": "user", "content": str(prompt_value or "")}]


def first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first
    return ""


def normalize_record(example: dict[str, Any], variant: str) -> dict[str, Any]:
    prompt_value = example.get("prompt")
    if prompt_value is None:
        prompt_value = example.get("source_prompt")
    messages = ensure_messages(prompt_value)

    reward_model = example.get("reward_model")
    reward_ground_truth = ""
    if isinstance(reward_model, dict):
        reward_ground_truth = first_nonempty(reward_model.get("ground_truth"))

    extra_info = example.get("extra_info")
    extra_ground_truth = ""
    if isinstance(extra_info, dict):
        extra_ground_truth = first_nonempty(
            extra_info.get("ground_truth"),
            extra_info.get("answer"),
        )

    ground_truth = first_nonempty(
        reward_ground_truth,
        example.get("ground_truth"),
        example.get("solution"),
        extra_ground_truth,
    )

    data_source = {
        "math": "math",
        "code": "olmo_rl_zero_code",
        "if": "olmo_rl_zero_if",
        "general": "olmo_rl_zero_general",
        "mix": "olmo_rl_zero_mix",
        "instruct": "olmo_instruct_rl",
    }[variant]

    ability = {
        "math": "math",
        "code": "code",
        "if": "instruction_following",
        "general": "general_chat",
        "mix": "mixed",
        "instruct": "mixed",
    }[variant]

    return {
        "data_source": data_source,
        "prompt": messages,
        "messages": messages,
        "ability": ability,
        "ground_truth": ground_truth,
        "reward_model": {
            "style": "external_dataset",
            "ground_truth": ground_truth,
            "variant": variant,
        },
        "extra_info": {
            "variant": variant,
            "original_dataset_id": VARIANT_SPECS[variant]["dataset_id"],
            "original_row": example,
        },
    }


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    spec = VARIANT_SPECS[args.variant]

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(f"~/scratch/forgetting-llms/data/olmo_rl_{args.variant}").expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(spec["dataset_id"], split=args.split)
    split_ds = ds.train_test_split(test_size=args.test_fraction, seed=args.seed, shuffle=True)
    train_raw = split_ds["train"]
    test_raw = split_ds["test"]

    train_norm = Dataset.from_list([normalize_record(row, args.variant) for row in train_raw])
    test_norm = Dataset.from_list([normalize_record(row, args.variant) for row in test_raw])

    train_raw_path = output_dir / "train_raw.parquet"
    test_raw_path = output_dir / "test_raw.parquet"
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    train_raw.to_parquet(str(train_raw_path))
    test_raw.to_parquet(str(test_raw_path))
    train_norm.to_parquet(str(train_path))
    test_norm.to_parquet(str(test_path))

    metadata = {
        "variant": args.variant,
        "dataset_id": spec["dataset_id"],
        "source_split": args.split,
        "train_rows": len(train_norm),
        "test_rows": len(test_norm),
        "reward_support": spec["reward_support"],
        "notes": spec["notes"],
        "paths": {
            "train": str(train_path),
            "test": str(test_path),
            "train_raw": str(train_raw_path),
            "test_raw": str(test_raw_path),
        },
    }
    if spec["reward_support"] != "full":
        metadata["warning"] = (
            "This variant is imported successfully, but the current repo reward path "
            "does not implement the corresponding verifier/judge. Do not launch it "
            "through scripts/run_grpo_dataset_dir.sh unless you add the correct reward."
        )

    write_metadata(output_dir / "metadata.json", metadata)

    print(f"Imported OLMo RL variant: {args.variant}")
    print(f"Dataset id: {spec['dataset_id']}")
    print(f"Train rows: {len(train_norm)} -> {train_path}")
    print(f"Test rows:  {len(test_norm)} -> {test_path}")
    print(f"Raw train:  {train_raw_path}")
    print(f"Raw test:   {test_raw_path}")
    print(f"Reward support: {spec['reward_support']}")
    if spec["reward_support"] != "full":
        print(metadata["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
