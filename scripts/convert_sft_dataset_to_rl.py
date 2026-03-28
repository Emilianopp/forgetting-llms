#!/usr/bin/env python3
"""Convert repo SFT parquet datasets into exact-match RL parquet datasets.

This is intended for custom datasets that already exist in the repo's SFT shape:

    {
      "data_source": "...",
      "extra_info": {
        "question": "...",
        "answer": "..."
      }
    }

The output uses the local GRPO / baseline parquet shape:

    {
      "data_source": "...",
      "prompt": [{"role": "user", "content": "..."}],
      "messages": [{"role": "user", "content": "..."}],
      "ground_truth": "...",
      "reward_model": {"style": "...", "ground_truth": "..."},
      "extra_info": {...}
    }

The reward semantics are "exact match proxy" unless you override the data source
to one already supported by `src/rewards/unified_reward.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-source",
        default=None,
        help="Optional override for the output data_source field.",
    )
    parser.add_argument(
        "--ability",
        default="general",
        help="Ability label stored in the RL parquet rows.",
    )
    parser.add_argument(
        "--reward-style",
        default="exact_match_proxy",
        help="reward_model.style written into the converted rows.",
    )
    return parser.parse_args()


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def nested_get(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def to_messages(question: Any) -> list[dict[str, str]]:
    if isinstance(question, list):
        messages: list[dict[str, str]] = []
        for item in question:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            messages.append({"role": role, "content": content})
        if messages:
            return messages
    return [{"role": "user", "content": str(question or "")}]


def convert_split(
    input_path: Path,
    output_path: Path,
    *,
    split: str,
    data_source_override: str | None,
    ability: str,
    reward_style: str,
) -> tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing SFT parquet: {input_path}")

    df = pd.read_parquet(input_path)
    rows: list[dict[str, Any]] = []
    skipped = 0

    for record in df.to_dict(orient="records"):
        question = nested_get(record, "extra_info.question")
        answer = nested_get(record, "extra_info.answer")
        if question in (None, "") or answer in (None, ""):
            skipped += 1
            continue

        data_source = data_source_override or str(record.get("data_source") or "").strip() or "custom"
        messages = to_messages(question)
        extra_info = record.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info = dict(extra_info)
        extra_info.setdefault("split", split)
        extra_info["converted_from"] = "sft"

        rows.append(
            {
                "data_source": data_source,
                "prompt": messages,
                "messages": messages,
                "ability": ability,
                "ground_truth": str(answer),
                "reward_model": {
                    "style": reward_style,
                    "ground_truth": str(answer),
                },
                "extra_info": extra_info,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)
    return len(rows), skipped


def main() -> int:
    args = parse_args()
    input_dir = expand(args.input_dir)
    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "conversion": "sft_to_rl",
        "data_source_override": args.data_source,
        "ability": args.ability,
        "reward_style": args.reward_style,
        "splits": {},
        "reward_support": "exact_match_proxy",
    }

    for split in ("train", "test"):
        output_rows, skipped = convert_split(
            input_dir / f"{split}.parquet",
            output_dir / f"{split}.parquet",
            split=split,
            data_source_override=args.data_source,
            ability=args.ability,
            reward_style=args.reward_style,
        )
        summary["splits"][split] = {
            "rows_written": output_rows,
            "rows_skipped": skipped,
        }

    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Converted SFT dataset -> RL dataset: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
