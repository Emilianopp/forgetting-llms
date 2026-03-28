#!/usr/bin/env python3
"""Convert repo RL parquet datasets into SFT parquet datasets.

This is useful for RL-first datasets such as imported OLMo RL-Zero math data.
The conversion keeps the prompt text as `extra_info.question` and the reference
answer as `extra_info.answer`.
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


def stringify_messages(messages: list[dict[str, Any]]) -> str:
    prompt_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        prompt_parts.append(f"{role}: {content}")
    return "\n\n".join(prompt_parts)


def resolve_question(record: dict[str, Any]) -> str:
    prompt_value = record.get("prompt")
    if isinstance(prompt_value, str) and prompt_value.strip():
        return prompt_value
    if isinstance(prompt_value, list) and prompt_value:
        rendered = stringify_messages(prompt_value)
        if rendered:
            return rendered

    messages_value = record.get("messages")
    if isinstance(messages_value, list) and messages_value:
        rendered = stringify_messages(messages_value)
        if rendered:
            return rendered

    extra_question = nested_get(record, "extra_info.question")
    if isinstance(extra_question, str) and extra_question.strip():
        return extra_question

    return ""


def resolve_answer(record: dict[str, Any]) -> str:
    for key in ("ground_truth", "extra_info.answer"):
        value = nested_get(record, key)
        if isinstance(value, str) and value.strip():
            return value
    reward_ground_truth = nested_get(record, "reward_model.ground_truth")
    if isinstance(reward_ground_truth, str) and reward_ground_truth.strip():
        return reward_ground_truth
    return ""


def convert_split(
    input_path: Path,
    output_path: Path,
    *,
    split: str,
    data_source_override: str | None,
) -> tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing RL parquet: {input_path}")

    df = pd.read_parquet(input_path)
    rows: list[dict[str, Any]] = []
    skipped = 0

    for record in df.to_dict(orient="records"):
        question = resolve_question(record)
        answer = resolve_answer(record)
        if not question or not answer:
            skipped += 1
            continue

        data_source = data_source_override or str(record.get("data_source") or "").strip() or "custom"
        extra_info = record.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info = dict(extra_info)
        extra_info["question"] = question
        extra_info["answer"] = answer
        extra_info.setdefault("split", split)
        extra_info["converted_from"] = "rl"

        rows.append(
            {
                "data_source": data_source,
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
        "conversion": "rl_to_sft",
        "data_source_override": args.data_source,
        "splits": {},
    }

    for split in ("train", "test"):
        output_rows, skipped = convert_split(
            input_dir / f"{split}.parquet",
            output_dir / f"{split}.parquet",
            split=split,
            data_source_override=args.data_source,
        )
        summary["splits"][split] = {
            "rows_written": output_rows,
            "rows_skipped": skipped,
        }

    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Converted RL dataset -> SFT dataset: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
