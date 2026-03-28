#!/usr/bin/env python3
"""Export a shareable SFT dataset from intermediate checkpoint/status files.

This is useful when trajectory generation is still running or when you want a
small portable dataset to send to someone else without the full scratch tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing checkpoint.parquet and status.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the exported dataset bundle.",
    )
    parser.add_argument(
        "--solutions-per-question",
        type=int,
        default=8,
        help="Keep exactly this many positive solutions per retained question.",
    )
    parser.add_argument(
        "--min-correct-per-question",
        type=int,
        default=8,
        help="Keep only questions with at least this many correct solutions.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also export a JSONL copy of the train dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = input_dir / "checkpoint.parquet"
    status_path = input_dir / "status.parquet"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")
    if not status_path.exists():
        raise FileNotFoundError(f"Missing status file: {status_path}")

    positives_df = pd.read_parquet(checkpoint_path)
    status_df = pd.read_parquet(status_path)

    required_correct = max(args.solutions_per_question, args.min_correct_per_question)
    eligible_ids = {
        int(row["idx"])
        for _, row in status_df.iterrows()
        if int(row.get("n_correct", 0)) >= required_correct
    }

    filtered = positives_df[positives_df["idx"].isin(eligible_ids)].copy()
    filtered = filtered.sort_values(["idx", "attempt_round", "answer"]).reset_index(drop=True)
    filtered = filtered.groupby("idx", group_keys=False).head(args.solutions_per_question)

    train_records: list[dict] = []
    for _, row in filtered.iterrows():
        train_records.append(
            {
                "data_source": row["data_source"],
                "extra_info": {
                    "question": row["question"],
                    "answer": row["answer"],
                    "source_question_id": int(row["idx"]),
                    "ground_truth": row["ground_truth"],
                    "split": "train",
                },
            }
        )

    train_df = pd.DataFrame(train_records)
    train_path = output_dir / "train.parquet"
    train_df.to_parquet(train_path)

    retained_questions = (
        filtered.groupby("idx", as_index=False)
        .agg(
            question=("question", "first"),
            ground_truth=("ground_truth", "first"),
            retained_positive_solutions=("normalized_answer", "nunique"),
        )
        .sort_values("idx")
        .reset_index(drop=True)
    )
    retained_questions = retained_questions.merge(
        status_df[["idx", "attempts_used", "n_correct", "attempt_round"]].drop_duplicates("idx", keep="last"),
        on="idx",
        how="left",
    )
    retained_questions.to_parquet(output_dir / "retained_questions.parquet")

    with (output_dir / "retained_question_ids.json").open("w") as handle:
        json.dump(retained_questions["idx"].tolist(), handle, indent=2)
        handle.write("\n")

    summary = {
        "source_input_dir": str(input_dir),
        "questions_with_positive_examples": int(positives_df["idx"].nunique()),
        "questions_kept": int(filtered["idx"].nunique()),
        "required_correct_per_question": required_correct,
        "solutions_per_question": args.solutions_per_question,
        "min_correct_per_question": args.min_correct_per_question,
        "total_positive_examples": int(len(positives_df)),
        "exported_train_examples": int(len(filtered)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.jsonl:
        with (output_dir / "train.jsonl").open("w") as handle:
            for record in train_records:
                handle.write(json.dumps(record) + "\n")

    print(f"Wrote exported dataset to {output_dir}")
    print(f"Train parquet: {train_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
