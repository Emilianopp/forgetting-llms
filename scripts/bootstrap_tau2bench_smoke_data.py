#!/usr/bin/env python3
"""Create a tiny tau2bench-shaped SFT parquet dataset for smoke testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TRAIN_ROWS = [
    {
        "data_source": "tau2bench",
        "extra_info": {
            "question": (
                "User: Cancel reservation R-1042 for tomorrow and confirm the refund window.\n"
                "Assistant:"
            ),
            "answer": "Reservation R-1042 canceled. Refund window: 3-5 business days.",
            "domain": "airline",
            "task_id": "tau2bench_smoke_train_0001",
        },
    },
    {
        "data_source": "tau2bench",
        "extra_info": {
            "question": (
                "User: Change order O-88 shipping address to 22 Market Street, Montreal.\n"
                "Assistant:"
            ),
            "answer": "Order O-88 shipping address updated to 22 Market Street, Montreal.",
            "domain": "retail",
            "task_id": "tau2bench_smoke_train_0002",
        },
    },
    {
        "data_source": "tau2bench",
        "extra_info": {
            "question": (
                "User: Upgrade hotel booking H-771 to a king room if available.\n"
                "Assistant:"
            ),
            "answer": "Booking H-771 upgraded to a king room.",
            "domain": "hotel",
            "task_id": "tau2bench_smoke_train_0003",
        },
    },
    {
        "data_source": "tau2bench",
        "extra_info": {
            "question": (
                "User: What is the status of support ticket T-555?\n"
                "Assistant:"
            ),
            "answer": "Support ticket T-555 is in progress.",
            "domain": "support",
            "task_id": "tau2bench_smoke_train_0004",
        },
    },
]

TEST_ROWS = [
    {
        "data_source": "tau2bench",
        "extra_info": {
            "question": (
                "User: Mark invoice I-202 as paid and confirm the balance.\n"
                "Assistant:"
            ),
            "answer": "Invoice I-202 marked as paid. Remaining balance: 0.",
            "domain": "billing",
            "task_id": "tau2bench_smoke_test_0001",
        },
    },
    {
        "data_source": "tau2bench",
        "extra_info": {
            "question": (
                "User: Pause subscription S-91 until next month.\n"
                "Assistant:"
            ),
            "answer": "Subscription S-91 paused until next month.",
            "domain": "subscription",
            "task_id": "tau2bench_smoke_test_0002",
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing train/test parquet files if they already exist.",
    )
    return parser.parse_args()


def write_split(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    if not args.force and (train_path.exists() or test_path.exists()):
        raise SystemExit(
            f"Refusing to overwrite existing tau2bench smoke parquet under {output_dir}. Use --force to replace it."
        )

    write_split(train_path, TRAIN_ROWS)
    write_split(test_path, TEST_ROWS)
    metadata = {
        "dataset": "tau2bench",
        "format": "sft",
        "source": "tau2bench_smoke_bootstrap",
        "train_rows": len(TRAIN_ROWS),
        "test_rows": len(TEST_ROWS),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote tau2bench smoke parquet to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
