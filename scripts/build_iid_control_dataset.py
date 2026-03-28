#!/usr/bin/env python3
"""Build two IID shards from a balanced pooled distribution of two datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", required=True, choices=("gsm8k", "math", "triviaqa", "polaris_math", "openr1_math"))
    parser.add_argument("--dataset-b", required=True, choices=("gsm8k", "math", "triviaqa", "polaris_math", "openr1_math"))
    parser.add_argument("--data-root", default="~/scratch/forgetting-llms/data")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(data_root: Path, dataset: str, split: str) -> pd.DataFrame:
    path = data_root / dataset / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split parquet: {path}")
    df = pd.read_parquet(path).copy()
    df["source_dataset"] = dataset
    df["source_row_id"] = [
        hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        for record in df.to_dict(orient="records")
    ]
    return df


def split_in_half(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    midpoint = len(df) // 2
    return df.iloc[:midpoint].reset_index(drop=True), df.iloc[midpoint:].reset_index(drop=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def main() -> int:
    args = parse_args()
    if args.dataset_a == args.dataset_b:
        raise SystemExit("dataset-a and dataset-b must differ")

    data_root = Path(args.data_root).expanduser()
    pair_name = f"{args.dataset_a}_{args.dataset_b}"
    output_root = Path(args.output_root).expanduser() if args.output_root else data_root / f"{pair_name}_iid"
    shard1_dir = output_root / "shard1"
    shard2_dir = output_root / "shard2"
    shard1_dir.mkdir(parents=True, exist_ok=True)
    shard2_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "dataset_a": args.dataset_a,
        "dataset_b": args.dataset_b,
        "seed": args.seed,
        "splits": {},
    }

    for split in ("train", "test"):
        a_df = load_split(data_root, args.dataset_a, split).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        b_df = load_split(data_root, args.dataset_b, split).sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)
        target = min(len(a_df), len(b_df))
        a_df = a_df.iloc[:target].reset_index(drop=True)
        b_df = b_df.iloc[:target].reset_index(drop=True)
        a1, a2 = split_in_half(a_df)
        b1, b2 = split_in_half(b_df)
        shard1 = pd.concat([a1, b1], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        shard2 = pd.concat([a2, b2], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        write_parquet(shard1, shard1_dir / f"{split}.parquet")
        write_parquet(shard2, shard2_dir / f"{split}.parquet")
        shard1[["source_dataset", "source_row_id"]].to_parquet(shard1_dir / f"{split}_selection.parquet")
        shard2[["source_dataset", "source_row_id"]].to_parquet(shard2_dir / f"{split}_selection.parquet")
        manifest["splits"][split] = {
            "source_paths": {
                args.dataset_a: str(data_root / args.dataset_a / f"{split}.parquet"),
                args.dataset_b: str(data_root / args.dataset_b / f"{split}.parquet"),
            },
            "original_counts": {
                args.dataset_a: len(a_df),
                args.dataset_b: len(b_df),
            },
            "per_dataset": target,
            "shard1_total": len(shard1),
            "shard2_total": len(shard2),
        }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote IID control shards to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
