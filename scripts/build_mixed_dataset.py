#!/usr/bin/env python3
"""Build a mixed parquet dataset from existing dataset directories.

Supports two modes:

1. Legacy two-dataset mode:
   --dataset-a gsm8k --dataset-b triviaqa [--balance min|none]

2. Explicit weighted mode:
   --dataset-weight math=0.3 --dataset-weight gsm8k=0.7

In weighted mode, the builder downsamples each source split to the largest
mixture that satisfies the requested proportions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


SUPPORTED_DATASETS = (
    "gsm8k",
    "math",
    "triviaqa",
    "polaris_math",
    "openr1_math",
    "synthetic2_sft_verified",
    "dolci_think_sft_7b",
    "dolci_think_sft_32b",
    "tau2bench",
    "olmo_rl_zero_math",
    "olmo_rl_zero_code",
    "olmo_rl_zero_if",
    "olmo_rl_zero_general",
    "olmo_rl_zero_mix",
    "olmo_instruct_rl",
    "dolci_rl_zero_math",
    "dolci_rl_zero_code",
    "dolci_rl_zero_if",
    "dolci_rl_zero_general",
    "dolci_rl_zero_mix",
    "dolci_instruct_rl",
)


OLMO_STAGE_ALIASES = {
    "dolci_rl_zero_math": "olmo_rl_zero_math",
    "dolci_rl_zero_code": "olmo_rl_zero_code",
    "dolci_rl_zero_if": "olmo_rl_zero_if",
    "dolci_rl_zero_general": "olmo_rl_zero_general",
    "dolci_rl_zero_mix": "olmo_rl_zero_mix",
    "dolci_instruct_rl": "olmo_instruct_rl",
}


OLMO_STAGE_VARIANTS = {
    "olmo_rl_zero_math": "math",
    "olmo_rl_zero_code": "code",
    "olmo_rl_zero_if": "if",
    "olmo_rl_zero_general": "general",
    "olmo_rl_zero_mix": "mix",
    "olmo_instruct_rl": "instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", choices=SUPPORTED_DATASETS)
    parser.add_argument("--dataset-b", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--dataset-weight",
        action="append",
        default=[],
        help=(
            "Dataset weight assignment in the form dataset=weight. "
            "Repeat for each dataset you want in the mixture, e.g. "
            "--dataset-weight math=0.3 --dataset-weight gsm8k=0.7"
        ),
    )
    parser.add_argument("--data-root", default="~/scratch/forgetting-llms/data")
    parser.add_argument(
        "--source-suffix",
        default="",
        help=(
            "Optional suffix appended to each source dataset directory under data-root. "
            "Examples: _sft or _sf_sft."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on the final mixed train split row count.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap on the final mixed test split row count.",
    )
    parser.add_argument(
        "--balance",
        default="min",
        choices=("min", "none"),
        help=(
            "Legacy two-dataset behavior. Downsample each source split to the "
            "smaller dataset size or keep all rows. Ignored when --dataset-weight "
            "is used."
        ),
    )
    return parser.parse_args()


def canonical_dataset_name(dataset: str) -> str:
    return OLMO_STAGE_ALIASES.get(dataset, dataset)


def source_dir_name(dataset: str, *, source_suffix: str) -> str:
    canonical = canonical_dataset_name(dataset)
    variant = OLMO_STAGE_VARIANTS.get(canonical)
    if variant is None:
        return f"{canonical}{source_suffix}"
    if source_suffix:
        return f"{canonical}{source_suffix}"
    return f"olmo_rl_{variant}"


def load_split(data_root: Path, dataset: str, split: str, *, source_suffix: str) -> tuple[pd.DataFrame, Path]:
    path = data_root / source_dir_name(dataset, source_suffix=source_suffix) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split parquet: {path}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["source_dataset"] = dataset
    df["source_row_id"] = [
        hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        for record in df.to_dict(orient="records")
    ]
    return df, path


def parse_weight_specs(specs: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --dataset-weight {spec!r}. Expected dataset=weight.")
        dataset, raw_weight = spec.split("=", 1)
        dataset = dataset.strip()
        if dataset not in SUPPORTED_DATASETS:
            raise SystemExit(
                f"Unsupported dataset in --dataset-weight: {dataset!r}. "
                f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}"
            )
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise SystemExit(f"Invalid weight for dataset {dataset!r}: {raw_weight!r}") from exc
        if weight <= 0:
            raise SystemExit(f"Weight for dataset {dataset!r} must be > 0.")
        if dataset in weights:
            raise SystemExit(f"Duplicate --dataset-weight provided for dataset {dataset!r}.")
        weights[dataset] = weight
    if len(weights) < 2:
        raise SystemExit("Weighted mixes require at least two --dataset-weight entries.")
    return weights


def resolve_datasets(args: argparse.Namespace) -> tuple[list[str], dict[str, float] | None]:
    if args.dataset_weight:
        if args.dataset_a or args.dataset_b:
            raise SystemExit("Use either --dataset-a/--dataset-b or --dataset-weight, not both.")
        weights = parse_weight_specs(args.dataset_weight)
        return list(weights.keys()), weights

    if not args.dataset_a or not args.dataset_b:
        raise SystemExit("Legacy mode requires both --dataset-a and --dataset-b.")
    if args.dataset_a == args.dataset_b:
        raise SystemExit("dataset-a and dataset-b must differ")
    return [args.dataset_a, args.dataset_b], None


def equal_pair_sample(
    frames: dict[str, pd.DataFrame],
    *,
    seed: int,
    mode: str,
) -> dict[str, pd.DataFrame]:
    items = list(frames.items())
    if mode == "none":
        return {name: df.reset_index(drop=True) for name, df in items}
    target = min(len(df) for _, df in items)
    return {
        name: df.sample(n=target, random_state=seed + idx).reset_index(drop=True)
        for idx, (name, df) in enumerate(items)
    }


def weighted_sample(
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
    *,
    seed: int,
) -> dict[str, pd.DataFrame]:
    normalized = {
        name: weight / sum(weights.values())
        for name, weight in weights.items()
    }
    max_total = min(len(frames[name]) / normalized[name] for name in normalized)
    total = int(math.floor(max_total))
    if total < len(normalized):
        raise SystemExit(
            "Requested weights leave too few examples after downsampling. "
            "Reduce the skew or use larger source datasets."
        )

    raw_counts = {name: total * normalized[name] for name in normalized}
    counts = {name: int(math.floor(value)) for name, value in raw_counts.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        normalized,
        key=lambda name: (raw_counts[name] - counts[name], normalized[name], name),
        reverse=True,
    )
    while remainder > 0:
        advanced = False
        for name in order:
            if counts[name] >= len(frames[name]):
                continue
            counts[name] += 1
            remainder -= 1
            advanced = True
            if remainder == 0:
                break
        if not advanced:
            break

    sampled: dict[str, pd.DataFrame] = {}
    for idx, (name, df) in enumerate(frames.items()):
        sampled[name] = df.sample(n=counts[name], random_state=seed + idx).reset_index(drop=True)
    return sampled


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def maybe_cap_rows(df: pd.DataFrame, limit: int | None, *, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    datasets, explicit_weights = resolve_datasets(args)

    data_root = Path(args.data_root).expanduser()
    mix_name = "_".join(datasets)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else data_root / f"{mix_name}_mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "datasets": datasets,
        "seed": args.seed,
        "mode": "weighted" if explicit_weights is not None else "legacy_pair",
        "balance": args.balance if explicit_weights is None else None,
        "weights": explicit_weights,
        "splits": {},
    }

    for split in ("train", "test"):
        loaded = {
            dataset: load_split(data_root, dataset, split, source_suffix=args.source_suffix)
            for dataset in datasets
        }
        split_frames = {dataset: frame for dataset, (frame, _path) in loaded.items()}
        if explicit_weights is None:
            sampled = equal_pair_sample(split_frames, seed=args.seed, mode=args.balance)
            effective_weights = {
                name: len(sampled[name]) / sum(len(df) for df in sampled.values())
                for name in sampled
            }
        else:
            sampled = weighted_sample(split_frames, explicit_weights, seed=args.seed)
            effective_weights = {
                name: len(sampled[name]) / sum(len(df) for df in sampled.values())
                for name in sampled
            }

        mixed = pd.concat(list(sampled.values()), ignore_index=True)
        mixed = mixed.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        split_cap = args.max_train_rows if split == "train" else args.max_test_rows
        mixed = maybe_cap_rows(mixed, split_cap, seed=args.seed)
        write_parquet(mixed, output_dir / f"{split}.parquet")
        mixed[["source_dataset", "source_row_id"]].to_parquet(output_dir / f"{split}_selection.parquet")

        manifest["splits"][split] = {
            "source_paths": {
                dataset: str(path)
                for dataset, (_frame, path) in loaded.items()
            },
            "original_counts": {
                dataset: len(split_frames[dataset])
                for dataset in datasets
            },
            "selected_counts": {
                dataset: len(sampled[dataset])
                for dataset in datasets
            },
            "effective_shares": effective_weights,
            "total": len(mixed),
        }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote mixed dataset to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
