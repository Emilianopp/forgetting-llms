"""Preprocess datasets into VeRL parquet format.

Usage:
    python scripts/preprocess_data.py --dataset gsm8k --output_dir ~/scratch/forgetting-llms/data/gsm8k
    python scripts/preprocess_data.py --dataset math --output_dir ~/scratch/forgetting-llms/data/math
"""

import argparse
import os
import re

from datasets import load_dataset


def extract_gsm8k_answer(solution: str) -> str:
    """Extract the numerical answer from a GSM8K solution string (after ####)."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", solution)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def extract_math_answer(solution: str) -> str:
    """Extract the answer from a MATH solution string (inside \\boxed{}).

    MATH answers can be symbolic (fractions, expressions), not just numbers.
    """
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, solution)
    if matches:
        return matches[-1].strip()
    return ""


def preprocess_gsm8k(output_dir: str):
    """Preprocess GSM8K into VeRL parquet format."""
    dataset = load_dataset("openai/gsm8k", "main")

    def make_map_fn(split: str):
        def process(example):
            question = example["question"]
            answer = example["answer"]
            solution = extract_gsm8k_answer(answer)

            prompt = (
                f"{question}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )

            return {
                "data_source": "gsm8k",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "original_question": question,
                    "original_answer": answer,
                },
            }

        return process

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = dataset["train"].map(
        make_map_fn("train"), remove_columns=dataset["train"].column_names
    )
    test_dataset = dataset["test"].map(
        make_map_fn("test"), remove_columns=dataset["test"].column_names
    )

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train: {len(train_dataset)} samples -> {train_path}")
    print(f"Test:  {len(test_dataset)} samples -> {test_path}")


def preprocess_math(output_dir: str):
    """Preprocess MATH (hendrycks) into VeRL parquet format."""
    dataset = load_dataset("hendrycks/competition_math")

    def make_map_fn(split: str):
        def process(example):
            problem = example["problem"]
            solution_text = example["solution"]
            answer = extract_math_answer(solution_text)

            prompt = (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )

            return {
                "data_source": "math",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "level": example.get("level", ""),
                    "type": example.get("type", ""),
                    "original_problem": problem,
                    "original_solution": solution_text,
                },
            }

        return process

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = dataset["train"].map(
        make_map_fn("train"), remove_columns=dataset["train"].column_names
    )
    test_dataset = dataset["test"].map(
        make_map_fn("test"), remove_columns=dataset["test"].column_names
    )

    # Filter out examples where we couldn't extract an answer
    train_before = len(train_dataset)
    test_before = len(test_dataset)
    train_dataset = train_dataset.filter(
        lambda x: x["reward_model"]["ground_truth"] != ""
    )
    test_dataset = test_dataset.filter(
        lambda x: x["reward_model"]["ground_truth"] != ""
    )

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train: {len(train_dataset)} samples -> {train_path} (filtered {train_before - len(train_dataset)})")
    print(f"Test:  {len(test_dataset)} samples -> {test_path} (filtered {test_before - len(test_dataset)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math"],
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for parquet files (default: ~/scratch/forgetting-llms/data/<dataset>)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.expanduser(
            f"~/scratch/forgetting-llms/data/{args.dataset}"
        )

    if args.dataset == "gsm8k":
        preprocess_gsm8k(args.output_dir)
    elif args.dataset == "math":
        preprocess_math(args.output_dir)
