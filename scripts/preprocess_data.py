"""Preprocess datasets into VeRL parquet format.

Usage:
    python scripts/preprocess_data.py --dataset gsm8k --output_dir ~/scratch/forgetting-llms/data/gsm8k
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k"],
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/scratch/forgetting-llms/data/gsm8k"),
        help="Output directory for parquet files",
    )
    args = parser.parse_args()

    if args.dataset == "gsm8k":
        preprocess_gsm8k(args.output_dir)
