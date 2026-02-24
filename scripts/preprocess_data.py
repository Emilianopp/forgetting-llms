"""Preprocess datasets into VeRL parquet format.

Usage:
    # GRPO format (default)
    python scripts/preprocess_data.py --dataset gsm8k --output_dir ~/scratch/forgetting-llms/data/gsm8k
    python scripts/preprocess_data.py --dataset math --output_dir ~/scratch/forgetting-llms/data/math

    # SFT format (for GT-SFT training)
    python scripts/preprocess_data.py --dataset gsm8k --format sft --output_dir ~/scratch/forgetting-llms/data/gsm8k_sft
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


def format_gsm8k_solution_with_boxed(solution: str) -> str:
    """Reformat GSM8K solution to end with \\boxed{N}.

    GSM8K solutions look like: "reasoning...\n#### 36"
    We keep the reasoning and append \\boxed{36} so the model learns
    the same output format as GRPO.
    """
    answer = extract_gsm8k_answer(solution)
    # Remove the #### line and append \boxed{}
    text = re.sub(r"\n?####\s*(-?[\d,]+\.?\d*)\s*$", "", solution).strip()
    return f"{text}\n\\boxed{{{answer}}}"


def preprocess_gsm8k_sft(output_dir: str):
    """Preprocess GSM8K into VeRL SFT parquet format.

    SFT format uses extra_info with 'question' and 'answer' keys,
    referenced by VeRL's fsdp_sft_trainer via prompt_dict_keys/response_dict_keys.
    """
    dataset = load_dataset("openai/gsm8k", "main")

    def make_map_fn(split: str):
        def process(example):
            question = example["question"]
            raw_answer = example["answer"]

            prompt = (
                f"{question}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            solution = format_gsm8k_solution_with_boxed(raw_answer)

            return {
                "data_source": "gsm8k",
                "extra_info": {
                    "question": prompt,
                    "answer": solution,
                    "split": split,
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

    print(f"[SFT] Train: {len(train_dataset)} samples -> {train_path}")
    print(f"[SFT] Test:  {len(test_dataset)} samples -> {test_path}")


def preprocess_math_sft(output_dir: str):
    """Preprocess MATH (hendrycks) into VeRL SFT parquet format."""
    dataset = load_dataset("hendrycks/competition_math")

    def make_map_fn(split: str):
        def process(example):
            problem = example["problem"]
            solution_text = example["solution"]

            prompt = (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )

            return {
                "data_source": "math",
                "extra_info": {
                    "question": prompt,
                    "answer": solution_text,
                    "split": split,
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
    answer_exists = lambda x: extract_math_answer(x["extra_info"]["answer"]) != ""
    train_before = len(train_dataset)
    test_before = len(test_dataset)
    train_dataset = train_dataset.filter(answer_exists)
    test_dataset = test_dataset.filter(answer_exists)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"[SFT] Train: {len(train_dataset)} samples -> {train_path} (filtered {train_before - len(train_dataset)})")
    print(f"[SFT] Test:  {len(test_dataset)} samples -> {test_path} (filtered {test_before - len(test_dataset)})")


def preprocess_triviaqa(output_dir: str, max_train: int = 7500, max_test: int = 1500):
    """Preprocess TriviaQA (closed-book, no context) into VeRL parquet format."""
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    def make_map_fn(split: str):
        def process(example):
            question = example["question"]
            aliases = example["answer"]["aliases"]
            # Join all aliases with ||| for multi-match in reward
            ground_truth = "|||".join(aliases) if aliases else example["answer"]["value"]

            prompt = (
                f"{question}\n\n"
                "Answer the question directly. Put your final answer after 'The answer is: '."
            )

            return {
                "data_source": "triviaqa",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "original_question": question,
                },
            }

        return process

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = dataset["train"].map(
        make_map_fn("train"), remove_columns=dataset["train"].column_names
    )
    test_dataset = dataset["validation"].map(
        make_map_fn("test"), remove_columns=dataset["validation"].column_names
    )

    # Subsample to keep sizes manageable
    if len(train_dataset) > max_train:
        train_dataset = train_dataset.shuffle(seed=42).select(range(max_train))
    if len(test_dataset) > max_test:
        test_dataset = test_dataset.shuffle(seed=42).select(range(max_test))

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train: {len(train_dataset)} samples -> {train_path}")
    print(f"Test:  {len(test_dataset)} samples -> {test_path}")


def preprocess_triviaqa_sft(output_dir: str, max_train: int = 7500, max_test: int = 1500):
    """Preprocess TriviaQA (closed-book) into VeRL SFT parquet format."""
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    def make_map_fn(split: str):
        def process(example):
            question = example["question"]
            primary_answer = example["answer"]["value"]

            prompt = (
                f"{question}\n\n"
                "Answer the question directly. Put your final answer after 'The answer is: '."
            )
            answer = f"The answer is: {primary_answer}"

            return {
                "data_source": "triviaqa",
                "extra_info": {
                    "question": prompt,
                    "answer": answer,
                    "split": split,
                },
            }

        return process

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = dataset["train"].map(
        make_map_fn("train"), remove_columns=dataset["train"].column_names
    )
    test_dataset = dataset["validation"].map(
        make_map_fn("test"), remove_columns=dataset["validation"].column_names
    )

    # Subsample
    if len(train_dataset) > max_train:
        train_dataset = train_dataset.shuffle(seed=42).select(range(max_train))
    if len(test_dataset) > max_test:
        test_dataset = test_dataset.shuffle(seed=42).select(range(max_test))

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"[SFT] Train: {len(train_dataset)} samples -> {train_path}")
    print(f"[SFT] Test:  {len(test_dataset)} samples -> {test_path}")


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
        choices=["gsm8k", "math", "triviaqa"],
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="grpo",
        choices=["grpo", "sft"],
        help="Output format: grpo (reward model fields) or sft (question/answer in extra_info)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for parquet files (default: ~/scratch/forgetting-llms/data/<dataset>[_sft])",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "_sft" if args.format == "sft" else ""
        args.output_dir = os.path.expanduser(
            f"~/scratch/forgetting-llms/data/{args.dataset}{suffix}"
        )

    if args.format == "sft":
        if args.dataset == "gsm8k":
            preprocess_gsm8k_sft(args.output_dir)
        elif args.dataset == "math":
            preprocess_math_sft(args.output_dir)
        elif args.dataset == "triviaqa":
            preprocess_triviaqa_sft(args.output_dir)
    else:
        if args.dataset == "gsm8k":
            preprocess_gsm8k(args.output_dir)
        elif args.dataset == "math":
            preprocess_math(args.output_dir)
        elif args.dataset == "triviaqa":
            preprocess_triviaqa(args.output_dir)
