"""Generate teacher solutions using vLLM offline batch inference.

For each question in the training set, generates N candidate solutions from a
teacher model, verifies correctness using the math reward function, and saves
the first correct solution as the training target.

Supports resumption: processes questions in chunks and appends results to a
checkpoint file after each chunk. On restart, already-completed questions are
skipped automatically.

Usage:
    python src/data/generate_teacher_solutions.py \
        --model Qwen/Qwen3-32B \
        --dataset gsm8k \
        --n_samples 4 \
        --output_dir ~/scratch/forgetting-llms/data/gsm8k_sf_sft
"""

import argparse
import os
import re
import sys

import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Add repo root to path so we can import reward functions
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from src.rewards.unified_reward import compute_score


# EleutherAI mirror of hendrycks/competition_math (original was DMCA'd)
MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def load_questions(dataset_name: str) -> list[dict]:
    """Load questions and ground truth answers from a dataset."""
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        questions = []
        for i, example in enumerate(dataset):
            question = example["question"]
            raw_answer = example["answer"]
            match = re.search(r"####\s*(-?[\d,]+\.?\d*)", raw_answer)
            ground_truth = match.group(1).replace(",", "").strip() if match else ""

            prompt = (
                f"{question}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            questions.append({
                "idx": i,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "original_question": question,
            })
        return questions

    elif dataset_name == "math":
        from datasets import concatenate_datasets
        train_parts = []
        for config in MATH_CONFIGS:
            ds = load_dataset("EleutherAI/hendrycks_math", config)
            train_parts.append(ds["train"])
        dataset = concatenate_datasets(train_parts)

        questions = []
        idx = 0
        for example in dataset:
            problem = example["problem"]
            solution_text = example["solution"]
            pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
            matches = re.findall(pattern, solution_text)
            ground_truth = matches[-1].strip() if matches else ""
            if not ground_truth:
                continue

            prompt = (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            questions.append({
                "idx": idx,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "original_question": problem,
            })
            idx += 1
        return questions

    elif dataset_name == "triviaqa":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
        # Subsample to 7500 like in preprocess_data.py
        if len(dataset) > 7500:
            dataset = dataset.shuffle(seed=42).select(range(7500))

        questions = []
        for i, example in enumerate(dataset):
            question = example["question"]
            aliases = example["answer"]["aliases"]
            ground_truth = "|||".join(aliases) if aliases else example["answer"]["value"]

            prompt = (
                f"{question}\n\n"
                "Answer the question directly. Put your final answer after 'The answer is: '."
            )
            questions.append({
                "idx": i,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "original_question": question,
            })
        return questions

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_checkpoint(checkpoint_path: str) -> set[int]:
    """Load completed question indices from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return set()
    df = pd.read_parquet(checkpoint_path)
    completed = set(df["idx"].tolist())
    print(f"Resuming: {len(completed)} questions already completed")
    return completed


def append_to_checkpoint(results: list[dict], checkpoint_path: str):
    """Append new results to the checkpoint parquet file."""
    new_df = pd.DataFrame(results)
    if os.path.exists(checkpoint_path):
        existing_df = pd.read_parquet(checkpoint_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(checkpoint_path)


def generate_and_filter_chunk(
    llm: LLM,
    sampling_params: SamplingParams,
    questions: list[dict],
    dataset_name: str,
) -> list[dict]:
    """Generate solutions for a chunk of questions and filter correct ones."""
    conversations = [
        [{"role": "user", "content": q["prompt"]}] for q in questions
    ]

    outputs = llm.chat(conversations, sampling_params)

    results = []
    n_correct = 0
    for question, output in zip(questions, outputs):
        gt = question["ground_truth"]
        correct_solution = None

        for o in output.outputs:
            if compute_score(dataset_name, o.text, gt) > 0.5:
                correct_solution = o.text
                break

        if correct_solution is not None:
            n_correct += 1
            results.append({
                "idx": question["idx"],
                "data_source": dataset_name,
                "question": question["prompt"],
                "answer": correct_solution,
                "ground_truth": gt,
            })

    return results, n_correct


def finalize_output(checkpoint_path: str, output_dir: str, dataset_name: str):
    """Convert checkpoint to final SFT parquet format."""
    df = pd.read_parquet(checkpoint_path)

    # Build SFT format
    records = []
    for _, row in df.iterrows():
        records.append({
            "data_source": row["data_source"],
            "extra_info": {
                "question": row["question"],
                "answer": row["answer"],
                "split": "train",
            },
        })

    train_df = pd.DataFrame(records)
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    print(f"Saved {len(train_df)} training examples to {train_path}")

    # Create test split from original test set
    test_records = []
    if dataset_name == "gsm8k":
        test_dataset = load_dataset("openai/gsm8k", "main", split="test")
        for example in test_dataset:
            question = example["question"]
            raw_answer = example["answer"]
            match = re.search(r"####\s*(-?[\d,]+\.?\d*)", raw_answer)
            numeric_answer = match.group(1).replace(",", "").strip() if match else ""
            text = re.sub(r"\n?####\s*(-?[\d,]+\.?\d*)\s*$", "", raw_answer).strip()
            formatted_answer = f"{text}\n\\boxed{{{numeric_answer}}}"
            prompt = (
                f"{question}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            test_records.append({
                "data_source": dataset_name,
                "extra_info": {"question": prompt, "answer": formatted_answer, "split": "test"},
            })

    elif dataset_name == "math":
        from datasets import concatenate_datasets
        test_parts = []
        for config in MATH_CONFIGS:
            ds = load_dataset("EleutherAI/hendrycks_math", config)
            test_parts.append(ds["test"])
        test_dataset = concatenate_datasets(test_parts)
        for example in test_dataset:
            problem = example["problem"]
            solution_text = example["solution"]
            prompt = (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            test_records.append({
                "data_source": dataset_name,
                "extra_info": {"question": prompt, "answer": solution_text, "split": "test"},
            })

    elif dataset_name == "triviaqa":
        test_dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        if len(test_dataset) > 1500:
            test_dataset = test_dataset.shuffle(seed=42).select(range(1500))
        for example in test_dataset:
            question = example["question"]
            primary_answer = example["answer"]["value"]
            prompt = (
                f"{question}\n\n"
                "Answer the question directly. Put your final answer after 'The answer is: '."
            )
            test_records.append({
                "data_source": dataset_name,
                "extra_info": {"question": prompt, "answer": f"The answer is: {primary_answer}", "split": "test"},
            })

    if test_records:
        test_df = pd.DataFrame(test_records)
        test_path = os.path.join(output_dir, "test.parquet")
        test_df.to_parquet(test_path)
        print(f"Saved {len(test_df)} test examples to {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher solutions for SF-SFT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B",
                        help="Teacher model (HF ID or local path)")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math", "triviaqa"],
                        help="Dataset to generate solutions for")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of candidate solutions per question")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="vLLM tensor parallel size")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens per generated solution")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Number of questions per chunk (saves after each chunk)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ~/scratch/forgetting-llms/data/<dataset>_sf_sft)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.expanduser(
            f"~/scratch/forgetting-llms/data/{args.dataset}_sf_sft"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.parquet")

    # Load questions
    print(f"Loading {args.dataset} questions...")
    questions = load_questions(args.dataset)
    print(f"Loaded {len(questions)} questions")

    # Check for existing progress
    completed_idxs = load_checkpoint(checkpoint_path)
    remaining = [q for q in questions if q["idx"] not in completed_idxs]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("All questions already processed! Finalizing output...")
        finalize_output(checkpoint_path, args.output_dir, args.dataset)
        print("Done!")
        return

    # Initialize model
    print(f"Loading model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    # Process in chunks
    total_correct = len(completed_idxs)
    total_processed = len(completed_idxs)

    for chunk_start in range(0, len(remaining), args.chunk_size):
        chunk = remaining[chunk_start:chunk_start + args.chunk_size]
        chunk_num = chunk_start // args.chunk_size + 1
        n_chunks = (len(remaining) + args.chunk_size - 1) // args.chunk_size

        print(f"\n--- Chunk {chunk_num}/{n_chunks}: {len(chunk)} questions ---")

        results, n_correct = generate_and_filter_chunk(
            llm, sampling_params, chunk, args.dataset
        )

        # Save immediately
        if results:
            append_to_checkpoint(results, checkpoint_path)

        total_correct += n_correct
        total_processed += len(chunk)
        print(f"Chunk {chunk_num}: {n_correct}/{len(chunk)} correct "
              f"(running total: {total_correct}/{total_processed}, "
              f"{100*total_correct/total_processed:.1f}%)")

    # Finalize
    print(f"\nAll chunks complete. Total: {total_correct}/{total_processed} "
          f"({100*total_correct/total_processed:.1f}%)")
    finalize_output(checkpoint_path, args.output_dir, args.dataset)
    print("Done!")


if __name__ == "__main__":
    main()
