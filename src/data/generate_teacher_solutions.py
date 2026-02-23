"""Generate teacher solutions using vLLM offline batch inference.

For each question in the training set, generates N candidate solutions from a
teacher model, verifies correctness using the math reward function, and saves
the first correct solution as the training target.

Usage:
    python src/data/generate_teacher_solutions.py \
        --model Qwen/Qwen3-32B \
        --dataset gsm8k \
        --n_samples 4 \
        --output_dir ~/scratch/forgetting-llms/data/gsm8k_sf_sft
"""

import argparse
import os
import sys

import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Add repo root to path so we can import reward functions
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from src.rewards.math_reward import compute_score


def load_questions(dataset_name: str) -> list[dict]:
    """Load questions and ground truth answers from a dataset."""
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        questions = []
        for example in dataset:
            question = example["question"]
            raw_answer = example["answer"]
            # Extract numeric answer for verification
            import re
            match = re.search(r"####\s*(-?[\d,]+\.?\d*)", raw_answer)
            ground_truth = match.group(1).replace(",", "").strip() if match else ""

            prompt = (
                f"{question}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            questions.append({
                "prompt": prompt,
                "ground_truth": ground_truth,
                "original_question": question,
            })
        return questions

    elif dataset_name == "math":
        dataset = load_dataset("hendrycks/competition_math", split="train")
        questions = []
        for example in dataset:
            problem = example["problem"]
            solution_text = example["solution"]
            import re
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
                "prompt": prompt,
                "ground_truth": ground_truth,
                "original_question": problem,
            })
        return questions

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def generate_solutions(
    model_name: str,
    questions: list[dict],
    n_samples: int,
    tensor_parallel_size: int,
    max_tokens: int,
) -> list[list[str]]:
    """Generate N candidate solutions per question using vLLM."""
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    # Build conversations in chat format
    conversations = [
        [{"role": "user", "content": q["prompt"]}] for q in questions
    ]

    print(f"Generating {n_samples} solutions for {len(questions)} questions...")
    outputs = llm.chat(conversations, sampling_params)

    # Extract text from each output
    all_solutions = []
    for output in outputs:
        solutions = [o.text for o in output.outputs]
        all_solutions.append(solutions)

    return all_solutions


def filter_correct_solutions(
    questions: list[dict],
    all_solutions: list[list[str]],
    dataset_name: str,
) -> tuple[list[dict], int, int]:
    """Filter to keep only the first correct solution per question.

    Returns:
        results: List of dicts with question, answer, ground_truth for correct ones.
        n_with_correct: Number of questions with at least one correct solution.
        n_total: Total number of questions.
    """
    results = []
    n_with_correct = 0

    for question, solutions in zip(questions, all_solutions):
        gt = question["ground_truth"]
        correct_solution = None

        for solution in solutions:
            score = compute_score(dataset_name, solution, gt)
            if score > 0.5:
                correct_solution = solution
                break

        if correct_solution is not None:
            n_with_correct += 1
            results.append({
                "question": question["prompt"],
                "answer": correct_solution,
                "ground_truth": gt,
            })

    return results, n_with_correct, len(questions)


def save_as_sft_parquet(results: list[dict], output_dir: str, dataset_name: str):
    """Save results in VeRL SFT parquet format."""
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for r in results:
        records.append({
            "data_source": dataset_name,
            "extra_info": {
                "question": r["question"],
                "answer": r["answer"],
                "split": "train",
            },
        })

    df = pd.DataFrame(records)
    train_path = os.path.join(output_dir, "train.parquet")
    df.to_parquet(train_path)
    print(f"Saved {len(df)} training examples to {train_path}")

    # Also create a test split from the original test set (GT format for validation loss)
    if dataset_name == "gsm8k":
        test_dataset = load_dataset("openai/gsm8k", "main", split="test")
        test_records = []
        import re
        for example in test_dataset:
            question = example["question"]
            raw_answer = example["answer"]
            # Reformat answer with \boxed{}
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
                "extra_info": {
                    "question": prompt,
                    "answer": formatted_answer,
                    "split": "test",
                },
            })
        test_df = pd.DataFrame(test_records)
        test_path = os.path.join(output_dir, "test.parquet")
        test_df.to_parquet(test_path)
        print(f"Saved {len(test_df)} test examples to {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher solutions for SF-SFT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B",
                        help="Teacher model (HF ID or local path)")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"],
                        help="Dataset to generate solutions for")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of candidate solutions per question")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="vLLM tensor parallel size")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens per generated solution")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ~/scratch/forgetting-llms/data/<dataset>_sf_sft)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.expanduser(
            f"~/scratch/forgetting-llms/data/{args.dataset}_sf_sft"
        )

    # Load questions
    print(f"Loading {args.dataset} questions...")
    questions = load_questions(args.dataset)
    print(f"Loaded {len(questions)} questions")

    # Generate solutions
    all_solutions = generate_solutions(
        model_name=args.model,
        questions=questions,
        n_samples=args.n_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
    )

    # Filter correct solutions
    print("Verifying solutions...")
    results, n_correct, n_total = filter_correct_solutions(
        questions, all_solutions, args.dataset
    )
    print(f"Correct: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")

    # Save
    save_as_sft_parquet(results, args.output_dir, args.dataset)
    print("Done!")


if __name__ == "__main__":
    main()
