"""Generate positive-only teacher solutions for SFT using vLLM inference.

This script is designed for hard datasets where a fixed sample budget is often
insufficient. It adaptively samples multiple rounds per question until either:

1. enough unique correct solutions are found, or
2. the maximum sample budget for that question is exhausted.

Only correct solutions are kept. Final SFT output keeps an equal number of
solutions per retained question.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd
from datasets import load_dataset

# Add repo root to path so we can import reward functions
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from src.rewards.unified_reward import compute_score


# EleutherAI mirror of hendrycks/competition_math (original was DMCA'd)
MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

CONFIGURABLE_MATH_DATASETS = {
    "polaris_math": {
        "dataset_id": None,
        "config": None,
        "train_split": "train",
        "test_split": "test",
        "question_fields": ["problem", "question", "prompt"],
        "solution_fields": ["solution", "response", "completion"],
        "answer_fields": ["answer", "final_answer"],
    },
    "openr1_math": {
        "dataset_id": "open-r1/OpenR1-Math-220k",
        "config": "default",
        "train_split": "train[:-2000]",
        "test_split": "train[-2000:]",
        "question_fields": ["problem", "question", "prompt"],
        "solution_fields": ["solution", "response", "completion"],
        "answer_fields": ["answer", "final_answer"],
    },
}


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


# vLLM's standalone compile path has been brittle on Mila with the current
# torch/vllm stack. Default to the more stable eager path for trace generation.
os.environ.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")


def env_or_default(cli_value: str | None, env_name: str, default: str | None = None) -> str | None:
    if cli_value:
        return cli_value
    if env_name in os.environ and os.environ[env_name]:
        return os.environ[env_name]
    return default


def split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def dataset_env_prefix(dataset_name: str) -> str:
    return dataset_name.upper()


def first_present(example: dict, candidates: list[str]) -> str:
    for field in candidates:
        if field in example and example[field] not in (None, ""):
            return str(example[field])
    return ""


def extract_math_ground_truth(solution_text: str, explicit_answer: str = "") -> str:
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return explicit_answer.strip()


def ensure_boxed_solution(solution_text: str, explicit_answer: str = "") -> str:
    solution_text = solution_text.strip()
    explicit_answer = explicit_answer.strip()
    if not explicit_answer:
        return solution_text
    if extract_math_ground_truth(solution_text):
        return solution_text
    if not solution_text:
        return f"\\boxed{{{explicit_answer}}}"
    return f"{solution_text}\n\\boxed{{{explicit_answer}}}"


def prompt_instruction(dataset_name: str, answer_format: str) -> str:
    if answer_format == "tagged":
        if dataset_name in ("gsm8k", "math", "polaris_math", "openr1_math"):
            return (
                "Please reason step by step inside <think></think>, and put your final "
                "answer inside <answer></answer>."
            )
        return (
            "Think briefly inside <think></think>, and put your final answer inside "
            "<answer></answer>."
        )
    if dataset_name in ("gsm8k", "math", "polaris_math", "openr1_math"):
        return "Please reason step by step, and put your final answer within \\boxed{}."
    return "Answer the question directly. Put your final answer after 'The answer is: '."


def extract_boxed_fragment(text: str) -> str:
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def format_reference_answer(dataset_name: str, answer_format: str, answer_text: str) -> str:
    if answer_format == "tagged":
        if dataset_name in ("gsm8k", "math", "polaris_math", "openr1_math"):
            return f"<answer>{extract_boxed_fragment(answer_text)}</answer>"
        cleaned = re.sub(r"^[Tt]he answer is:\s*", "", answer_text).strip()
        return f"<answer>{cleaned}</answer>"
    return answer_text


def load_configurable_math_dataset(dataset_name: str) -> tuple:
    defaults = CONFIGURABLE_MATH_DATASETS[dataset_name]
    env_prefix = dataset_env_prefix(dataset_name)
    dataset_id = env_or_default(None, f"{env_prefix}_DATASET_ID", defaults["dataset_id"])
    if not dataset_id:
        raise ValueError(
            f"{dataset_name} requires {env_prefix}_DATASET_ID to be set."
        )
    config = env_or_default(None, f"{env_prefix}_DATASET_CONFIG", defaults["config"])
    train_split = env_or_default(None, f"{env_prefix}_TRAIN_SPLIT", defaults["train_split"])
    test_split = env_or_default(None, f"{env_prefix}_TEST_SPLIT", defaults["test_split"])

    load_kwargs = {}
    if config:
        load_kwargs["name"] = config
    train_dataset = load_dataset(dataset_id, split=train_split, **load_kwargs)
    test_dataset = load_dataset(dataset_id, split=test_split, **load_kwargs)
    return train_dataset, test_dataset


def load_polaris_math_dataset() -> tuple:
    return load_configurable_math_dataset("polaris_math")


def load_openr1_math_dataset() -> tuple:
    return load_configurable_math_dataset("openr1_math")


def resolve_configurable_math_fields(dataset_name: str, example: dict) -> tuple[str, str, str]:
    defaults = CONFIGURABLE_MATH_DATASETS[dataset_name]
    env_prefix = dataset_env_prefix(dataset_name)
    question = first_present(
        example,
        split_csv(os.environ.get(f"{env_prefix}_QUESTION_FIELD"), defaults["question_fields"]),
    )
    solution_text = first_present(
        example,
        split_csv(os.environ.get(f"{env_prefix}_SOLUTION_FIELD"), defaults["solution_fields"]),
    )
    explicit_answer = first_present(
        example,
        split_csv(os.environ.get(f"{env_prefix}_ANSWER_FIELD"), defaults["answer_fields"]),
    )
    return question, solution_text, explicit_answer


def resolve_polaris_fields(example: dict) -> tuple[str, str, str]:
    return resolve_configurable_math_fields("polaris_math", example)


def load_questions(dataset_name: str, answer_format: str = "dataset_default") -> list[dict]:
    """Load questions and ground truth answers from a dataset."""
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        questions = []
        for i, example in enumerate(dataset):
            question = example["question"]
            raw_answer = example["answer"]
            match = re.search(r"####\s*(-?[\d,]+\.?\d*)", raw_answer)
            ground_truth = match.group(1).replace(",", "").strip() if match else ""

            prompt = f"{question}\n\n{prompt_instruction(dataset_name, answer_format)}"
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

            prompt = f"{problem}\n\n{prompt_instruction(dataset_name, answer_format)}"
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

            prompt = f"{question}\n\n{prompt_instruction(dataset_name, answer_format)}"
            questions.append({
                "idx": i,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "original_question": question,
            })
        return questions

    elif dataset_name in CONFIGURABLE_MATH_DATASETS:
        train_dataset, _ = load_configurable_math_dataset(dataset_name)
        questions = []
        idx = 0
        for example in train_dataset:
            problem, solution_text, explicit_answer = resolve_configurable_math_fields(dataset_name, example)
            ground_truth = extract_math_ground_truth(solution_text, explicit_answer)
            if not problem or not ground_truth:
                continue

            prompt = f"{problem}\n\n{prompt_instruction(dataset_name, answer_format)}"
            questions.append({
                "idx": idx,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "original_question": problem,
            })
            idx += 1
        return questions

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def normalize_solution_text(text: str) -> str:
    """Normalize solution text for deduplication."""
    return re.sub(r"\s+", " ", text.strip())


def load_positive_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    """Load positive examples accumulated so far."""
    if not os.path.exists(checkpoint_path):
        return pd.DataFrame()
    return pd.read_parquet(checkpoint_path)


def load_status(status_path: str) -> pd.DataFrame:
    """Load per-question generation status."""
    if not os.path.exists(status_path):
        return pd.DataFrame()
    return pd.read_parquet(status_path)


def append_to_checkpoint(results: list[dict], checkpoint_path: str):
    """Append positive examples to the checkpoint parquet file, deduplicated."""
    new_df = pd.DataFrame(results)
    if new_df.empty:
        return
    if os.path.exists(checkpoint_path):
        existing_df = pd.read_parquet(checkpoint_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["idx", "normalized_answer"], keep="first")
    combined.to_parquet(checkpoint_path)


def upsert_status(status_rows: list[dict], status_path: str):
    """Persist the latest per-question status rows."""
    new_df = pd.DataFrame(status_rows)
    if new_df.empty:
        return
    if os.path.exists(status_path):
        existing_df = pd.read_parquet(status_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values(["idx", "attempt_round"]).drop_duplicates(subset=["idx"], keep="last")
    combined.to_parquet(status_path)


def build_existing_positive_index(df: pd.DataFrame) -> dict[int, set[str]]:
    """Build idx -> normalized correct answer set from a checkpoint dataframe."""
    if df.empty:
        return {}
    index: dict[int, set[str]] = {}
    for _, row in df.iterrows():
        index.setdefault(int(row["idx"]), set()).add(str(row["normalized_answer"]))
    return index


def build_status_index(df: pd.DataFrame) -> dict[int, dict]:
    """Build idx -> latest status row."""
    if df.empty:
        return {}
    return {int(row["idx"]): row.to_dict() for _, row in df.iterrows()}


def generate_round(
    llm: LLM,
    sampling_params: SamplingParams,
    questions: list[dict],
    dataset_name: str,
) -> dict[int, list[str]]:
    """Generate one sampling round and return unique correct solutions by idx."""
    conversations = [
        [{"role": "user", "content": q["prompt"]}] for q in questions
    ]

    outputs = llm.chat(conversations, sampling_params)

    by_idx: dict[int, list[str]] = {}
    for question, output in zip(questions, outputs):
        gt = question["ground_truth"]
        for o in output.outputs:
            if compute_score(dataset_name, o.text, gt) > 0.5:
                by_idx.setdefault(question["idx"], []).append(o.text)
    return by_idx


def generate_positive_examples_for_chunk(
    llm: LLM,
    questions: list[dict],
    dataset_name: str,
    samples_per_round: int,
    max_total_samples: int,
    target_correct_per_question: int,
    min_correct_per_question: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    existing_positive_index: dict[int, set[str]],
    status_index: dict[int, dict],
) -> tuple[list[dict], list[dict], dict]:
    """Adaptively sample until questions are solved or exhausted."""
    from vllm import SamplingParams

    new_positive_rows: list[dict] = []
    latest_status_rows: list[dict] = []

    active_questions = []
    for question in questions:
        prior = status_index.get(question["idx"], {})
        attempts_so_far = int(prior.get("attempts_used", 0))
        n_correct_so_far = int(prior.get("n_correct", len(existing_positive_index.get(question["idx"], set()))))
        complete = bool(prior.get("complete", False))
        if complete:
            latest_status_rows.append(prior)
            continue
        if attempts_so_far >= max_total_samples or n_correct_so_far >= target_correct_per_question:
            latest_status_rows.append({
                "idx": question["idx"],
                "attempts_used": attempts_so_far,
                "n_correct": n_correct_so_far,
                "complete": True,
                "kept": n_correct_so_far >= min_correct_per_question,
                "attempt_round": int(prior.get("attempt_round", 0)),
            })
            continue
        active_questions.append(question)

    round_num = 0
    while active_questions:
        round_num += 1
        sampling_params = SamplingParams(
            n=samples_per_round,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        correct_by_idx = generate_round(llm, sampling_params, active_questions, dataset_name)

        next_active: list[dict] = []
        for question in active_questions:
            idx = question["idx"]
            previous = status_index.get(idx, {})
            normalized_existing = existing_positive_index.setdefault(idx, set())
            attempts_used = int(previous.get("attempts_used", 0)) + samples_per_round

            for answer_text in correct_by_idx.get(idx, []):
                normalized = normalize_solution_text(answer_text)
                if normalized in normalized_existing:
                    continue
                normalized_existing.add(normalized)
                new_positive_rows.append({
                    "idx": idx,
                    "data_source": dataset_name,
                    "question": question["prompt"],
                    "answer": answer_text,
                    "normalized_answer": normalized,
                    "ground_truth": question["ground_truth"],
                    "attempt_round": round_num,
                    "attempts_used": attempts_used,
                })

            n_correct = len(normalized_existing)
            complete = n_correct >= target_correct_per_question or attempts_used >= max_total_samples
            status_row = {
                "idx": idx,
                "attempts_used": attempts_used,
                "n_correct": n_correct,
                "complete": complete,
                "kept": n_correct >= min_correct_per_question,
                "attempt_round": round_num,
            }
            status_index[idx] = status_row
            latest_status_rows.append(status_row)
            if not complete:
                next_active.append(question)
        active_questions = next_active

    summary = {
        "new_positive_rows": len(new_positive_rows),
        "questions_processed": len(questions),
        "questions_kept": sum(1 for row in latest_status_rows if row.get("kept")),
        "questions_completed": sum(1 for row in latest_status_rows if row.get("complete")),
    }
    return new_positive_rows, latest_status_rows, summary


def finalize_output(
    checkpoint_path: str,
    status_path: str,
    output_dir: str,
    dataset_name: str,
    solutions_per_question: int,
    min_correct_per_question: int,
    answer_format: str,
):
    """Convert checkpoint to final SFT parquet format."""
    positives_df = pd.read_parquet(checkpoint_path) if os.path.exists(checkpoint_path) else pd.DataFrame()
    status_df = pd.read_parquet(status_path) if os.path.exists(status_path) else pd.DataFrame()
    if positives_df.empty:
        raise RuntimeError("No positive examples were generated; cannot finalize SFT output.")

    required_correct = max(solutions_per_question, min_correct_per_question)
    eligible_ids = set()
    if not status_df.empty:
        eligible_ids = {
            int(row["idx"])
            for _, row in status_df.iterrows()
            if int(row.get("n_correct", 0)) >= required_correct
        }
    else:
        counts = positives_df.groupby("idx")["normalized_answer"].nunique()
        eligible_ids = set(counts[counts >= required_correct].index.tolist())

    filtered = positives_df[positives_df["idx"].isin(eligible_ids)].copy()
    filtered = filtered.sort_values(["idx", "attempt_round", "answer"]).reset_index(drop=True)
    filtered = filtered.groupby("idx", group_keys=False).head(solutions_per_question)

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
    if not status_df.empty:
        retained_questions = retained_questions.merge(
            status_df[["idx", "attempts_used", "n_correct", "attempt_round"]].drop_duplicates("idx", keep="last"),
            on="idx",
            how="left",
        )
    retained_questions_path = os.path.join(output_dir, "retained_questions.parquet")
    retained_questions.to_parquet(retained_questions_path)
    with open(os.path.join(output_dir, "retained_question_ids.json"), "w") as handle:
        json.dump(retained_questions["idx"].tolist(), handle, indent=2)
        handle.write("\n")

    # Build SFT format
    records = []
    for _, row in filtered.iterrows():
        records.append({
            "data_source": row["data_source"],
            "extra_info": {
                "question": row["question"],
                "answer": row["answer"],
                "source_question_id": int(row["idx"]),
                "ground_truth": row["ground_truth"],
                "split": "train",
            },
        })

    train_df = pd.DataFrame(records)
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    print(f"Saved {len(train_df)} training examples to {train_path}")
    summary = {
        "dataset": dataset_name,
        "answer_format": answer_format,
        "questions_with_positive_examples": int(positives_df["idx"].nunique()),
        "questions_kept": int(filtered["idx"].nunique()),
        "required_correct_per_question": required_correct,
        "solutions_per_question": solutions_per_question,
        "min_correct_per_question": min_correct_per_question,
        "total_positive_examples": int(len(positives_df)),
        "final_train_examples": int(len(filtered)),
        "retained_questions_path": retained_questions_path,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

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
            prompt = f"{question}\n\n{prompt_instruction(dataset_name, answer_format)}"
            test_records.append({
                "data_source": dataset_name,
                "extra_info": {
                    "question": prompt,
                    "answer": format_reference_answer(dataset_name, answer_format, formatted_answer),
                    "split": "test",
                },
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
            prompt = f"{problem}\n\n{prompt_instruction(dataset_name, answer_format)}"
            test_records.append({
                "data_source": dataset_name,
                "extra_info": {
                    "question": prompt,
                    "answer": format_reference_answer(dataset_name, answer_format, solution_text),
                    "split": "test",
                },
            })

    elif dataset_name == "triviaqa":
        test_dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        if len(test_dataset) > 1500:
            test_dataset = test_dataset.shuffle(seed=42).select(range(1500))
        for example in test_dataset:
            question = example["question"]
            primary_answer = example["answer"]["value"]
            prompt = f"{question}\n\n{prompt_instruction(dataset_name, answer_format)}"
            test_records.append({
                "data_source": dataset_name,
                "extra_info": {
                    "question": prompt,
                    "answer": format_reference_answer(dataset_name, answer_format, f"The answer is: {primary_answer}"),
                    "split": "test",
                },
            })

    elif dataset_name in CONFIGURABLE_MATH_DATASETS:
        _, test_dataset = load_configurable_math_dataset(dataset_name)
        for example in test_dataset:
            problem, solution_text, explicit_answer = resolve_configurable_math_fields(dataset_name, example)
            if not problem:
                continue
            solution_text = ensure_boxed_solution(solution_text, explicit_answer)
            if not solution_text:
                continue
            prompt = f"{problem}\n\n{prompt_instruction(dataset_name, answer_format)}"
            test_records.append({
                "data_source": dataset_name,
                "extra_info": {
                    "question": prompt,
                    "answer": format_reference_answer(dataset_name, answer_format, solution_text),
                    "split": "test",
                },
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
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math", "triviaqa", "polaris_math", "openr1_math"],
                        help="Dataset to generate solutions for")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Deprecated alias for --samples_per_round")
    parser.add_argument("--samples_per_round", type=int, default=None,
                        help="Number of candidate solutions per question per sampling round")
    parser.add_argument("--max_total_samples", type=int, default=16,
                        help="Maximum total samples per question across all rounds")
    parser.add_argument("--target_correct_per_question", type=int, default=2,
                        help="Stop sampling once this many unique correct solutions are found")
    parser.add_argument("--min_correct_per_question", type=int, default=2,
                        help="Keep only questions with at least this many unique correct solutions")
    parser.add_argument("--solutions_per_question", type=int, default=2,
                        help="Keep exactly this many positive solutions per retained question")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="vLLM tensor parallel size")
    parser.add_argument("--answer_format", type=str, default="dataset_default",
                        choices=["dataset_default", "tagged"],
                        help="Output format to request from the teacher")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum model context length passed to vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="vLLM GPU memory utilization fraction")
    parser.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction,
                        default=env_flag("VLLM_ENFORCE_EAGER", True),
                        help="Run vLLM in eager mode to avoid torch compile instability")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens per generated solution")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Number of questions per chunk (saves after each chunk)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Sampling top-p")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ~/scratch/forgetting-llms/data/<dataset>_sf_sft)")
    args = parser.parse_args()
    if args.samples_per_round is None:
        args.samples_per_round = args.n_samples

    if args.output_dir is None:
        args.output_dir = os.path.expanduser(
            f"~/scratch/forgetting-llms/data/{args.dataset}_sf_sft"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.parquet")
    status_path = os.path.join(args.output_dir, "status.parquet")

    # Load questions
    print(f"Loading {args.dataset} questions...")
    questions = load_questions(args.dataset, answer_format=args.answer_format)
    print(f"Loaded {len(questions)} questions")

    # Check for existing progress
    positive_df = load_positive_checkpoint(checkpoint_path)
    status_df = load_status(status_path)
    existing_positive_index = build_existing_positive_index(positive_df)
    status_index = build_status_index(status_df)
    completed_idxs = {idx for idx, row in status_index.items() if bool(row.get("complete", False))}
    remaining = [q for q in questions if q["idx"] not in completed_idxs]
    print(f"Completed questions: {len(completed_idxs)}")
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("All questions already processed! Finalizing output...")
        finalize_output(
            checkpoint_path,
            status_path,
            args.output_dir,
            args.dataset,
            solutions_per_question=args.solutions_per_question,
            min_correct_per_question=args.min_correct_per_question,
            answer_format=args.answer_format,
        )
        print("Done!")
        return

    # Initialize model
    from vllm import LLM

    print(f"Loading model {args.model}...")
    print(
        "vLLM settings: "
        f"enforce_eager={args.enforce_eager}, "
        f"VLLM_USE_STANDALONE_COMPILE={os.environ.get('VLLM_USE_STANDALONE_COMPILE')}, "
        f"VLLM_DISABLE_COMPILE_CACHE={os.environ.get('VLLM_DISABLE_COMPILE_CACHE')}"
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )

    # Process in chunks
    total_positive = len(positive_df)
    total_processed = len(completed_idxs)

    for chunk_start in range(0, len(remaining), args.chunk_size):
        chunk = remaining[chunk_start:chunk_start + args.chunk_size]
        chunk_num = chunk_start // args.chunk_size + 1
        n_chunks = (len(remaining) + args.chunk_size - 1) // args.chunk_size

        print(f"\n--- Chunk {chunk_num}/{n_chunks}: {len(chunk)} questions ---")

        new_positive_rows, status_rows, summary = generate_positive_examples_for_chunk(
            llm=llm,
            questions=chunk,
            dataset_name=args.dataset,
            samples_per_round=args.samples_per_round,
            max_total_samples=args.max_total_samples,
            target_correct_per_question=args.target_correct_per_question,
            min_correct_per_question=args.min_correct_per_question,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            existing_positive_index=existing_positive_index,
            status_index=status_index,
        )

        # Save immediately
        append_to_checkpoint(new_positive_rows, checkpoint_path)
        upsert_status(status_rows, status_path)

        total_positive += len(new_positive_rows)
        total_processed += len(chunk)
        print(
            f"Chunk {chunk_num}: +{len(new_positive_rows)} new positive solutions, "
            f"{summary['questions_kept']}/{summary['questions_processed']} questions currently keepable "
            f"(running total positives: {total_positive})"
        )

    # Finalize
    print(f"\nAll chunks complete. Processed {total_processed} questions, "
          f"collected {total_positive} positive solutions.")
    finalize_output(
        checkpoint_path,
        status_path,
        args.output_dir,
        args.dataset,
        solutions_per_question=args.solutions_per_question,
        min_correct_per_question=args.min_correct_per_question,
        answer_format=args.answer_format,
    )
    print("Done!")


if __name__ == "__main__":
    main()
