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

from datasets import concatenate_datasets, load_dataset


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
        "level_fields": ["level", "difficulty"],
        "type_fields": ["type", "subject", "category"],
    },
    "openr1_math": {
        "dataset_id": "open-r1/OpenR1-Math-220k",
        "config": "default",
        # OpenR1-Math-220k is train-only; keep a deterministic held-out slice.
        "train_split": "train[:-2000]",
        "test_split": "train[-2000:]",
        "question_fields": ["problem", "question", "prompt"],
        "solution_fields": ["solution", "response", "completion"],
        "answer_fields": ["answer", "final_answer"],
        "level_fields": ["difficulty", "level"],
        "type_fields": ["source", "domain", "category", "type"],
    },
}


def load_math_dataset():
    """Load MATH dataset from EleutherAI mirror, combining all subject configs."""
    train_parts, test_parts = [], []
    for config in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", config)
        train_parts.append(ds["train"])
        test_parts.append(ds["test"])
    return {
        "train": concatenate_datasets(train_parts),
        "test": concatenate_datasets(test_parts),
    }


def env_or_default(cli_value: str | None, env_name: str, default: str | None = None) -> str | None:
    """Use CLI value first, then env var, then fallback default."""
    if cli_value:
        return cli_value
    if env_name in os.environ and os.environ[env_name]:
        return os.environ[env_name]
    return default


def dataset_env_prefix(dataset_name: str) -> str:
    return dataset_name.upper()


def split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def first_present(example, candidates: list[str]) -> str:
    for field in candidates:
        if field in example and example[field] not in (None, ""):
            return str(example[field])
    return ""


def configurable_math_defaults(dataset_name: str) -> dict[str, object]:
    try:
        return CONFIGURABLE_MATH_DATASETS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown configurable math dataset: {dataset_name}") from exc


def load_configurable_math_dataset(
    dataset_name: str,
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
):
    """Load a configurable math dataset from Hugging Face."""
    defaults = configurable_math_defaults(dataset_name)
    env_prefix = dataset_env_prefix(dataset_name)

    dataset_id = env_or_default(dataset_id, f"{env_prefix}_DATASET_ID", defaults["dataset_id"])
    if not dataset_id:
        raise ValueError(
            f"{dataset_name} requires a dataset id. Set --hf_dataset or {env_prefix}_DATASET_ID."
        )

    config = env_or_default(config, f"{env_prefix}_DATASET_CONFIG", defaults["config"])
    train_split = env_or_default(train_split, f"{env_prefix}_TRAIN_SPLIT", defaults["train_split"])
    test_split = env_or_default(test_split, f"{env_prefix}_TEST_SPLIT", defaults["test_split"])

    load_kwargs = {}
    if config:
        load_kwargs["name"] = config

    return {
        "train": load_dataset(dataset_id, split=train_split, **load_kwargs),
        "test": load_dataset(dataset_id, split=test_split, **load_kwargs),
    }


def load_polaris_math_dataset(
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
):
    return load_configurable_math_dataset(
        "polaris_math",
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
    )


def load_openr1_math_dataset(
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
):
    return load_configurable_math_dataset(
        "openr1_math",
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
    )


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


def resolve_configurable_math_fields(
    dataset_name: str,
    example,
    question_field: str | None,
    solution_field: str | None,
    answer_field: str | None,
) -> tuple[str, str, str]:
    defaults = configurable_math_defaults(dataset_name)
    env_prefix = dataset_env_prefix(dataset_name)
    question = first_present(
        example,
        split_csv(
            env_or_default(question_field, f"{env_prefix}_QUESTION_FIELD"),
            defaults["question_fields"],
        ),
    )
    solution_text = first_present(
        example,
        split_csv(
            env_or_default(solution_field, f"{env_prefix}_SOLUTION_FIELD"),
            defaults["solution_fields"],
        ),
    )
    explicit_answer = first_present(
        example,
        split_csv(
            env_or_default(answer_field, f"{env_prefix}_ANSWER_FIELD"),
            defaults["answer_fields"],
        ),
    )
    return question, solution_text, explicit_answer


def resolve_polaris_fields(
    example,
    question_field: str | None,
    solution_field: str | None,
    answer_field: str | None,
) -> tuple[str, str, str]:
    return resolve_configurable_math_fields(
        "polaris_math",
        example,
        question_field,
        solution_field,
        answer_field,
    )


def ensure_boxed_solution(solution_text: str, explicit_answer: str) -> str:
    solution_text = solution_text.strip()
    explicit_answer = explicit_answer.strip()
    if not explicit_answer:
        return solution_text
    if not solution_text:
        return f"\\boxed{{{explicit_answer}}}"
    if extract_math_answer(solution_text):
        return solution_text
    return f"{solution_text}\n\\boxed{{{explicit_answer}}}"


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
    dataset = load_math_dataset()

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


def preprocess_configurable_math(
    dataset_name: str,
    output_dir: str,
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
    question_field: str | None = None,
    solution_field: str | None = None,
    answer_field: str | None = None,
    level_field: str | None = None,
    type_field: str | None = None,
):
    """Preprocess a configurable math dataset into VeRL GRPO format."""
    dataset = load_configurable_math_dataset(
        dataset_name,
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
    )

    defaults = configurable_math_defaults(dataset_name)
    env_prefix = dataset_env_prefix(dataset_name)
    level_candidates = split_csv(
        env_or_default(level_field, f"{env_prefix}_LEVEL_FIELD"),
        defaults["level_fields"],
    )
    type_candidates = split_csv(
        env_or_default(type_field, f"{env_prefix}_TYPE_FIELD"),
        defaults["type_fields"],
    )

    def make_map_fn(split: str):
        def process(example):
            problem, solution_text, explicit_answer = resolve_configurable_math_fields(
                dataset_name,
                example,
                question_field=question_field,
                solution_field=solution_field,
                answer_field=answer_field,
            )
            answer = extract_math_answer(solution_text) if solution_text else ""
            if not answer:
                answer = explicit_answer

            prompt = (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )

            return {
                "data_source": dataset_name,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "level": first_present(example, level_candidates),
                    "type": first_present(example, type_candidates),
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

    train_before = len(train_dataset)
    test_before = len(test_dataset)
    train_dataset = train_dataset.filter(lambda x: x["reward_model"]["ground_truth"] != "")
    test_dataset = test_dataset.filter(lambda x: x["reward_model"]["ground_truth"] != "")

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train: {len(train_dataset)} samples -> {train_path} (filtered {train_before - len(train_dataset)})")
    print(f"Test:  {len(test_dataset)} samples -> {test_path} (filtered {test_before - len(test_dataset)})")


def preprocess_polaris_math(
    output_dir: str,
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
    question_field: str | None = None,
    solution_field: str | None = None,
    answer_field: str | None = None,
    level_field: str | None = None,
    type_field: str | None = None,
):
    preprocess_configurable_math(
        "polaris_math",
        output_dir,
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
        question_field=question_field,
        solution_field=solution_field,
        answer_field=answer_field,
        level_field=level_field,
        type_field=type_field,
    )


def preprocess_openr1_math(
    output_dir: str,
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
    question_field: str | None = None,
    solution_field: str | None = None,
    answer_field: str | None = None,
    level_field: str | None = None,
    type_field: str | None = None,
):
    preprocess_configurable_math(
        "openr1_math",
        output_dir,
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
        question_field=question_field,
        solution_field=solution_field,
        answer_field=answer_field,
        level_field=level_field,
        type_field=type_field,
    )


def preprocess_configurable_math_sft(
    dataset_name: str,
    output_dir: str,
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
    question_field: str | None = None,
    solution_field: str | None = None,
    answer_field: str | None = None,
):
    """Preprocess a configurable math dataset into VeRL SFT format."""
    dataset = load_configurable_math_dataset(
        dataset_name,
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
    )

    def make_map_fn(split: str):
        def process(example):
            problem, solution_text, explicit_answer = resolve_configurable_math_fields(
                dataset_name,
                example,
                question_field=question_field,
                solution_field=solution_field,
                answer_field=answer_field,
            )
            solution_text = ensure_boxed_solution(solution_text, explicit_answer)
            prompt = (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            return {
                "data_source": dataset_name,
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

    train_before = len(train_dataset)
    test_before = len(test_dataset)
    train_dataset = train_dataset.filter(lambda x: extract_math_answer(x["extra_info"]["answer"]) != "")
    test_dataset = test_dataset.filter(lambda x: extract_math_answer(x["extra_info"]["answer"]) != "")

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"[SFT] Train: {len(train_dataset)} samples -> {train_path} (filtered {train_before - len(train_dataset)})")
    print(f"[SFT] Test:  {len(test_dataset)} samples -> {test_path} (filtered {test_before - len(test_dataset)})")


def preprocess_polaris_math_sft(
    output_dir: str,
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
    question_field: str | None = None,
    solution_field: str | None = None,
    answer_field: str | None = None,
):
    preprocess_configurable_math_sft(
        "polaris_math",
        output_dir,
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
        question_field=question_field,
        solution_field=solution_field,
        answer_field=answer_field,
    )


def preprocess_openr1_math_sft(
    output_dir: str,
    dataset_id: str | None = None,
    config: str | None = None,
    train_split: str | None = None,
    test_split: str | None = None,
    question_field: str | None = None,
    solution_field: str | None = None,
    answer_field: str | None = None,
):
    preprocess_configurable_math_sft(
        "openr1_math",
        output_dir,
        dataset_id=dataset_id,
        config=config,
        train_split=train_split,
        test_split=test_split,
        question_field=question_field,
        solution_field=solution_field,
        answer_field=answer_field,
    )


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
    dataset = load_math_dataset()

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
        choices=["gsm8k", "math", "triviaqa", "polaris_math", "openr1_math"],
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
    parser.add_argument("--hf_dataset", type=str, default=None, help="HF dataset id/path override (used for configurable math datasets).")
    parser.add_argument("--hf_config", type=str, default=None, help="HF dataset config override (used for configurable math datasets).")
    parser.add_argument("--train_split", type=str, default=None, help="Train split override (used for configurable math datasets).")
    parser.add_argument("--test_split", type=str, default=None, help="Test split override (used for configurable math datasets).")
    parser.add_argument("--question_field", type=str, default=None, help="Question/problem field name override (used for configurable math datasets).")
    parser.add_argument("--solution_field", type=str, default=None, help="Solution field name override (used for configurable math datasets).")
    parser.add_argument("--answer_field", type=str, default=None, help="Explicit answer field name override (used for configurable math datasets).")
    parser.add_argument("--level_field", type=str, default=None, help="Level/difficulty field name override (used for configurable math datasets).")
    parser.add_argument("--type_field", type=str, default=None, help="Type/subject field name override (used for configurable math datasets).")
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
        elif args.dataset == "polaris_math":
            preprocess_polaris_math_sft(
                args.output_dir,
                dataset_id=args.hf_dataset,
                config=args.hf_config,
                train_split=args.train_split,
                test_split=args.test_split,
                question_field=args.question_field,
                solution_field=args.solution_field,
                answer_field=args.answer_field,
            )
        elif args.dataset == "openr1_math":
            preprocess_openr1_math_sft(
                args.output_dir,
                dataset_id=args.hf_dataset,
                config=args.hf_config,
                train_split=args.train_split,
                test_split=args.test_split,
                question_field=args.question_field,
                solution_field=args.solution_field,
                answer_field=args.answer_field,
            )
    else:
        if args.dataset == "gsm8k":
            preprocess_gsm8k(args.output_dir)
        elif args.dataset == "math":
            preprocess_math(args.output_dir)
        elif args.dataset == "triviaqa":
            preprocess_triviaqa(args.output_dir)
        elif args.dataset == "polaris_math":
            preprocess_polaris_math(
                args.output_dir,
                dataset_id=args.hf_dataset,
                config=args.hf_config,
                train_split=args.train_split,
                test_split=args.test_split,
                question_field=args.question_field,
                solution_field=args.solution_field,
                answer_field=args.answer_field,
                level_field=args.level_field,
                type_field=args.type_field,
            )
        elif args.dataset == "openr1_math":
            preprocess_openr1_math(
                args.output_dir,
                dataset_id=args.hf_dataset,
                config=args.hf_config,
                train_split=args.train_split,
                test_split=args.test_split,
                question_field=args.question_field,
                solution_field=args.solution_field,
                answer_field=args.answer_field,
                level_field=args.level_field,
                type_field=args.type_field,
            )
