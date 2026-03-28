#!/usr/bin/env python3
"""Run native SuperGPQA evaluation with batched local vLLM generation."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


LETTER_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and predictions.")
    parser.add_argument(
        "--dataset-name",
        default="m-a-p/SuperGPQA",
        help="Hugging Face dataset id for SuperGPQA.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to evaluate (default: train).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Prompt batch size for vLLM generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens per completion.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Max model length passed to vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size passed to vLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of evaluated examples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of generations per prompt.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=512,
        help="Requested pass@k value. Must be <= --num-samples.",
    )
    parser.add_argument(
        "--samples-per-call",
        type=int,
        default=16,
        help="Number of samples to request per vLLM call while accumulating pass@k.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.max_model_len < 1:
        parser.error("--max-model-len must be >= 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in the range (0, 1]")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the range (0, 1]")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be >= 1")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.pass_k < 1:
        parser.error("--pass-k must be >= 1")
    if args.pass_k > args.num_samples:
        parser.error("--pass-k must be <= --num-samples")
    if args.samples_per_call < 1:
        parser.error("--samples-per-call must be >= 1")
    if args.samples_per_call > args.num_samples:
        parser.error("--samples-per-call must be <= --num-samples")
    return args


def try_import_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return tqdm


def load_dataset_rows(args: argparse.Namespace) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("SuperGPQA runner requires `datasets` in the active environment.") from exc

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    return [dict(row) for row in dataset]


def build_question_prompt(question: str, options: list[str]) -> str:
    letters = LETTER_CHOICES[: len(options)]
    option_lines = "\n".join(
        f"{letter}. {option}" for letter, option in zip(letters, options)
    )
    return (
        "Answer the following graduate-level multiple-choice question.\n"
        "Select the single best option and return only the option letter.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Final answer:"
    )


def build_prompt(tokenizer, question: str, options: list[str]) -> str:
    user_prompt = build_question_prompt(question, options)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer multiple-choice questions. "
                "Return only the single best option letter."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{messages[0]['content']}\n\n{messages[1]['content']}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_answer_letter(response: str, options: list[str]) -> str | None:
    valid_letters = LETTER_CHOICES[: len(options)]
    upper = response.upper()

    patterns = [
        rf"ANSWER(?: IS|:)?\s*\(?([{valid_letters}])\)?",
        rf"FINAL ANSWER(?: IS|:)?\s*\(?([{valid_letters}])\)?",
        rf"OPTION\s*\(?([{valid_letters}])\)?",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, upper)
        if matches:
            return matches[-1]

    compact_letters = re.sub(r"[^A-Z]", "", upper)
    if len(compact_letters) == 1 and compact_letters in valid_letters:
        return compact_letters

    standalone = re.findall(rf"\b([{valid_letters}])\b", upper)
    if standalone:
        return standalone[-1]

    normalized_response = normalize_text(response)
    for letter, option in zip(valid_letters, options):
        option_text = normalize_text(str(option))
        if not option_text:
            continue
        if normalized_response == option_text:
            return letter
        if option_text in normalized_response:
            return letter
    return None


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def macro_metric(records: list[dict], group_key: str, value_key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        group_value = record.get(group_key)
        if group_value is None:
            continue
        grouped[str(group_value)].append(float(record[value_key]))
    return safe_mean([safe_mean(scores) for scores in grouped.values()])


def difficulty_breakdown(records: list[dict], value_key: str, metric_label: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("difficulty", "unknown"))].append(float(record[value_key]))
    return {
        f"{difficulty}_{metric_label}": safe_mean(scores)
        for difficulty, scores in sorted(grouped.items())
    }


def evaluate(args: argparse.Namespace) -> int:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "SuperGPQA runner requires `transformers` and `vllm` in the active environment."
        ) from exc

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset_name}:{args.split} ...")
    rows = load_dataset_rows(args)
    print(f"Loaded {len(rows)} SuperGPQA examples")
    print(
        f"Running batched generation: batch_size={args.batch_size}, "
        f"num_samples={args.num_samples}, pass@k={args.pass_k}, "
        f"samples_per_call={args.samples_per_call}"
    )
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    tqdm = try_import_tqdm()
    iterator = range(0, len(rows), args.batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="SuperGPQA prompt batches", unit="batch")

    predictions_path = output_dir / "predictions.jsonl"
    records: list[dict] = []
    with predictions_path.open("w") as handle:
        for start in iterator:
            batch_rows = rows[start : start + args.batch_size]
            prompts = [
                build_prompt(tokenizer, str(row["question"]), [str(item) for item in row["options"]])
                for row in batch_rows
            ]
            batch_records: list[dict] = []
            for row in batch_rows:
                options = [str(item) for item in row["options"]]
                gold_letter = str(row.get("answer_letter", "")).strip().upper() or None
                if gold_letter is None and "answer" in row:
                    gold_answer = normalize_text(str(row["answer"]))
                    for letter, option in zip(LETTER_CHOICES[: len(options)], options):
                        if normalize_text(option) == gold_answer:
                            gold_letter = letter
                            break
                batch_records.append(
                    {
                        "uuid": row.get("uuid"),
                        "discipline": row.get("discipline"),
                        "field": row.get("field"),
                        "subfield": row.get("subfield"),
                        "difficulty": row.get("difficulty"),
                        "gold_letter": gold_letter,
                        "_options": options,
                        "_candidate_texts": [],
                        "_predicted_letters": [],
                    }
                )

            for sample_start in range(0, args.num_samples, args.samples_per_call):
                current_n = min(args.samples_per_call, args.num_samples - sample_start)
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    n=current_n,
                )
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                for record, output in zip(batch_records, outputs):
                    candidate_texts = [candidate.text for candidate in output.outputs]
                    predicted_letters = [
                        extract_answer_letter(candidate_text, record["_options"])
                        for candidate_text in candidate_texts
                    ]
                    record["_candidate_texts"].extend(candidate_texts)
                    record["_predicted_letters"].extend(predicted_letters)

            for record in batch_records:
                predicted_letters = record.pop("_predicted_letters")
                candidate_texts = record.pop("_candidate_texts")
                record.pop("_options")
                first_letter = predicted_letters[0] if predicted_letters else None
                gold_letter = record["gold_letter"]
                accuracy_at_1 = bool(first_letter and gold_letter and first_letter == gold_letter)
                pass_at_k = any(
                    predicted_letter and gold_letter and predicted_letter == gold_letter
                    for predicted_letter in predicted_letters[: args.pass_k]
                )
                correct_sample_indices = [
                    idx
                    for idx, predicted_letter in enumerate(predicted_letters[: args.pass_k])
                    if predicted_letter and gold_letter and predicted_letter == gold_letter
                ]
                record.update(
                    {
                        "predicted_letter_at_1": first_letter,
                        "predicted_letters_preview": predicted_letters[:16],
                        "num_parsed_predictions": sum(
                            1 for predicted_letter in predicted_letters if predicted_letter is not None
                        ),
                        "accuracy_at_1": accuracy_at_1,
                        "pass_at_k": pass_at_k,
                        "correct_sample_indices_preview": correct_sample_indices[:16],
                        "response_at_1": candidate_texts[0] if candidate_texts else "",
                        "response_samples": candidate_texts[: min(4, len(candidate_texts))],
                    }
                )
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    parsed_at_1 = sum(1 for record in records if record["predicted_letter_at_1"] is not None)
    parsed_any = sum(1 for record in records if record["num_parsed_predictions"] > 0)
    accuracy_at_1 = safe_mean([float(record["accuracy_at_1"]) for record in records])
    pass_at_k = safe_mean([float(record["pass_at_k"]) for record in records])

    metrics = {
        "results": {
            "all": {
                "accuracy_at_1": accuracy_at_1,
                f"pass@k:k={args.pass_k}": pass_at_k,
                "subfield_macro_accuracy_at_1": macro_metric(records, "subfield", "accuracy_at_1"),
                f"subfield_macro_pass@k:k={args.pass_k}": macro_metric(records, "subfield", "pass_at_k"),
                "field_macro_accuracy_at_1": macro_metric(records, "field", "accuracy_at_1"),
                f"field_macro_pass@k:k={args.pass_k}": macro_metric(records, "field", "pass_at_k"),
                "discipline_macro_accuracy_at_1": macro_metric(records, "discipline", "accuracy_at_1"),
                f"discipline_macro_pass@k:k={args.pass_k}": macro_metric(records, "discipline", "pass_at_k"),
                "parsed_rate_at_1": parsed_at_1 / len(records) if records else 0.0,
                f"parsed_rate_at_k:k={args.pass_k}": parsed_any / len(records) if records else 0.0,
            },
            "difficulty_accuracy_at_1": difficulty_breakdown(records, "accuracy_at_1", "accuracy_at_1"),
            "difficulty_pass_at_k": difficulty_breakdown(records, "pass_at_k", f"pass@k:k={args.pass_k}"),
        },
        "counts": {
            "num_examples": len(records),
            "num_parsed_at_1": parsed_at_1,
            "num_parsed_at_k": parsed_any,
            "num_unparsed_at_1": len(records) - parsed_at_1,
            "num_unparsed_at_k": len(records) - parsed_any,
        },
        "config": {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "model": args.model,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_samples": args.max_samples,
            "num_samples": args.num_samples,
            "pass_k": args.pass_k,
            "samples_per_call": args.samples_per_call,
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(
        "SuperGPQA Accuracy@1: "
        f"{accuracy_at_1:.4f} "
        f"({sum(int(record['accuracy_at_1']) for record in records)}/{len(records)})"
    )
    print(
        f"SuperGPQA Pass@{args.pass_k}: "
        f"{pass_at_k:.4f} "
        f"({sum(int(record['pass_at_k']) for record in records)}/{len(records)})"
    )
    print(
        "SuperGPQA parsed rate@1: "
        f"{metrics['results']['all']['parsed_rate_at_1']:.4f} "
        f"({parsed_at_1}/{len(records)})"
    )
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")

    del llm
    gc.collect()
    return 0


def main() -> int:
    args = parse_args()
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
