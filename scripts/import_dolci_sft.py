#!/usr/bin/env python3
"""Import or regenerate message-based SFT datasets into the repo parquet format.

This script supports two modes:

1. direct import of released assistant traces
2. fresh trace generation from released prompt messages with a local model via vLLM

It is intentionally separate from `generate_teacher_solutions.py` because
datasets like Dolci and SYNTHETIC-2-SFT-verified expose prompt/trace pairs but
not the full verifier payload needed for correctness-gated re-sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset


SUPPORTED_DATASETS = (
    "allenai/Dolci-Think-SFT-7B",
    "allenai/Dolci-Think-SFT-32B",
    "PrimeIntellect/SYNTHETIC-2-SFT-verified",
)


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def extract_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a non-empty messages list.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Expected the final message to be from the assistant.")
    return messages[:-1]


def serialize_prompt_messages(messages: list[dict[str, Any]]) -> str:
    prompt_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        prompt_parts.append(f"{role}: {content}")
    if not prompt_parts:
        raise ValueError("No prompt content found before the assistant answer.")
    return "\n\n".join(prompt_parts)


def extract_prompt_and_answer(messages: list[dict[str, Any]]) -> tuple[str, str]:
    prompt_messages = extract_prompt_messages(messages)
    prompt = serialize_prompt_messages(prompt_messages)
    answer = str(messages[-1].get("content", "")).strip()
    if not answer:
        raise ValueError("Assistant answer is empty.")

    return prompt, answer


def maybe_wrap_answer_tags(answer: str) -> str:
    think_close = "</think>"
    if think_close not in answer:
        return answer
    if "<answer>" in answer.lower():
        return answer

    prefix, suffix = answer.rsplit(think_close, maxsplit=1)
    final_answer = suffix.strip()
    if not final_answer:
        return answer
    return f"{prefix}{think_close}\n<answer>{final_answer}</answer>"


def build_sft_rows(
    dataset: Dataset,
    *,
    dataset_label: str,
    wrap_answer_tags: bool,
    source_filter: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in dataset:
        source = str(record.get("source", record.get("dataset_source", ""))).strip()
        if source_filter and source not in source_filter:
            continue
        messages = record.get("messages")
        try:
            prompt, answer = extract_prompt_and_answer(messages)
        except ValueError:
            continue
        if wrap_answer_tags:
            answer = maybe_wrap_answer_tags(answer)

        source_prompt_id = str(record.get("problem_id") or record.get("id") or "")
        task_type = str(record.get("task_type", "")).strip()
        reward = record.get("reward")

        rows.append(
            {
                "data_source": dataset_label,
                "extra_info": {
                    "question": prompt,
                    "answer": answer,
                    "split": "train",
                    "source": source,
                    "source_prompt_id": source_prompt_id,
                    "task_type": task_type,
                    "reward": reward,
                },
            }
        )
    return rows


def build_generated_rows(
    dataset: Dataset,
    *,
    dataset_label: str,
    model: str,
    wrap_answer_tags: bool,
    source_filter: set[str] | None,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    samples_per_prompt: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        n=samples_per_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    records = list(dataset)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        prompt_rows: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
        for record in chunk:
            source = str(record.get("source", record.get("dataset_source", ""))).strip()
            if source_filter and source not in source_filter:
                continue
            try:
                prompt_messages = extract_prompt_messages(record["messages"])
                prompt = serialize_prompt_messages(prompt_messages)
            except (KeyError, ValueError):
                continue
            prompt_rows.append((record, prompt, prompt_messages))

        if not prompt_rows:
            continue

        outputs = llm.chat([prompt_messages for _, _, prompt_messages in prompt_rows], sampling_params)
        for (record, prompt, _prompt_messages), output in zip(prompt_rows, outputs, strict=True):
            source = str(record.get("source", record.get("dataset_source", ""))).strip()
            source_prompt_id = str(record.get("problem_id") or record.get("id") or "")
            task_type = str(record.get("task_type", "")).strip()
            reward = record.get("reward")
            for sample_idx, candidate in enumerate(output.outputs):
                answer = candidate.text.strip()
                if not answer:
                    continue
                if wrap_answer_tags:
                    answer = maybe_wrap_answer_tags(answer)
                rows.append(
                    {
                        "data_source": dataset_label,
                        "extra_info": {
                            "question": prompt,
                            "answer": answer,
                            "split": "train",
                            "source": source,
                            "source_prompt_id": source_prompt_id,
                            "task_type": task_type,
                            "reward": reward,
                            "sample_idx": sample_idx,
                            "generator_model": model,
                        },
                    }
                )
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    *,
    test_fraction: float,
    max_test_samples: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        extra = row.get("extra_info", {})
        group_key = str(extra.get("source_prompt_id") or extra.get("question"))
        grouped.setdefault(group_key, []).append(row)

    group_keys = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    test_group_count = int(len(group_keys) * test_fraction)
    if max_test_samples is not None:
        capped_groups = 0
        capped_rows = 0
        for key in group_keys:
            next_size = len(grouped[key])
            if capped_rows + next_size > max_test_samples and capped_groups > 0:
                break
            capped_groups += 1
            capped_rows += next_size
        test_group_count = min(test_group_count, capped_groups)
    test_group_count = max(1, test_group_count) if len(group_keys) > 1 else 0

    test_keys = set(group_keys[:test_group_count])
    test_rows = []
    train_rows = []
    for key, group_rows in grouped.items():
        if key in test_keys:
            for row in group_rows:
                row["extra_info"]["split"] = "test"
                test_rows.append(row)
        else:
            train_rows.extend(group_rows)
    return train_rows, test_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dataset",
        default="allenai/Dolci-Think-SFT-7B",
        help="HF dataset id. Supported and tested for the Dolci Think SFT datasets.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--source-filter",
        action="append",
        default=None,
        help="Optional repeatable source filter matching the dataset's `source` field exactly.",
    )
    parser.add_argument(
        "--wrap-answer-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the text after </think> in <answer>...</answer> when missing.",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Optional model path or HF id. When set, generate fresh traces from Dolci prompts instead of importing released answers.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_filter = set(args.source_filter) if args.source_filter else None
    if args.hf_dataset not in SUPPORTED_DATASETS:
        print(f"Warning: continuing with unsupported dataset id {args.hf_dataset!r}.")

    dataset_label = (
        args.hf_dataset.split("/", 1)[1]
        .lower()
        .replace("-", "_")
    )

    dataset = load_dataset(args.hf_dataset, split=args.split)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    if args.generator_model:
        rows = build_generated_rows(
            dataset,
            dataset_label=dataset_label,
            model=args.generator_model,
            wrap_answer_tags=args.wrap_answer_tags,
            source_filter=source_filter,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
        )
    else:
        rows = build_sft_rows(
            dataset,
            dataset_label=dataset_label,
            wrap_answer_tags=args.wrap_answer_tags,
            source_filter=source_filter,
        )
    train_rows, test_rows = split_rows(
        rows,
        test_fraction=args.test_fraction,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    summary_path = output_dir / "summary.json"

    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(test_rows).to_parquet(test_path)

    summary = {
        "hf_dataset": args.hf_dataset,
        "split": args.split,
        "wrap_answer_tags": args.wrap_answer_tags,
        "generator_model": args.generator_model,
        "source_filter": sorted(source_filter) if source_filter else None,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "output_dir": str(output_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Saved train parquet to {train_path}")
    print(f"Saved test parquet to {test_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
