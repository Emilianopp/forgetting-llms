#!/usr/bin/env python3
"""Self-distillation trainer with privileged-answer teacher conditioning.

By default this trainer does not consume an existing SFT trace as the target.
Instead, for each question it:

1. samples one student trace from the unprivileged student
2. samples one privileged teacher trace that must grade as correct
3. trains the student against the teacher trace with CE
4. applies forward/reverse/interpolated KL on generated supports

Forward KL is computed on the teacher-generated support.
Reverse KL is computed on the student-generated support.

An escape hatch remains for legacy "reuse the existing parquet trace" training
via ``--trace-source existing``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rewards.unified_reward import compute_score

MISSING_IMPORT: ImportError | None = None

try:  # pragma: no cover - import availability depends on the runtime env.
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_scheduler,
    )
except ImportError as exc:  # pragma: no cover
    MISSING_IMPORT = exc
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Accelerator = None  # type: ignore[assignment]
    AdamW = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    get_scheduler = None  # type: ignore[assignment]

try:  # pragma: no cover - peft availability depends on the runtime env.
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - peft is expected in the cluster env.
    LoraConfig = None
    TaskType = None
    get_peft_model = None


DEFAULT_TEACHER_TEMPLATE = (
    "You are a privileged teacher for self-distillation. "
    "The correct final answer for the current problem is:\n"
    "{correct_answer}\n\n"
    "Use this hidden information to produce a correct reasoning trace that "
    "ends with the correct final answer inside <answer>...</answer>. "
    "Do not mention that you were given hidden information."
)
DEFAULT_STUDENT_TEMPLATE = (
    "Solve the user's problem. You may reason step by step, but you must put "
    "your final answer inside <answer>...</answer>."
)


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_field_path(path: str) -> list[str]:
    return [part for part in path.split(".") if part]


def nested_get(record: dict[str, Any], field_path: str, default: Any = None) -> Any:
    current: Any = record
    for part in parse_field_path(field_path):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def serialize_chat_fallback(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].capitalize()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("Assistant:")
    return "\n\n".join(rendered)


def chat_token_ids(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return list(ids)

    text = serialize_chat_fallback(messages, add_generation_prompt=add_generation_prompt)
    return tokenizer(text, add_special_tokens=True).input_ids


def build_teacher_messages(
    question: str,
    privileged_answer: str,
    template: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": template.format(correct_answer=privileged_answer),
        },
        {"role": "user", "content": question},
    ]


def build_student_messages(question: str, template: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if template.strip():
        messages.append({"role": "system", "content": template})
    messages.append({"role": "user", "content": question})
    return messages


def truncate_left(
    input_ids: list[int],
    labels: list[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    if len(input_ids) <= max_length:
        return input_ids, labels
    overflow = len(input_ids) - max_length
    return input_ids[overflow:], labels[overflow:]


def build_sequence(
    tokenizer: AutoTokenizer,
    prompt_messages: list[dict[str, str]],
    answer: str,
    *,
    max_length: int,
) -> dict[str, list[int]] | None:
    prompt_ids = chat_token_ids(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_ids = chat_token_ids(
        tokenizer,
        prompt_messages + [{"role": "assistant", "content": answer}],
        add_generation_prompt=False,
    )
    if len(full_ids) <= len(prompt_ids):
        return None
    labels = [-100] * len(full_ids)
    for idx in range(len(prompt_ids), len(full_ids)):
        labels[idx] = full_ids[idx]
    full_ids, labels = truncate_left(full_ids, labels, max_length=max_length)
    if sum(1 for token in labels if token != -100) == 0:
        return None
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def truncate_prompt(input_ids: list[int], max_prompt_length: int) -> list[int]:
    if len(input_ids) <= max_prompt_length:
        return input_ids
    return input_ids[-max_prompt_length:]


def build_prompt_batch(
    tokenizer: AutoTokenizer,
    messages_batch: list[list[dict[str, str]]],
    *,
    add_generation_prompt: bool,
    max_prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    prompt_id_rows: list[list[int]] = []
    prompt_lengths: list[int] = []
    for messages in messages_batch:
        prompt_ids = chat_token_ids(
            tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        prompt_ids = truncate_prompt(prompt_ids, max_prompt_length)
        prompt_id_rows.append(prompt_ids)
        prompt_lengths.append(len(prompt_ids))

    max_len = max(prompt_lengths)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_rows = []
    mask_rows = []
    for prompt_ids in prompt_id_rows:
        pad_len = max_len - len(prompt_ids)
        input_rows.append(prompt_ids + [pad_token_id] * pad_len)
        mask_rows.append([1] * len(prompt_ids) + [0] * pad_len)
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.long),
        prompt_lengths,
    )


def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages_batch: list[list[dict[str, str]]],
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
) -> list[str]:
    if not messages_batch:
        return []
    max_prompt_length = max(32, max_length - max_new_tokens)
    input_ids, attention_mask, prompt_lengths = build_prompt_batch(
        tokenizer,
        messages_batch,
        add_generation_prompt=True,
        max_prompt_length=max_prompt_length,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    do_sample = temperature > 0.0
    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    was_training = model.training
    model.eval()
    with torch.no_grad():
        sequences = model.generate(**generate_kwargs)
    if was_training:
        model.train()

    completions: list[str] = []
    for row_idx, prompt_length in enumerate(prompt_lengths):
        completion_ids = sequences[row_idx, prompt_length:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        completions.append(completion_text)
    return completions


def generate_correct_teacher_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages_batch: list[list[dict[str, str]]],
    *,
    ground_truths: list[str],
    data_sources: list[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    max_attempts: int,
) -> tuple[list[str | None], list[int]]:
    accepted: list[str | None] = [None] * len(messages_batch)
    attempts_used: list[int] = [0] * len(messages_batch)
    pending = list(range(len(messages_batch)))

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        completions = generate_completions(
            model,
            tokenizer,
            [messages_batch[idx] for idx in pending],
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
        )
        next_pending: list[int] = []
        for idx, completion in zip(pending, completions, strict=True):
            attempts_used[idx] = attempt
            score = float(compute_score(data_sources[idx], completion, ground_truths[idx]))
            if score > 0.5:
                accepted[idx] = completion
            else:
                next_pending.append(idx)
        pending = next_pending
    return accepted, attempts_used


def build_aligned_support_pack(
    tokenizer: AutoTokenizer,
    *,
    student_messages: list[dict[str, str]],
    teacher_messages: list[dict[str, str]],
    completion: str,
    max_length: int,
) -> dict[str, list[int]] | None:
    student_pack = build_sequence(
        tokenizer,
        student_messages,
        completion,
        max_length=max_length,
    )
    teacher_pack = build_sequence(
        tokenizer,
        teacher_messages,
        completion,
        max_length=max_length,
    )
    if student_pack is None or teacher_pack is None:
        return None

    student_target_count = sum(1 for token in student_pack["labels"] if token != -100)
    teacher_target_count = sum(1 for token in teacher_pack["labels"] if token != -100)
    if student_target_count == 0 or student_target_count != teacher_target_count:
        return None

    return {
        "student_input_ids": student_pack["input_ids"],
        "student_attention_mask": student_pack["attention_mask"],
        "student_labels": student_pack["labels"],
        "teacher_input_ids": teacher_pack["input_ids"],
        "teacher_attention_mask": teacher_pack["attention_mask"],
        "teacher_labels": teacher_pack["labels"],
    }


@dataclass(frozen=True)
class DistillExample:
    question: str
    answer: str
    ground_truth: str
    privileged_answer: str
    data_source: str


class DistillDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        *,
        question_field: str,
        answer_field: str,
        ground_truth_field: str,
        data_source_field: str,
        privileged_source: str,
        trace_source: str,
        max_samples: int | None,
        seed: int,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed)

        self.examples: list[DistillExample] = []
        skipped = 0
        for record in df.to_dict("records"):
            question = nested_get(record, question_field)
            answer = nested_get(record, answer_field)
            ground_truth = nested_get(record, ground_truth_field)
            data_source = nested_get(record, data_source_field, "gsm8k")
            if is_missing(question):
                skipped += 1
                continue
            question = str(question)
            answer = "" if is_missing(answer) else str(answer)
            ground_truth_str = "" if is_missing(ground_truth) else str(ground_truth)
            if privileged_source == "ground_truth":
                privileged_answer = ground_truth_str
            elif privileged_source == "answer":
                privileged_answer = answer
            else:
                privileged_answer = ground_truth_str or answer
            if not privileged_answer:
                skipped += 1
                continue
            if trace_source == "existing" and is_missing(answer):
                skipped += 1
                continue
            if trace_source == "generate" and not ground_truth_str:
                skipped += 1
                continue

            self.examples.append(
                DistillExample(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth_str,
                    privileged_answer=privileged_answer,
                    data_source=str(data_source),
                )
            )
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DistillExample:
        return self.examples[idx]


class DistillCollator:
    def __call__(self, batch: list[DistillExample]) -> dict[str, list[str]]:
        return {
            "question": [item.question for item in batch],
            "answer": [item.answer for item in batch],
            "ground_truth": [item.ground_truth for item in batch],
            "privileged_answer": [item.privileged_answer for item in batch],
            "data_source": [item.data_source for item in batch],
        }


class TokenPackCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _pad(self, items: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        max_len = max(item.shape[0] for item in items)
        padded = []
        for item in items:
            if item.shape[0] == max_len:
                padded.append(item)
                continue
            pad = torch.full((max_len - item.shape[0],), padding_value, dtype=item.dtype)
            padded.append(torch.cat([item, pad], dim=0))
        return torch.stack(padded, dim=0)

    def __call__(self, batch: list[dict[str, list[int] | int]]) -> dict[str, torch.Tensor]:
        collated: dict[str, torch.Tensor] = {}
        for prefix in ("teacher_support", "student_support"):
            collated[f"{prefix}_student_input_ids"] = self._pad(
                [torch.tensor(item[f"{prefix}_student_input_ids"], dtype=torch.long) for item in batch],
                self.pad_token_id,
            )
            collated[f"{prefix}_student_attention_mask"] = self._pad(
                [torch.tensor(item[f"{prefix}_student_attention_mask"], dtype=torch.long) for item in batch],
                0,
            )
            collated[f"{prefix}_student_labels"] = self._pad(
                [torch.tensor(item[f"{prefix}_student_labels"], dtype=torch.long) for item in batch],
                -100,
            )
            collated[f"{prefix}_teacher_input_ids"] = self._pad(
                [torch.tensor(item[f"{prefix}_teacher_input_ids"], dtype=torch.long) for item in batch],
                self.pad_token_id,
            )
            collated[f"{prefix}_teacher_attention_mask"] = self._pad(
                [torch.tensor(item[f"{prefix}_teacher_attention_mask"], dtype=torch.long) for item in batch],
                0,
            )
            collated[f"{prefix}_teacher_labels"] = self._pad(
                [torch.tensor(item[f"{prefix}_teacher_labels"], dtype=torch.long) for item in batch],
                -100,
            )
        return collated


def move_batch_to_devices(
    batch: dict[str, torch.Tensor],
    student_device: torch.device,
    teacher_device: torch.device,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if "_teacher_" in key:
            moved[key] = value.to(teacher_device)
        else:
            moved[key] = value.to(student_device)
    return moved


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device_name}, but CUDA is not available.")
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device {device_name}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return device


def sync_teacher_from_student(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
) -> None:
    teacher_model.load_state_dict(student_model.state_dict(), strict=True)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False


def select_privileged_support_answer(example: DistillExample) -> str:
    return example.privileged_answer


def prepare_training_batch(
    raw_batch: dict[str, list[str]],
    *,
    tokenizer: AutoTokenizer,
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    student_device: torch.device,
    teacher_device: torch.device,
    student_template: str,
    teacher_template: str,
    trace_source: str,
    max_length: int,
    student_max_new_tokens: int,
    teacher_max_new_tokens: int,
    student_temperature: float,
    teacher_temperature: float,
    student_top_p: float,
    teacher_top_p: float,
    teacher_max_attempts: int,
    parallel_trace_generation: bool,
    token_pack_collator: TokenPackCollator,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, float]]:
    examples = [
        DistillExample(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            privileged_answer=privileged_answer,
            data_source=data_source,
        )
        for question, answer, ground_truth, privileged_answer, data_source in zip(
            raw_batch["question"],
            raw_batch["answer"],
            raw_batch["ground_truth"],
            raw_batch["privileged_answer"],
            raw_batch["data_source"],
            strict=True,
        )
    ]

    student_messages_batch = [build_student_messages(example.question, student_template) for example in examples]
    teacher_messages_batch = [
        build_teacher_messages(
            example.question,
            select_privileged_support_answer(example),
            teacher_template,
        )
        for example in examples
    ]

    if trace_source == "generate":
        if parallel_trace_generation and student_device != teacher_device:
            with ThreadPoolExecutor(max_workers=2) as executor:
                student_future = executor.submit(
                    generate_completions,
                    student_model,
                    tokenizer,
                    student_messages_batch,
                    device=student_device,
                    max_new_tokens=student_max_new_tokens,
                    temperature=student_temperature,
                    top_p=student_top_p,
                    max_length=max_length,
                )
                teacher_future = executor.submit(
                    generate_correct_teacher_completions,
                    teacher_model,
                    tokenizer,
                    teacher_messages_batch,
                    ground_truths=[example.ground_truth for example in examples],
                    data_sources=[example.data_source for example in examples],
                    device=teacher_device,
                    max_new_tokens=teacher_max_new_tokens,
                    temperature=teacher_temperature,
                    top_p=teacher_top_p,
                    max_length=max_length,
                    max_attempts=teacher_max_attempts,
                )
                student_completions = student_future.result()
                teacher_completions, teacher_attempts = teacher_future.result()
        else:
            student_completions = generate_completions(
                student_model,
                tokenizer,
                student_messages_batch,
                device=student_device,
                max_new_tokens=student_max_new_tokens,
                temperature=student_temperature,
                top_p=student_top_p,
                max_length=max_length,
            )
            teacher_completions, teacher_attempts = generate_correct_teacher_completions(
                teacher_model,
                tokenizer,
                teacher_messages_batch,
                ground_truths=[example.ground_truth for example in examples],
                data_sources=[example.data_source for example in examples],
                device=teacher_device,
                max_new_tokens=teacher_max_new_tokens,
                temperature=teacher_temperature,
                top_p=teacher_top_p,
                max_length=max_length,
                max_attempts=teacher_max_attempts,
            )
    else:
        student_completions = [example.answer for example in examples]
        teacher_completions = [example.answer for example in examples]
        teacher_attempts = [1 if example.answer else 0 for example in examples]

    packs: list[dict[str, list[int] | int]] = []
    student_correct = 0
    teacher_correct = 0
    skipped = 0
    teacher_attempt_total = 0
    for example, student_completion, teacher_completion, teacher_attempt in zip(
        examples,
        student_completions,
        teacher_completions,
        teacher_attempts,
        strict=True,
    ):
        teacher_attempt_total += teacher_attempt
        if teacher_completion is None or not teacher_completion.strip():
            skipped += 1
            continue

        student_score = float(compute_score(example.data_source, student_completion, example.ground_truth))
        teacher_score = float(compute_score(example.data_source, teacher_completion, example.ground_truth))
        student_correct += int(student_score > 0.5)
        teacher_correct += int(teacher_score > 0.5)
        if teacher_score <= 0.5:
            skipped += 1
            continue

        forward_support = build_aligned_support_pack(
            tokenizer,
            student_messages=build_student_messages(example.question, student_template),
            teacher_messages=build_teacher_messages(example.question, select_privileged_support_answer(example), teacher_template),
            completion=teacher_completion,
            max_length=max_length,
        )
        reverse_support = build_aligned_support_pack(
            tokenizer,
            student_messages=build_student_messages(example.question, student_template),
            teacher_messages=build_teacher_messages(example.question, select_privileged_support_answer(example), teacher_template),
            completion=student_completion,
            max_length=max_length,
        )
        if forward_support is None or reverse_support is None:
            skipped += 1
            continue

        pack: dict[str, list[int] | int] = {}
        for key, value in forward_support.items():
            pack[f"teacher_support_{key}"] = value
        for key, value in reverse_support.items():
            pack[f"student_support_{key}"] = value
        packs.append(pack)

    metrics = {
        "source_examples": float(len(examples)),
        "usable_examples": float(len(packs)),
        "skipped_examples": float(skipped),
        "teacher_correct_rate": float(teacher_correct / len(examples)) if examples else 0.0,
        "student_correct_rate": float(student_correct / len(examples)) if examples else 0.0,
        "teacher_attempts_mean": float(teacher_attempt_total / len(examples)) if examples else 0.0,
    }
    if not packs:
        return None, metrics

    return token_pack_collator(packs), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="Training parquet file.")
    parser.add_argument("--student-model", required=True, help="Student model path or HF id.")
    parser.add_argument("--teacher-model", default=None, help="Teacher model path or HF id. Defaults to --student-model.")
    parser.add_argument("--output-dir", required=True, help="Scratch output directory for checkpoints and logs.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--question-field", default="extra_info.question")
    parser.add_argument(
        "--answer-field",
        default="extra_info.answer",
        help="Legacy trace field. Used for --trace-source existing and as an optional privileged fallback.",
    )
    parser.add_argument("--ground-truth-field", default="extra_info.ground_truth")
    parser.add_argument("--data-source-field", default="data_source")
    parser.add_argument(
        "--trace-source",
        choices=("generate", "existing"),
        default="generate",
        help="Default generate mode samples fresh student and teacher traces. Existing reuses the parquet trace.",
    )
    parser.add_argument(
        "--privileged-source",
        choices=("auto", "ground_truth", "answer"),
        default="auto",
        help="Which field should be exposed to the teacher as privileged information.",
    )
    parser.add_argument(
        "--student-template",
        default=DEFAULT_STUDENT_TEMPLATE,
        help="System prompt injected for the student generation prompt.",
    )
    parser.add_argument(
        "--teacher-template",
        default=DEFAULT_TEACHER_TEMPLATE,
        help="Prompt template injected as a system message for the teacher. Use {correct_answer}.",
    )
    parser.add_argument("--student-max-new-tokens", type=int, default=1024)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=1024)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--student-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-top-p", type=float, default=1.0)
    parser.add_argument(
        "--teacher-max-attempts",
        type=int,
        default=4,
        help="Maximum privileged teacher rollouts to try per example before the example is skipped.",
    )
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument(
        "--teacher-sync-mode",
        choices=("off", "step"),
        default="step",
        help="Whether the teacher should be refreshed from the current student weights.",
    )
    parser.add_argument(
        "--teacher-sync-every",
        type=int,
        default=1,
        help="Refresh the teacher every N optimizer steps when --teacher-sync-mode=step.",
    )
    parser.add_argument(
        "--parallel-trace-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate student and teacher traces concurrently when they are on different devices.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument(
        "--kl-token-chunk-size",
        type=int,
        default=128,
        help="Chunk selected support tokens during KL to reduce peak memory use.",
    )
    parser.add_argument("--kl-mode", choices=("forward", "reverse", "interpolate"), default="forward")
    parser.add_argument(
        "--kl-interp-alpha",
        type=float,
        default=0.5,
        help="Reverse-KL weight when --kl-mode=interpolate. final=(1-alpha)*forward + alpha*reverse.",
    )
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--wandb-project", default="forgetting-llms")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="disabled", choices=("disabled", "offline", "online"))
    return parser.parse_args()


def dtype_from_precision(mixed_precision: str) -> torch.dtype | None:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def maybe_apply_lora(model: AutoModelForCausalLM, args: argparse.Namespace) -> AutoModelForCausalLM:
    if args.lora_rank <= 0:
        return model
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise RuntimeError("peft is required for LoRA but is not installed.")
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def shift_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_mask = shifted_labels.ne(-100)
    return shifted_logits, shifted_labels, shifted_mask


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits, shifted_labels, _ = shift_logits_and_labels(logits, labels)
    vocab = shifted_logits.shape[-1]
    return F.cross_entropy(
        shifted_logits.view(-1, vocab),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def compute_directional_kl_loss(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    *,
    direction: str,
    temperature: float,
    token_chunk_size: int,
) -> torch.Tensor:
    student_shifted, _student_shifted_labels, student_mask = shift_logits_and_labels(student_logits, student_labels)
    teacher_shifted, _teacher_shifted_labels, teacher_mask = shift_logits_and_labels(teacher_logits, teacher_labels)

    student_selected = student_shifted[student_mask]
    teacher_selected = teacher_shifted[teacher_mask]
    if student_selected.shape[0] != teacher_selected.shape[0]:
        raise RuntimeError(
            "Student and teacher target-token counts diverged after tokenization. "
            f"student={student_selected.shape[0]} teacher={teacher_selected.shape[0]}"
        )

    if student_selected.shape[0] == 0:
        return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    temperature = max(temperature, 1e-6)
    token_chunk_size = max(int(token_chunk_size), 1)
    total_tokens = student_selected.shape[0]
    total_kl = torch.zeros((), device=student_selected.device, dtype=torch.float32)

    try:
        for start in range(0, total_tokens, token_chunk_size):
            end = min(start + token_chunk_size, total_tokens)
            student_chunk = student_selected[start:end]
            teacher_chunk = teacher_selected[start:end].to(student_selected.device, non_blocking=True)

            student_log_probs = F.log_softmax(student_chunk / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_chunk / temperature, dim=-1)

            if direction == "forward":
                teacher_probs = teacher_log_probs.exp()
                chunk_kl = torch.sum(
                    teacher_probs * (teacher_log_probs - student_log_probs),
                    dim=-1,
                )
            elif direction == "reverse":
                student_probs = student_log_probs.exp()
                chunk_kl = torch.sum(
                    student_probs * (student_log_probs - teacher_log_probs),
                    dim=-1,
                )
            else:
                raise ValueError(f"Unsupported KL direction: {direction}")

            total_kl = total_kl + chunk_kl.float().sum()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if student_selected.device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "OOM while computing KL over generated supports. "
                "Reduce --kl-token-chunk-size, --max-length, or generation lengths."
            ) from exc
        raise

    mean_kl = total_kl / float(total_tokens)
    return mean_kl.to(student_logits.dtype) * (temperature ** 2)


def save_student_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    step: int,
    metadata: dict[str, Any],
) -> None:
    checkpoint_dir = output_dir / f"global_step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        checkpoint_dir,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "trainer_state.json").write_text(json.dumps(metadata, indent=2) + "\n")


def maybe_log_wandb(accelerator: Accelerator, payload: dict[str, Any], step: int) -> None:
    if accelerator.is_main_process:
        accelerator.log(payload, step=step)


def main() -> int:
    args = parse_args()
    if MISSING_IMPORT is not None:
        raise RuntimeError(
            "self_distill.py requires torch, transformers, and accelerate in the active environment."
        ) from MISSING_IMPORT
    train_file = expand(args.train_file)
    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.teacher_model is None:
        args.teacher_model = args.student_model
    if not 0.0 <= args.kl_interp_alpha <= 1.0:
        raise ValueError("--kl-interp-alpha must be in [0, 1].")
    if args.trace_source == "generate" and args.teacher_max_attempts < 1:
        raise ValueError("--teacher-max-attempts must be >= 1 in generate mode.")
    if args.teacher_sync_mode == "step" and args.teacher_sync_every < 1:
        raise ValueError("--teacher-sync-every must be >= 1 when teacher sync is enabled.")

    set_seed(args.seed)
    student_device = resolve_device(args.student_device)
    teacher_device = resolve_device(args.teacher_device)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="wandb" if args.wandb_mode != "disabled" else None,
        project_dir=str(output_dir),
    )

    if accelerator.is_main_process:
        manifest = {
            "command": "self_distill",
            "args": vars(args),
            "train_file": str(train_file),
            "started_at": datetime_now_iso(),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.wandb_mode != "disabled":
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "entity": args.wandb_entity,
                    "mode": args.wandb_mode,
                    "name": args.run_name or output_dir.name,
                    "dir": str(output_dir / "wandb"),
                }
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = DistillDataset(
        train_file,
        question_field=args.question_field,
        answer_field=args.answer_field,
        ground_truth_field=args.ground_truth_field,
        data_source_field=args.data_source_field,
        privileged_source=args.privileged_source,
        trace_source=args.trace_source,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError("No usable training examples were found for self-distillation.")

    collator = DistillCollator()
    token_pack_collator = TokenPackCollator(tokenizer.pad_token_id)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    model_dtype = dtype_from_precision(args.mixed_precision)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
    ).to(student_device)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
    ).to(teacher_device)

    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.config.use_cache = False

    student_model = maybe_apply_lora(student_model, args)
    if args.lora_rank > 0 and args.teacher_sync_mode == "step":
        teacher_model = maybe_apply_lora(teacher_model, args)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    if args.teacher_sync_mode == "step":
        sync_teacher_from_student(student_model, teacher_model)

    optimizer = AdamW(
        [param for param in student_model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_training_steps = args.max_steps or (args.epochs * update_steps_per_epoch)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    trainable_params, total_params = count_trainable_parameters(student_model)
    if accelerator.is_main_process:
        print(
            f"Loaded {len(dataset)} training examples (skipped {dataset.skipped}) "
            f"trace_source={args.trace_source}"
        )
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        print(f"Student device: {student_device}")
        print(f"Teacher device: {teacher_device}")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_ce = 0.0
    running_distill = 0.0
    running_forward = 0.0
    running_reverse = 0.0
    running_teacher_success = 0.0
    running_student_success = 0.0
    running_skipped_examples = 0.0
    running_usable_examples = 0.0
    running_teacher_attempts = 0.0
    log_counter = 0

    for epoch in range(args.epochs):
        student_model.train()
        for raw_batch in train_dataloader:
            prepared_batch, generation_metrics = prepare_training_batch(
                raw_batch,
                tokenizer=tokenizer,
                student_model=student_model,
                teacher_model=teacher_model,
                student_device=student_device,
                teacher_device=teacher_device,
                student_template=args.student_template,
                teacher_template=args.teacher_template,
                trace_source=args.trace_source,
                max_length=args.max_length,
                student_max_new_tokens=args.student_max_new_tokens,
                teacher_max_new_tokens=args.teacher_max_new_tokens,
                student_temperature=args.student_temperature,
                teacher_temperature=args.teacher_temperature,
                student_top_p=args.student_top_p,
                teacher_top_p=args.teacher_top_p,
                teacher_max_attempts=args.teacher_max_attempts,
                parallel_trace_generation=args.parallel_trace_generation,
                token_pack_collator=token_pack_collator,
            )
            if prepared_batch is None:
                if accelerator.is_main_process:
                    print(
                        "Skipping batch with no usable generated supports: "
                        f"source_examples={int(generation_metrics['source_examples'])} "
                        f"skipped={int(generation_metrics['skipped_examples'])}"
                    )
                continue
            batch = move_batch_to_devices(prepared_batch, student_device, teacher_device)
            with accelerator.accumulate(student_model):
                teacher_support_student_outputs = student_model(
                    input_ids=batch["teacher_support_student_input_ids"],
                    attention_mask=batch["teacher_support_student_attention_mask"],
                )
                with torch.no_grad():
                    teacher_support_teacher_outputs = teacher_model(
                        input_ids=batch["teacher_support_teacher_input_ids"],
                        attention_mask=batch["teacher_support_teacher_attention_mask"],
                    )
                if args.kl_mode in ("reverse", "interpolate"):
                    reverse_support_student_outputs = student_model(
                        input_ids=batch["student_support_student_input_ids"],
                        attention_mask=batch["student_support_student_attention_mask"],
                    )
                    with torch.no_grad():
                        reverse_support_teacher_outputs = teacher_model(
                            input_ids=batch["student_support_teacher_input_ids"],
                            attention_mask=batch["student_support_teacher_attention_mask"],
                        )
                else:
                    reverse_support_student_outputs = None
                    reverse_support_teacher_outputs = None

                ce_loss = compute_ce_loss(
                    teacher_support_student_outputs.logits,
                    batch["teacher_support_student_labels"],
                )
                forward_kl = compute_directional_kl_loss(
                    teacher_support_student_outputs.logits,
                    batch["teacher_support_student_labels"],
                    teacher_support_teacher_outputs.logits,
                    batch["teacher_support_teacher_labels"],
                    direction="forward",
                    temperature=args.distill_temperature,
                    token_chunk_size=args.kl_token_chunk_size,
                )
                if reverse_support_student_outputs is not None and reverse_support_teacher_outputs is not None:
                    reverse_kl = compute_directional_kl_loss(
                        reverse_support_student_outputs.logits,
                        batch["student_support_student_labels"],
                        reverse_support_teacher_outputs.logits,
                        batch["student_support_teacher_labels"],
                        direction="reverse",
                        temperature=args.distill_temperature,
                        token_chunk_size=args.kl_token_chunk_size,
                    )
                else:
                    reverse_kl = torch.zeros_like(forward_kl)

                if args.kl_mode == "forward":
                    distill_loss = forward_kl
                elif args.kl_mode == "reverse":
                    distill_loss = reverse_kl
                else:
                    distill_loss = (
                        (1.0 - args.kl_interp_alpha) * forward_kl
                        + args.kl_interp_alpha * reverse_kl
                    )
                total_loss = args.ce_weight * ce_loss + args.distill_weight * distill_loss

                accelerator.backward(total_loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            reduced_loss = float(total_loss.detach().item())
            reduced_ce = float(ce_loss.detach().item())
            reduced_distill = float(distill_loss.detach().item())
            reduced_forward = float(forward_kl.detach().item())
            reduced_reverse = float(reverse_kl.detach().item())
            running_loss += reduced_loss
            running_ce += reduced_ce
            running_distill += reduced_distill
            running_forward += reduced_forward
            running_reverse += reduced_reverse
            running_teacher_success += generation_metrics["teacher_correct_rate"]
            running_student_success += generation_metrics["student_correct_rate"]
            running_skipped_examples += generation_metrics["skipped_examples"]
            running_usable_examples += generation_metrics["usable_examples"]
            running_teacher_attempts += generation_metrics["teacher_attempts_mean"]
            log_counter += 1

            if accelerator.sync_gradients:
                global_step += 1
                if (
                    args.teacher_sync_mode == "step"
                    and global_step % args.teacher_sync_every == 0
                ):
                    sync_teacher_from_student(student_model, teacher_model)

                if global_step % args.logging_steps == 0:
                    payload = {
                        "train/loss": running_loss / log_counter,
                        "train/ce_loss": running_ce / log_counter,
                        "train/distill_loss": running_distill / log_counter,
                        "train/forward_kl": running_forward / log_counter,
                        "train/reverse_kl": running_reverse / log_counter,
                        "train/teacher_trace_success_rate": running_teacher_success / log_counter,
                        "train/student_trace_correct_rate": running_student_success / log_counter,
                        "train/skipped_examples": running_skipped_examples / log_counter,
                        "train/usable_examples": running_usable_examples / log_counter,
                        "train/teacher_attempts_mean": running_teacher_attempts / log_counter,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": float(epoch),
                    }
                    maybe_log_wandb(accelerator, payload, global_step)
                    if accelerator.is_main_process:
                        print(
                            f"step={global_step} "
                            f"loss={payload['train/loss']:.4f} "
                            f"ce={payload['train/ce_loss']:.4f} "
                            f"distill={payload['train/distill_loss']:.4f} "
                            f"fkl={payload['train/forward_kl']:.4f} "
                            f"rkl={payload['train/reverse_kl']:.4f} "
                            f"teacher_ok={payload['train/teacher_trace_success_rate']:.3f} "
                            f"student_ok={payload['train/student_trace_correct_rate']:.3f}"
                        )
                    running_loss = 0.0
                    running_ce = 0.0
                    running_distill = 0.0
                    running_forward = 0.0
                    running_reverse = 0.0
                    running_teacher_success = 0.0
                    running_student_success = 0.0
                    running_skipped_examples = 0.0
                    running_usable_examples = 0.0
                    running_teacher_attempts = 0.0
                    log_counter = 0

                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_student_checkpoint(
                            accelerator,
                            student_model,
                            tokenizer,
                            output_dir,
                            global_step,
                            metadata={
                                "global_step": global_step,
                                "epoch": epoch,
                                "kl_mode": args.kl_mode,
                                "kl_interp_alpha": args.kl_interp_alpha,
                                "ce_weight": args.ce_weight,
                                "distill_weight": args.distill_weight,
                            },
                        )

                if args.max_steps is not None and global_step >= args.max_steps:
                    break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_student_checkpoint(
            accelerator,
            student_model,
            tokenizer,
            output_dir,
            global_step,
            metadata={
                "global_step": global_step,
                "completed": True,
                "kl_mode": args.kl_mode,
                "kl_interp_alpha": args.kl_interp_alpha,
                "ce_weight": args.ce_weight,
                "distill_weight": args.distill_weight,
                "train_examples": len(dataset),
                "skipped_examples": dataset.skipped,
            },
        )
        (output_dir / "completed.marker").write_text("completed\n")
    accelerator.end_training()
    return 0


def datetime_now_iso() -> str:
    import datetime as _datetime

    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
