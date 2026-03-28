#!/usr/bin/env python3
"""Native parquet-backed SFT trainer for sequential stage experiments.

This is the repo's non-VeRL SFT path. It trains directly on the parquet format
used throughout the project:

- ``data_source`` at the row root
- ``extra_info.question`` for the prompt
- ``extra_info.answer`` for the target answer/trace

The trainer supports:
- full fine-tuning or optional LoRA
- resume from the latest checkpoint
- fixed-step stage training for sequential experiments
- multi-GPU launches via ``torchrun``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def nested_get(record: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = record
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for child in output_dir.iterdir() if output_dir.exists() else []:
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                checkpoints.append((int(child.name.split("-", 1)[1]), child))
            except (IndexError, ValueError):
                continue
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def export_merged_lora_model(
    *,
    base_model: str,
    adapter_dir: Path,
    tokenizer: AutoTokenizer,
    torch_dtype: torch.dtype | None,
) -> Path:
    from peft import PeftModel

    merged_dir = adapter_dir / "merged_hf"
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    (merged_dir / "merge_manifest.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "adapter_dir": str(adapter_dir),
                "merged_dir": str(merged_dir),
            },
            indent=2,
        )
        + "\n"
    )
    return merged_dir


class SFTParquetDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        tokenizer: AutoTokenizer,
        *,
        max_length: int,
        question_field: str,
        answer_field: str,
        separator: str,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []

        df = pd.read_parquet(parquet_path)
        for record in df.to_dict("records"):
            question = nested_get(record, question_field, "")
            answer = nested_get(record, answer_field, "")
            if not isinstance(question, str) or not question.strip():
                continue
            if not isinstance(answer, str) or not answer.strip():
                continue

            prompt = question.strip()
            target = answer.strip()
            prompt_prefix = f"{prompt}{separator}"
            full_text = f"{prompt_prefix}{target}"

            prompt_tokens = tokenizer(
                prompt_prefix,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            full_tokens = tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )

            input_ids = list(full_tokens["input_ids"])
            attention_mask = list(full_tokens["attention_mask"])
            labels = list(input_ids)

            prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
            for idx in range(prompt_len):
                labels[idx] = -100

            if all(label == -100 for label in labels):
                continue

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        if not self.examples:
            raise RuntimeError(f"No usable examples found in {parquet_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


class PackedCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="plain_sft")
    parser.add_argument("--question-field", default="extra_info.question")
    parser.add_argument("--answer-field", default="extra_info.answer")
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--wandb-project", default="forgetting-llms")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="disabled", choices=("disabled", "offline", "online"))
    return parser.parse_args()


def maybe_apply_lora(model: AutoModelForCausalLM, args: argparse.Namespace) -> AutoModelForCausalLM:
    if args.lora_rank <= 0:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    return get_peft_model(model, config)


def main() -> int:
    args = parse_args()
    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model = maybe_apply_lora(model, args)

    train_dataset = SFTParquetDataset(
        expand(args.train_file),
        tokenizer,
        max_length=args.max_length,
        question_field=args.question_field,
        answer_field=args.answer_field,
        separator=args.separator,
    )
    eval_dataset = None
    if args.eval_file:
        eval_path = expand(args.eval_file)
        if eval_path.exists():
            eval_dataset = SFTParquetDataset(
                eval_path,
                tokenizer,
                max_length=args.max_length,
                question_field=args.question_field,
                answer_field=args.answer_field,
                separator=args.separator,
            )

    report_to = [] if args.wandb_mode == "disabled" else ["wandb"]
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if eval_dataset is not None else None,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": args.mixed_precision == "bf16",
        "fp16": args.mixed_precision == "fp16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "report_to": report_to,
        "run_name": args.run_name,
        "seed": args.seed,
        "ddp_find_unused_parameters": False,
        "save_safetensors": True,
    }
    strategy_value = "steps" if eval_dataset is not None else "no"
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = strategy_value
    elif "eval_strategy" in training_args_signature.parameters:
        training_args_kwargs["eval_strategy"] = strategy_value

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PackedCollator(tokenizer.pad_token_id),
        tokenizer=tokenizer,
    )

    resume_from = None
    if args.resume_from_checkpoint == "auto":
        latest = latest_checkpoint(output_dir)
        if latest is not None:
            resume_from = str(latest)
    elif args.resume_from_checkpoint not in ("", "none", "None"):
        resume_from = str(expand(args.resume_from_checkpoint))

    manifest = {
        "command": "plain_sft",
        "model": args.model,
        "train_file": str(expand(args.train_file)),
        "eval_file": str(expand(args.eval_file)) if args.eval_file else None,
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "max_length": args.max_length,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "lora_rank": args.lora_rank,
        "resume_from_checkpoint": resume_from,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    distributed_barrier()
    merged_export = None
    if args.lora_rank > 0 and trainer.is_world_process_zero():
        # Benchmarks and downstream RL expect a standalone HF checkpoint, not just adapter weights.
        merged_export = export_merged_lora_model(
            base_model=args.model,
            adapter_dir=output_dir,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
        )
    distributed_barrier()
    train_result = trainer.state.log_history
    result_payload = {
        "log_history": train_result,
        "merged_export": str(merged_export) if merged_export is not None else None,
    }
    (output_dir / "train_result.json").write_text(json.dumps(result_payload, indent=2) + "\n")
    (output_dir / "completed.marker").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
