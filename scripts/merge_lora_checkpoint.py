#!/usr/bin/env python3
"""Merge a PEFT/LoRA adapter checkpoint into a standalone HF model directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    return parser.parse_args()


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def resolve_base_model(adapter_dir: Path, explicit_base_model: str | None) -> str:
    if explicit_base_model:
        return explicit_base_model
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing adapter config: {config_path}")
    payload = json.loads(config_path.read_text())
    base_model = payload.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise SystemExit(f"Could not determine base model from {config_path}")
    return base_model


def resolve_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return None


def main() -> int:
    args = parse_args()
    adapter_dir = expand(args.adapter_dir)
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter dir not found: {adapter_dir}")

    output_dir = expand(args.output_dir) if args.output_dir else adapter_dir / "merged_hf"
    base_model = resolve_base_model(adapter_dir, args.base_model)

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged_model = peft_model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(
            {
                "adapter_dir": str(adapter_dir),
                "base_model": base_model,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
        + "\n"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
