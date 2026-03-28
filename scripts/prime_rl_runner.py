#!/usr/bin/env python3
"""Baseline evaluation and PRIME-RL run-bundle launcher.

This script is intentionally split into two workflows:

1. ``baseline`` runs local generation against the repo's current parquet datasets
   and saves predictions + metrics to scratch.
2. ``prime`` prepares a scratch-local PRIME-RL bundle (configs, env vars,
   launch script, manifest) and can optionally execute it.

The PRIME-RL integration here is process-level on purpose. This repo does not
ship a native PRIME-RL trainer implementation, and the local dev environment may
not have ``prime-rl`` or ``verifiers`` installed. The generated bundle is the
stable handoff point.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.rewards.unified_reward import compute_score  # noqa: E402
from src.utils.model_context import infer_model_max_len  # noqa: E402


SUPPORTED_DATASETS = ("gsm8k", "math", "triviaqa", "polaris_math", "openr1_math")


@dataclass
class RunPaths:
    run_dir: Path
    configs_dir: Path
    logs_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    checkpoints_dir: Path
    wandb_dir: Path
    wandb_cache_dir: Path
    hf_home: Path
    models_dir: Path


@dataclass
class PrimeLaunchOverrides:
    inference_gpu_ids: list[str]
    trainer_gpu_ids: list[str]
    inference_dp: int | None
    inference_tp: int | None
    inference_gpu_memory_utilization: float | None
    inference_server_port: int | None
    orchestrator_client_base_url: str | None
    passthrough_args: list[str]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def slugify(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def default_run_name(prefix: str, dataset: str | None = None, model: str | None = None) -> str:
    model_slug = slugify((model or "model").split("/")[-1])
    parts = [prefix]
    if dataset:
        parts.append(slugify(dataset))
    parts.append(model_slug)
    parts.append(now_stamp())
    return "_".join(parts)


def build_run_paths(output_root: Path, run_name: str, hf_home: Path) -> RunPaths:
    run_dir = output_root / run_name
    return RunPaths(
        run_dir=run_dir,
        configs_dir=run_dir / "configs",
        logs_dir=run_dir / "logs",
        metrics_dir=run_dir / "metrics",
        predictions_dir=run_dir / "predictions",
        checkpoints_dir=run_dir / "checkpoints",
        wandb_dir=run_dir / "wandb",
        wandb_cache_dir=run_dir / "wandb-cache",
        hf_home=hf_home,
        models_dir=output_root / "models",
    )


def ensure_dirs(paths: RunPaths) -> None:
    for path in (
        paths.run_dir,
        paths.configs_dir,
        paths.logs_dir,
        paths.metrics_dir,
        paths.predictions_dir,
        paths.checkpoints_dir,
        paths.wandb_dir,
        paths.wandb_cache_dir,
        paths.hf_home,
        paths.models_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return f"[missing] {path}"
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"[unreadable] {path}: {exc}"
    snippet = lines[-max_lines:]
    return "\n".join(snippet) if snippet else "[empty log]"


def prime_failure_log_paths(paths: RunPaths) -> list[Path]:
    candidates = [
        paths.logs_dir / "orchestrator.log",
        paths.logs_dir / "trainer.log",
        paths.logs_dir / "inference.log",
        paths.run_dir / "orchestrator.stdout",
        paths.run_dir / "trainer.stdout",
        paths.run_dir / "inference.stdout",
        paths.run_dir / "rl.log",
    ]
    existing = [path for path in candidates if path.exists()]
    return existing or candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run a baseline generation/eval pass.")
    baseline.add_argument("--model", required=True, help="HF model id or local model path.")
    baseline.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset label used for bookkeeping. Built-in datasets are "
            f"{', '.join(SUPPORTED_DATASETS)}. For custom parquet evals, this is "
            "the display label unless --data-source is set separately."
        ),
    )
    baseline.add_argument("--split", default="test", choices=("train", "test"))
    baseline.add_argument("--data-root", default="~/scratch/forgetting-llms/data")
    baseline.add_argument(
        "--dataset-path",
        default=None,
        help="Optional custom parquet path. If set, baseline eval loads this file instead of data-root/dataset/split.parquet.",
    )
    baseline.add_argument(
        "--data-source",
        default=None,
        help="Optional reward routing key. Defaults to row['data_source'] or --dataset.",
    )
    baseline.add_argument("--messages-field", default="messages")
    baseline.add_argument("--prompt-field", default="prompt")
    baseline.add_argument("--ground-truth-field", default="ground_truth")
    baseline.add_argument("--output-root", default="~/scratch/forgetting-llms/runs")
    baseline.add_argument("--hf-home", default="~/scratch/huggingface")
    baseline.add_argument("--run-name", default=None)
    baseline.add_argument("--max-samples", type=int, default=None)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--tensor-parallel-size", type=int, default=1)
    baseline.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    baseline.add_argument("--max-model-len", type=int, default=None)
    baseline.add_argument("--max-tokens", type=int, default=8192)
    baseline.add_argument("--temperature", type=float, default=0.0)
    baseline.add_argument("--top-p", type=float, default=1.0)
    baseline.add_argument("--rollouts-per-prompt", type=int, default=1)
    baseline.add_argument("--wandb-project", default="forgetting-llms")
    baseline.add_argument("--wandb-entity", default=None)
    baseline.add_argument("--wandb-mode", default="disabled", choices=("disabled", "offline", "online"))

    prime = subparsers.add_parser("prime", help="Prepare or run a PRIME-RL bundle.")
    prime.add_argument("--model", required=True, help="HF model id or local model path.")
    prime.add_argument("--environment-name", required=True, help="Verifiers environment name.")
    prime.add_argument("--output-root", default="~/scratch/forgetting-llms/prime_runs")
    prime.add_argument("--hf-home", default="~/scratch/huggingface")
    prime.add_argument("--run-name", default=None)
    prime.add_argument("--max-steps", type=int, default=1000)
    prime.add_argument("--max-async-level", type=int, default=1)
    prime.add_argument("--batch-size", type=int, default=16)
    prime.add_argument("--seq-len", type=int, default=8192)
    prime.add_argument("--max-tokens", type=int, default=1024)
    prime.add_argument("--temperature", type=float, default=1.0)
    prime.add_argument("--top-p", type=float, default=1.0)
    prime.add_argument("--rollouts-per-prompt", type=int, default=8)
    prime.add_argument("--fake-data-batch-size", type=int, default=16)
    prime.add_argument("--ckpt-interval", type=int, default=5)
    prime.add_argument("--ckpt-keep-last", type=int, default=3)
    prime.add_argument("--ckpt-keep-interval", type=int, default=50)
    prime.add_argument("--resume-step", type=int, default=None)
    prime.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not auto-detect the latest PRIME checkpoint step for resume.",
    )
    prime.add_argument("--enforce-eager", action="store_true")
    prime.add_argument("--trainer-config", default=None, help="Existing PRIME-RL trainer TOML.")
    prime.add_argument("--orchestrator-config", default=None, help="Existing PRIME-RL orchestrator TOML.")
    prime.add_argument("--inference-config", default=None, help="Existing PRIME-RL inference TOML.")
    prime.add_argument("--combined-config", default=None, help="Existing combined PRIME-RL TOML.")
    prime.add_argument("--prime-command", default="uv run rl", help="Command used in the launch script.")
    prime.add_argument(
        "--prime-extra-args",
        default="",
        help="Raw extra argument string forwarded to the PRIME rl entrypoint.",
    )
    prime.add_argument("--wandb-project", default="forgetting-llms")
    prime.add_argument("--wandb-entity", default=None)
    prime.add_argument("--wandb-mode", default="online", choices=("disabled", "offline", "online"))
    prime.add_argument("--execute", action="store_true", help="Run the generated launch script.")

    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def resolve_dataset_parquet(
    dataset: str,
    split: str,
    data_root: Path,
    dataset_path: str | None,
) -> Path:
    if dataset_path:
        path = expand(dataset_path)
        if path.is_dir():
            return path / f"{split}.parquet"
        return path
    return data_root / dataset / f"{split}.parquet"


def resolve_messages(row: pd.Series, prompt_field: str, messages_field: str) -> list[dict[str, Any]]:
    messages_value = row.get(messages_field)
    if isinstance(messages_value, list):
        return messages_value

    prompt_value = row.get(prompt_field)
    if isinstance(prompt_value, list):
        return prompt_value
    if not is_missing(prompt_value):
        return [{"role": "user", "content": str(prompt_value)}]
    raise ValueError(
        f"Could not resolve prompt/messages from row; expected {messages_field} or {prompt_field}."
    )


def resolve_ground_truth(row: pd.Series, ground_truth_field: str) -> str:
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, str) and ground_truth.strip():
            return ground_truth

    direct_ground_truth = row.get(ground_truth_field)
    if not is_missing(direct_ground_truth):
        return str(direct_ground_truth)

    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("ground_truth", "answer"):
            value = extra_info.get(key)
            if not is_missing(value):
                return str(value)

    return ""


def load_local_dataset(
    dataset: str,
    split: str,
    data_root: Path,
    max_samples: int | None,
    seed: int,
    *,
    dataset_path: str | None = None,
    data_source: str | None = None,
    prompt_field: str = "prompt",
    messages_field: str = "messages",
    ground_truth_field: str = "ground_truth",
) -> list[dict[str, Any]]:
    parquet_path = resolve_dataset_parquet(dataset, split, data_root, dataset_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        messages = resolve_messages(row, prompt_field=prompt_field, messages_field=messages_field)
        ground_truth = resolve_ground_truth(row, ground_truth_field=ground_truth_field)
        row_data_source = row.get("data_source")
        if is_missing(row_data_source):
            row_data_source = None
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        records.append(
            {
                "messages": messages,
                "user_msg": user_msg,
                "ground_truth": ground_truth,
                "data_source": data_source or (str(row_data_source) if row_data_source is not None else dataset),
            }
        )
    return records


def baseline_run(args: argparse.Namespace) -> int:
    os.environ.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Baseline mode requires `transformers` and `vllm` in the environment."
        ) from exc

    run_name = args.run_name or default_run_name("baseline", dataset=args.dataset, model=args.model)
    output_root = expand(args.output_root)
    hf_home = expand(args.hf_home)
    paths = build_run_paths(output_root=output_root, run_name=run_name, hf_home=hf_home)
    ensure_dirs(paths)

    os.environ["HF_HOME"] = str(paths.hf_home)
    os.environ["WANDB_DIR"] = str(paths.wandb_dir)
    os.environ["WANDB_CACHE_DIR"] = str(paths.wandb_cache_dir)

    samples = load_local_dataset(
        dataset=args.dataset,
        split=args.split,
        data_root=expand(args.data_root),
        max_samples=args.max_samples,
        seed=args.seed,
        dataset_path=args.dataset_path,
        data_source=args.data_source,
        prompt_field=args.prompt_field,
        messages_field=args.messages_field,
        ground_truth_field=args.ground_truth_field,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
        for sample in samples
    ]

    effective_max_model_len = args.max_model_len or infer_model_max_len(args.model)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=effective_max_model_len,
    )
    sampling_params = SamplingParams(
        n=args.rollouts_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    results: list[dict[str, Any]] = []
    total_rollouts = 0
    total_reward = 0.0
    first_correct = 0
    any_correct = 0
    for sample, output in zip(samples, outputs):
        completions = [candidate.text for candidate in output.outputs]
        rewards = [
            float(compute_score(sample["data_source"], completion, sample["ground_truth"]))
            for completion in completions
        ]
        total_rollouts += len(completions)
        total_reward += sum(rewards)
        first_correct += int(bool(rewards and rewards[0] > 0.5))
        any_correct += int(any(reward > 0.5 for reward in rewards))
        results.append(
            {
                "prompt": sample["user_msg"],
                "ground_truth": sample["ground_truth"],
                "completions": completions,
                "rewards": rewards,
                "first_correct": bool(rewards and rewards[0] > 0.5),
                "any_correct": any(reward > 0.5 for reward in rewards),
            }
        )

    metrics = {
        "run_name": run_name,
        "mode": "baseline",
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "num_examples": len(samples),
        "rollouts_per_prompt": args.rollouts_per_prompt,
        "accuracy_at_1": first_correct / len(samples) if samples else 0.0,
        "pass_at_k": any_correct / len(samples) if samples else 0.0,
        "mean_rollout_reward": total_reward / total_rollouts if total_rollouts else 0.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    predictions_path = paths.predictions_dir / "predictions.jsonl"
    with predictions_path.open("w") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")
    write_json(paths.metrics_dir / "metrics.json", metrics)
    write_json(
        paths.run_dir / "manifest.json",
        {
            "command": "baseline",
            "args": vars(args),
            "paths": {key: str(value) for key, value in asdict(paths).items()},
        },
    )

    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError:
            print("wandb is not installed; skipping remote logging", file=sys.stderr)
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                dir=str(paths.wandb_dir),
                mode=args.wandb_mode,
                config={**vars(args), **metrics},
            )
            wandb.log(metrics)
            wandb.finish()

    print(
        "Accuracy@1: "
        f"{metrics['accuracy_at_1']:.4f} "
        f"({first_correct}/{len(samples)})"
    )
    print(
        f"Pass@{args.rollouts_per_prompt}: "
        f"{metrics['pass_at_k']:.4f} "
        f"({any_correct}/{len(samples)})"
    )
    print(f"Mean rollout reward: {metrics['mean_rollout_reward']:.4f}")
    print(f"Baseline run complete: {paths.run_dir}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {paths.metrics_dir / 'metrics.json'}")
    return 0


def copy_or_write_config(source: str | None, destination: Path, fallback_content: str) -> None:
    if source:
        shutil.copyfile(expand(source), destination)
    else:
        destination.write_text(fallback_content)


def trainer_stub(args: argparse.Namespace) -> str:
    return (
        "# Generated PRIME-RL trainer stub.\n"
        "# Prefer passing --trainer-config from a working `vf-setup` or project config.\n\n"
        f"max_steps = {args.max_steps}\n"
        f"max_async_level = {args.max_async_level}\n\n"
        "[model]\n"
        f'name = "{args.model}"\n'
        f"seq_len = {args.seq_len}\n\n"
        'optimization_dtype = "bfloat16"\n'
        'reduce_dtype = "bfloat16"\n\n'
        "[data.fake]\n"
        f"batch_size = {args.fake_data_batch_size}\n"
    )


def orchestrator_stub(args: argparse.Namespace) -> str:
    return (
        "# Generated PRIME-RL orchestrator stub.\n"
        "# Environment-specific settings generally belong in a working base config.\n\n"
        f'max_steps = {args.max_steps}\n'
        f"max_async_level = {args.max_async_level}\n"
        f"batch_size = {args.batch_size}\n\n"
        f"rollouts_per_example = {args.rollouts_per_prompt}\n\n"
        "[model]\n"
        f'name = "{args.model}"\n\n'
        "[sampling]\n"
        f"max_tokens = {args.max_tokens}\n"
        f"temperature = {args.temperature}\n"
    )


def inference_stub(args: argparse.Namespace) -> str:
    eager = "true" if args.enforce_eager else "false"
    return (
        "# Generated PRIME-RL inference stub.\n\n"
        "[model]\n"
        f'name = "{args.model}"\n'
        f"enforce_eager = {eager}\n"
    )


def combined_stub(args: argparse.Namespace) -> str:
    return (
        "# Generated PRIME-RL combined stub.\n"
        "# Replace this with a real config produced by `uv run vf-setup` when possible.\n\n"
        f'# environment_name = "{args.environment_name}"\n'
        f"max_steps = {args.max_steps}\n\n"
        "[model]\n"
        f'name = "{args.model}"\n'
        f"seq_len = {args.seq_len}\n\n"
        "[sampling]\n"
        f"max_tokens = {args.max_tokens}\n"
        f"temperature = {args.temperature}\n"
        f"top_p = {args.top_p}\n"
        f"n = {args.rollouts_per_prompt}\n"
    )


def latest_checkpoint_step(checkpoints_dir: Path) -> int | None:
    latest: int | None = None
    for step_dir in checkpoints_dir.glob("step_*"):
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        latest = step if latest is None else max(latest, step)
    return latest


def parse_prime_extra_args(raw_args: str) -> PrimeLaunchOverrides:
    tokens = shlex.split(raw_args)
    passthrough: list[str] = []
    inference_gpu_ids: list[str] = []
    trainer_gpu_ids: list[str] = []
    inference_dp: int | None = None
    inference_tp: int | None = None
    inference_gpu_memory_utilization: float | None = None
    inference_server_port: int | None = None
    orchestrator_client_base_url: str | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        next_token = tokens[idx + 1] if idx + 1 < len(tokens) else None

        def read_value() -> tuple[str | None, int]:
            if "=" in token:
                return token.split("=", 1)[1], 1
            return next_token, 2 if next_token is not None else 1

        if token.startswith("--inference-gpu-ids"):
            value, consumed = read_value()
            inference_gpu_ids = [item for item in (value or "").split(",") if item]
            idx += consumed
            continue
        if token.startswith("--trainer-gpu-ids"):
            value, consumed = read_value()
            trainer_gpu_ids = [item for item in (value or "").split(",") if item]
            idx += consumed
            continue
        if token.startswith("--inference.parallel.dp"):
            value, consumed = read_value()
            inference_dp = int(value) if value is not None else None
            idx += consumed
            continue
        if token.startswith("--inference.parallel.tp"):
            value, consumed = read_value()
            inference_tp = int(value) if value is not None else None
            idx += consumed
            continue
        if token.startswith("--inference.gpu-memory-utilization"):
            value, consumed = read_value()
            inference_gpu_memory_utilization = float(value) if value is not None else None
            idx += consumed
            continue
        if token.startswith("--inference.server.port"):
            value, consumed = read_value()
            inference_server_port = int(value) if value is not None else None
            idx += consumed
            continue
        if token.startswith("--orchestrator.client.base-url"):
            value, consumed = read_value()
            orchestrator_client_base_url = value
            idx += consumed
            continue

        passthrough.append(token)
        idx += 1

    return PrimeLaunchOverrides(
        inference_gpu_ids=inference_gpu_ids,
        trainer_gpu_ids=trainer_gpu_ids,
        inference_dp=inference_dp,
        inference_tp=inference_tp,
        inference_gpu_memory_utilization=inference_gpu_memory_utilization,
        inference_server_port=inference_server_port,
        orchestrator_client_base_url=orchestrator_client_base_url,
        passthrough_args=passthrough,
    )


def prime_run_prefix(prime_command: str) -> list[str] | None:
    tokens = shlex.split(prime_command)
    if not tokens:
        return None
    if "run" in tokens:
        run_index = len(tokens) - 1 - tokens[::-1].index("run")
        return tokens[: run_index + 1]
    if tokens[-1] == "rl":
        return tokens[:-1]
    return None


def prime_project_root(prime_command: str) -> Path | None:
    tokens = shlex.split(prime_command)
    for idx, token in enumerate(tokens):
        if token == "--project" and idx + 1 < len(tokens):
            return Path(tokens[idx + 1]).expanduser().resolve()
        if token.startswith("--project="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    return None


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_launch_script(
    args: argparse.Namespace,
    paths: RunPaths,
    use_combined: bool,
    resume_step: int | None,
) -> str:
    vllm_config_root = paths.run_dir / "vllm"
    scratch_root = paths.hf_home.parent
    cache_root = scratch_root / ".cache"
    combined_config_path = paths.configs_dir / "prime_rl.toml"
    trainer_config_path = paths.configs_dir / "trainer.toml"
    orchestrator_config_path = paths.configs_dir / "orchestrator.toml"
    inference_config_path = paths.configs_dir / "inference.toml"
    base_exports = [
        "set -euo pipefail",
        f'export HF_HOME="{paths.hf_home}"',
        'export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"',
        'export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"',
        'export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"',
        'export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"',
        f'export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-{cache_root}}}"',
        'export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"',
        'export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"',
        'export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"',
        'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"',
        'export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"',
        'export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$XDG_CACHE_HOME/pycache}"',
        'export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"',
        f'export TMPDIR="${{TMPDIR:-{scratch_root / "tmp"}}}"',
        'export TMP="${TMP:-$TMPDIR}"',
        'export TEMP="${TEMP:-$TMPDIR}"',
        f'export WANDB_DIR="{paths.wandb_dir}"',
        f'export WANDB_CACHE_DIR="{paths.wandb_cache_dir}"',
        f'export VLLM_CONFIG_ROOT="{vllm_config_root}"',
        f'export PRIME_RUN_DIR="{paths.run_dir}"',
        f'export PRIME_ENV_NAME="{args.environment_name}"',
        f'export PRIME_CHECKPOINT_DIR="{paths.checkpoints_dir}"',
        f'export PRIME_METRICS_DIR="{paths.metrics_dir}"',
        f'export WANDB_PROJECT="{args.wandb_project}"',
        'export WANDB_MODE="' + args.wandb_mode + '"',
        'export PYTHONUNBUFFERED="1"',
        'mkdir -p '
        f'"{paths.logs_dir}" "{paths.metrics_dir}" "{paths.checkpoints_dir}" '
        f'"{paths.wandb_dir}" "{paths.wandb_cache_dir}" "{paths.hf_home}" '
        '"$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" '
        '"$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" '
        '"$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" '
        '"$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" '
        '"$TMPDIR"',
    ]
    if args.wandb_entity:
        base_exports.append(f'export WANDB_ENTITY="{args.wandb_entity}"')
    extra_args = [
        "--model.name",
        str(args.model),
        "--output-dir",
        str(paths.run_dir),
        "--max-steps",
        str(args.max_steps),
        "--wandb.project",
        args.wandb_project,
        "--wandb.name",
        args.run_name,
        "--ckpt",
        "--ckpt.interval",
        str(args.ckpt_interval),
        "--ckpt.keep-last",
        str(args.ckpt_keep_last),
        "--ckpt.keep-interval",
        str(args.ckpt_keep_interval),
    ]
    overrides = parse_prime_extra_args(args.prime_extra_args)
    if overrides.passthrough_args:
        extra_args.extend(overrides.passthrough_args)
    if resume_step is not None:
        extra_args.extend(["--ckpt.resume-step", str(resume_step)])

    if overrides.inference_gpu_ids or overrides.trainer_gpu_ids:
        run_prefix = prime_run_prefix(args.prime_command)
        project_root = prime_project_root(args.prime_command)
        if run_prefix is None:
            raise ValueError(
                f"Unable to derive PRIME subcommands from prime command: {args.prime_command}"
            )

        inference_gpu_ids = overrides.inference_gpu_ids or ["0"]
        trainer_gpu_ids = overrides.trainer_gpu_ids or ["1"]
        inference_port = overrides.inference_server_port or 8000
        client_base_url = (
            overrides.orchestrator_client_base_url or f"http://127.0.0.1:{inference_port}/v1"
        )

        inference_tp = overrides.inference_tp
        inference_dp = overrides.inference_dp
        if inference_tp is None and inference_dp is None:
            inference_tp = max(1, len(inference_gpu_ids))
        if inference_tp is None:
            inference_tp = 1
        if inference_dp is None:
            inference_dp = 1

        inference_cmd = run_prefix + ["inference", "@", str(inference_config_path)]
        inference_cmd += ["--model.name", str(args.model)]
        inference_cmd += ["--server.port", str(inference_port)]
        if overrides.inference_gpu_memory_utilization is not None:
            inference_cmd += [
                "--gpu-memory-utilization",
                str(overrides.inference_gpu_memory_utilization),
            ]
        if inference_tp > 1:
            inference_cmd += ["--parallel.tp", str(inference_tp)]
        if inference_dp > 1:
            inference_cmd += ["--parallel.dp", str(inference_dp)]
            inference_cmd += ["--data-parallel-size-local", str(inference_dp)]

        orchestrator_cmd = run_prefix + ["orchestrator", "@", str(orchestrator_config_path)]
        orchestrator_cmd += ["--model.name", str(args.model)]
        orchestrator_cmd += ["--output-dir", str(paths.run_dir)]
        orchestrator_cmd += ["--client.base-url", client_base_url]

        trainer_base = [
            "--output-dir",
            str(paths.run_dir),
            "--wandb.project",
            args.wandb_project,
            "--wandb.name",
            args.run_name,
            "--ckpt",
            "--ckpt.interval",
            str(args.ckpt_interval),
            "--ckpt.keep-last",
            str(args.ckpt_keep_last),
            "--ckpt.keep-interval",
            str(args.ckpt_keep_interval),
        ]
        if resume_step is not None:
            trainer_base += ["--ckpt.resume-step", str(resume_step)]

        if len(trainer_gpu_ids) > 1:
            trainer_entrypoint = "src/prime_rl/trainer/rl/train.py"
            if project_root is not None:
                trainer_entrypoint = str(project_root / trainer_entrypoint)
            trainer_cmd = run_prefix + [
                "torchrun",
                "--nproc-per-node",
                str(len(trainer_gpu_ids)),
                trainer_entrypoint,
                "@",
                str(trainer_config_path),
            ]
        else:
            trainer_cmd = run_prefix + ["trainer", "@", str(trainer_config_path)]
        trainer_cmd += ["--model.name", str(args.model)]
        trainer_cmd += trainer_base

        inference_log = paths.logs_dir / "inference.log"
        orchestrator_log = paths.logs_dir / "orchestrator.log"
        trainer_log = paths.logs_dir / "trainer.log"
        return """#!/usr/bin/env bash
set -euo pipefail
{exports}

INFERENCE_READY_TIMEOUT_SECS="${{PRIME_INFERENCE_READY_TIMEOUT_SECS:-900}}"
INFERENCE_READY_POLL_SECS="${{PRIME_INFERENCE_READY_POLL_SECS:-2}}"

cleanup() {{
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "${{INFERENCE_PID:-}}" ]]; then kill "${{INFERENCE_PID}}" >/dev/null 2>&1 || true; fi
  if [[ -n "${{ORCHESTRATOR_PID:-}}" ]]; then kill "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1 || true; fi
  wait "${{INFERENCE_PID:-}}" >/dev/null 2>&1 || true
  wait "${{ORCHESTRATOR_PID:-}}" >/dev/null 2>&1 || true
  exit "$code"
}}
trap cleanup EXIT INT TERM

echo "Starting PRIME inference on GPUs {inference_gpu_ids}"
echo "Inference log: {inference_log}"
(export CUDA_VISIBLE_DEVICES="{inference_gpu_ids}"; exec {inference_cmd}) > {inference_log} 2>&1 &
INFERENCE_PID=$!

ready=0
attempts=$(( (INFERENCE_READY_TIMEOUT_SECS + INFERENCE_READY_POLL_SECS - 1) / INFERENCE_READY_POLL_SECS ))
for _ in $(seq 1 "$attempts"); do
  if curl -fsS http://127.0.0.1:{inference_port}/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep "$INFERENCE_READY_POLL_SECS"
done
if [[ "$ready" -ne 1 ]]; then
  echo "Inference server did not become ready; tail follows:" >&2
  tail -n 200 {inference_log} >&2 || true
  exit 1
fi

echo "Starting PRIME orchestrator"
({orchestrator_cmd}) > {orchestrator_log} 2>&1 &
ORCHESTRATOR_PID=$!

echo "Starting PRIME trainer on GPUs {trainer_gpu_ids}"
(export CUDA_VISIBLE_DEVICES="{trainer_gpu_ids}"; exec {trainer_cmd}) 2>&1 | tee {trainer_log}
""".format(
            exports="\n".join(base_exports),
            inference_gpu_ids=",".join(inference_gpu_ids),
            trainer_gpu_ids=",".join(trainer_gpu_ids),
            inference_cmd=shell_join(inference_cmd),
            orchestrator_cmd=shell_join(orchestrator_cmd),
            trainer_cmd=shell_join(trainer_cmd),
            inference_log=shlex.quote(str(inference_log)),
            orchestrator_log=shlex.quote(str(orchestrator_log)),
            trainer_log=shlex.quote(str(trainer_log)),
            inference_port=inference_port,
        )

    quoted_extra = " ".join(shlex.quote(item) for item in extra_args)
    if use_combined:
        command = (
            f'{args.prime_command} @ "{combined_config_path}" '
            f"{quoted_extra} " + '"$@"'
        )
    else:
        command = (
            f'{args.prime_command} '
            f'--trainer @ "{trainer_config_path}" '
            f'--orchestrator @ "{orchestrator_config_path}" '
            f'--inference @ "{inference_config_path}" '
            f"{quoted_extra} " + '"$@"'
        )
    return "#!/usr/bin/env bash\n" + "\n".join(base_exports) + "\n\n" + command + "\n"


def prime_run(args: argparse.Namespace) -> int:
    run_name = args.run_name or default_run_name("prime", dataset=args.environment_name, model=args.model)
    args.run_name = run_name
    output_root = expand(args.output_root)
    hf_home = expand(args.hf_home)
    paths = build_run_paths(output_root=output_root, run_name=run_name, hf_home=hf_home)
    ensure_dirs(paths)

    resume_step = args.resume_step
    if resume_step is None and not args.no_auto_resume:
        resume_step = latest_checkpoint_step(paths.checkpoints_dir)
    use_combined = bool(args.combined_config)
    if use_combined:
        copy_or_write_config(
            source=args.combined_config,
            destination=paths.configs_dir / "prime_rl.toml",
            fallback_content=combined_stub(args),
        )
    else:
        copy_or_write_config(args.trainer_config, paths.configs_dir / "trainer.toml", trainer_stub(args))
        copy_or_write_config(
            args.orchestrator_config,
            paths.configs_dir / "orchestrator.toml",
            orchestrator_stub(args),
        )
        copy_or_write_config(
            args.inference_config,
            paths.configs_dir / "inference.toml",
            inference_stub(args),
        )

    manifest = {
        "command": "prime",
        "args": vars(args),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "mode": "combined_config" if use_combined else "split_configs",
        "resume_step": resume_step,
        "checkpointing": {
            "enabled": True,
            "interval": args.ckpt_interval,
            "keep_last": args.ckpt_keep_last,
            "keep_interval": args.ckpt_keep_interval,
        },
        "notes": [
            "If you have working PRIME-RL configs from `uv run vf-setup`, pass them explicitly.",
            "The generated stubs are suitable as starting points, not guaranteed production configs.",
            "Metrics, logs, checkpoints, and WandB local state are all directed into scratch.",
        ],
    }
    write_json(paths.run_dir / "manifest.json", manifest)

    launch_script = paths.run_dir / "launch_prime_rl.sh"
    launch_script.write_text(
        build_launch_script(
            args,
            paths,
            use_combined=use_combined,
            resume_step=resume_step,
        )
    )
    launch_script.chmod(0o755)

    print(f"Prepared PRIME-RL bundle: {paths.run_dir}")
    print(f"Launch script: {launch_script}")
    if resume_step is not None:
        print(f"Resume step: {resume_step}")
    if args.execute:
        print("Executing launch script...")
        try:
            subprocess.run([str(launch_script)], check=True)
        except subprocess.CalledProcessError:
            for log_path in prime_failure_log_paths(paths):
                print(f"\n===== {log_path} =====", file=sys.stderr)
                print(tail_text(log_path), file=sys.stderr)
            raise
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "baseline":
        return baseline_run(args)
    if args.command == "prime":
        return prime_run(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
