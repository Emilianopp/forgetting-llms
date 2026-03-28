#!/usr/bin/env python3
"""Run safety-eval with a patched native vLLM loader.

The upstream safety-eval CLI does not expose vLLM's gpu_memory_utilization.
This wrapper imports the safety-eval modules directly and monkeypatches the
vLLM constructor that safety-eval uses so the repo can run safety benchmarks
without immediately exhausting the eval GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety-root", required=True, help="Path to the safety-eval checkout.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-input-template-path-or-name", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--report-output-path", required=True)
    parser.add_argument("--save-individual-results-path", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in the range (0, 1]")

    safety_root = Path(args.safety_root).expanduser().resolve()
    if not (safety_root / "evaluation" / "eval.py").exists():
        raise SystemExit(f"Expected safety-eval entrypoint under {safety_root}/evaluation/eval.py")

    os.chdir(safety_root)
    sys.path.insert(0, str(safety_root))
    sys.path.insert(0, str(safety_root / "evaluation"))
    sys.path.insert(0, str(safety_root / "src"))

    import vllm

    original_llm = vllm.LLM

    def patched_llm(*llm_args, **llm_kwargs):
        llm_kwargs.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
        return original_llm(*llm_args, **llm_kwargs)

    vllm.LLM = patched_llm

    try:
        import src.generation_utils as generation_utils

        if hasattr(generation_utils, "vllm") and hasattr(generation_utils.vllm, "LLM"):
            generation_utils.vllm.LLM = patched_llm
    except Exception:
        pass

    from evaluation.eval import generators

    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    report_path = Path(args.report_output_path).expanduser()
    individual_path = Path(args.save_individual_results_path).expanduser()
    task_root = individual_path.with_suffix("")
    task_root.mkdir(parents=True, exist_ok=True)

    aggregate_metrics: dict[str, object] = {"per_task": {}, "skipped_subtasks": []}
    aggregate_individual: dict[str, object] = {"per_task": {}, "skipped_subtasks": []}

    def is_wildguard_gated_error(exc: BaseException) -> bool:
        payload = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).lower()
        return "allenai/wildguard" in payload and "gated" in payload

    def safe_task_slug(task: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in task)

    for task in requested_tasks:
        task_slug = safe_task_slug(task)
        task_dir = task_root / task_slug
        task_dir.mkdir(parents=True, exist_ok=True)
        task_metrics_path = task_dir / "metrics.json"
        task_individual_path = task_dir / "all.json"
        try:
            generators(
                model_name_or_path=args.model_name_or_path,
                tasks=task,
                model_input_template_path_or_name=args.model_input_template_path_or_name,
                report_output_path=str(task_metrics_path),
                save_individual_results_path=str(task_individual_path),
                use_vllm=True,
                batch_size=args.batch_size,
                hf_revision=args.hf_revision,
                hparam_overrides=json.loads("{}"),
            )
        except Exception as exc:
            if is_wildguard_gated_error(exc):
                skipped = {
                    "task": task,
                    "reason": "requires gated model allenai/wildguard",
                }
                print(
                    "Skipping safety subtask "
                    f"{task}: requires gated model allenai/wildguard",
                    file=sys.stderr,
                )
                aggregate_metrics["skipped_subtasks"].append(skipped)
                aggregate_individual["skipped_subtasks"].append(skipped)
                continue
            raise

        task_metrics_payload: object = {}
        task_individual_payload: object = {}
        if task_metrics_path.exists():
            task_metrics_payload = json.loads(task_metrics_path.read_text())
        if task_individual_path.exists():
            task_individual_payload = json.loads(task_individual_path.read_text())
        aggregate_metrics["per_task"][task] = task_metrics_payload
        aggregate_individual["per_task"][task] = task_individual_payload

    report_path.parent.mkdir(parents=True, exist_ok=True)
    individual_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(aggregate_metrics, indent=2, sort_keys=True))
    individual_path.write_text(json.dumps(aggregate_individual, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
