#!/usr/bin/env python3
"""Unified evaluation entry point for forgetting-llms.

Supports two workflows:
1. Evaluate a single Hugging Face model path or checkpoint.
2. Sweep a training run directory containing ``global_step_*`` checkpoints.

The default forgetting suite runs through lm-eval. The broader ``tasks_md``
suite is wired to concrete external backends so the repo has a single place
that launches GPQA/SuperGPQA/AIME/SimpleQA, code benchmarks, function-calling evals,
and safety evals.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import time
from functools import lru_cache
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse
import urllib.error
import urllib.request


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    runner: str
    task: str | None = None
    fewshot: int = 0
    notes: str = ""
    display_name: str | None = None
    example_count: int | None = None
    example_count_note: str | None = None
    lighteval_max_new_tokens: int | None = None


@dataclass(frozen=True)
class EvalTarget:
    label: str
    model_ref: str
    checkpoint_dir: Path | None = None


BENCHMARK_REGISTRY: dict[str, dict[str, object]] = {
    "forgetting": {
        "description": "Current 10-benchmark OOD forgetting suite used in the repo.",
        "benchmarks": [
            BenchmarkSpec("arc_challenge", runner="lm_eval", task="arc_challenge"),
            BenchmarkSpec("arc_easy", runner="lm_eval", task="arc_easy"),
            BenchmarkSpec("hellaswag", runner="lm_eval", task="hellaswag"),
            BenchmarkSpec("winogrande", runner="lm_eval", task="winogrande"),
            BenchmarkSpec("piqa", runner="lm_eval", task="piqa"),
            BenchmarkSpec("boolq", runner="lm_eval", task="boolq"),
            BenchmarkSpec("openbookqa", runner="lm_eval", task="openbookqa"),
            BenchmarkSpec("truthfulqa_mc2", runner="lm_eval", task="truthfulqa_mc2"),
            BenchmarkSpec("mmlu", runner="lm_eval", task="mmlu"),
            BenchmarkSpec("ifeval", runner="lm_eval", task="ifeval"),
        ],
    },
    "tasks_md": {
        "description": (
            "Research checklist mirrored from tasks.md, wired to concrete "
            "external runners or explicit command hooks."
        ),
        "benchmarks": [
            BenchmarkSpec(
                "gpqa",
                runner="lighteval",
                task="gpqa:diamond",
                display_name="GPQA Diamond",
                example_count=198,
                lighteval_max_new_tokens=256,
            ),
            BenchmarkSpec(
                "supergpqa",
                runner="supergpqa",
                task="m-a-p/SuperGPQA",
                display_name="SuperGPQA",
                example_count=26529,
                example_count_note="train split",
            ),
            BenchmarkSpec(
                "rg_mix",
                runner="rg_mix",
                display_name="RG-mix",
                notes="Project-local reasoning-gym mix benchmark evaluated via the served checkpoint endpoint.",
                example_count=2048,
            ),
            BenchmarkSpec(
                "aime",
                runner="lighteval",
                task="aime24",
                display_name="AIME 2024",
                example_count=30,
                lighteval_max_new_tokens=512,
            ),
            BenchmarkSpec(
                "livecodebench_v6",
                runner="lighteval",
                task="lcb:codegeneration",
                display_name="LiveCodeBench v6",
                example_count=1055,
                notes="Uses LightEval's LiveCodeBench task. Override via --benchmark-command if you need the standalone LCB repo flow.",
                lighteval_max_new_tokens=512,
            ),
            BenchmarkSpec(
                "simpleqa",
                runner="lighteval",
                task="simpleqa",
                display_name="SimpleQA",
                example_count=4321,
                example_count_note="test split",
                lighteval_max_new_tokens=128,
            ),
            BenchmarkSpec(
                "humaneval_plus",
                runner="evalplus",
                task="humaneval",
                display_name="HumanEval+",
                example_count=164,
            ),
            BenchmarkSpec(
                "mbpp_plus",
                runner="evalplus",
                task="mbpp",
                display_name="MBPP+",
                example_count=378,
            ),
            BenchmarkSpec(
                "litqa2",
                runner="labbench",
                task="LitQA2",
                display_name="LitQA2",
                example_count=248,
            ),
            BenchmarkSpec(
                "bfcl",
                runner="bfcl",
                task="all",
                display_name="BFCL",
                example_count=2000,
                example_count_note="BFCL all-category bundle",
            ),
            BenchmarkSpec(
                "safety",
                runner="safety_eval",
                task="wildguardtest,harmbench,xstest,toxigen:tiny",
                display_name="Safety",
                example_count_note="composite runner bundle: wildguardtest + harmbench + xstest + toxigen:tiny",
            ),
        ],
    },
    "tasks_md_core": {
        "description": (
            "Subset of tasks_md that excludes the heaviest external benchmark "
            "dependencies (RG-mix, BFCL, Safety)."
        ),
        "benchmarks": [
            BenchmarkSpec(
                "supergpqa",
                runner="supergpqa",
                task="m-a-p/SuperGPQA",
                display_name="SuperGPQA",
                example_count=26529,
                example_count_note="train split",
            ),
            BenchmarkSpec(
                "aime",
                runner="lighteval",
                task="aime24",
                display_name="AIME 2024",
                example_count=30,
                lighteval_max_new_tokens=512,
            ),
            BenchmarkSpec(
                "livecodebench_v6",
                runner="lighteval",
                task="lcb:codegeneration",
                display_name="LiveCodeBench v6",
                example_count=1055,
                notes="Uses LightEval's LiveCodeBench task. Override via --benchmark-command if you need the standalone LCB repo flow.",
                lighteval_max_new_tokens=512,
            ),
            BenchmarkSpec(
                "simpleqa",
                runner="lighteval",
                task="simpleqa",
                display_name="SimpleQA",
                example_count=4321,
                example_count_note="test split",
                lighteval_max_new_tokens=128,
            ),
            BenchmarkSpec(
                "humaneval_plus",
                runner="evalplus",
                task="humaneval",
                display_name="HumanEval+",
                example_count=164,
            ),
            BenchmarkSpec(
                "mbpp_plus",
                runner="evalplus",
                task="mbpp",
                display_name="MBPP+",
                example_count=378,
            ),
            BenchmarkSpec(
                "litqa2",
                runner="labbench",
                task="LitQA2",
                display_name="LitQA2",
                example_count=248,
            ),
        ],
    },
}

PRIMARY_METRIC_HINTS = (
    "pass@k",
    "pass@1",
    "pass_at_k",
    "pass_at_1",
    "accuracy_at_1",
    "accuracy",
    "acc_norm,none",
    "acc,none",
    "prompt_level_strict_acc,none",
    "exact_match",
    "qem",
    "em",
    "score",
    "overall",
)

PRIMARY_METRIC_IGNORE_HINTS = (
    "config_general",
    "config_tasks",
    "summary_general",
    "summary_tasks",
    "versions/",
    "/versions",
    "hashes",
    "stderr",
    "num_fewshot",
    "max_samples",
    "start_time",
    "end_time",
    "total_evaluation_time",
    "padded",
    "non_padded",
    "effective_few_shots",
    "effective_num_docs",
    "original_num_docs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        type=str,
        help=(
            "HF model id, merged checkpoint path, or a training run directory "
            "containing global_step_* checkpoints."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Suite name to run. Repeatable. Use 'all' to expand all suites.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional label stored in the summary metadata.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model to evaluate alongside checkpoints when model_path is a sweep dir.",
    )
    parser.add_argument(
        "--benchmark-task",
        action="append",
        default=[],
        metavar="NAME=TASK_ID",
        help=(
            "Override a benchmark task identifier. Works for lm-eval, LightEval, "
            "EvalPlus, safety bundles, or other task-based runners."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional root path for external benchmark repos. Repeatable, for example "
            "--benchmark-root safety=/path/to/safety-eval."
        ),
    )
    parser.add_argument(
        "--benchmark-command",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help=(
            "Custom command template for a benchmark. Supports placeholders like "
            "{model}, {benchmark_dir}, {served_model}, {target_label}. Repeatable."
        ),
    )
    parser.add_argument(
        "--include-runner",
        action="append",
        default=[],
        metavar="RUNNER",
        help=(
            "Restrict execution to benchmarks using the named runner. Repeatable, "
            "for example --include-runner lighteval."
        ),
    )
    parser.add_argument(
        "--include-benchmark",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Restrict execution to named benchmarks only. Repeatable, for example "
            "--include-benchmark gpqa."
        ),
    )
    parser.add_argument(
        "--lm_eval_model",
        type=str,
        default="hf",
        help="lm-eval model backend (default: hf).",
    )
    parser.add_argument(
        "--lm_eval_model_args",
        type=str,
        default="",
        help="Extra lm-eval model_args appended after pretrained=...",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="lm-eval batch size (default: auto).",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Pass --log_samples to lm_eval.",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Pass --apply_chat_template to lm_eval.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=os.environ.get("OPENAI_MODEL_NAME") or os.environ.get("SERVED_MODEL_NAME"),
        help=(
            "Model name exposed by an OpenAI-compatible endpoint. Used by EvalPlus, "
            "LAB-Bench, BFCL, and custom-command hooks when they target a served model."
        ),
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        help="OpenAI-compatible base URL for endpoint-backed benchmark runners.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="OpenAI-compatible API key for endpoint-backed benchmark runners.",
    )
    parser.add_argument(
        "--lighteval-max-model-length",
        type=int,
        default=None,
        help="Optional max model length passed to LightEval's vLLM backend.",
    )
    parser.add_argument(
        "--lighteval-max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens passed to LightEval's vLLM backend.",
    )
    parser.add_argument(
        "--lighteval-num-samples",
        type=int,
        default=int(os.environ.get("LIGHTEVAL_NUM_SAMPLES", "512")),
        help=(
            "Number of stochastic generations per prompt for LightEval generative "
            "benchmarks. This controls the effective pass@k sample count."
        ),
    )
    parser.add_argument(
        "--lighteval-pass-k",
        type=int,
        default=int(os.environ.get("LIGHTEVAL_PASS_K", os.environ.get("LIGHTEVAL_NUM_SAMPLES", "512"))),
        help=(
            "Requested pass@k value for LightEval generative benchmarks. Must be <= "
            "--lighteval-num-samples."
        ),
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used for benchmark generation backends.",
    )
    parser.add_argument(
        "--sampling-top-p",
        type=float,
        default=1.0,
        help="Sampling top-p used for benchmark generation backends.",
    )
    parser.add_argument(
        "--evalplus-n-samples",
        type=int,
        default=int(os.environ.get("EVALPLUS_NUM_SAMPLES", "512")),
        help=(
            "Number of stochastic samples per EvalPlus task. Keep this above 1 "
            "if you do not want EvalPlus to collapse its effective batch size to 1."
        ),
    )
    parser.add_argument(
        "--evalplus-backend",
        type=str,
        default=os.environ.get("EVALPLUS_BACKEND", "auto"),
        choices=("auto", "vllm", "openai"),
        help=(
            "EvalPlus generation backend. In this repo, EvalPlus is forced onto "
            "the OpenAI-compatible endpoint path because native EvalPlus vllm has "
            "reproducibly crashed on Mila."
        ),
    )
    parser.add_argument(
        "--evalplus-batch-size",
        type=int,
        default=int(os.environ.get("EVALPLUS_BATCH_SIZE", "32")),
        help="Batch size passed to EvalPlus code generation.",
    )
    parser.add_argument(
        "--evalplus-pass-k",
        type=int,
        default=int(os.environ.get("EVALPLUS_PASS_K", os.environ.get("EVALPLUS_NUM_SAMPLES", "512"))),
        help=(
            "Requested pass@k summary for EvalPlus. Must be <= --evalplus-n-samples. "
            "This repo derives the requested pass@k from EvalPlus's detailed eval results."
        ),
    )
    parser.add_argument(
        "--lighteval-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization passed to LightEval's vLLM backend.",
    )
    parser.add_argument(
        "--lighteval-endpoint-max-concurrent-requests",
        type=int,
        default=1,
        help="Max concurrent requests passed to LightEval endpoint backends when supported.",
    )
    parser.add_argument(
        "--lighteval-endpoint-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds for endpoint-backed LightEval/LiteLLM calls.",
    )
    parser.add_argument(
        "--lighteval-endpoint-api-max-retry",
        type=int,
        default=2,
        help="Max retries for endpoint-backed LightEval/LiteLLM calls.",
    )
    parser.add_argument(
        "--no-lighteval-chat-template",
        action="store_true",
        help="Do not pass --use-chat-template to LightEval.",
    )
    parser.add_argument(
        "--allow-lighteval-empty-responses",
        action="store_true",
        help=(
            "Allow endpoint-backed LightEval benchmarks to complete even if LiteLLM "
            "reported empty-response request failures."
        ),
    )
    parser.add_argument(
        "--safety-template",
        type=str,
        default=os.environ.get("SAFETY_EVAL_TEMPLATE", "auto"),
        help=(
            "Template name/path passed to safety-eval. Use 'auto' to synthesize an "
            "instruction-compatible prompt template from the model tokenizer."
        ),
    )
    parser.add_argument(
        "--safety-batch-size",
        type=int,
        default=int(os.environ.get("SAFETY_BATCH_SIZE", os.environ.get("EVAL_SERVER_MAX_NUM_SEQS", "32"))),
        help="Batch size passed to safety-eval generation (default: SAFETY_BATCH_SIZE or EVAL_SERVER_MAX_NUM_SEQS or 32).",
    )
    parser.add_argument(
        "--safety-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("SAFETY_GPU_MEMORY_UTILIZATION", "0.7")),
        help="GPU memory utilization used by the native safety-eval vLLM loader (default: 0.7).",
    )
    parser.add_argument(
        "--bfcl-test-category",
        type=str,
        default="all",
        help="BFCL test category bundle (default: all).",
    )
    parser.add_argument(
        "--bfcl-model-name",
        type=str,
        default=os.environ.get("BFCL_MODEL_NAME"),
        help="BFCL model identifier override used for MODEL_CONFIG_MAPPING lookup.",
    )
    parser.add_argument(
        "--labbench-threads",
        type=int,
        default=20,
        help="Thread count for the LAB-Bench LitQA2 wrapper.",
    )
    parser.add_argument(
        "--supergpqa-split",
        type=str,
        default=os.environ.get("SUPERGPQA_SPLIT", "train"),
        help="Dataset split used for the native SuperGPQA runner (default: train).",
    )
    parser.add_argument(
        "--supergpqa-batch-size",
        type=int,
        default=int(os.environ.get("SUPERGPQA_BATCH_SIZE", "32")),
        help="Prompt batch size for the native SuperGPQA vLLM runner.",
    )
    parser.add_argument(
        "--supergpqa-max-new-tokens",
        type=int,
        default=int(os.environ.get("SUPERGPQA_MAX_NEW_TOKENS", "64")),
        help="Max new tokens for SuperGPQA generations.",
    )
    parser.add_argument(
        "--supergpqa-max-model-len",
        type=int,
        default=int(os.environ.get("SUPERGPQA_MAX_MODEL_LEN", "2048")),
        help="Max model length for the native SuperGPQA vLLM runner.",
    )
    parser.add_argument(
        "--supergpqa-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("SUPERGPQA_GPU_MEMORY_UTILIZATION", "0.9")),
        help="GPU memory utilization for the native SuperGPQA vLLM runner.",
    )
    parser.add_argument(
        "--supergpqa-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("SUPERGPQA_TENSOR_PARALLEL_SIZE", "1")),
        help="Tensor parallel size for the native SuperGPQA vLLM runner.",
    )
    parser.add_argument(
        "--supergpqa-max-samples",
        type=int,
        default=int(os.environ["SUPERGPQA_MAX_SAMPLES"]) if os.environ.get("SUPERGPQA_MAX_SAMPLES") else None,
        help="Optional cap on the number of SuperGPQA examples for smoke tests.",
    )
    parser.add_argument(
        "--supergpqa-num-samples",
        type=int,
        default=int(os.environ.get("SUPERGPQA_NUM_SAMPLES", "512")),
        help="Number of SuperGPQA generations per prompt.",
    )
    parser.add_argument(
        "--supergpqa-pass-k",
        type=int,
        default=int(os.environ.get("SUPERGPQA_PASS_K", os.environ.get("SUPERGPQA_NUM_SAMPLES", "512"))),
        help="Requested pass@k value for SuperGPQA. Must be <= --supergpqa-num-samples.",
    )
    parser.add_argument(
        "--supergpqa-samples-per-call",
        type=int,
        default=int(os.environ.get("SUPERGPQA_SAMPLES_PER_CALL", "16")),
        help=(
            "Number of SuperGPQA samples to request per vLLM call. The runner "
            "accumulates calls until --supergpqa-num-samples is reached."
        ),
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep evaluating remaining benchmarks if one benchmark fails.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Delete prior per-benchmark outputs under --output_dir and rerun all selected benchmarks.",
    )
    parser.add_argument(
        "--no_base_model",
        action="store_true",
        help="Do not evaluate base_model when model_path is a checkpoint sweep.",
    )
    parser.add_argument(
        "--list_suites",
        action="store_true",
        help="Print available suites and exit.",
    )
    args = parser.parse_args()

    if args.list_suites:
        return args

    if not args.model_path:
        parser.error("--model_path is required unless --list_suites is used")
    if not args.output_dir:
        parser.error("--output_dir is required unless --list_suites is used")
    if args.sampling_temperature <= 0:
        parser.error("--sampling-temperature must be > 0")
    if not 0 < args.sampling_top_p <= 1:
        parser.error("--sampling-top-p must be in the range (0, 1]")
    if args.lighteval_num_samples < 1:
        parser.error("--lighteval-num-samples must be >= 1")
    if args.lighteval_pass_k < 1:
        parser.error("--lighteval-pass-k must be >= 1")
    if args.lighteval_pass_k > args.lighteval_num_samples:
        parser.error("--lighteval-pass-k must be <= --lighteval-num-samples")
    if args.evalplus_n_samples < 1:
        parser.error("--evalplus-n-samples must be >= 1")
    if args.evalplus_batch_size < 1:
        parser.error("--evalplus-batch-size must be >= 1")
    if args.evalplus_pass_k < 1:
        parser.error("--evalplus-pass-k must be >= 1")
    if args.evalplus_pass_k > args.evalplus_n_samples:
        parser.error("--evalplus-pass-k must be <= --evalplus-n-samples")
    if args.safety_batch_size < 1:
        parser.error("--safety-batch-size must be >= 1")
    if not 0 < args.safety_gpu_memory_utilization <= 1:
        parser.error("--safety-gpu-memory-utilization must be in the range (0, 1]")
    if args.supergpqa_batch_size < 1:
        parser.error("--supergpqa-batch-size must be >= 1")
    if args.supergpqa_max_new_tokens < 1:
        parser.error("--supergpqa-max-new-tokens must be >= 1")
    if args.supergpqa_max_model_len < 1:
        parser.error("--supergpqa-max-model-len must be >= 1")
    if not 0 < args.supergpqa_gpu_memory_utilization <= 1:
        parser.error("--supergpqa-gpu-memory-utilization must be in the range (0, 1]")
    if args.supergpqa_tensor_parallel_size < 1:
        parser.error("--supergpqa-tensor-parallel-size must be >= 1")
    if args.supergpqa_max_samples is not None and args.supergpqa_max_samples < 1:
        parser.error("--supergpqa-max-samples must be >= 1")
    if args.supergpqa_num_samples < 1:
        parser.error("--supergpqa-num-samples must be >= 1")
    if args.supergpqa_pass_k < 1:
        parser.error("--supergpqa-pass-k must be >= 1")
    if args.supergpqa_pass_k > args.supergpqa_num_samples:
        parser.error("--supergpqa-pass-k must be <= --supergpqa-num-samples")
    if args.supergpqa_samples_per_call < 1:
        parser.error("--supergpqa-samples-per-call must be >= 1")
    if args.supergpqa_samples_per_call > args.supergpqa_num_samples:
        parser.error("--supergpqa-samples-per-call must be <= --supergpqa-num-samples")

    return args


def print_suites() -> None:
    for suite_name, suite_data in BENCHMARK_REGISTRY.items():
        print(f"{suite_name}: {suite_data['description']}")
        for spec in suite_data["benchmarks"]:
            suffix = f" -> {spec.task}" if spec.task else ""
            print(f"  - {spec.name} [{spec.runner}]{suffix}")


def expand_suites(selected: list[str] | None) -> list[str]:
    suite_names = selected or ["forgetting"]
    expanded: list[str] = []
    for suite_name in suite_names:
        if suite_name == "all":
            for known in BENCHMARK_REGISTRY:
                if known not in expanded:
                    expanded.append(known)
            continue
        if suite_name not in BENCHMARK_REGISTRY:
            raise ValueError(f"Unknown suite: {suite_name}")
        if suite_name not in expanded:
            expanded.append(suite_name)
    return expanded


def filter_registry(
    registry: dict[str, dict[str, object]],
    selected_suites: list[str],
    include_runners: list[str],
    include_benchmarks: list[str],
) -> dict[str, dict[str, object]]:
    if not include_runners and not include_benchmarks:
        return registry

    runner_set = {item.strip() for item in include_runners if item.strip()}
    benchmark_set = {item.strip() for item in include_benchmarks if item.strip()}
    filtered = deepcopy(registry)
    for suite_name in selected_suites:
        suite_data = filtered[suite_name]
        benchmarks = suite_data["benchmarks"]
        kept: list[BenchmarkSpec] = []
        for benchmark in benchmarks:
            if runner_set and benchmark.runner not in runner_set:
                continue
            if benchmark_set and benchmark.name not in benchmark_set:
                continue
            kept.append(benchmark)
        suite_data["benchmarks"] = kept
    return filtered


def parse_task_overrides(raw_overrides: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected NAME=TASK_ID.")
        name, task_id = item.split("=", 1)
        name = name.strip()
        task_id = task_id.strip()
        if not name or not task_id:
            raise ValueError(f"Invalid override '{item}'. Expected NAME=TASK_ID.")
        overrides[name] = task_id
    return overrides


def parse_kv_overrides(raw_overrides: list[str], label: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid {label} override '{item}'. Expected NAME=VALUE.")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid {label} override '{item}'. Expected NAME=VALUE.")
        overrides[name] = value
    return overrides


def get_registry_with_overrides(overrides: dict[str, str]) -> dict[str, dict[str, object]]:
    registry = deepcopy(BENCHMARK_REGISTRY)
    for suite_data in registry.values():
        updated_specs: list[BenchmarkSpec] = []
        for spec in suite_data["benchmarks"]:
            if spec.name in overrides:
                updated_specs.append(
                    BenchmarkSpec(
                        name=spec.name,
                        runner=spec.runner,
                        task=overrides[spec.name],
                        fewshot=spec.fewshot,
                        notes=spec.notes,
                        display_name=spec.display_name,
                        example_count=spec.example_count,
                        example_count_note=spec.example_count_note,
                        lighteval_max_new_tokens=spec.lighteval_max_new_tokens,
                    )
                )
            else:
                updated_specs.append(spec)
        suite_data["benchmarks"] = updated_specs
    return registry


def normalize_lighteval_task(task_id: str) -> str:
    parts = [part.strip() for part in task_id.split("|")]
    if len(parts) >= 2 and parts[0] in {"lighteval", "extended", "custom"}:
        return parts[1]
    return task_id.strip()


def parse_version_tuple(raw: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for part in raw.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def installed_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def lighteval_vllm_compatible() -> tuple[bool, str | None]:
    raw = installed_package_version("vllm")
    if raw is None:
        return False, None
    parsed = parse_version_tuple(raw)
    lower = parse_version_tuple("0.10.0")
    upper = parse_version_tuple("0.10.2")
    return lower <= parsed < upper, raw


def lighteval_litellm_compatible() -> tuple[bool, str | None]:
    raw = installed_package_version("litellm")
    if raw is None:
        return False, None
    parsed = parse_version_tuple(raw)
    lower = parse_version_tuple("1.66.0")
    return parsed >= lower, raw


def rg_mix_installed() -> bool:
    return installed_package_version("rg-mix-env") is not None or installed_package_version("rg_mix_env") is not None


def validate_benchmark_requirements(
    selected_suites: list[str],
    registry: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> None:
    errors: list[str] = []
    hf_home = Path(os.environ.get("HF_HOME", "~/scratch/huggingface")).expanduser()
    hf_token_path = Path(os.environ.get("HF_TOKEN_PATH", str(hf_home / "token"))).expanduser()
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    hf_auth_available = bool(hf_token) or hf_token_path.exists()

    for suite_name in selected_suites:
        suite_data = registry[suite_name]
        for benchmark in suite_data["benchmarks"]:
            if benchmark.runner == "custom_command":
                if not benchmark_command_override(benchmark.name, args.benchmark_command_map):
                    errors.append(
                        f"{suite_name}/{benchmark.name}: missing benchmark command override "
                        f"(set --benchmark-command {benchmark.name}='...' or {benchmark_env_key('BENCHMARK_COMMAND', benchmark.name)})"
                    )
            elif benchmark.runner == "evalplus":
                if not args.openai_base_url:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: requires --openai-base-url. "
                        "This repo disables native EvalPlus vllm because it crashes "
                        "during vLLM engine startup on Mila."
                    )
            elif benchmark.runner == "labbench":
                if not args.openai_base_url:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: requires --openai-base-url "
                        "for the endpoint-backed LAB-Bench runner"
                    )
            elif benchmark.runner == "rg_mix":
                if not args.openai_base_url:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: requires --openai-base-url "
                        "for the endpoint-backed RG-mix runner"
                    )
                root = benchmark_root("rg_mix", args.benchmark_root_map)
                installed_rg_mix = rg_mix_installed()
                if root is None and not installed_rg_mix:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: requires either an installed rg-mix-env verifier package "
                        "or --benchmark-root rg_mix=/path/to/rg_mix_env / RG_MIX_ROOT"
                    )
                elif root is not None and not root.exists() and not installed_rg_mix:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: RG-mix root does not exist: {root}"
                    )
                elif root is not None and root.exists() and not (root / "rg_mix_env.py").exists() and not installed_rg_mix:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: expected rg_mix_env.py under {root}"
                    )
            elif benchmark.runner == "bfcl":
                root = benchmark_root("bfcl", args.benchmark_root_map)
                if root is None:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: requires --benchmark-root bfcl=/path/to/berkeley-function-call-leaderboard or BFCL_ROOT "
                        "(or run bash scripts/setup_tasks_md_benchmarks.sh)"
                    )
                elif not root.exists():
                    errors.append(
                        f"{suite_name}/{benchmark.name}: BFCL root does not exist: {root} "
                        "(run bash scripts/setup_tasks_md_benchmarks.sh)"
                    )
                elif shutil.which("bfcl") is None:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: bfcl CLI not found in PATH "
                        "(run bash scripts/setup_tasks_md_benchmarks.sh inside your active env)"
                    )
            elif benchmark.runner == "safety_eval":
                root = benchmark_root("safety", args.benchmark_root_map)
                if root is None:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: requires --benchmark-root safety=/path/to/safety-eval or SAFETY_EVAL_ROOT "
                        "(or run bash scripts/setup_tasks_md_benchmarks.sh)"
                    )
                elif not root.exists():
                    errors.append(
                        f"{suite_name}/{benchmark.name}: safety-eval root does not exist: {root} "
                        "(run bash scripts/setup_tasks_md_benchmarks.sh)"
                    )
                elif not (root / "evaluation" / "eval.py").exists():
                    errors.append(
                        f"{suite_name}/{benchmark.name}: expected safety-eval entrypoint at {root / 'evaluation' / 'eval.py'} "
                        "(run bash scripts/setup_tasks_md_benchmarks.sh)"
                    )
            elif benchmark.runner == "lighteval":
                if shutil.which("lighteval") is None:
                    errors.append(f"{suite_name}/{benchmark.name}: lighteval CLI not found in PATH")
                if benchmark.name == "gpqa" and not hf_auth_available:
                    errors.append(
                        f"{suite_name}/{benchmark.name}: requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) "
                        "with access to the gated dataset Idavidrein/gpqa"
                    )
                if args.openai_base_url:
                    compatible_litellm, raw_litellm = lighteval_litellm_compatible()
                    if not compatible_litellm:
                        if raw_litellm is None:
                            errors.append(
                                f"{suite_name}/{benchmark.name}: endpoint-backed LightEval requires litellm[caching]>=1.66.0 "
                                "(install it in the active env)"
                            )
                        else:
                            errors.append(
                                f"{suite_name}/{benchmark.name}: installed litellm=={raw_litellm} is incompatible; "
                                "endpoint-backed LightEval requires litellm[caching]>=1.66.0"
                            )
                else:
                    compatible_vllm, raw_vllm = lighteval_vllm_compatible()
                    if not compatible_vllm:
                        if raw_vllm is None:
                            errors.append(
                                f"{suite_name}/{benchmark.name}: requires vllm>=0.10.0,<0.10.2 for the current LightEval vllm runner"
                            )
                        else:
                            errors.append(
                                f"{suite_name}/{benchmark.name}: installed vllm=={raw_vllm} is incompatible; "
                                "LightEval currently requires vllm>=0.10.0,<0.10.2 for this runner"
                            )

    if errors:
        message = "Benchmark suite is not fully runnable:\n  - " + "\n  - ".join(errors)
        raise RuntimeError(message)


def benchmark_env_key(prefix: str, benchmark_name: str) -> str:
    return f"{prefix}_{benchmark_name.upper()}"


def benchmark_root(
    benchmark_name: str,
    cli_overrides: dict[str, str],
) -> Path | None:
    raw_value = cli_overrides.get(benchmark_name) or os.environ.get(
        benchmark_env_key("BENCHMARK_ROOT", benchmark_name)
    )
    if not raw_value:
        special = {
            "safety": os.environ.get("SAFETY_EVAL_ROOT"),
            "bfcl": os.environ.get("BFCL_ROOT"),
            "rg_mix": os.environ.get("RG_MIX_ROOT"),
        }.get(benchmark_name)
        raw_value = special
    if not raw_value and benchmark_name == "rg_mix":
        repo_local = Path(__file__).resolve().parents[2] / "environments" / "rg_mix_env"
        if repo_local.exists():
            return repo_local
        sibling = Path(__file__).resolve().parents[3] / "contextual_learning2" / "environments" / "rg_mix_env"
        if sibling.exists():
            return sibling
    return Path(raw_value).expanduser() if raw_value else None


def resolve_hf_token(env: Mapping[str, str] | None = None) -> str | None:
    scope = env or os.environ
    token = (
        scope.get("HF_TOKEN")
        or scope.get("HUGGING_FACE_HUB_TOKEN")
        or scope.get("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        stripped = token.strip()
        return stripped or None
    hf_home = Path(scope.get("HF_HOME", "~/scratch/huggingface")).expanduser()
    hf_token_path = Path(scope.get("HF_TOKEN_PATH", str(hf_home / "token"))).expanduser()
    try:
        if hf_token_path.exists():
            stripped = hf_token_path.read_text().strip()
            return stripped or None
    except OSError:
        return None
    return None


def hf_model_access_check(repo_id: str, env: Mapping[str, str] | None = None) -> tuple[bool | None, str | None]:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except Exception:
        return None, None

    token = resolve_hf_token(env)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer_config.json",
            token=token,
        )
        return True, None
    except GatedRepoError:
        return False, f"missing accepted access to gated model {repo_id}"
    except RepositoryNotFoundError:
        return False, f"model {repo_id} is not accessible with the current token"
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            return False, f"HTTP {status} while accessing gated model {repo_id}"
        return None, None
    except Exception:
        return None, None


def resolve_safety_tasks(task_string: str, env: Mapping[str, str]) -> tuple[list[str], list[dict[str, str]]]:
    tasks = [task.strip() for task in task_string.split(",") if task.strip()]
    skipped: list[dict[str, str]] = []
    if "wildguardtest" in tasks:
        accessible, reason = hf_model_access_check("allenai/wildguard", env)
        if accessible is False:
            tasks = [task for task in tasks if task != "wildguardtest"]
            skipped.append(
                {
                    "task": "wildguardtest",
                    "reason": reason or "gated WildGuard model access unavailable",
                }
            )
    return tasks, skipped


def benchmark_command_override(
    benchmark_name: str,
    cli_overrides: dict[str, str],
) -> str | None:
    return cli_overrides.get(benchmark_name) or os.environ.get(
        benchmark_env_key("BENCHMARK_COMMAND", benchmark_name)
    )


def benchmark_shell_prefix(benchmark: BenchmarkSpec) -> str | None:
    benchmark_value = os.environ.get(benchmark_env_key("BENCHMARK_SHELL_PREFIX", benchmark.name))
    if benchmark_value:
        return benchmark_value.strip()
    runner_key = f"BENCHMARK_RUNNER_SHELL_PREFIX_{benchmark.runner.upper()}"
    runner_value = os.environ.get(runner_key)
    if runner_value:
        return runner_value.strip()
    return None


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def discover_targets(model_path: str, base_model: str | None, include_base_model: bool) -> list[EvalTarget]:
    expanded = Path(model_path).expanduser()
    if expanded.exists() and expanded.is_dir():
        merged_sft_dir = expanded / "merged_hf"
        if is_nonempty_dir(merged_sft_dir) and (merged_sft_dir / "config.json").exists():
            return [EvalTarget(label=merged_sft_dir.name, model_ref=str(merged_sft_dir))]
        checkpoints = sorted(expanded.glob("global_step_*"), key=checkpoint_step)
        if checkpoints:
            targets: list[EvalTarget] = []
            if include_base_model and base_model:
                targets.append(EvalTarget(label="base_model", model_ref=base_model))
            for checkpoint in checkpoints:
                targets.append(
                    EvalTarget(
                        label=checkpoint.name,
                        model_ref=str(checkpoint),
                        checkpoint_dir=checkpoint,
                    )
                )
            return targets

    label = expanded.name if expanded.name else Path(model_path).name
    if not label:
        label = "model"
    return [EvalTarget(label=label, model_ref=model_path)]


def is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def merge_checkpoint_if_needed(target: EvalTarget) -> tuple[str, Path | None]:
    if target.checkpoint_dir is None:
        return target.model_ref, None

    checkpoint_dir = target.checkpoint_dir
    actor_dir = checkpoint_dir / "actor"
    merged_dir: Path | None = None
    fsdp_dir: Path | None = None

    if actor_dir.is_dir():
        fsdp_dir = actor_dir
        merged_dir = checkpoint_dir / "actor_merged"
    elif list(checkpoint_dir.glob("model_world_size_*_rank_*.pt")):
        fsdp_dir = checkpoint_dir
        merged_dir = checkpoint_dir / "merged"
    elif (checkpoint_dir / "config.json").exists():
        return str(checkpoint_dir), None
    else:
        return str(checkpoint_dir), None

    if merged_dir is not None and is_nonempty_dir(merged_dir):
        return str(merged_dir), None

    if fsdp_dir is None or merged_dir is None:
        return str(checkpoint_dir), None

    raise RuntimeError(
        "Legacy VeRL/FSDP checkpoint detected without a merged export. "
        f"Expected an already-merged model under {merged_dir}. "
        "This repo's supported path is PRIME-RL only; evaluate a PRIME-exported "
        "checkpoint or a pre-merged HF checkpoint instead."
    )


def build_model_args(model_ref: str, extra_args: str) -> str:
    model_args = f"pretrained={model_ref},trust_remote_code=True,attn_implementation=sdpa"
    if extra_args.strip():
        model_args = f"{model_args},{extra_args.strip().lstrip(',')}"
    return model_args


def lighteval_max_new_tokens_for_benchmark(
    benchmark: BenchmarkSpec,
    args: argparse.Namespace,
) -> int:
    if benchmark.lighteval_max_new_tokens is not None:
        return benchmark.lighteval_max_new_tokens
    return args.lighteval_max_new_tokens


def build_lighteval_model_args(
    model_ref: str,
    benchmark: BenchmarkSpec,
    args: argparse.Namespace,
) -> str:
    max_new_tokens = lighteval_max_new_tokens_for_benchmark(benchmark, args)
    parts = [
        f"model_name={model_ref}",
        "dtype=bfloat16",
        f"gpu_memory_utilization={args.lighteval_gpu_memory_utilization}",
        (
            f"generation_parameters={{max_new_tokens:{max_new_tokens},"
            f"temperature:{args.sampling_temperature},top_p:{args.sampling_top_p}}}"
        ),
    ]
    if args.lighteval_max_model_length is not None:
        parts.insert(2, f"max_model_length={args.lighteval_max_model_length}")
    return ",".join(parts)


def build_lighteval_endpoint_model_args(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    args: argparse.Namespace,
) -> str:
    max_new_tokens = lighteval_max_new_tokens_for_benchmark(benchmark, args)
    return (
        f"provider=openai,"
        f"model_name={served_model_name(target, model_ref, args)},"
        f"base_url={args.openai_base_url},"
        f"api_key={args.openai_api_key},"
        f"api_max_retry={args.lighteval_endpoint_api_max_retry},"
        f"timeout={args.lighteval_endpoint_timeout},"
        f"generation_parameters={{max_new_tokens:{max_new_tokens},"
        f"temperature:{args.sampling_temperature},top_p:{args.sampling_top_p}}}"
    )


def sanitize_lighteval_task_name(task_id: str) -> str:
    sanitized = [
        ch if ch.isalnum() else "_"
        for ch in task_id.strip()
    ]
    collapsed = "".join(sanitized).strip("_")
    return collapsed or "task"


def lighteval_custom_task_name(task_id: str, num_samples: int) -> str:
    return f"{sanitize_lighteval_task_name(task_id)}__samples_{num_samples}"


def ensure_custom_lighteval_task_module(
    benchmark: BenchmarkSpec,
    benchmark_dir: Path,
    task_id: str,
    args: argparse.Namespace,
) -> tuple[Path | None, str]:
    if args.lighteval_num_samples == 1 and args.lighteval_pass_k == 1:
        return None, task_id

    custom_task_name = lighteval_custom_task_name(task_id, args.lighteval_num_samples)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    module_path = benchmark_dir / "_lighteval_custom_tasks.py"
    module_source = f"""# Auto-generated by src/evaluation/run_eval.py.
from copy import deepcopy
import re

from lighteval.tasks.registry import Registry

BASE_TASK_NAME = {task_id!r}
CUSTOM_TASK_NAME = {custom_task_name!r}
NUM_SAMPLES = {args.lighteval_num_samples}
PASS_K = {args.lighteval_pass_k}

_TASK_CONFIGS = Registry.load_all_task_configs(custom_tasks=None)
_BASE_CONFIG = deepcopy(_TASK_CONFIGS.get(BASE_TASK_NAME))
if _BASE_CONFIG is None:
    raise ValueError(f"Unknown LightEval task: {{BASE_TASK_NAME}}")

_BASE_CONFIG.name = CUSTOM_TASK_NAME
_BASE_CONFIG.num_samples = [NUM_SAMPLES]
_METRICS_ATTR = "metrics" if hasattr(_BASE_CONFIG, "metrics") else "metric"
_METRICS = list(getattr(_BASE_CONFIG, _METRICS_ATTR, []) or [])
for _metric in _METRICS:
    _metric_name = getattr(_metric, "metric_name", "")
    if "pass@k" not in _metric_name and "pass_at_k" not in _metric_name:
        continue
    _sample_level = getattr(_metric, "sample_level_fn", None)
    if _sample_level is not None:
        if hasattr(_sample_level, "k"):
            _sample_level.k = PASS_K
        if hasattr(_sample_level, "n"):
            _sample_level.n = NUM_SAMPLES
    if hasattr(_metric, "metric_name"):
        _updated_name = re.sub(r"k=\\d+", f"k={{PASS_K}}", _metric_name)
        _updated_name = re.sub(r"@\\d+", f"@{{PASS_K}}", _updated_name)
        _metric.metric_name = _updated_name
setattr(_BASE_CONFIG, _METRICS_ATTR, _METRICS)
TASKS_TABLE = [_BASE_CONFIG]
"""
    module_path.write_text(module_source)
    return module_path, custom_task_name


def benchmark_display_name(benchmark: BenchmarkSpec) -> str:
    return benchmark.display_name or benchmark.name


def benchmark_example_metadata(benchmark: BenchmarkSpec, args: argparse.Namespace) -> tuple[int | None, str | None]:
    if benchmark.name == "bfcl" and args.bfcl_test_category != "all":
        return None, f"BFCL category={args.bfcl_test_category}"
    if benchmark.name == "supergpqa" and args.supergpqa_max_samples is not None:
        note = f"{args.supergpqa_split} split smoke subset"
        return args.supergpqa_max_samples, note
    return benchmark.example_count, benchmark.example_count_note


def benchmark_metadata_payload(benchmark: BenchmarkSpec, args: argparse.Namespace) -> dict[str, object]:
    example_count, example_note = benchmark_example_metadata(benchmark, args)
    payload: dict[str, object] = {
        "display_name": benchmark_display_name(benchmark),
    }
    if example_count is not None:
        payload["expected_examples"] = example_count
    if example_note:
        payload["expected_examples_note"] = example_note
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def flatten_numeric_metrics(
    value: object,
    prefix: list[str],
    sink: dict[str, float],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten_numeric_metrics(item, prefix + [str(key)], sink)
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            flatten_numeric_metrics(item, prefix + [str(idx)], sink)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        sink["/".join(prefix)] = float(value)


def extract_metrics_from_benchmark_dir(
    benchmark_dir: Path,
    *,
    max_file_mb: float = 10.0,
) -> list[tuple[str, float, str]]:
    metrics: list[tuple[str, float, str]] = []
    max_bytes = int(max_file_mb * 1024 * 1024)
    for json_path in sorted(benchmark_dir.rglob("*.json")):
        if json_path.name in {"status.json", "eval_summary.json"}:
            continue
        try:
            if json_path.stat().st_size > max_bytes:
                continue
            payload = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        flattened: dict[str, float] = {}
        flatten_numeric_metrics(payload, [], flattened)
        source_file = str(json_path.relative_to(benchmark_dir))
        for metric_key, metric_value in flattened.items():
            metrics.append((metric_key, metric_value, source_file))
    return metrics


def select_primary_metric(metrics: list[tuple[str, float, str]]) -> tuple[str, float, str] | None:
    if not metrics:
        return None
    ranked: list[tuple[int, str, float, str]] = []
    fallback: list[tuple[str, float, str]] = []

    for metric_key, metric_value, metric_source in metrics:
        key = metric_key.lower()
        if any(ignore in key for ignore in PRIMARY_METRIC_IGNORE_HINTS):
            continue
        if "stderr" in key:
            continue

        fallback.append((metric_key, metric_value, metric_source))

        score = 0
        if key.startswith("results/"):
            score += 100
        if "/results/" in key:
            score += 80
        if key.startswith("metrics/") or "/metrics/" in key:
            score += 70
        if key.startswith("result/") or "/result/" in key:
            score += 60
        if key.startswith("results/all/") or "/all/" in key:
            score += 10

        for rank, hint in enumerate(PRIMARY_METRIC_HINTS):
            if key.endswith(hint) or f"/{hint}" in key or hint in key:
                score += 50 - rank
                break

        ranked.append((score, metric_key, metric_value, metric_source))

    if ranked:
        ranked.sort(key=lambda item: (-item[0], len(item[1]), item[1], item[3]))
        _, metric_key, metric_value, metric_source = ranked[0]
        return metric_key, metric_value, metric_source
    if fallback:
        return fallback[0]
    return metrics[0]


def benchmark_has_complete_artifacts(benchmark: BenchmarkSpec, benchmark_dir: Path) -> bool:
    expected_files: list[Path] = []
    if benchmark.runner == "rg_mix":
        expected_files = [benchmark_dir / "metrics.json", benchmark_dir / "all.json"]
    elif benchmark.runner == "labbench":
        expected_files = [benchmark_dir / "results.json"]
    elif benchmark.runner == "safety_eval":
        expected_files = [benchmark_dir / "metrics.json", benchmark_dir / "all.json"]

    if expected_files:
        return any(path.exists() for path in expected_files)

    return any(
        path.name not in {"status.json", "eval_summary.json"}
        for path in benchmark_dir.rglob("*.json")
    )


def summarize_benchmark_metrics(benchmark_dir: Path) -> tuple[str, float, str] | None:
    metrics = extract_metrics_from_benchmark_dir(benchmark_dir)
    return select_primary_metric(metrics)


def print_benchmark_metric_summary(benchmark_dir: Path) -> tuple[str, float, str] | None:
    primary_metric = summarize_benchmark_metrics(benchmark_dir)
    if primary_metric is None:
        print("  primary metric: not found")
        sys.stdout.flush()
        return None
    metric_name, metric_value, metric_source = primary_metric
    print(f"  primary metric: {metric_name}={metric_value:.4f} ({metric_source})")
    sys.stdout.flush()
    return primary_metric


def print_evaluation_metric_summary(summary: dict[str, object]) -> None:
    printed_header = False
    targets = summary.get("targets", {})
    if not isinstance(targets, dict):
        return

    for target_label, target_summary in targets.items():
        if not isinstance(target_summary, dict):
            continue
        for suite_name, suite_results in target_summary.items():
            if not isinstance(suite_results, dict):
                continue
            for benchmark_name, result in suite_results.items():
                if not isinstance(result, dict):
                    continue
                if result.get("status") != "complete":
                    continue
                metric_name = result.get("primary_metric_name")
                metric_value = result.get("primary_metric_value")
                if metric_name is None or metric_value is None:
                    continue
                if not printed_header:
                    print("Completed benchmark metrics:")
                    printed_header = True
                print(f"  {target_label}/{suite_name}/{benchmark_name}: {metric_name}={float(metric_value):.4f}")

    if printed_header:
        sys.stdout.flush()


def apply_scratch_runtime_defaults(env: dict[str, str]) -> dict[str, str]:
    scratch_root = Path(env.get("SCRATCH_ROOT", "~/scratch")).expanduser()
    cache_root = scratch_root / ".cache"
    project_root = scratch_root / "forgetting-llms"

    defaults = {
        "HF_HOME": scratch_root / "huggingface",
        "HF_DATASETS_CACHE": scratch_root / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": scratch_root / "huggingface" / "transformers",
        "HF_HUB_CACHE": scratch_root / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": scratch_root / "huggingface" / "hub",
        "XDG_CACHE_HOME": cache_root,
        "PIP_CACHE_DIR": cache_root / "pip",
        "UV_CACHE_DIR": cache_root / "uv",
        "TORCH_HOME": cache_root / "torch",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "TMPDIR": scratch_root / "tmp",
        "WANDB_DIR": project_root / "wandb",
        "WANDB_CACHE_DIR": project_root / "wandb_cache",
        "NLTK_DATA": project_root / "nltk_data",
    }
    for key, value in defaults.items():
        env.setdefault(key, str(value))

    env.setdefault("TMP", env["TMPDIR"])
    env.setdefault("TEMP", env["TMPDIR"])

    runtime_dirs = [
        env["HF_HOME"],
        env["HF_DATASETS_CACHE"],
        env["TRANSFORMERS_CACHE"],
        env["HF_HUB_CACHE"],
        env["HUGGINGFACE_HUB_CACHE"],
        env["XDG_CACHE_HOME"],
        env["PIP_CACHE_DIR"],
        env["UV_CACHE_DIR"],
        env["TORCH_HOME"],
        env["TRITON_CACHE_DIR"],
        env["PYTHONPYCACHEPREFIX"],
        env["MPLCONFIGDIR"],
        env["TMPDIR"],
        env["WANDB_DIR"],
        env["WANDB_CACHE_DIR"],
        env["NLTK_DATA"],
    ]
    for path in runtime_dirs:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)

    return env


def apply_cert_bundle_defaults(env: dict[str, str]) -> dict[str, str]:
    try:
        import certifi
    except Exception:
        return env

    cert_bundle = certifi.where()
    if cert_bundle:
        env.setdefault("SSL_CERT_FILE", cert_bundle)
        env.setdefault("REQUESTS_CA_BUNDLE", cert_bundle)
        env.setdefault("CURL_CA_BUNDLE", cert_bundle)
    return env


def parse_visible_gpu_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def query_gpu_free_memory_mib() -> dict[str, int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    result: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        index, free_mem = parts
        try:
            result[index] = int(free_mem)
        except ValueError:
            continue
    return result


def choose_safety_visible_gpus(env: dict[str, str]) -> str | None:
    explicit = (
        env.get("SAFETY_EVAL_GPU")
        or env.get("SAFETY_EVAL_CUDA_VISIBLE_DEVICES")
        or env.get("BENCHMARK_EVAL_GPU")
        or env.get("ASYNC_EVAL_GPU")
    )
    if explicit:
        return explicit.strip()

    current_visible = parse_visible_gpu_list(env.get("CUDA_VISIBLE_DEVICES"))
    if len(current_visible) <= 1:
        return current_visible[0] if current_visible else None

    free_mem_by_gpu = query_gpu_free_memory_mib()
    if free_mem_by_gpu:
        ranked = sorted(
            current_visible,
            key=lambda gpu: (free_mem_by_gpu.get(gpu, -1), gpu),
            reverse=True,
        )
        if ranked and free_mem_by_gpu.get(ranked[0], -1) >= 0:
            return ranked[0]

    return current_visible[0]


def default_runner_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    apply_scratch_runtime_defaults(env)
    apply_cert_bundle_defaults(env)
    hf_token = resolve_hf_token(env)
    if hf_token:
        env.setdefault("HF_TOKEN", hf_token)
        env.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        env.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    # Native benchmark runners can hit the same vLLM standalone-compile crash
    # observed on the local server path. Default to the hardened settings unless
    # the caller explicitly overrides them.
    env.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.openai_base_url:
        env["OPENAI_BASE_URL"] = args.openai_base_url
        env["OPENAI_API_BASE"] = args.openai_base_url
        parsed = urlparse(args.openai_base_url)
        if parsed.hostname:
            env.setdefault("VLLM_ENDPOINT", parsed.hostname)
            if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
                local_hosts = ["127.0.0.1", "localhost", "::1"]
                existing = env.get("NO_PROXY") or env.get("no_proxy") or ""
                existing_items = [item.strip() for item in existing.split(",") if item.strip()]
                merged = []
                for item in existing_items + local_hosts:
                    if item not in merged:
                        merged.append(item)
                no_proxy_value = ",".join(merged)
                env["NO_PROXY"] = no_proxy_value
                env["no_proxy"] = no_proxy_value
                for proxy_var in (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                ):
                    env.pop(proxy_var, None)
        if parsed.port:
            env.setdefault("VLLM_PORT", str(parsed.port))
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key
    return env


def evalplus_dataset_locator(
    benchmark: BenchmarkSpec,
    env: dict[str, str],
) -> tuple[str, Path, str | None] | None:
    dataset = benchmark.task or ""
    if dataset == "mbpp":
        module_name = "evalplus.data.mbpp"
        version_name = "MBPP_PLUS_VERSION"
        dataset_name = "MbppPlus"
    elif dataset == "humaneval":
        module_name = "evalplus.data.humaneval"
        version_name = "HUMANEVAL_PLUS_VERSION"
        dataset_name = "HumanEvalPlus"
    else:
        return None

    script = f"""
import json
from evalplus.data.utils import get_dataset_metadata
from {module_name} import {version_name}

try:
    import certifi
    cert_bundle = certifi.where()
except Exception:
    cert_bundle = None

version = {version_name}
url, cache_path = get_dataset_metadata("{dataset_name}", version, False, False)
print(json.dumps({{"url": url, "cache_path": cache_path, "version": version, "cert_bundle": cert_bundle}}))
""".strip()
    shell_prefix = benchmark_shell_prefix(benchmark)
    if shell_prefix:
        cmd = ["bash", "-lc", f"{shell_prefix} && python3 - <<'PY'\n{script}\nPY"]
    else:
        cmd = [sys.executable, "-c", script]

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None
    url = payload.get("url")
    cache_path = payload.get("cache_path")
    cert_bundle = payload.get("cert_bundle")
    if not isinstance(url, str) or not isinstance(cache_path, str):
        return None
    if cert_bundle is not None and not isinstance(cert_bundle, str):
        cert_bundle = None
    return url, Path(cache_path).expanduser(), cert_bundle


def prefetch_evalplus_dataset(
    benchmark: BenchmarkSpec,
    env: dict[str, str],
) -> Path | None:
    located = evalplus_dataset_locator(benchmark, env)
    if located is None:
        return None
    url, cache_path, cert_bundle = located
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "forgetting-llms/1.0"})
    cert_path = env.get("SSL_CERT_FILE") or cert_bundle
    if cert_path:
        context = ssl.create_default_context(cafile=cert_path)
    else:
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to prefetch EvalPlus dataset for benchmark={benchmark.name} from {url} "
            f"using SSL_CERT_FILE={cert_path!r}: {exc}"
        ) from exc
    if url.endswith(".gz"):
        payload = gzip.decompress(payload)
    cache_path.write_bytes(payload)
    return cache_path


def served_model_name(target: EvalTarget, model_ref: str, args: argparse.Namespace) -> str:
    if args.served_model_name:
        return args.served_model_name
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return model_path.name
    if "/" in model_ref:
        return model_ref.rsplit("/", 1)[-1]
    return target.label


def ensure_endpoint_ready(
    base_url: str,
    api_key: str,
    served_model: str,
    benchmark: BenchmarkSpec,
) -> None:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"{benchmark.runner} requires a valid OpenAI-compatible base URL, got: {base_url!r}")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        models_path = path + "/models"
    elif path:
        models_path = path + "/v1/models"
    else:
        models_path = "/v1/models"
    models_url = parsed._replace(path=models_path, params="", query="", fragment="").geturl()

    request = urllib.request.Request(models_url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"{benchmark.runner} endpoint preflight failed for benchmark={benchmark.name}: "
                    f"GET {models_url} returned HTTP {response.status}"
                )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"{benchmark.runner} endpoint preflight failed for benchmark={benchmark.name}: "
            f"cannot reach {models_url} ({exc})"
        ) from exc

    if path.endswith("/v1"):
        completion_path = path + "/chat/completions"
    elif path:
        completion_path = path + "/v1/chat/completions"
    else:
        completion_path = "/v1/chat/completions"
    completion_url = parsed._replace(path=completion_path, params="", query="", fragment="").geturl()
    payload = json.dumps(
        {
            "model": served_model,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 8,
        }
    ).encode("utf-8")
    completion_request = urllib.request.Request(
        completion_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        completion_request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(completion_request, timeout=20) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"{benchmark.runner} endpoint completion probe failed for benchmark={benchmark.name}: "
                    f"POST {completion_url} returned HTTP {response.status}"
                )
            body = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"{benchmark.runner} endpoint completion probe failed for benchmark={benchmark.name}: "
            f"cannot reach {completion_url} ({exc})"
        ) from exc

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"{benchmark.runner} endpoint completion probe failed for benchmark={benchmark.name}: "
            "response did not contain any choices"
        )
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if content in (None, ""):
        raise RuntimeError(
            f"{benchmark.runner} endpoint completion probe failed for benchmark={benchmark.name}: "
            "chat completion returned empty content"
        )


def run_command(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    benchmark_dir: Path,
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    args: argparse.Namespace | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    shell_prefix = benchmark_shell_prefix(benchmark)
    effective_cmd = cmd
    if shell_prefix:
        quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
        effective_cmd = ["bash", "-lc", f"{shell_prefix} && exec {quoted_cmd}"]

    metadata: dict[str, object] = {
        "status": "running",
        "benchmark": benchmark.name,
        "task": benchmark.task,
        "runner": benchmark.runner,
        "target": target.label,
        "command": effective_cmd,
    }
    if shell_prefix:
        metadata["shell_prefix"] = shell_prefix
    if args is not None:
        metadata.update(benchmark_metadata_payload(benchmark, args))
    if cwd is not None:
        metadata["cwd"] = str(cwd)
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(benchmark_dir / "status.json", metadata)

    expected_examples = metadata.get("expected_examples")
    evalplus_progress_active = False

    def maybe_finish_evalplus_progress() -> None:
        nonlocal evalplus_progress_active
        if evalplus_progress_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            evalplus_progress_active = False

    def maybe_render_evalplus_progress(line: str) -> bool:
        nonlocal evalplus_progress_active
        if benchmark.runner != "evalplus":
            return False
        match = re.match(r"^Codegen:\s+(.+?)/(\d+)\s+@\s+(.+?)\s*$", line)
        if match is None:
            return False
        dataset_name = match.group(1)
        current_index = int(match.group(2))
        model_name = match.group(3)
        completed = current_index + 1
        if isinstance(expected_examples, int) and expected_examples > 0:
            total = expected_examples
            completed = min(completed, total)
            ratio = completed / total
            bar_width = 30
            filled = min(bar_width, max(0, int(round(ratio * bar_width))))
            bar = "#" * filled + "-" * (bar_width - filled)
            progress_line = (
                f"\r  Codegen {dataset_name}: [{bar}] {completed}/{total} "
                f"({ratio * 100:5.1f}%) @ {model_name}"
            )
        else:
            progress_line = f"\r  Codegen {dataset_name}: {completed} @ {model_name}"
        sys.stdout.write(progress_line)
        sys.stdout.flush()
        evalplus_progress_active = True
        return True

    lighteval_empty_response_errors = 0
    with subprocess.Popen(
        effective_cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    ) as completed:
        assert completed.stdout is not None
        for line in completed.stdout:
            if maybe_render_evalplus_progress(line):
                continue
            maybe_finish_evalplus_progress()
            sys.stdout.write(line)
            sys.stdout.flush()
            if "API call failed after" in line and "returning empty response" in line:
                lighteval_empty_response_errors += 1
        returncode = completed.wait()
    maybe_finish_evalplus_progress()

    if lighteval_empty_response_errors:
        metadata["lighteval_empty_response_errors"] = lighteval_empty_response_errors

    if returncode != 0:
        metadata["status"] = "failed"
        metadata["returncode"] = returncode
        write_json(benchmark_dir / "status.json", metadata)
        raise RuntimeError(
            f"{benchmark.runner} failed for benchmark={benchmark.name} target={target.label} "
            f"with exit code {returncode}"
        )

    if (
        benchmark.runner == "lighteval"
        and args is not None
        and args.openai_base_url
        and lighteval_empty_response_errors > 0
        and not args.allow_lighteval_empty_responses
    ):
        metadata["status"] = "failed"
        metadata["returncode"] = returncode
        metadata["failure_reason"] = (
            f"Observed {lighteval_empty_response_errors} LiteLLM empty-response request failures."
        )
        write_json(benchmark_dir / "status.json", metadata)
        raise RuntimeError(
            f"{benchmark.runner} produced {lighteval_empty_response_errors} empty-response request failures "
            f"for benchmark={benchmark.name} target={target.label}; refusing to score partial outputs"
        )

    metadata["status"] = "complete"
    metadata["returncode"] = returncode
    primary_metric = summarize_benchmark_metrics(benchmark_dir)
    if primary_metric is not None:
        metric_name, metric_value, metric_source = primary_metric
        metadata["primary_metric_name"] = metric_name
        metadata["primary_metric_value"] = metric_value
        metadata["primary_metric_source"] = metric_source
    write_json(benchmark_dir / "status.json", metadata)
    print("  status: complete")
    sys.stdout.flush()
    print_benchmark_metric_summary(benchmark_dir)
    return metadata


def run_lighteval_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    normalized_task = normalize_lighteval_task(benchmark.task or "")
    custom_tasks_path, effective_task = ensure_custom_lighteval_task_module(
        benchmark,
        benchmark_dir,
        normalized_task,
        args,
    )
    served_model = served_model_name(target, model_ref, args)
    if args.openai_base_url:
        ensure_endpoint_ready(args.openai_base_url, args.openai_api_key, served_model, benchmark)
        if args.no_lighteval_chat_template:
            raise RuntimeError(
                "endpoint-backed LightEval cannot run with --no-lighteval-chat-template. "
                "Hugging Face documents --use-chat-template as required for LiteLLM backends."
            )
        cmd = [
            "lighteval",
            "endpoint",
            "litellm",
            build_lighteval_endpoint_model_args(benchmark, target, model_ref, args),
            effective_task,
            "--output-dir",
            str(benchmark_dir),
            "--save-details",
        ]
        if custom_tasks_path is not None:
            cmd.extend(["--custom-tasks", str(custom_tasks_path)])
        if lighteval_supports_max_concurrent_requests("endpoint", "litellm"):
            cmd.extend(
                [
                    "--max-concurrent-requests",
                    str(args.lighteval_endpoint_max_concurrent_requests),
                ]
            )
        if not args.no_lighteval_chat_template and lighteval_supports_chat_template("endpoint", "litellm"):
            cmd.append("--use-chat-template")
        backend = "lighteval_endpoint_litellm"
    else:
        cmd = [
            "lighteval",
            "vllm",
            build_lighteval_model_args(model_ref, benchmark, args),
            effective_task,
            "--output-dir",
            str(benchmark_dir),
            "--save-details",
        ]
        if custom_tasks_path is not None:
            cmd.extend(["--custom-tasks", str(custom_tasks_path)])
        if not args.no_lighteval_chat_template and lighteval_supports_chat_template("vllm"):
            cmd.append("--use-chat-template")
        backend = "lighteval_vllm"
    return run_command(
        benchmark,
        target,
        benchmark_dir,
        cmd,
        env=default_runner_env(args),
        args=args,
        extra_metadata={
            "normalized_task": normalized_task,
            "effective_task": effective_task,
            "backend": backend,
            "lighteval_num_samples": args.lighteval_num_samples,
            "lighteval_pass_k": args.lighteval_pass_k,
        },
    )


@lru_cache(maxsize=None)
def lighteval_help_text(*backend_parts: str) -> str:
    executable = shutil.which("lighteval")
    if executable is None:
        return ""
    try:
        completed = subprocess.run(
            [executable, *backend_parts, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return f"{completed.stdout}\n{completed.stderr}"


def lighteval_supports_chat_template(*backend_parts: str) -> bool:
    return "--use-chat-template" in lighteval_help_text(*backend_parts)


def lighteval_supports_max_concurrent_requests(*backend_parts: str) -> bool:
    return "--max-concurrent-requests" in lighteval_help_text(*backend_parts)


def effective_evalplus_backend(args: argparse.Namespace) -> str:
    return "openai"


def evalplus_status_passed(status: object) -> bool:
    return isinstance(status, str) and status.strip().lower() == "pass"


def estimate_pass_at_k(total_samples: int, correct_samples: int, k: int) -> float:
    if total_samples <= 0 or correct_samples <= 0 or k <= 0:
        return 0.0
    if k > total_samples:
        k = total_samples
    if total_samples - correct_samples < k:
        return 1.0
    product = 1.0
    for value in range(total_samples - correct_samples + 1, total_samples + 1):
        product *= 1.0 - float(k) / float(value)
    return 1.0 - product


def derive_evalplus_requested_pass_at_k(
    evalplus_root: Path,
    requested_k: int,
) -> tuple[dict[str, float], list[Path]]:
    derived_metrics: dict[str, float] = {}
    updated_files: list[Path] = []
    if requested_k < 1:
        return derived_metrics, updated_files

    for eval_results_path in sorted(evalplus_root.rglob("eval_results.json")):
        try:
            payload = json.loads(eval_results_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        eval_payload = payload.get("eval")
        if not isinstance(eval_payload, dict):
            continue

        base_totals: list[int] = []
        base_correct: list[int] = []
        plus_totals: list[int] = []
        plus_correct: list[int] = []

        for task_results in eval_payload.values():
            if not isinstance(task_results, list):
                continue
            task_total = 0
            task_base_correct = 0
            task_plus_correct = 0
            task_has_plus = False

            for sample in task_results:
                if not isinstance(sample, list) or len(sample) < 2 or not isinstance(sample[1], dict):
                    continue
                sample_result = sample[1]
                task_total += 1
                base_ok = evalplus_status_passed(sample_result.get("base_status"))
                plus_status = sample_result.get("plus_status")
                plus_ok = base_ok and evalplus_status_passed(plus_status)
                if base_ok:
                    task_base_correct += 1
                if plus_status is not None:
                    task_has_plus = True
                    if plus_ok:
                        task_plus_correct += 1

            if task_total <= 0:
                continue

            base_totals.append(task_total)
            base_correct.append(task_base_correct)
            if task_has_plus:
                plus_totals.append(task_total)
                plus_correct.append(task_plus_correct)

        requested_metrics: dict[str, float] = {}
        if base_totals and all(total >= requested_k for total in base_totals):
            base_values = [
                estimate_pass_at_k(total, correct, requested_k)
                for total, correct in zip(base_totals, base_correct, strict=True)
            ]
            requested_metrics["base"] = sum(base_values) / len(base_values)
        if plus_totals and all(total >= requested_k for total in plus_totals):
            plus_values = [
                estimate_pass_at_k(total, correct, requested_k)
                for total, correct in zip(plus_totals, plus_correct, strict=True)
            ]
            requested_metrics["plus"] = sum(plus_values) / len(plus_values)

        if not requested_metrics:
            continue

        metric_name = f"pass@k:k={requested_k}"
        derived_payload = {
            "requested_k": requested_k,
            "results": {
                "base": {metric_name: requested_metrics.get("base")},
            },
        }
        if "plus" in requested_metrics:
            derived_payload["results"]["plus"] = {metric_name: requested_metrics["plus"]}
            derived_payload["results"]["all"] = {metric_name: requested_metrics["plus"]}
            derived_metrics[f"results/all/{metric_name}"] = requested_metrics["plus"]
        elif "base" in requested_metrics:
            derived_payload["results"]["all"] = {metric_name: requested_metrics["base"]}
            derived_metrics[f"results/all/{metric_name}"] = requested_metrics["base"]
        if requested_metrics.get("base") is not None:
            derived_metrics[f"results/base/{metric_name}"] = requested_metrics["base"]
        if requested_metrics.get("plus") is not None:
            derived_metrics[f"results/plus/{metric_name}"] = requested_metrics["plus"]

        derived_path = eval_results_path.with_name("derived_pass_at_k.json")
        write_json(derived_path, derived_payload)
        updated_files.append(derived_path)

    return derived_metrics, updated_files


def run_evalplus_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    def build_evalplus_command(backend: str, model_name: str) -> list[str]:
        cmd = [
            "python3",
            "-m",
            "evalplus.evaluate",
            "--model",
            model_name,
            "--dataset",
            benchmark.task or "",
            "--backend",
            backend,
            "--root",
            str(evalplus_root),
            "--temperature",
            str(args.sampling_temperature),
            "--n-samples",
            str(args.evalplus_n_samples),
            "--bs",
            str(args.evalplus_batch_size),
        ]
        if backend == "openai" and args.openai_base_url:
            cmd.extend(["--base-url", args.openai_base_url])
        return cmd

    def evalplus_metadata(backend: str, env: dict[str, str]) -> dict[str, object]:
        return {
            "backend": f"evalplus_{backend}",
            "evalplus_backend": backend,
            "evalplus_n_samples": args.evalplus_n_samples,
            "evalplus_batch_size": args.evalplus_batch_size,
            "vllm_use_v1": env.get("VLLM_USE_V1"),
        }

    evalplus_root = benchmark_dir / "evalplus_results"
    requested_backend = args.evalplus_backend
    backend = effective_evalplus_backend(args)
    if not args.openai_base_url:
        raise RuntimeError(
            "EvalPlus native vllm is disabled in this repo because it reproducibly "
            "crashes during vLLM engine startup on Mila. Re-run with "
            "--openai-base-url pointing at a local OpenAI-compatible vLLM server, "
            "or use scripts/run_eval_with_local_server.sh."
        )
    model_name = model_ref
    env = default_runner_env(args)
    prefetched_dataset = prefetch_evalplus_dataset(benchmark, env)
    if prefetched_dataset is not None:
        print(f"  Prefetched EvalPlus dataset cache: {prefetched_dataset}")
        sys.stdout.flush()
    ensure_endpoint_ready(
        args.openai_base_url,
        args.openai_api_key,
        served_model_name(target, model_ref, args),
        benchmark,
    )
    model_name = served_model_name(target, model_ref, args)
    if requested_backend != backend:
        print(f"  EvalPlus backend resolved: requested={requested_backend}, effective={backend}")
        sys.stdout.flush()
    launch_cmd_preview = " ".join(
        shlex.quote(part) for part in build_evalplus_command(backend, model_name)
    )
    print(f"  EvalPlus launch command: {launch_cmd_preview}")
    sys.stdout.flush()

    metadata = run_command(
        benchmark,
        target,
        benchmark_dir,
        build_evalplus_command(backend, model_name),
        cwd=benchmark_dir.parent,
        env=env,
        args=args,
        extra_metadata={
            **evalplus_metadata(backend, env),
            "evalplus_requested_backend": requested_backend,
            "evalplus_native_vllm_disabled": True,
            "evalplus_dataset_cache": str(prefetched_dataset) if prefetched_dataset is not None else None,
        },
    )
    derived_metrics, derived_files = derive_evalplus_requested_pass_at_k(
        evalplus_root,
        args.evalplus_pass_k,
    )
    if derived_files:
        print(
            f"  EvalPlus derived pass@k:k={args.evalplus_pass_k} written to "
            f"{len(derived_files)} file(s)"
        )
        sys.stdout.flush()
        metadata["evalplus_pass_k"] = args.evalplus_pass_k
        metadata["evalplus_derived_metric_files"] = [str(path.relative_to(benchmark_dir)) for path in derived_files]
        metadata["evalplus_derived_metrics"] = derived_metrics
        primary_metric = summarize_benchmark_metrics(benchmark_dir)
        if primary_metric is not None:
            metric_name, metric_value, metric_source = primary_metric
            metadata["primary_metric_name"] = metric_name
            metadata["primary_metric_value"] = metric_value
            metadata["primary_metric_source"] = metric_source
        write_json(benchmark_dir / "status.json", metadata)
        print_benchmark_metric_summary(benchmark_dir)
    return metadata


def run_labbench_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    if not args.openai_base_url:
        raise RuntimeError("LAB-Bench requires --openai-base-url or OPENAI_BASE_URL/OPENAI_API_BASE.")
    ensure_endpoint_ready(
        args.openai_base_url,
        args.openai_api_key,
        served_model_name(target, model_ref, args),
        benchmark,
    )
    cmd = [
        "python3",
        str(Path(__file__).resolve().parents[2] / "scripts" / "run_labbench_eval.py"),
        "--eval",
        benchmark.task or "LitQA2",
        "--model-name",
        served_model_name(target, model_ref, args),
        "--base-url",
        args.openai_base_url,
        "--api-key",
        args.openai_api_key,
        "--output-path",
        str(benchmark_dir / "results.json"),
        "--threads",
        str(args.labbench_threads),
    ]
    return run_command(
        benchmark,
        target,
        benchmark_dir,
        cmd,
        env=default_runner_env(args),
        args=args,
    )


def run_supergpqa_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    cmd = [
        "python3",
        str(Path(__file__).resolve().parents[2] / "scripts" / "run_supergpqa_eval.py"),
        "--model",
        model_ref,
        "--output-dir",
        str(benchmark_dir),
        "--dataset-name",
        benchmark.task or "m-a-p/SuperGPQA",
        "--split",
        args.supergpqa_split,
        "--batch-size",
        str(args.supergpqa_batch_size),
        "--max-new-tokens",
        str(args.supergpqa_max_new_tokens),
        "--max-model-len",
        str(args.supergpqa_max_model_len),
        "--gpu-memory-utilization",
        str(args.supergpqa_gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.supergpqa_tensor_parallel_size),
        "--num-samples",
        str(args.supergpqa_num_samples),
        "--pass-k",
        str(args.supergpqa_pass_k),
        "--samples-per-call",
        str(args.supergpqa_samples_per_call),
        "--temperature",
        str(args.sampling_temperature),
        "--top-p",
        str(args.sampling_top_p),
    ]
    if args.supergpqa_max_samples is not None:
        cmd.extend(["--max-samples", str(args.supergpqa_max_samples)])
    return run_command(
        benchmark,
        target,
        benchmark_dir,
        cmd,
        env=default_runner_env(args),
        args=args,
        extra_metadata={
            "dataset_name": benchmark.task or "m-a-p/SuperGPQA",
            "split": args.supergpqa_split,
            "batch_size": args.supergpqa_batch_size,
            "supergpqa_num_samples": args.supergpqa_num_samples,
            "supergpqa_pass_k": args.supergpqa_pass_k,
            "supergpqa_samples_per_call": args.supergpqa_samples_per_call,
        },
    )


def run_bfcl_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    root = benchmark_root("bfcl", args.benchmark_root_map)
    if root is None:
        raise RuntimeError("BFCL requires --benchmark-root bfcl=/path/to/berkeley-function-call-leaderboard or BFCL_ROOT.")

    result_dir = benchmark_dir / "result"
    score_dir = benchmark_dir / "score"
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    relative_result = os.path.relpath(result_dir, root)
    relative_score = os.path.relpath(score_dir, root)

    model_name = args.bfcl_model_name or served_model_name(target, model_ref, args)
    env = default_runner_env(args)
    shell_prefix = benchmark_shell_prefix(benchmark)

    generate_cmd = [
        "bfcl",
        "generate",
        "--model",
        model_name,
        "--test-category",
        args.bfcl_test_category,
        "--result-dir",
        relative_result,
    ]
    if Path(model_ref).expanduser().exists():
        generate_cmd.extend(["--backend", "vllm", "--local-model-path", model_ref, "--skip-server-setup"])

    eval_cmd = [
        "bfcl",
        "evaluate",
        "--model",
        model_name,
        "--test-category",
        args.bfcl_test_category,
        "--result-dir",
        relative_result,
        "--score-dir",
        relative_score,
    ]

    effective_generate_cmd = generate_cmd
    effective_eval_cmd = eval_cmd
    if shell_prefix:
        quoted_generate = " ".join(shlex.quote(part) for part in generate_cmd)
        quoted_evaluate = " ".join(shlex.quote(part) for part in eval_cmd)
        effective_generate_cmd = ["bash", "-lc", f"{shell_prefix} && exec {quoted_generate}"]
        effective_eval_cmd = ["bash", "-lc", f"{shell_prefix} && exec {quoted_evaluate}"]

    metadata = {
        "status": "running",
        "benchmark": benchmark.name,
        "task": benchmark.task,
        "runner": benchmark.runner,
        "target": target.label,
        "cwd": str(root),
        "generate_command": effective_generate_cmd,
        "evaluate_command": effective_eval_cmd,
    }
    if shell_prefix:
        metadata["shell_prefix"] = shell_prefix
    metadata.update(benchmark_metadata_payload(benchmark, args))
    write_json(benchmark_dir / "status.json", metadata)

    def count_stage_files(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for child in path.rglob("*") if child.is_file())

    def run_bfcl_stage(stage_name: str, cmd: list[str], watch_dir: Path) -> subprocess.Popen[bytes]:
        print(f"  BFCL {stage_name} started")
        sys.stdout.flush()
        process = subprocess.Popen(cmd, cwd=root, env=env)
        start = time.monotonic()
        last_count = -1
        spinner = "|/-\\"
        spinner_idx = 0
        while True:
            returncode = process.poll()
            current_count = count_stage_files(watch_dir)
            if current_count != last_count or returncode is None:
                elapsed = int(time.monotonic() - start)
                minutes, seconds = divmod(elapsed, 60)
                bar_width = 16
                filled = spinner_idx % bar_width
                bar = "#" * filled + "-" * (bar_width - filled)
                sys.stdout.write(
                    f"\r  BFCL {stage_name}: [{bar}] files={current_count} elapsed={minutes:02d}:{seconds:02d} "
                    f"{spinner[spinner_idx % len(spinner)]}"
                )
                sys.stdout.flush()
                spinner_idx += 1
                last_count = current_count
            if returncode is not None:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return process
            time.sleep(5)

    generate = run_bfcl_stage("generate", effective_generate_cmd, result_dir)
    if generate.returncode != 0:
        metadata["status"] = "failed"
        metadata["returncode"] = generate.returncode
        metadata["failed_stage"] = "generate"
        write_json(benchmark_dir / "status.json", metadata)
        raise RuntimeError(f"bfcl generate failed for benchmark={benchmark.name} target={target.label}")

    evaluate = run_bfcl_stage("evaluate", effective_eval_cmd, score_dir)
    if evaluate.returncode != 0:
        metadata["status"] = "failed"
        metadata["returncode"] = evaluate.returncode
        metadata["failed_stage"] = "evaluate"
        write_json(benchmark_dir / "status.json", metadata)
        raise RuntimeError(f"bfcl evaluate failed for benchmark={benchmark.name} target={target.label}")

    metadata["status"] = "complete"
    metadata["returncode"] = 0
    write_json(benchmark_dir / "status.json", metadata)
    return metadata


def run_rg_mix_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    root = benchmark_root("rg_mix", args.benchmark_root_map)
    installed_rg_mix = rg_mix_installed()
    if root is None and not installed_rg_mix:
        raise RuntimeError(
            "RG-mix requires either an installed rg-mix-env verifier package "
            "or --benchmark-root rg_mix=/path/to/rg_mix_env / RG_MIX_ROOT."
        )
    if root is not None and not root.exists():
        if installed_rg_mix:
            root = None
        else:
            raise RuntimeError(f"RG-mix root does not exist: {root}")
    if root is not None and not (root / "rg_mix_env.py").exists():
        if installed_rg_mix:
            root = None
        else:
            raise RuntimeError(f"Expected rg_mix_env.py under RG-mix root: {root}")
    local_model_path = Path(model_ref).expanduser()
    use_local_server = local_model_path.exists()
    if not args.openai_base_url and not use_local_server:
        raise RuntimeError(
            "RG-mix requires either --openai-base-url / OPENAI_BASE_URL / OPENAI_API_BASE "
            "or a local model/checkpoint path that can be served directly."
        )
    if args.openai_base_url:
        ensure_endpoint_ready(
            args.openai_base_url,
            args.openai_api_key,
            served_model_name(target, model_ref, args),
            benchmark,
        )
    cmd = [
        "python3",
        str(Path(__file__).resolve().parents[2] / "scripts" / "run_rg_mix_benchmark.py"),
        "--model-name",
        served_model_name(target, model_ref, args),
        "--api-key",
        args.openai_api_key,
        "--output-dir",
        str(benchmark_dir),
    ]
    if args.openai_base_url:
        cmd.extend(["--base-url", args.openai_base_url])
    else:
        cmd.extend(["--model-path", model_ref])
    if root is not None:
        cmd.extend(["--rg-mix-root", str(root)])
    return run_command(
        benchmark,
        target,
        benchmark_dir,
        cmd,
        env=default_runner_env(args),
        args=args,
    )


def run_safety_eval_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    root = benchmark_root("safety", args.benchmark_root_map)
    if root is None:
        raise RuntimeError("Safety eval requires --benchmark-root safety=/path/to/safety-eval or SAFETY_EVAL_ROOT.")

    def slugify_filename(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"

    def resolve_safety_template(root: Path, env: dict[str, str]) -> str:
        requested = args.safety_template.strip() if args.safety_template else "auto"
        if requested.lower() != "auto":
            return requested

        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Safety eval template resolution requires transformers in the active env. "
                "Re-run with --safety-template NAME_OR_PATH if you cannot load the tokenizer."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"Safety eval could not load tokenizer for model={model_ref!r} to infer "
                "a safety prompt template. Re-run with --safety-template NAME_OR_PATH."
            ) from exc

        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                f"Safety eval tokenizer for model={model_ref!r} does not expose apply_chat_template(). "
                "Re-run with --safety-template NAME_OR_PATH."
            )

        instruction_placeholder = "__FORGETTING_LLMS_INSTRUCTION_PLACEHOLDER__"
        try:
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction_placeholder}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction_placeholder}],
                tokenize=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Safety eval could not render an instruction prompt template for model={model_ref!r}. "
                "Re-run with --safety-template NAME_OR_PATH."
            ) from exc

        if instruction_placeholder not in rendered_prompt:
            raise RuntimeError(
                f"Safety eval rendered prompt for model={model_ref!r} does not preserve the "
                "instruction placeholder. Re-run with --safety-template NAME_OR_PATH."
            )

        prompt_template = rendered_prompt.replace(instruction_placeholder, "{instruction}")
        if "{instruction}" not in prompt_template:
            raise RuntimeError(
                f"Safety eval auto-template for model={model_ref!r} is missing the required "
                "{{instruction}} field."
            )

        # The installed safety-eval loader only treats file paths specially when
        # they end with ".txt". Other suffixes are interpreted as registered
        # template names and fail immediately.
        template_name = f"forgetting_llms_auto_{slugify_filename(target.label)}.txt"
        template_path = benchmark_dir / template_name
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(prompt_template)
        print(f"  Safety eval template resolved to instruction prompt template: {template_path}")
        sys.stdout.flush()
        return str(template_path)

    env = default_runner_env(args)
    original_cuda_visible = env.get("CUDA_VISIBLE_DEVICES")
    chosen_safety_gpus = choose_safety_visible_gpus(env)
    if chosen_safety_gpus:
        env["CUDA_VISIBLE_DEVICES"] = chosen_safety_gpus
        print(
            "  Safety eval CUDA_VISIBLE_DEVICES="
            f"{chosen_safety_gpus}"
            f" (was: {original_cuda_visible or '<all visible>'})"
        )
        sys.stdout.flush()
    backend = "safety_eval_vllm"
    model_name = model_ref
    requested_tasks = benchmark.task or ""
    effective_tasks, skipped_safety_tasks = resolve_safety_tasks(requested_tasks, env)
    if skipped_safety_tasks:
        for skipped_task in skipped_safety_tasks:
            print(
                "  Safety eval skipping subtask "
                f"{skipped_task['task']}: {skipped_task['reason']}"
            )
        sys.stdout.flush()
    if not effective_tasks:
        payload = {
            "status": "skipped",
            "reason": "No runnable safety subtasks remain after access checks.",
            "benchmark": benchmark.name,
            "task": requested_tasks,
            "effective_task": "",
            "runner": benchmark.runner,
            "target": target.label,
            "skipped_subtasks": skipped_safety_tasks,
        }
        payload.update(benchmark_metadata_payload(benchmark, args))
        write_json(benchmark_dir / "status.json", payload)
        print("  status: skipped (no runnable safety subtasks remain)")
        sys.stdout.flush()
        return payload
    resolved_template = resolve_safety_template(root, env)
    if args.openai_base_url:
        print(
            "  Safety eval ignoring --openai-base-url and using native --use_vllm. "
            "The installed safety-eval API backend crashes in query_openai_chat_model() "
            "for WildGuardTest on this stack."
        )
        sys.stdout.flush()
    cmd = [
        "python3",
        str(Path(__file__).resolve().parents[2] / "scripts" / "run_safety_eval_wrapper.py"),
        "--safety-root",
        str(root),
        "--model-name-or-path",
        model_name,
        "--model-input-template-path-or-name",
        resolved_template,
        "--batch-size",
        str(args.safety_batch_size),
        "--gpu-memory-utilization",
        str(args.safety_gpu_memory_utilization),
        "--tasks",
        ",".join(effective_tasks),
        "--report-output-path",
        str(benchmark_dir / "metrics.json"),
        "--save-individual-results-path",
        str(benchmark_dir / "all.json"),
    ]
    return run_command(
        benchmark,
        target,
        benchmark_dir,
        cmd,
        cwd=root,
        env=env,
        args=args,
        extra_metadata={
            "backend": backend,
            "safety_template": resolved_template,
            "served_model_name": None,
            "requested_task": requested_tasks,
            "effective_task": ",".join(effective_tasks),
            "skipped_subtasks": skipped_safety_tasks,
            "safety_gpu_memory_utilization": args.safety_gpu_memory_utilization,
        },
    )


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def run_custom_command_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    template = benchmark_command_override(benchmark.name, args.benchmark_command_map)
    if not template:
        payload = {
            "status": "skipped",
            "reason": benchmark.notes or "No benchmark command override provided.",
            "benchmark": benchmark.name,
            "runner": benchmark.runner,
            "target": target.label,
        }
        payload.update(benchmark_metadata_payload(benchmark, args))
        write_json(benchmark_dir / "status.json", payload)
        return payload

    root = benchmark_root(benchmark.name, args.benchmark_root_map)
    served_model = served_model_name(target, model_ref, args)
    formatted = template.format_map(
        SafeFormatDict(
            model=model_ref,
            benchmark_dir=str(benchmark_dir),
            output_dir=str(benchmark_dir),
            target_label=target.label,
            served_model=served_model,
            run_name=args.run_name or "",
        )
    )
    cmd = ["bash", "-lc", formatted]
    return run_command(
        benchmark,
        target,
        benchmark_dir,
        cmd,
        cwd=root,
        env=default_runner_env(args),
        args=args,
    )


def run_benchmark(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    model_ref: str,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if benchmark.runner == "lm_eval" and benchmark.task:
        cmd = [
            "lm_eval",
            "--model",
            args.lm_eval_model,
            "--model_args",
            build_model_args(model_ref, args.lm_eval_model_args),
            "--tasks",
            benchmark.task,
            "--num_fewshot",
            str(benchmark.fewshot),
            "--batch_size",
            args.batch_size,
            "--output_path",
            str(benchmark_dir),
        ]
        if args.log_samples:
            cmd.append("--log_samples")
        if args.apply_chat_template:
            cmd.append("--apply_chat_template")
        return run_command(benchmark, target, benchmark_dir, cmd, args=args)

    if benchmark.runner == "lighteval" and benchmark.task:
        return run_lighteval_benchmark(benchmark, target, model_ref, benchmark_dir, args)
    if benchmark.runner == "evalplus" and benchmark.task:
        return run_evalplus_benchmark(benchmark, target, model_ref, benchmark_dir, args)
    if benchmark.runner == "labbench" and benchmark.task:
        return run_labbench_benchmark(benchmark, target, model_ref, benchmark_dir, args)
    if benchmark.runner == "supergpqa" and benchmark.task:
        return run_supergpqa_benchmark(benchmark, target, model_ref, benchmark_dir, args)
    if benchmark.runner == "rg_mix":
        return run_rg_mix_benchmark(benchmark, target, model_ref, benchmark_dir, args)
    if benchmark.runner == "bfcl" and benchmark.task:
        return run_bfcl_benchmark(benchmark, target, model_ref, benchmark_dir, args)
    if benchmark.runner == "safety_eval" and benchmark.task:
        return run_safety_eval_benchmark(benchmark, target, model_ref, benchmark_dir, args)
    if benchmark.runner == "custom_command":
        return run_custom_command_benchmark(benchmark, target, model_ref, benchmark_dir, args)

    if not benchmark.task:
        payload = {
            "status": "skipped",
            "reason": benchmark.notes or "Benchmark does not have a runnable task id.",
            "benchmark": benchmark.name,
            "runner": benchmark.runner,
            "target": target.label,
        }
        payload.update(benchmark_metadata_payload(benchmark, args))
        write_json(benchmark_dir / "status.json", payload)
        return payload

    payload = {
        "status": "skipped",
        "reason": benchmark.notes or f"Unsupported runner type: {benchmark.runner}",
        "benchmark": benchmark.name,
        "runner": benchmark.runner,
        "target": target.label,
    }
    payload.update(benchmark_metadata_payload(benchmark, args))
    write_json(benchmark_dir / "status.json", payload)
    return payload


def benchmark_progress_banner(
    benchmark: BenchmarkSpec,
    target: EvalTarget,
    suite_name: str,
    index: int,
    total: int,
    args: argparse.Namespace,
) -> None:
    example_count, example_note = benchmark_example_metadata(benchmark, args)
    print(
        f"[Eval {index}/{total}] suite={suite_name} benchmark={benchmark.name} "
        f"dataset=\"{benchmark_display_name(benchmark)}\" target={target.label}"
    )
    if benchmark.task:
        print(f"  task: {benchmark.task}")
    print(f"  runner: {benchmark.runner}")
    if benchmark.runner == "lighteval":
        print(
            f"  LightEval samples per prompt: {args.lighteval_num_samples} "
            f"(pass@k={args.lighteval_pass_k})"
        )
    if benchmark.runner == "evalplus":
        effective_backend = effective_evalplus_backend(args)
        backend_label = effective_backend
        if effective_backend != args.evalplus_backend:
            backend_label = f"{args.evalplus_backend}->{effective_backend}"
        print(
            f"  EvalPlus samples per problem: {args.evalplus_n_samples} "
            f"(backend={backend_label}, batch={args.evalplus_batch_size}, "
            f"pass@k={args.evalplus_pass_k})"
        )
    if benchmark.runner == "supergpqa":
        print(
            f"  SuperGPQA samples per prompt: {args.supergpqa_num_samples} "
            f"(pass@k={args.supergpqa_pass_k}, batch={args.supergpqa_batch_size}, "
            f"samples/call={args.supergpqa_samples_per_call})"
        )
    if example_count is not None and example_note:
        print(f"  examples: {example_count} ({example_note})")
    elif example_count is not None:
        print(f"  examples: {example_count}")
    elif example_note:
        print(f"  examples: unknown ({example_note})")
    else:
        print("  examples: unknown")
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    if args.list_suites:
        print_suites()
        return 0

    selected_suites = expand_suites(args.suite)
    registry = get_registry_with_overrides(parse_task_overrides(args.benchmark_task))
    registry = filter_registry(
        registry,
        selected_suites=selected_suites,
        include_runners=args.include_runner,
        include_benchmarks=args.include_benchmark,
    )
    args.benchmark_root_map = parse_kv_overrides(args.benchmark_root, "benchmark root")
    args.benchmark_command_map = parse_kv_overrides(args.benchmark_command, "benchmark command")
    validate_benchmark_requirements(selected_suites, registry, args)
    targets = discover_targets(
        model_path=args.model_path,
        base_model=args.base_model,
        include_base_model=not args.no_base_model,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    has_any_benchmarks = any(registry[suite_name]["benchmarks"] for suite_name in selected_suites)
    if not has_any_benchmarks:
        print("No benchmarks selected after applying suite/runner/benchmark filters. Nothing to do.")
        return 0

    summary_path = output_dir / "eval_summary.json"
    if not args.force_rerun and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = {}
    else:
        summary = {}
    summary.update(
        {
            "run_name": args.run_name,
            "model_path": args.model_path,
            "base_model": args.base_model,
            "suites": selected_suites,
        }
    )
    if not isinstance(summary.get("targets"), dict):
        summary["targets"] = {}

    failures: list[str] = []

    for target in targets:
        target_summary = summary["targets"].get(target.label, {})
        if not isinstance(target_summary, dict):
            target_summary = {}
        temporary_merge_dir: Path | None = None
        model_ref = target.model_ref
        try:
            model_ref, temporary_merge_dir = merge_checkpoint_if_needed(target)
            for suite_name in selected_suites:
                suite_data = registry[suite_name]
                suite_results = target_summary.get(suite_name, {})
                if not isinstance(suite_results, dict):
                    suite_results = {}
                suite_dir = output_dir / target.label / suite_name
                benchmarks = suite_data["benchmarks"]
                total_benchmarks = len(benchmarks)
                for benchmark_index, benchmark in enumerate(benchmarks, start=1):
                    benchmark_progress_banner(
                        benchmark=benchmark,
                        target=target,
                        suite_name=suite_name,
                        index=benchmark_index,
                        total=total_benchmarks,
                        args=args,
                    )
                    benchmark_dir = suite_dir / benchmark.name
                    if args.force_rerun and benchmark_dir.exists():
                        print("  status: force rerun requested, clearing prior benchmark outputs")
                        sys.stdout.flush()
                        shutil.rmtree(benchmark_dir, ignore_errors=True)
                    status_path = benchmark_dir / "status.json"
                    if status_path.exists():
                        existing = json.loads(status_path.read_text())
                        if existing.get("status") in {"complete", "skipped"}:
                            if existing.get("status") == "complete" and not benchmark_has_complete_artifacts(
                                benchmark, benchmark_dir
                            ):
                                print("  status: stale complete marker without result artifacts, rerunning")
                                sys.stdout.flush()
                            else:
                                print("  status: already complete, skipping")
                                if existing.get("status") == "complete":
                                    if "primary_metric_name" in existing and "primary_metric_value" in existing:
                                        metric_source = existing.get("primary_metric_source")
                                        metric_suffix = f" ({metric_source})" if metric_source else ""
                                        print(
                                            "  primary metric: "
                                            f"{existing['primary_metric_name']}={float(existing['primary_metric_value']):.4f}"
                                            f"{metric_suffix}"
                                        )
                                        sys.stdout.flush()
                                    else:
                                        print_benchmark_metric_summary(benchmark_dir)
                                sys.stdout.flush()
                                suite_results[benchmark.name] = existing
                                continue

                    try:
                        suite_results[benchmark.name] = run_benchmark(
                            benchmark=benchmark,
                            target=target,
                            model_ref=model_ref,
                            benchmark_dir=benchmark_dir,
                            args=args,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failure_msg = f"{target.label}/{suite_name}/{benchmark.name}: {exc}"
                        failures.append(failure_msg)
                        suite_results[benchmark.name] = {
                            "status": "failed",
                            "error": str(exc),
                        }
                        write_json(benchmark_dir / "status.json", suite_results[benchmark.name])
                        if not args.continue_on_error:
                            raise
                target_summary[suite_name] = suite_results
        finally:
            if temporary_merge_dir is not None and temporary_merge_dir.exists():
                shutil.rmtree(temporary_merge_dir, ignore_errors=True)

        summary["targets"][target.label] = target_summary
        write_json(output_dir / "eval_summary.json", summary)

    print_evaluation_metric_summary(summary)

    if failures:
        print("Evaluation completed with failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"Saved evaluation summary to {output_dir / 'eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
