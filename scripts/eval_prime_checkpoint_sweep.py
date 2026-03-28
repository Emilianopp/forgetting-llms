#!/usr/bin/env python3
"""Sweep PRIME-RL checkpoints through benchmark-plan and task pass@k evals.

This script is the missing automation layer on top of:

- ``src/evaluation/run_eval.py`` for the benchmark-plan suite
- ``scripts/prime_rl_runner.py baseline`` for local task pass@k evaluation

It is designed around PRIME-RL run bundles under ``~/scratch/forgetting-llms/prime_runs``.
For each discovered run it:

1. Finds exported model checkpoints inside ``step_*`` directories.
2. Runs benchmark-plan evals (default: ``tasks_md`` suite) for checkpoints that
   have not already been recorded as complete.
3. Runs local task evals with ``rollouts_per_prompt = k`` (default ``k=512``),
   including backward-task evals for sequential runs when they can be inferred.
4. Stores run/checkpoint/eval metadata and metrics in a SQLite database.
5. Regenerates lightweight plots after each processed checkpoint.

The database is the source of truth for "which checkpoints have been evaluated".
Existing output directories are also imported so the script can resume safely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib").resolve()),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL_SCRIPT = REPO_ROOT / "src" / "evaluation" / "run_eval.py"
PRIME_RL_RUNNER = REPO_ROOT / "scripts" / "prime_rl_runner.py"
RUN_EVAL_WITH_LOCAL_SERVER = REPO_ROOT / "scripts" / "run_eval_with_local_server.sh"
SUPPORTED_DATASETS = ("gsm8k", "math", "triviaqa", "polaris_math", "openr1_math")
PRIMARY_METRIC_HINTS = (
    "pass@1",
    "pass_at_1",
    "pass_at_k",
    "accuracy_at_1",
    "acc_norm,none",
    "acc,none",
    "prompt_level_strict_acc,none",
    "exact_match",
    "score",
    "overall",
)


@dataclass(frozen=True)
class PrimeRun:
    name: str
    run_dir: Path
    manifest_path: Path | None
    environment_name: str | None
    source_model: str | None
    completed: bool


@dataclass(frozen=True)
class CheckpointTarget:
    run_name: str
    checkpoint_label: str
    checkpoint_dir: Path
    model_path: Path
    step: int | None
    kind: str


@dataclass(frozen=True)
class TaskEvalSpec:
    label: str
    dataset: str
    dataset_path: Path | None
    data_source: str | None


class SweepError(RuntimeError):
    """Raised when a subprocess-backed evaluation step fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def step_value(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def path_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prime-runs-root",
        default="~/scratch/forgetting-llms/prime_runs",
        help="Root containing PRIME-RL run bundle directories.",
    )
    parser.add_argument(
        "--output-root",
        default="~/scratch/forgetting-llms/benchmark_plan_evals",
        help="Root for benchmark/task eval outputs, database, and plots.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional SQLite path. Defaults to <output-root>/eval_registry.sqlite.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional plot output directory. Defaults to <output-root>/plots.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Benchmark suite(s) to run via run_eval.py. Repeatable. Default: tasks_md.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Restrict to specific PRIME run names. Repeatable.",
    )
    parser.add_argument(
        "--only-completed",
        action="store_true",
        help="Only evaluate PRIME runs that have completed.marker.",
    )
    parser.add_argument(
        "--only-latest",
        action="store_true",
        help="Only evaluate the latest discovered checkpoint per run.",
    )
    parser.add_argument(
        "--max-checkpoints-per-run",
        type=int,
        default=None,
        help="Limit discovered checkpoints per run after sorting by step.",
    )
    parser.add_argument(
        "--skip-task-evals",
        action="store_true",
        help="Skip local dataset task evals.",
    )
    parser.add_argument(
        "--skip-benchmark-evals",
        action="store_true",
        help="Skip benchmark-plan evals via run_eval.py.",
    )
    parser.add_argument(
        "--task-pass-k",
        type=int,
        default=512,
        help="rollouts_per_prompt used for local task evals (default: 512).",
    )
    parser.add_argument("--task-max-model-len", type=int, default=None)
    parser.add_argument("--task-max-tokens", type=int, default=8192)
    parser.add_argument("--task-temperature", type=float, default=1.0)
    parser.add_argument("--task-top-p", type=float, default=1.0)
    parser.add_argument("--task-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--task-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--task-max-samples", type=int, default=None)
    parser.add_argument(
        "--task-eval-dataset",
        action="append",
        default=[],
        help=(
            "Explicit local task eval dataset label. Repeatable. When provided, "
            "these labels are evaluated in the order given instead of relying on "
            "run-name inference."
        ),
    )
    parser.add_argument(
        "--task-eval-parquet",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Explicit custom parquet-backed local task eval. Repeatable. "
            "Example: --task-eval-parquet olmo=/scratch/.../test.parquet"
        ),
    )
    parser.add_argument(
        "--task-eval-data-source",
        action="append",
        default=[],
        metavar="NAME=SOURCE",
        help=(
            "Optional reward routing override for a task eval label. Repeatable. "
            "Useful for custom parquet datasets whose scoring should route through "
            "gsm8k/math/triviaqa style rewards."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Forwarded to run_eval.py. Repeatable.",
    )
    parser.add_argument(
        "--benchmark-command",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="Forwarded to run_eval.py. Repeatable.",
    )
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--benchmark-env-file",
        default=os.environ.get("BENCHMARK_ENV_FILE", "~/scratch/forgetting-llms/benchmark_env.sh"),
        help="Optional env file sourced by the local-server eval wrapper.",
    )
    parser.add_argument(
        "--auto-start-eval-server",
        action="store_true",
        default=os.environ.get("AUTO_START_EVAL_SERVER", "0") == "1",
        help=(
            "Serve each benchmark checkpoint on the current CUDA_VISIBLE_DEVICES "
            "through a local vLLM OpenAI-compatible endpoint before running run_eval.py."
        ),
    )
    parser.add_argument(
        "--eval-server-port",
        type=int,
        default=int(os.environ.get("EVAL_SERVER_PORT", "8000")),
        help="Port used by the local vLLM eval server when --auto-start-eval-server is enabled.",
    )
    parser.add_argument(
        "--eval-server-tp",
        type=int,
        default=int(os.environ.get("EVAL_SERVER_TP", "1")),
        help="Tensor parallel size for the local vLLM eval server.",
    )
    parser.add_argument(
        "--eval-server-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("EVAL_SERVER_GPU_MEMORY_UTILIZATION", "0.90")),
        help="GPU memory utilization for the local vLLM eval server.",
    )
    parser.add_argument(
        "--eval-server-max-model-len",
        type=int,
        default=int(os.environ["EVAL_SERVER_MAX_MODEL_LEN"]) if os.environ.get("EVAL_SERVER_MAX_MODEL_LEN") else None,
        help="Optional max model length for the local vLLM eval server.",
    )
    parser.add_argument(
        "--eval-server-startup-timeout",
        type=int,
        default=int(os.environ.get("EVAL_SERVER_STARTUP_TIMEOUT", "600")),
        help="Startup timeout in seconds for the local vLLM eval server.",
    )
    parser.add_argument(
        "--eval-server-api-key",
        default=os.environ.get("EVAL_SERVER_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key exposed by the local vLLM eval server.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default="forgetting-llms",
        help="Project used when --wandb-mode is not disabled.",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="disabled",
        choices=("disabled", "offline", "online"),
    )
    parser.add_argument("--wandb-dir", default="~/scratch/forgetting-llms/wandb")
    parser.add_argument(
        "--max-json-metric-file-mb",
        type=float,
        default=8.0,
        help="Ignore larger JSON files when scraping numeric metrics.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            name TEXT PRIMARY KEY,
            run_dir TEXT NOT NULL,
            manifest_path TEXT,
            environment_name TEXT,
            source_model TEXT,
            completed INTEGER NOT NULL,
            discovered_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL,
            checkpoint_label TEXT NOT NULL,
            checkpoint_dir TEXT NOT NULL,
            model_path TEXT NOT NULL,
            step INTEGER,
            kind TEXT NOT NULL,
            discovered_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            UNIQUE(run_name, checkpoint_label, model_path)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_eval_results (
            checkpoint_id INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            requested_k INTEGER NOT NULL,
            status TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            metrics_path TEXT,
            accuracy_at_1 REAL,
            pass_at_k REAL,
            mean_rollout_reward REAL,
            evaluated_at TEXT NOT NULL,
            error TEXT,
            PRIMARY KEY (checkpoint_id, dataset, requested_k)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_suite_runs (
            checkpoint_id INTEGER NOT NULL,
            suite TEXT NOT NULL,
            status TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            eval_summary_path TEXT,
            evaluated_at TEXT NOT NULL,
            error TEXT,
            PRIMARY KEY (checkpoint_id, suite)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_statuses (
            checkpoint_id INTEGER NOT NULL,
            suite TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            status TEXT NOT NULL,
            runner TEXT,
            benchmark_dir TEXT NOT NULL,
            status_path TEXT,
            primary_metric_name TEXT,
            primary_metric_value REAL,
            error TEXT,
            evaluated_at TEXT NOT NULL,
            PRIMARY KEY (checkpoint_id, suite, benchmark)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_metrics (
            checkpoint_id INTEGER NOT NULL,
            suite TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            metric_key TEXT NOT NULL,
            metric_value REAL NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (checkpoint_id, suite, benchmark, metric_key, source_file)
        )
        """
    )
    conn.commit()


def parse_name_value(items: list[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {label} '{item}'. Expected NAME=VALUE.")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid {label} '{item}'. Expected NAME=VALUE.")
        parsed[name] = value
    return parsed


def upsert_run(conn: sqlite3.Connection, run: PrimeRun) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO runs (
            name, run_dir, manifest_path, environment_name, source_model,
            completed, discovered_at, last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            run_dir=excluded.run_dir,
            manifest_path=excluded.manifest_path,
            environment_name=excluded.environment_name,
            source_model=excluded.source_model,
            completed=excluded.completed,
            last_seen_at=excluded.last_seen_at
        """,
        (
            run.name,
            str(run.run_dir),
            str(run.manifest_path) if run.manifest_path else None,
            run.environment_name,
            run.source_model,
            int(run.completed),
            now,
            now,
        ),
    )
    conn.commit()


def upsert_checkpoint(conn: sqlite3.Connection, target: CheckpointTarget) -> int:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO checkpoints (
            run_name, checkpoint_label, checkpoint_dir, model_path, step, kind,
            discovered_at, last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_name, checkpoint_label, model_path) DO UPDATE SET
            checkpoint_dir=excluded.checkpoint_dir,
            step=excluded.step,
            kind=excluded.kind,
            last_seen_at=excluded.last_seen_at
        """,
        (
            target.run_name,
            target.checkpoint_label,
            str(target.checkpoint_dir),
            str(target.model_path),
            target.step,
            target.kind,
            now,
            now,
        ),
    )
    row = conn.execute(
        """
        SELECT id FROM checkpoints
        WHERE run_name = ? AND checkpoint_label = ? AND model_path = ?
        """,
        (target.run_name, target.checkpoint_label, str(target.model_path)),
    ).fetchone()
    conn.commit()
    assert row is not None
    return int(row[0])


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_prime_run(path: Path) -> PrimeRun | None:
    manifest_path = path / "manifest.json"
    environment_name = None
    source_model = None
    if manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
        except json.JSONDecodeError:
            return None
        if manifest.get("command") != "prime":
            return None
        args = manifest.get("args", {})
        environment_name = args.get("environment_name")
        source_model = args.get("model")
    elif not (path / "checkpoints").exists():
        return None

    return PrimeRun(
        name=path.name,
        run_dir=path,
        manifest_path=manifest_path if manifest_path.exists() else None,
        environment_name=environment_name,
        source_model=source_model,
        completed=(path / "completed.marker").exists(),
    )


def discover_runs(root: Path, selected_names: set[str], only_completed: bool) -> list[PrimeRun]:
    runs: list[PrimeRun] = []
    if not root.exists():
        return runs
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if selected_names and child.name not in selected_names:
            continue
        run = load_prime_run(child)
        if run is None:
            continue
        if only_completed and not run.completed:
            continue
        runs.append(run)
    return runs


def collect_model_dirs(search_root: Path) -> list[Path]:
    if not search_root.exists():
        return []
    candidates = {path.parent.resolve() for path in search_root.rglob("config.json")}
    preferred_tokens = ("final", "latest", "merged", "hf", "export")
    return sorted(
        candidates,
        key=lambda path: (
            0 if any(token in str(path).lower() for token in preferred_tokens) else 1,
            len(path.parts),
            str(path),
        ),
    )


def discover_checkpoint_targets(run: PrimeRun) -> list[CheckpointTarget]:
    checkpoints_root = run.run_dir / "checkpoints"
    search_root = checkpoints_root if checkpoints_root.exists() else run.run_dir
    discovered: list[CheckpointTarget] = []
    seen_model_paths: set[Path] = set()

    step_dirs = sorted(
        [path for path in search_root.glob("step_*") if path.is_dir()],
        key=lambda path: step_value(path.name, "step_") if step_value(path.name, "step_") is not None else math.inf,
    )
    for step_dir in step_dirs:
        model_dirs = collect_model_dirs(step_dir)
        if not model_dirs:
            continue
        model_path = model_dirs[0]
        if model_path in seen_model_paths:
            continue
        seen_model_paths.add(model_path)
        discovered.append(
            CheckpointTarget(
                run_name=run.name,
                checkpoint_label=step_dir.name,
                checkpoint_dir=step_dir,
                model_path=model_path,
                step=step_value(step_dir.name, "step_"),
                kind="step",
            )
        )

    root_model_dirs = collect_model_dirs(search_root)
    for model_path in root_model_dirs:
        if any(model_path == target.model_path for target in discovered):
            continue
        relative_parts = model_path.relative_to(search_root).parts if model_path.is_relative_to(search_root) else ()
        label = "final_export"
        if relative_parts:
            label = "__".join(relative_parts)
        discovered.append(
            CheckpointTarget(
                run_name=run.name,
                checkpoint_label=label,
                checkpoint_dir=model_path,
                model_path=model_path,
                step=None,
                kind="export",
            )
        )

    return discovered


def trim_targets(
    targets: list[CheckpointTarget],
    *,
    only_latest: bool,
    max_checkpoints_per_run: int | None,
) -> list[CheckpointTarget]:
    ordered = sorted(
        targets,
        key=lambda item: (item.step is None, item.step if item.step is not None else math.inf, item.checkpoint_label),
    )
    if only_latest and ordered:
        return [ordered[-1]]
    if max_checkpoints_per_run is not None:
        return ordered[-max_checkpoints_per_run:]
    return ordered


def infer_task_eval_datasets(run: PrimeRun) -> list[str]:
    run_name = run.name.lower()
    found: list[tuple[int, str]] = []
    for dataset in SUPPORTED_DATASETS:
        for match in re.finditer(re.escape(dataset), run_name):
            found.append((match.start(), dataset))
    ordered_names: list[str] = []
    for _position, dataset in sorted(found):
        if dataset not in ordered_names:
            ordered_names.append(dataset)

    datasets: list[str] = []
    if "_then_" in run_name and len(ordered_names) >= 2:
        datasets.extend(reversed(ordered_names))
    elif any(token in run_name for token in ("mixed", "iid")) and len(ordered_names) >= 2:
        datasets.extend(ordered_names[:2])
    elif ordered_names:
        datasets.extend(ordered_names[:1])

    if run.environment_name in SUPPORTED_DATASETS and run.environment_name not in datasets:
        datasets.insert(0, run.environment_name)

    deduped: list[str] = []
    for dataset in datasets:
        if dataset not in deduped:
            deduped.append(dataset)
    return deduped


def task_eval_specs_for_run(
    args: argparse.Namespace,
    run: PrimeRun,
) -> list[TaskEvalSpec]:
    specs: list[TaskEvalSpec] = []
    seen: set[str] = set()
    explicit_dataset_paths = parse_name_value(args.task_eval_parquet, "task eval parquet")
    explicit_data_sources = parse_name_value(args.task_eval_data_source, "task eval data source")

    def add_spec(label: str, dataset_path: Path | None = None, data_source: str | None = None) -> None:
        if label in seen:
            return
        seen.add(label)
        specs.append(
            TaskEvalSpec(
                label=label,
                dataset=label,
                dataset_path=dataset_path,
                data_source=data_source,
            )
        )

    for label in args.task_eval_dataset:
        add_spec(label, data_source=explicit_data_sources.get(label))
    for label, raw_path in explicit_dataset_paths.items():
        add_spec(label, dataset_path=expand(raw_path), data_source=explicit_data_sources.get(label))

    if specs:
        return specs

    for dataset in infer_task_eval_datasets(run):
        add_spec(dataset, data_source=explicit_data_sources.get(dataset))
    return specs


def task_eval_already_complete(
    conn: sqlite3.Connection,
    checkpoint_id: int,
    dataset: str,
    requested_k: int,
) -> bool:
    row = conn.execute(
        """
        SELECT status, metrics_path FROM task_eval_results
        WHERE checkpoint_id = ? AND dataset = ? AND requested_k = ?
        """,
        (checkpoint_id, dataset, requested_k),
    ).fetchone()
    if row is None:
        return False
    status, metrics_path = row
    return status == "complete" and metrics_path and Path(metrics_path).exists()


def benchmark_suite_already_complete(
    conn: sqlite3.Connection,
    checkpoint_id: int,
    suite: str,
) -> bool:
    row = conn.execute(
        """
        SELECT status, eval_summary_path FROM benchmark_suite_runs
        WHERE checkpoint_id = ? AND suite = ?
        """,
        (checkpoint_id, suite),
    ).fetchone()
    if row is None:
        return False
    status, eval_summary_path = row
    return status == "complete" and eval_summary_path and Path(eval_summary_path).exists()


def sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:/-]+", "_", name).strip("_")


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise SweepError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def record_task_eval_result(
    conn: sqlite3.Connection,
    checkpoint_id: int,
    dataset: str,
    requested_k: int,
    status: str,
    output_dir: Path,
    metrics_path: Path | None,
    metrics: dict[str, Any] | None,
    error: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO task_eval_results (
            checkpoint_id, dataset, requested_k, status, output_dir, metrics_path,
            accuracy_at_1, pass_at_k, mean_rollout_reward, evaluated_at, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(checkpoint_id, dataset, requested_k) DO UPDATE SET
            status=excluded.status,
            output_dir=excluded.output_dir,
            metrics_path=excluded.metrics_path,
            accuracy_at_1=excluded.accuracy_at_1,
            pass_at_k=excluded.pass_at_k,
            mean_rollout_reward=excluded.mean_rollout_reward,
            evaluated_at=excluded.evaluated_at,
            error=excluded.error
        """,
        (
            checkpoint_id,
            dataset,
            requested_k,
            status,
            str(output_dir),
            str(metrics_path) if metrics_path else None,
            metrics.get("accuracy_at_1") if metrics else None,
            metrics.get("pass_at_k") if metrics else None,
            metrics.get("mean_rollout_reward") if metrics else None,
            utc_now(),
            error,
        ),
    )
    conn.commit()


def run_task_eval(
    args: argparse.Namespace,
    conn: sqlite3.Connection,
    checkpoint_id: int,
    target: CheckpointTarget,
    spec: TaskEvalSpec,
) -> dict[str, Any] | None:
    if task_eval_already_complete(conn, checkpoint_id, spec.label, args.task_pass_k):
        return None

    run_name = f"{target.run_name}__{target.checkpoint_label}__taskeval__{spec.label}"
    task_runs_root = args.output_root / "task_eval_runs"
    output_dir = task_runs_root / run_name
    metrics_path = output_dir / "metrics" / "metrics.json"

    if metrics_path.exists():
        metrics = load_json(metrics_path)
        record_task_eval_result(
            conn,
            checkpoint_id=checkpoint_id,
            dataset=spec.label,
            requested_k=args.task_pass_k,
            status="complete",
            output_dir=output_dir,
            metrics_path=metrics_path,
            metrics=metrics,
            error=None,
        )
        return metrics

    cmd = [
        sys.executable,
        str(PRIME_RL_RUNNER),
        "baseline",
        "--model",
        str(target.model_path),
        "--dataset",
        spec.dataset,
        "--output-root",
        str(task_runs_root),
        "--run-name",
        run_name,
        "--rollouts-per-prompt",
        str(args.task_pass_k),
        "--temperature",
        str(args.task_temperature),
        "--top-p",
        str(args.task_top_p),
        "--max-tokens",
        str(args.task_max_tokens),
        "--tensor-parallel-size",
        str(args.task_tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.task_gpu_memory_utilization),
        "--wandb-mode",
        "disabled",
    ]
    if args.task_max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.task_max_model_len)])
    if spec.dataset_path is not None:
        cmd.extend(["--dataset-path", str(spec.dataset_path)])
    if spec.data_source:
        cmd.extend(["--data-source", spec.data_source])
    if args.task_max_samples is not None:
        cmd.extend(["--max-samples", str(args.task_max_samples)])

    if args.dry_run:
        record_task_eval_result(
            conn,
            checkpoint_id=checkpoint_id,
            dataset=spec.label,
            requested_k=args.task_pass_k,
            status="planned",
            output_dir=output_dir,
            metrics_path=metrics_path,
            metrics=None,
            error=None,
        )
        return None

    try:
        run_subprocess(cmd)
        metrics = load_json(metrics_path)
    except Exception as exc:  # noqa: BLE001
        record_task_eval_result(
            conn,
            checkpoint_id=checkpoint_id,
            dataset=spec.label,
            requested_k=args.task_pass_k,
            status="failed",
            output_dir=output_dir,
            metrics_path=metrics_path if metrics_path.exists() else None,
            metrics=None,
            error=str(exc),
        )
        if not args.continue_on_error:
            raise
        return None

    record_task_eval_result(
        conn,
        checkpoint_id=checkpoint_id,
        dataset=spec.label,
        requested_k=args.task_pass_k,
        status="complete",
        output_dir=output_dir,
        metrics_path=metrics_path,
        metrics=metrics,
        error=None,
    )
    return metrics


def flatten_numeric_metrics(
    value: Any,
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
    max_file_mb: float,
) -> list[tuple[str, float, str]]:
    metrics: list[tuple[str, float, str]] = []
    max_bytes = int(max_file_mb * 1024 * 1024)
    for json_path in sorted(benchmark_dir.rglob("*.json")):
        if json_path.name in {"status.json", "eval_summary.json"}:
            continue
        try:
            if json_path.stat().st_size > max_bytes:
                continue
            payload = load_json(json_path)
        except (OSError, json.JSONDecodeError):
            continue
        flattened: dict[str, float] = {}
        flatten_numeric_metrics(payload, [], flattened)
        source_file = str(json_path.relative_to(benchmark_dir))
        for metric_key, metric_value in flattened.items():
            metrics.append((metric_key, metric_value, source_file))
    return metrics


def select_primary_metric(metrics: Iterable[tuple[str, float, str]]) -> tuple[str, float] | None:
    metric_list = list(metrics)
    if not metric_list:
        return None
    for hint in PRIMARY_METRIC_HINTS:
        for metric_key, metric_value, _source in metric_list:
            if metric_key.endswith(hint) or hint in metric_key:
                return metric_key, metric_value
    metric_key, metric_value, _source = metric_list[0]
    return metric_key, metric_value


def record_benchmark_suite_run(
    conn: sqlite3.Connection,
    checkpoint_id: int,
    suite: str,
    status: str,
    output_dir: Path,
    eval_summary_path: Path | None,
    error: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO benchmark_suite_runs (
            checkpoint_id, suite, status, output_dir, eval_summary_path, evaluated_at, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(checkpoint_id, suite) DO UPDATE SET
            status=excluded.status,
            output_dir=excluded.output_dir,
            eval_summary_path=excluded.eval_summary_path,
            evaluated_at=excluded.evaluated_at,
            error=excluded.error
        """,
        (
            checkpoint_id,
            suite,
            status,
            str(output_dir),
            str(eval_summary_path) if eval_summary_path else None,
            utc_now(),
            error,
        ),
    )
    conn.commit()


def record_benchmark_statuses(
    conn: sqlite3.Connection,
    checkpoint_id: int,
    suite: str,
    target_output_dir: Path,
    suite_results: dict[str, Any],
    max_file_mb: float,
) -> list[tuple[str, str, float]]:
    conn.execute(
        "DELETE FROM benchmark_metrics WHERE checkpoint_id = ? AND suite = ?",
        (checkpoint_id, suite),
    )
    logged_primary: list[tuple[str, str, float]] = []
    for benchmark, result in suite_results.items():
        benchmark_dir = target_output_dir / suite / benchmark
        status_path = benchmark_dir / "status.json"
        status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
        runner = result.get("runner") if isinstance(result, dict) else None
        error = result.get("error") if isinstance(result, dict) else None
        extracted_metrics = extract_metrics_from_benchmark_dir(benchmark_dir, max_file_mb=max_file_mb)
        primary_metric = select_primary_metric(extracted_metrics)
        primary_name = primary_metric[0] if primary_metric else None
        primary_value = primary_metric[1] if primary_metric else None
        conn.execute(
            """
            INSERT INTO benchmark_statuses (
                checkpoint_id, suite, benchmark, status, runner, benchmark_dir,
                status_path, primary_metric_name, primary_metric_value, error, evaluated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(checkpoint_id, suite, benchmark) DO UPDATE SET
                status=excluded.status,
                runner=excluded.runner,
                benchmark_dir=excluded.benchmark_dir,
                status_path=excluded.status_path,
                primary_metric_name=excluded.primary_metric_name,
                primary_metric_value=excluded.primary_metric_value,
                error=excluded.error,
                evaluated_at=excluded.evaluated_at
            """,
            (
                checkpoint_id,
                suite,
                benchmark,
                status,
                runner,
                str(benchmark_dir),
                str(status_path) if status_path.exists() else None,
                primary_name,
                primary_value,
                error,
                utc_now(),
            ),
        )
        for metric_key, metric_value, source_file in extracted_metrics:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_metrics (
                    checkpoint_id, suite, benchmark, metric_key, metric_value, source_file
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (checkpoint_id, suite, benchmark, metric_key, metric_value, source_file),
            )
        if primary_metric is not None:
            logged_primary.append((benchmark, primary_metric[0], primary_metric[1]))
    conn.commit()
    return logged_primary


def run_benchmark_suite(
    args: argparse.Namespace,
    conn: sqlite3.Connection,
    checkpoint_id: int,
    target: CheckpointTarget,
    suite: str,
) -> list[tuple[str, str, float]]:
    if benchmark_suite_already_complete(conn, checkpoint_id, suite):
        output_dir = args.output_root / "benchmark_evals" / target.run_name / target.checkpoint_label / suite
        return import_existing_benchmark_results(
            conn,
            checkpoint_id=checkpoint_id,
            suite=suite,
            output_dir=output_dir,
            max_file_mb=args.max_json_metric_file_mb,
        )

    output_dir = args.output_root / "benchmark_evals" / target.run_name / target.checkpoint_label / suite
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eval_summary.json"

    if summary_path.exists():
        record_benchmark_suite_run(
            conn,
            checkpoint_id=checkpoint_id,
            suite=suite,
            status="complete",
            output_dir=output_dir,
            eval_summary_path=summary_path,
            error=None,
        )
        return import_existing_benchmark_results(
            conn,
            checkpoint_id=checkpoint_id,
            suite=suite,
            output_dir=output_dir,
            max_file_mb=args.max_json_metric_file_mb,
        )

    extra_args: list[str] = []
    for benchmark_root in args.benchmark_root:
        extra_args.extend(["--benchmark-root", benchmark_root])
    for benchmark_command in args.benchmark_command:
        extra_args.extend(["--benchmark-command", benchmark_command])
    if args.served_model_name:
        extra_args.extend(["--served-model-name", args.served_model_name])
    if args.openai_base_url:
        extra_args.extend(["--openai-base-url", args.openai_base_url])
    if args.openai_api_key:
        extra_args.extend(["--openai-api-key", args.openai_api_key])

    if args.continue_on_error:
        extra_args.append("--continue_on_error")

    run_name = f"{target.run_name}__{target.checkpoint_label}"
    if args.auto_start_eval_server and not args.openai_base_url:
        env = os.environ.copy()
        env["EVAL_SUITE"] = suite
        env["CONTINUE_ON_ERROR"] = "1" if args.continue_on_error else "0"
        env["BENCHMARK_ENV_FILE"] = str(expand(args.benchmark_env_file))
        env["EVAL_SERVER_PORT"] = str(args.eval_server_port)
        env["EVAL_SERVER_TP"] = str(args.eval_server_tp)
        env["EVAL_SERVER_GPU_MEMORY_UTILIZATION"] = str(args.eval_server_gpu_memory_utilization)
        if args.eval_server_max_model_len is not None:
            env["EVAL_SERVER_MAX_MODEL_LEN"] = str(args.eval_server_max_model_len)
        env["EVAL_SERVER_STARTUP_TIMEOUT"] = str(args.eval_server_startup_timeout)
        env["EVAL_SERVER_API_KEY"] = str(args.eval_server_api_key)
        cmd = [
            "bash",
            str(RUN_EVAL_WITH_LOCAL_SERVER),
            str(target.model_path),
            str(output_dir),
            run_name,
            *extra_args,
        ]
    else:
        cmd = [
            sys.executable,
            str(RUN_EVAL_SCRIPT),
            "--model_path",
            str(target.model_path),
            "--suite",
            suite,
            "--output_dir",
            str(output_dir),
            "--run_name",
            run_name,
            *extra_args,
        ]
        env = None

    if args.dry_run:
        record_benchmark_suite_run(
            conn,
            checkpoint_id=checkpoint_id,
            suite=suite,
            status="planned",
            output_dir=output_dir,
            eval_summary_path=summary_path,
            error=None,
        )
        return []

    try:
        run_subprocess(cmd, env=env)
    except Exception as exc:  # noqa: BLE001
        record_benchmark_suite_run(
            conn,
            checkpoint_id=checkpoint_id,
            suite=suite,
            status="failed",
            output_dir=output_dir,
            eval_summary_path=summary_path if summary_path.exists() else None,
            error=str(exc),
        )
        if not args.continue_on_error:
            raise
        return []

    record_benchmark_suite_run(
        conn,
        checkpoint_id=checkpoint_id,
        suite=suite,
        status="complete",
        output_dir=output_dir,
        eval_summary_path=summary_path,
        error=None,
    )
    return import_existing_benchmark_results(
        conn,
        checkpoint_id=checkpoint_id,
        suite=suite,
        output_dir=output_dir,
        max_file_mb=args.max_json_metric_file_mb,
    )


def import_existing_benchmark_results(
    conn: sqlite3.Connection,
    checkpoint_id: int,
    suite: str,
    output_dir: Path,
    max_file_mb: float,
) -> list[tuple[str, str, float]]:
    summary_path = output_dir / "eval_summary.json"
    if not summary_path.exists():
        return []
    summary = load_json(summary_path)
    targets = summary.get("targets", {})
    if not targets:
        return []
    target_label, suite_map = next(iter(targets.items()))
    suite_results = suite_map.get(suite, {})
    target_output_dir = output_dir / target_label
    return record_benchmark_statuses(
        conn,
        checkpoint_id=checkpoint_id,
        suite=suite,
        target_output_dir=target_output_dir,
        suite_results=suite_results,
        max_file_mb=max_file_mb,
    )


class WandbRelay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._wandb = None
        self._runs: dict[str, Any] = {}
        if args.wandb_mode == "disabled":
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        self._wandb = wandb
        self._wandb_dir = str(expand(args.wandb_dir))

    def available(self) -> bool:
        return self._wandb is not None

    def run(self, run_name: str) -> Any | None:
        if not self.available():
            return None
        if run_name in self._runs:
            return self._runs[run_name]
        handle = self._wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=f"eval_{run_name}",
            group="prime_checkpoint_eval",
            dir=self._wandb_dir,
            mode=self.args.wandb_mode,
            reinit=True,
            config={"eval_script": "eval_prime_checkpoint_sweep.py"},
        )
        self._runs[run_name] = handle
        return handle

    def log_checkpoint_metrics(
        self,
        run_name: str,
        step: int | None,
        task_metrics: list[tuple[str, dict[str, Any]]],
        benchmark_metrics: list[tuple[str, str, float]],
    ) -> None:
        handle = self.run(run_name)
        if handle is None:
            return
        payload: dict[str, float] = {}
        for dataset, metrics in task_metrics:
            pass_key = f"task_eval/{dataset}/pass_at_{int(metrics.get('rollouts_per_prompt', 0) or 0)}"
            payload[pass_key] = float(metrics.get("pass_at_k", 0.0))
            payload[f"task_eval/{dataset}/accuracy_at_1"] = float(metrics.get("accuracy_at_1", 0.0))
        for benchmark, metric_name, metric_value in benchmark_metrics:
            safe_name = sanitize_metric_name(metric_name.replace("/", "__"))
            payload[f"benchmark/{benchmark}/{safe_name}"] = metric_value
        if not payload:
            return
        handle.log(payload, step=step if step is not None else 0)

    def finish(self) -> None:
        if not self.available():
            return
        for handle in self._runs.values():
            handle.finish()
        self._runs.clear()


def plot_task_eval_curves(conn: sqlite3.Connection, plot_dir: Path) -> None:
    rows = conn.execute(
        """
        SELECT c.run_name, c.step, t.dataset, t.accuracy_at_1, t.pass_at_k, t.requested_k
        FROM task_eval_results t
        JOIN checkpoints c ON c.id = t.checkpoint_id
        WHERE t.status = 'complete'
        ORDER BY c.run_name, c.step, t.dataset
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

    task_dir = plot_dir / "task_eval"
    task_dir.mkdir(parents=True, exist_ok=True)

    for run_name, run_rows in grouped.items():
        datasets = sorted({row[2] for row in run_rows})
        steps = sorted({row[1] if row[1] is not None else 0 for row in run_rows})
        if not steps:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for dataset in datasets:
            dataset_rows = [row for row in run_rows if row[2] == dataset]
            dataset_steps = [row[1] if row[1] is not None else 0 for row in dataset_rows]
            dataset_acc = [row[3] for row in dataset_rows]
            dataset_pass = [row[4] for row in dataset_rows]
            axes[0].plot(dataset_steps, dataset_acc, marker="o", label=dataset)
            axes[1].plot(dataset_steps, dataset_pass, marker="o", label=dataset)
        requested_k = next((row[5] for row in run_rows if row[5] is not None), None)
        axes[0].set_title(f"{run_name}: accuracy@1")
        axes[1].set_title(f"{run_name}: pass@{requested_k or 'k'}")
        for axis in axes:
            axis.set_xlabel("Checkpoint step")
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.3)
            axis.legend()
        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("Pass rate")
        fig.tight_layout()
        fig.savefig(task_dir / f"{sanitize_metric_name(run_name)}.png", dpi=150)
        plt.close(fig)


def plot_benchmark_progress(conn: sqlite3.Connection, plot_dir: Path) -> None:
    rows = conn.execute(
        """
        SELECT c.run_name, c.step, COUNT(*)
        FROM benchmark_statuses b
        JOIN checkpoints c ON c.id = b.checkpoint_id
        WHERE b.status = 'complete'
        GROUP BY c.run_name, c.step
        ORDER BY c.run_name, c.step
        """
    ).fetchall()
    grouped: dict[str, list[tuple[int, int]]] = {}
    for run_name, step, complete_count in rows:
        grouped.setdefault(run_name, []).append((step if step is not None else 0, complete_count))

    benchmark_dir = plot_dir / "benchmark_progress"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for run_name, series in grouped.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = [step for step, _count in series]
        ys = [count for _step, count in series]
        ax.plot(xs, ys, marker="o")
        ax.set_title(f"{run_name}: completed benchmark-plan evals")
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Benchmarks complete")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(benchmark_dir / f"{sanitize_metric_name(run_name)}.png", dpi=150)
        plt.close(fig)


def plot_primary_benchmark_metrics(conn: sqlite3.Connection, plot_dir: Path) -> None:
    rows = conn.execute(
        """
        SELECT c.run_name, c.step, b.benchmark, b.primary_metric_name, b.primary_metric_value
        FROM benchmark_statuses b
        JOIN checkpoints c ON c.id = b.checkpoint_id
        WHERE b.status = 'complete' AND b.primary_metric_value IS NOT NULL
        ORDER BY c.run_name, b.benchmark, c.step
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

    metrics_dir = plot_dir / "benchmark_primary"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for run_name, run_rows in grouped.items():
        benchmarks = sorted({row[2] for row in run_rows})
        if not benchmarks:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        for benchmark in benchmarks:
            benchmark_rows = [row for row in run_rows if row[2] == benchmark]
            xs = [row[1] if row[1] is not None else 0 for row in benchmark_rows]
            ys = [row[4] for row in benchmark_rows]
            ax.plot(xs, ys, marker="o", label=benchmark)
        ax.set_title(f"{run_name}: benchmark primary metrics")
        ax.set_xlabel("Checkpoint step")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(metrics_dir / f"{sanitize_metric_name(run_name)}.png", dpi=150)
        plt.close(fig)


def regenerate_plots(conn: sqlite3.Connection, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_task_eval_curves(conn, plot_dir)
    plot_benchmark_progress(conn, plot_dir)
    plot_primary_benchmark_metrics(conn, plot_dir)


def summarize_processed(
    run: PrimeRun,
    target: CheckpointTarget,
    task_metrics: list[tuple[str, dict[str, Any]]],
    benchmark_metrics: list[tuple[str, str, float]],
) -> str:
    task_summary = ", ".join(
        f"{dataset}:pass={metrics.get('pass_at_k', 0.0):.3f}"
        for dataset, metrics in task_metrics
    )
    benchmark_summary = ", ".join(
        f"{benchmark}:{metric_value:.3f}"
        for benchmark, _metric_name, metric_value in benchmark_metrics[:5]
    )
    parts = [f"{run.name}/{target.checkpoint_label}"]
    if task_summary:
        parts.append(task_summary)
    if benchmark_summary:
        parts.append(benchmark_summary)
    return " | ".join(parts)


def main() -> int:
    args = parse_args()
    args.prime_runs_root = expand(args.prime_runs_root)
    args.output_root = expand(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.db_path = expand(args.db_path) if args.db_path else args.output_root / "eval_registry.sqlite"
    args.plot_dir = expand(args.plot_dir) if args.plot_dir else args.output_root / "plots"

    selected_suites = args.suite or ["tasks_md"]
    conn = sqlite3.connect(args.db_path)
    ensure_db(conn)
    wandb_relay = WandbRelay(args)

    try:
        runs = discover_runs(
            root=args.prime_runs_root,
            selected_names=set(args.run),
            only_completed=args.only_completed,
        )
        processed = 0
        for run in runs:
            upsert_run(conn, run)
            targets = trim_targets(
                discover_checkpoint_targets(run),
                only_latest=args.only_latest,
                max_checkpoints_per_run=args.max_checkpoints_per_run,
            )
            for target in targets:
                checkpoint_id = upsert_checkpoint(conn, target)
                task_metrics_for_wandb: list[tuple[str, dict[str, Any]]] = []
                benchmark_metrics_for_wandb: list[tuple[str, str, float]] = []

                if not args.skip_task_evals:
                    for spec in task_eval_specs_for_run(args, run):
                        result = run_task_eval(args, conn, checkpoint_id, target, spec)
                        if result is not None:
                            task_metrics_for_wandb.append((spec.label, result))

                if not args.skip_benchmark_evals:
                    for suite in selected_suites:
                        benchmark_metrics_for_wandb.extend(
                            run_benchmark_suite(args, conn, checkpoint_id, target, suite)
                        )

                regenerate_plots(conn, args.plot_dir)
                wandb_relay.log_checkpoint_metrics(
                    run_name=run.name,
                    step=target.step,
                    task_metrics=task_metrics_for_wandb,
                    benchmark_metrics=benchmark_metrics_for_wandb,
                )
                processed += 1
                print(
                    summarize_processed(
                        run,
                        target,
                        task_metrics_for_wandb,
                        benchmark_metrics_for_wandb,
                    )
                )

        print(f"Processed {processed} checkpoint target(s)")
        print(f"SQLite DB: {args.db_path}")
        print(f"Plots: {args.plot_dir}")
        return 0
    finally:
        wandb_relay.finish()
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
