#!/usr/bin/env python3
"""Generate a lightweight markdown board from checkpoints and eval outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunStatus:
    name: str
    checkpoint_count: int
    latest_step: int | None
    ood_completed: int
    ood_expected: int | None
    task_eval_completed: int | None
    task_eval_expected: int | None
    last_updated: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        default="~/scratch/forgetting-llms/checkpoints",
        help="Root directory containing run checkpoint folders.",
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default="~/scratch/forgetting-llms/eval_results",
        help="Root directory containing evaluation result folders.",
    )
    parser.add_argument(
        "--prime_runs_root",
        type=str,
        default="~/scratch/forgetting-llms/prime_runs",
        help="Optional PRIME-RL runs root containing bundle dirs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="EXPERIMENT_BOARD.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def latest_mtime(paths: list[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else None


def count_nonempty_dirs(paths: list[Path]) -> int:
    return sum(1 for path in paths if path.is_dir() and any(path.iterdir()))


def load_task_eval_counts(eval_dir: Path) -> tuple[int | None, int | None]:
    if not eval_dir.exists():
        return None, None
    task_eval_files = sorted(eval_dir.glob("task_accuracy*.json"))
    if not task_eval_files:
        return None, None
    return len(task_eval_files), None


def collect_run_status(run_name: str, checkpoints_root: Path, eval_root: Path, prime_runs_root: Path) -> RunStatus:
    checkpoint_dir = checkpoints_root / run_name
    eval_dir = eval_root / run_name
    prime_run_dir = prime_runs_root / run_name

    checkpoint_paths = sorted(checkpoint_dir.glob("global_step_*"), key=checkpoint_step) if checkpoint_dir.exists() else []
    checkpoint_count = len(checkpoint_paths)
    latest_step = checkpoint_step(checkpoint_paths[-1]) if checkpoint_paths else None

    if checkpoint_count == 0 and (prime_run_dir / "checkpoints").exists():
        prime_checkpoint_dir = prime_run_dir / "checkpoints"
        checkpoint_paths = [path.parent for path in prime_checkpoint_dir.rglob("config.json")]
        checkpoint_paths = sorted(set(checkpoint_paths), key=lambda p: str(p))
        checkpoint_count = len(checkpoint_paths)
        if checkpoint_count == 0:
            prime_entries = list(prime_checkpoint_dir.iterdir())
            checkpoint_count = len(prime_entries)
        latest_step = None

    ood_dirs: list[Path] = []
    if eval_dir.exists():
        for child in eval_dir.iterdir():
            if not child.is_dir() or child.name == "plots":
                continue
            ood_dirs.append(child)

    ood_completed = count_nonempty_dirs(ood_dirs)
    ood_expected = checkpoint_count + 1 if checkpoint_count else (ood_completed or None)

    task_eval_completed, task_eval_expected = load_task_eval_counts(eval_dir)

    last_updated = latest_mtime([checkpoint_dir, eval_dir, prime_run_dir, *checkpoint_paths, *ood_dirs])

    return RunStatus(
        name=run_name,
        checkpoint_count=checkpoint_count,
        latest_step=latest_step,
        ood_completed=ood_completed,
        ood_expected=ood_expected,
        task_eval_completed=task_eval_completed,
        task_eval_expected=task_eval_expected,
        last_updated=last_updated,
    )


def format_count(completed: int | None, expected: int | None) -> str:
    if completed is None and expected is None:
        return "not started"
    if completed is None:
        return f"0/{expected}"
    if expected is None:
        return str(completed)
    return f"{completed}/{expected}"


def format_timestamp(epoch_seconds: float | None) -> str:
    if epoch_seconds is None:
        return "n/a"
    return datetime.fromtimestamp(epoch_seconds).strftime("%Y-%m-%d %H:%M")


def render_markdown(runs: list[RunStatus], checkpoints_root: Path, eval_root: Path, prime_runs_root: Path) -> str:
    total_runs = len(runs)
    with_checkpoints = sum(1 for run in runs if run.checkpoint_count > 0)
    with_ood = sum(1 for run in runs if run.ood_completed > 0)
    with_task_eval = sum(1 for run in runs if run.task_eval_completed)

    lines = [
        "# Experiment Board",
        "",
        f"- Checkpoints root: `{checkpoints_root}`",
        f"- Eval root: `{eval_root}`",
        f"- PRIME runs root: `{prime_runs_root}`",
        f"- Total runs discovered: **{total_runs}**",
        f"- Runs with checkpoints: **{with_checkpoints}**",
        f"- Runs with OOD eval output: **{with_ood}**",
        f"- Runs with task-accuracy output: **{with_task_eval}**",
        "",
        "| Run | Ckpts | Latest step | OOD eval | Task eval | Updated |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]

    for run in runs:
        lines.append(
            "| "
            f"{run.name} | "
            f"{run.checkpoint_count} | "
            f"{run.latest_step if run.latest_step is not None else 'n/a'} | "
            f"{format_count(run.ood_completed, run.ood_expected)} | "
            f"{format_count(run.task_eval_completed, run.task_eval_expected)} | "
            f"{format_timestamp(run.last_updated)} |"
        )

    lines.append("")
    lines.append("Update this board with:")
    lines.append("```bash")
    lines.append("python scripts/update_experiment_board.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    checkpoints_root = Path(args.checkpoints_root).expanduser()
    eval_root = Path(args.eval_root).expanduser()
    prime_runs_root = Path(args.prime_runs_root).expanduser()
    output_path = Path(args.output)

    run_names = set()
    if checkpoints_root.exists():
        run_names.update(path.name for path in checkpoints_root.iterdir() if path.is_dir())
    if eval_root.exists():
        run_names.update(path.name for path in eval_root.iterdir() if path.is_dir())
    if prime_runs_root.exists():
        run_names.update(path.name for path in prime_runs_root.iterdir() if path.is_dir())

    runs = [
        collect_run_status(
            run_name,
            checkpoints_root=checkpoints_root,
            eval_root=eval_root,
            prime_runs_root=prime_runs_root,
        )
        for run_name in sorted(run_names)
    ]

    markdown = render_markdown(
        runs,
        checkpoints_root=checkpoints_root,
        eval_root=eval_root,
        prime_runs_root=prime_runs_root,
    )
    output_path.write_text(markdown)

    if args.json_output:
        json_output = Path(args.json_output)
        payload = {
            "checkpoints_root": str(checkpoints_root),
            "eval_root": str(eval_root),
            "prime_runs_root": str(prime_runs_root),
            "runs": [run.__dict__ for run in runs],
        }
        json_output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote experiment board to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
