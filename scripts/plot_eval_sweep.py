"""Plot forgetting curves from eval sweep results.

Reads lm_eval JSON outputs and produces:
1. Forgetting curves — all benchmarks vs training step
2. Delta from baseline — change from step 0, shading below 0
3. Heatmap — benchmarks × steps, color = delta
4. Average forgetting — composite score vs step

Usage:
    python scripts/plot_eval_sweep.py --results_dir ~/scratch/forgetting-llms/eval_results/grpo_full_qwen3_1.7b_gsm8k
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS = [
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "winogrande",
    "piqa",
    "boolq",
    "openbookqa",
    "truthfulqa_mc2",
    "mmlu",
    "ifeval",
]

# Metric to extract per benchmark (lm_eval default metric names)
METRIC_MAP = {
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "piqa": "acc_norm,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    "mmlu": "acc,none",
    "ifeval": "prompt_level_strict_acc,none",
}


def find_results_json(result_dir: Path) -> dict | None:
    """Find and load the lm_eval results JSON from a result directory."""
    # lm_eval saves results under results_dir/<model_hash>/results_*.json
    for root, _dirs, files in os.walk(result_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                with open(os.path.join(root, f)) as fp:
                    return json.load(fp)
    return None


def extract_step(dirname: str) -> int:
    """Extract step number from directory name like 'global_step_200'."""
    if dirname == "base_model":
        return 0
    match = re.search(r"global_step_(\d+)", dirname)
    return int(match.group(1)) if match else -1


def extract_scores(results_json: dict) -> dict[str, float]:
    """Extract benchmark scores from lm_eval results JSON."""
    scores = {}
    results = results_json.get("results", {})
    for bench in BENCHMARKS:
        if bench in results:
            metric_key = METRIC_MAP.get(bench, "acc,none")
            val = results[bench].get(metric_key)
            if val is None:
                # Fallback: try acc,none
                val = results[bench].get("acc,none")
            if val is not None:
                scores[bench] = float(val)
    return scores


def load_all_results(results_dir: Path) -> tuple[list[int], list[dict[str, float]]]:
    """Load all eval results, sorted by step."""
    entries = []
    for d in results_dir.iterdir():
        if not d.is_dir() or d.name == "plots":
            continue
        step = extract_step(d.name)
        if step < 0:
            continue
        data = find_results_json(d)
        if data is None:
            print(f"WARNING: No results JSON found in {d}")
            continue
        scores = extract_scores(data)
        if scores:
            entries.append((step, scores))

    entries.sort(key=lambda x: x[0])
    steps = [e[0] for e in entries]
    score_dicts = [e[1] for e in entries]
    return steps, score_dicts


def plot_forgetting_curves(steps, score_dicts, output_dir):
    """Plot 1: All benchmarks vs training step."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for bench in BENCHMARKS:
        vals = [sd.get(bench) for sd in score_dicts]
        if any(v is not None for v in vals):
            valid_steps = [s for s, v in zip(steps, vals) if v is not None]
            valid_vals = [v for v in vals if v is not None]
            ax.plot(valid_steps, valid_vals, "o-", label=bench, markersize=5, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Benchmark Performance During GRPO Training", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "forgetting_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'forgetting_curves.png'}")


def plot_deltas(steps, score_dicts, output_dir):
    """Plot 2: Delta from baseline with shading below 0."""
    if len(score_dicts) < 2:
        print("Not enough data points for delta plot")
        return

    baseline = score_dicts[0]
    fig, ax = plt.subplots(figsize=(12, 6))

    for bench in BENCHMARKS:
        base_val = baseline.get(bench)
        if base_val is None:
            continue
        deltas = []
        valid_steps = []
        for s, sd in zip(steps, score_dicts):
            v = sd.get(bench)
            if v is not None:
                valid_steps.append(s)
                deltas.append(v - base_val)

        line, = ax.plot(valid_steps, deltas, "o-", label=bench, markersize=5, linewidth=2)
        # Shade below zero
        ax.fill_between(
            valid_steps, deltas, 0,
            where=[d < 0 for d in deltas],
            alpha=0.1, color=line.get_color(),
        )

    ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Delta from Baseline", fontsize=12)
    ax.set_title("Change in Benchmark Performance (Forgetting = Below 0)", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "delta_from_baseline.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'delta_from_baseline.png'}")


def plot_heatmap(steps, score_dicts, output_dir):
    """Plot 3: Heatmap of benchmarks × steps, color = delta from baseline."""
    if len(score_dicts) < 2:
        print("Not enough data points for heatmap")
        return

    baseline = score_dicts[0]
    present_benchmarks = [b for b in BENCHMARKS if b in baseline]

    matrix = []
    for bench in present_benchmarks:
        row = []
        for sd in score_dicts:
            v = sd.get(bench)
            base_v = baseline.get(bench)
            if v is not None and base_v is not None:
                row.append(v - base_v)
            else:
                row.append(0.0)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 1.5), 6))

    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right")
    ax.set_yticks(range(len(present_benchmarks)))
    ax.set_yticklabels(present_benchmarks)
    ax.set_xlabel("Training Step")
    ax.set_title("Benchmark Delta Heatmap (Green = Improvement, Red = Forgetting)")

    # Annotate cells
    for i in range(len(present_benchmarks)):
        for j in range(len(steps)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(val) < vmax * 0.6 else "white")

    fig.colorbar(im, ax=ax, label="Delta from baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'heatmap.png'}")


def plot_average_forgetting(steps, score_dicts, output_dir):
    """Plot 4: Average forgetting (composite score) vs step."""
    if len(score_dicts) < 2:
        print("Not enough data points for average forgetting plot")
        return

    baseline = score_dicts[0]
    avg_deltas = []
    avg_scores = []

    for sd in score_dicts:
        deltas = []
        scores = []
        for bench in BENCHMARKS:
            v = sd.get(bench)
            base_v = baseline.get(bench)
            if v is not None and base_v is not None:
                deltas.append(v - base_v)
                scores.append(v)
        avg_deltas.append(sum(deltas) / len(deltas) if deltas else 0)
        avg_scores.append(sum(scores) / len(scores) if scores else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Average delta
    ax1.plot(steps, avg_deltas, "o-", color="crimson", linewidth=2, markersize=6)
    ax1.fill_between(steps, avg_deltas, 0,
                     where=[d < 0 for d in avg_deltas],
                     alpha=0.2, color="red")
    ax1.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Avg Delta from Baseline", fontsize=12)
    ax1.set_title("Average Forgetting Across Benchmarks", fontsize=13)
    ax1.grid(True, alpha=0.3)
    for s, d in zip(steps, avg_deltas):
        ax1.annotate(f"{d:+.3f}", (s, d), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

    # Average absolute score
    ax2.plot(steps, avg_scores, "o-", color="steelblue", linewidth=2, markersize=6)
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Avg Benchmark Accuracy", fontsize=12)
    ax2.set_title("Average Benchmark Score", fontsize=13)
    ax2.grid(True, alpha=0.3)
    for s, a in zip(steps, avg_scores):
        ax2.annotate(f"{a:.3f}", (s, a), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "average_forgetting.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'average_forgetting.png'}")


def write_summary(steps, score_dicts, output_dir):
    """Write summary JSON and print summary table."""
    if not score_dicts:
        return

    baseline = score_dicts[0] if score_dicts else {}
    summary = {"steps": steps, "benchmarks": {}, "baseline": baseline}

    for bench in BENCHMARKS:
        bench_data = []
        for step, sd in zip(steps, score_dicts):
            val = sd.get(bench)
            base_val = baseline.get(bench)
            bench_data.append({
                "step": step,
                "score": val,
                "delta": (val - base_val) if val is not None and base_val is not None else None,
            })
        summary["benchmarks"][bench] = bench_data

    summary_path = output_dir.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Print table
    print(f"\n{'='*80}")
    print(f"{'Benchmark':<20}", end="")
    for s in steps:
        print(f"{'Step '+str(s):>10}", end="")
    print()
    print("-" * 80)

    for bench in BENCHMARKS:
        if bench not in baseline:
            continue
        print(f"{bench:<20}", end="")
        for sd in score_dicts:
            v = sd.get(bench)
            if v is not None:
                delta = v - baseline[bench]
                marker = " " if abs(delta) < 0.001 else ("+" if delta > 0 else "-")
                print(f"{v:>9.4f}{marker}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print("-" * 80)
    # Average row
    print(f"{'AVERAGE':<20}", end="")
    for sd in score_dicts:
        vals = [sd.get(b) for b in BENCHMARKS if sd.get(b) is not None]
        avg = sum(vals) / len(vals) if vals else 0
        print(f"{avg:>10.4f}", end="")
    print()
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval sweep forgetting curves")
    parser.add_argument("--results_dir", required=True, help="Directory with eval results")
    parser.add_argument("--output_dir", default=None, help="Output directory for plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    steps, score_dicts = load_all_results(results_dir)

    if not steps:
        print("ERROR: No results found!")
        return

    print(f"Found {len(steps)} evaluations at steps: {steps}")

    plot_forgetting_curves(steps, score_dicts, output_dir)
    plot_deltas(steps, score_dicts, output_dir)
    plot_heatmap(steps, score_dicts, output_dir)
    plot_average_forgetting(steps, score_dicts, output_dir)
    write_summary(steps, score_dicts, output_dir)


if __name__ == "__main__":
    main()
