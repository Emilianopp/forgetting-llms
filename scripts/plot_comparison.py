"""Plot comparison of forgetting curves across training methods.

Overlays eval sweep results from multiple methods (GRPO, GT-SFT, SF-SFT, etc.)
on the same axes for direct comparison.

Usage:
    python scripts/plot_comparison.py \
        --methods grpo=~/scratch/forgetting-llms/eval_results/grpo_full_qwen3_1.7b_gsm8k \
                  gt_sft=~/scratch/forgetting-llms/eval_results/gt_sft_qwen3_1.7b_gsm8k \
        --output_dir ~/scratch/forgetting-llms/eval_results/comparison_plots \
        --max_step 1200
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
]

METRIC_MAP = {
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "piqa": "acc_norm,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
}

METHOD_STYLES = {
    "grpo": {"color": "#e74c3c", "marker": "o", "label": "GRPO (Online RL)"},
    "gt_sft": {"color": "#2ecc71", "marker": "s", "label": "GT-SFT"},
    "sf_sft": {"color": "#3498db", "marker": "^", "label": "SF-SFT (Teacher)"},
    "cf_sft": {"color": "#9b59b6", "marker": "D", "label": "CF-SFT (Cross-Family)"},
}


def find_results_json(result_dir: Path) -> dict | None:
    for root, _dirs, files in os.walk(result_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                with open(os.path.join(root, f)) as fp:
                    return json.load(fp)
    return None


def extract_step(dirname: str) -> int:
    if dirname == "base_model":
        return 0
    match = re.search(r"global_step_(\d+)", dirname)
    return int(match.group(1)) if match else -1


def extract_scores(results_json: dict) -> dict[str, float]:
    scores = {}
    results = results_json.get("results", {})
    for bench in BENCHMARKS:
        if bench in results:
            metric_key = METRIC_MAP.get(bench, "acc,none")
            val = results[bench].get(metric_key)
            if val is None:
                val = results[bench].get("acc,none")
            if val is not None:
                scores[bench] = float(val)
    return scores


def load_method_results(
    results_dir: Path, max_step: int | None = None
) -> tuple[list[int], list[dict[str, float]]]:
    entries = []
    for d in results_dir.iterdir():
        if not d.is_dir() or d.name in ("plots", "comparison_plots"):
            continue
        step = extract_step(d.name)
        if step < 0:
            continue
        if max_step is not None and step > max_step:
            continue
        data = find_results_json(d)
        if data is None:
            continue
        scores = extract_scores(data)
        if scores:
            entries.append((step, scores))

    entries.sort(key=lambda x: x[0])
    steps = [e[0] for e in entries]
    score_dicts = [e[1] for e in entries]
    return steps, score_dicts


def plot_avg_forgetting_comparison(methods_data, output_dir):
    """Plot average forgetting delta across all benchmarks, one line per method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, (steps, score_dicts) in methods_data.items():
        if not score_dicts:
            continue
        baseline = score_dicts[0]
        avg_deltas = []
        for sd in score_dicts:
            deltas = []
            for bench in BENCHMARKS:
                v = sd.get(bench)
                base_v = baseline.get(bench)
                if v is not None and base_v is not None:
                    deltas.append(v - base_v)
            avg_deltas.append(sum(deltas) / len(deltas) if deltas else 0)

        style = METHOD_STYLES.get(method_name, {"color": "gray", "marker": "x", "label": method_name})
        ax.plot(steps, avg_deltas, f"{style['marker']}-",
                color=style["color"], label=style["label"],
                linewidth=2, markersize=6)
        ax.fill_between(steps, avg_deltas, 0,
                        where=[d < 0 for d in avg_deltas],
                        alpha=0.1, color=style["color"])

    ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Avg Delta from Baseline", fontsize=12)
    ax.set_title("Average Forgetting: Method Comparison (Qwen3-1.7B / GSM8K)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_forgetting_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'avg_forgetting_comparison.png'}")


def plot_per_benchmark_comparison(methods_data, output_dir):
    """Plot per-benchmark forgetting curves, one subplot per benchmark."""
    n_benchmarks = len(BENCHMARKS)
    ncols = 4
    nrows = (n_benchmarks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), sharey=False)
    axes = axes.flatten()

    for idx, bench in enumerate(BENCHMARKS):
        ax = axes[idx]
        for method_name, (steps, score_dicts) in methods_data.items():
            if not score_dicts:
                continue
            baseline = score_dicts[0]
            base_v = baseline.get(bench)
            if base_v is None:
                continue
            deltas = []
            valid_steps = []
            for s, sd in zip(steps, score_dicts):
                v = sd.get(bench)
                if v is not None:
                    valid_steps.append(s)
                    deltas.append(v - base_v)

            style = METHOD_STYLES.get(method_name, {"color": "gray", "marker": "x", "label": method_name})
            ax.plot(valid_steps, deltas, f"{style['marker']}-",
                    color=style["color"], label=style["label"],
                    linewidth=1.5, markersize=4)

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title(bench, fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_benchmarks, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel("Training Step", fontsize=12)
    fig.supylabel("Delta from Baseline", fontsize=12)
    fig.suptitle("Per-Benchmark Forgetting: Method Comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "per_benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'per_benchmark_comparison.png'}")


def plot_absolute_scores_comparison(methods_data, output_dir):
    """Plot average absolute benchmark score vs step."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, (steps, score_dicts) in methods_data.items():
        if not score_dicts:
            continue
        avg_scores = []
        for sd in score_dicts:
            vals = [sd.get(b) for b in BENCHMARKS if sd.get(b) is not None]
            avg_scores.append(sum(vals) / len(vals) if vals else 0)

        style = METHOD_STYLES.get(method_name, {"color": "gray", "marker": "x", "label": method_name})
        ax.plot(steps, avg_scores, f"{style['marker']}-",
                color=style["color"], label=style["label"],
                linewidth=2, markersize=6)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Avg Benchmark Accuracy", fontsize=12)
    ax.set_title("Average Benchmark Score: Method Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "absolute_scores_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'absolute_scores_comparison.png'}")


def plot_heatmap_comparison(methods_data, output_dir):
    """Side-by-side heatmaps for each method."""
    n_methods = len(methods_data)
    fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    all_vmax = 0
    matrices = {}
    for method_name, (steps, score_dicts) in methods_data.items():
        if len(score_dicts) < 2:
            continue
        baseline = score_dicts[0]
        present = [b for b in BENCHMARKS if b in baseline]
        matrix = []
        for bench in present:
            row = []
            for sd in score_dicts:
                v = sd.get(bench)
                bv = baseline.get(bench)
                row.append((v - bv) if v is not None and bv is not None else 0.0)
            matrix.append(row)
        matrix = np.array(matrix)
        matrices[method_name] = (steps, present, matrix)
        all_vmax = max(all_vmax, abs(matrix.min()), abs(matrix.max()))

    all_vmax = max(all_vmax, 0.01)

    for ax, (method_name, (steps, benchmarks, matrix)) in zip(axes, matrices.items()):
        style = METHOD_STYLES.get(method_name, {"label": method_name})
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-all_vmax, vmax=all_vmax)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(benchmarks)))
        ax.set_yticklabels(benchmarks, fontsize=9)
        ax.set_title(style["label"], fontsize=12)

        for i in range(len(benchmarks)):
            for j in range(len(steps)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < all_vmax * 0.6 else "white")

    fig.colorbar(im, ax=axes, label="Delta from baseline", shrink=0.8)
    fig.suptitle("Forgetting Heatmaps: Method Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'heatmap_comparison.png'}")


def write_comparison_summary(methods_data, output_dir):
    """Print and save comparison summary table."""
    print(f"\n{'='*90}")
    print(f"{'Method Comparison Summary':^90}")
    print(f"{'='*90}")

    for method_name, (steps, score_dicts) in methods_data.items():
        if not score_dicts:
            continue
        baseline = score_dicts[0]
        style = METHOD_STYLES.get(method_name, {"label": method_name})
        print(f"\n--- {style['label']} ({len(steps)} evaluations) ---")

        # Final step deltas
        final_sd = score_dicts[-1]
        final_step = steps[-1]
        deltas = []
        for bench in BENCHMARKS:
            v = final_sd.get(bench)
            bv = baseline.get(bench)
            if v is not None and bv is not None:
                delta = v - bv
                deltas.append(delta)
                marker = "  " if abs(delta) < 0.001 else ("+ " if delta > 0 else "- ")
                print(f"  {bench:<20} {bv:.4f} -> {v:.4f} ({delta:+.4f}) {marker}")
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            print(f"  {'AVG FORGETTING':<20} {avg_delta:+.4f} (at step {final_step})")

    # Save summary JSON
    summary = {}
    for method_name, (steps, score_dicts) in methods_data.items():
        if not score_dicts:
            continue
        baseline = score_dicts[0]
        method_summary = {"steps": steps, "benchmarks": {}}
        for bench in BENCHMARKS:
            bench_data = []
            for step, sd in zip(steps, score_dicts):
                v = sd.get(bench)
                bv = baseline.get(bench)
                bench_data.append({
                    "step": step,
                    "score": v,
                    "delta": (v - bv) if v is not None and bv is not None else None,
                })
            method_summary["benchmarks"][bench] = bench_data
        summary[method_name] = method_summary

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Compare forgetting across methods")
    parser.add_argument(
        "--methods", nargs="+", required=True,
        help="Method=results_dir pairs (e.g., grpo=/path/to/results gt_sft=/path/to/results)",
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ~/scratch/forgetting-llms/eval_results/comparison_plots)")
    parser.add_argument("--max_step", type=int, default=None,
                        help="Only include steps up to this value")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.expanduser(
            "~/scratch/forgetting-llms/eval_results/comparison_plots"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse method=dir pairs
    methods_data = {}
    for spec in args.methods:
        parts = spec.split("=", 1)
        if len(parts) != 2:
            print(f"ERROR: Invalid method spec '{spec}', expected 'name=/path/to/dir'")
            continue
        method_name, results_path = parts
        results_dir = Path(os.path.expanduser(results_path))
        if not results_dir.exists():
            print(f"WARNING: Results dir not found for {method_name}: {results_dir}")
            continue
        steps, score_dicts = load_method_results(results_dir, args.max_step)
        if steps:
            methods_data[method_name] = (steps, score_dicts)
            print(f"Loaded {method_name}: {len(steps)} evaluations at steps {steps}")
        else:
            print(f"WARNING: No results found for {method_name}")

    if not methods_data:
        print("ERROR: No valid method data loaded!")
        return

    plot_avg_forgetting_comparison(methods_data, output_dir)
    plot_per_benchmark_comparison(methods_data, output_dir)
    plot_absolute_scores_comparison(methods_data, output_dir)
    plot_heatmap_comparison(methods_data, output_dir)
    write_comparison_summary(methods_data, output_dir)


if __name__ == "__main__":
    main()
