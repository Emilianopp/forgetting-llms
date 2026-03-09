"""Plot in-distribution task accuracy comparison across all methods."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_task_accuracy(results_dir):
    """Load task_accuracy.json and return (steps, accuracies)."""
    path = Path(results_dir) / "task_accuracy.json"
    with open(path) as f:
        data = json.load(f)

    steps_acc = []
    for step_str, result in data["steps"].items():
        steps_acc.append((int(step_str), result["accuracy"]))

    steps_acc.sort(key=lambda x: x[0])
    steps = [s for s, a in steps_acc]
    accs = [a for s, a in steps_acc]
    return steps, accs


COLORS = {
    "GRPO": "#d62728",
    "GT-SFT": "#2ca02c",
    "SF-SFT": "#1f77b4",
    "GT-SFT+GRPO": "#ff7f0e",
    "SF-SFT+GRPO": "#9467bd",
}
MARKERS = {
    "GRPO": "o",
    "GT-SFT": "s",
    "SF-SFT": "^",
    "GT-SFT+GRPO": "D",
    "SF-SFT+GRPO": "v",
}


def plot_task_accuracy_comparison(methods, output_dir, dataset_name="GSM8K"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, results_dir in methods.items():
        try:
            steps, accs = load_task_accuracy(results_dir)
            accs_pct = [a * 100 for a in accs]
            color = COLORS.get(label, "gray")
            marker = MARKERS.get(label, "x")
            ax.plot(steps, accs_pct, f"-", color=color, label=label,
                    linewidth=2, markersize=7, marker=marker)
            ax.annotate(f"{accs_pct[-1]:.1f}%",
                       (steps[-1], accs_pct[-1]),
                       textcoords="offset points", xytext=(10, 5),
                       fontsize=9, color=color, fontweight="bold")
        except Exception as e:
            print(f"Warning: Could not load {label}: {e}")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Task Accuracy (%)", fontsize=12)
    ax.set_title(f"In-Distribution Task Accuracy: {dataset_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_path = output_dir / f"task_accuracy_comparison_{dataset_name.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_multi_dataset_task_accuracy(datasets_methods, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_datasets = len(datasets_methods)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6), squeeze=False)

    for idx, (dataset_name, methods) in enumerate(datasets_methods.items()):
        ax = axes[0][idx]
        for label, results_dir in methods.items():
            try:
                steps, accs = load_task_accuracy(results_dir)
                accs_pct = [a * 100 for a in accs]
                color = COLORS.get(label, "gray")
                marker = MARKERS.get(label, "x")
                ax.plot(steps, accs_pct, "-", color=color, label=label,
                        linewidth=2, markersize=6, marker=marker)
                ax.annotate(f"{accs_pct[-1]:.1f}%",
                           (steps[-1], accs_pct[-1]),
                           textcoords="offset points", xytext=(8, 5),
                           fontsize=8, color=color, fontweight="bold")
            except Exception as e:
                print(f"Warning: Could not load {label} for {dataset_name}: {e}")

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Task Accuracy (%)", fontsize=11)
        ax.set_title(f"{dataset_name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    fig.suptitle("In-Distribution Task Accuracy by Dataset", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "task_accuracy_comparison_all.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def try_add(methods_dict, label, results_dir):
    """Add method if task_accuracy.json exists."""
    if os.path.exists(os.path.join(results_dir, "task_accuracy.json")):
        methods_dict[label] = results_dir
        return True
    return False


if __name__ == "__main__":
    base = os.path.expanduser("~/scratch/forgetting-llms/eval_results")
    output = os.path.expanduser("~/scratch/forgetting-llms/eval_results/comparison_plots")

    # Define all experiments per dataset
    experiment_map = {
        "GSM8K": {
            "GRPO": "grpo_full_qwen3_1.7b_gsm8k",
            "GT-SFT": "gt_sft_qwen3_1.7b_gsm8k",
            "SF-SFT": "sf_sft_qwen3_1.7b_gsm8k",
            "GT-SFT+GRPO": "gt_sft_grpo_qwen3_1.7b_gsm8k",
            "SF-SFT+GRPO": "sf_sft_grpo_qwen3_1.7b_gsm8k",
        },
        "MATH": {
            "GRPO": "grpo_qwen3_1.7b_math",
            "GT-SFT": "gt_sft_qwen3_1.7b_math",
            "SF-SFT": "sf_sft_qwen3_1.7b_math",
            "GT-SFT+GRPO": "gt_sft_grpo_qwen3_1.7b_math",
            "SF-SFT+GRPO": "sf_sft_grpo_qwen3_1.7b_math",
        },
        "TriviaQA": {
            "GRPO": "grpo_qwen3_1.7b_triviaqa",
            "GT-SFT": "gt_sft_qwen3_1.7b_triviaqa",
            "SF-SFT": "sf_sft_qwen3_1.7b_triviaqa",
            "GT-SFT+GRPO": "gt_sft_grpo_qwen3_1.7b_triviaqa",
            "SF-SFT+GRPO": "sf_sft_grpo_qwen3_1.7b_triviaqa",
        },
    }

    datasets = {}
    for dataset_name, exp_dirs in experiment_map.items():
        methods = {}
        for label, dirname in exp_dirs.items():
            try_add(methods, label, os.path.join(base, dirname))

        if methods:
            plot_task_accuracy_comparison(methods, output, dataset_name)
            datasets[dataset_name] = methods
            print(f"  {dataset_name}: {list(methods.keys())}")

    if len(datasets) > 1:
        plot_multi_dataset_task_accuracy(datasets, output)

    print("\nDone!")
