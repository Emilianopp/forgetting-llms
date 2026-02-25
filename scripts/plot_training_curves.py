"""Plot training curves for all SFT and GRPO runs from Slurm logs.

Usage:
    python scripts/plot_training_curves.py --log_dir ~/forgetting-llms/slurm_logs \
        --output_dir ~/scratch/forgetting-llms/training_curves
"""

import argparse
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_sft_log(log_path: str) -> dict:
    """Parse SFT training log for loss, lr, and validation loss per step."""
    steps, losses, lrs = [], [], []
    val_steps, val_losses = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"step:(\d+) - train/loss:([\d.]+) - train/lr\(1e-3\):([\d.eE+-]+)", line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)) * 1e-3)
            vm = re.search(r"step:(\d+) - val/loss:([\d.]+)", line)
            if vm and "train/loss" not in line:
                val_steps.append(int(vm.group(1)))
                val_losses.append(float(vm.group(2)))
    return {"steps": steps, "loss": losses, "lr": lrs,
            "val_steps": val_steps, "val_loss": val_losses}


def parse_grpo_log(log_path: str) -> dict:
    """Parse GRPO training log for metrics per step.

    Extracts both training batch score (critic/score/mean) and validation
    accuracy (val-core/.../acc/mean@1). Validation is logged at test_freq
    intervals and lives on the same long log line as training metrics.
    """
    steps, scores, entropies, grad_norms, resp_lens = [], [], [], [], []
    val_steps, val_accs = [], []
    with open(log_path) as f:
        for line in f:
            step_m = re.search(r"training/global_step:(\d+)", line)
            score_m = re.search(r"critic/score/mean:([\d.]+)", line)
            ent_m = re.search(r"actor/entropy:([\d.]+)", line)
            grad_m = re.search(r"actor/grad_norm:np\.float64\(([\d.]+)\)", line)
            resp_m = re.search(r"response_length/mean:([\d.]+)", line)
            val_m = re.search(r"val-core/\w+/acc/mean@1:np\.float64\(([\d.]+)\)", line)
            if step_m and score_m:
                steps.append(int(step_m.group(1)))
                scores.append(float(score_m.group(1)))
                entropies.append(float(ent_m.group(1)) if ent_m else 0)
                grad_norms.append(float(grad_m.group(1)) if grad_m else 0)
                resp_lens.append(float(resp_m.group(1)) if resp_m else 0)
            # Validation accuracy — may appear on step lines or standalone
            if val_m:
                vstep_m = re.search(r"step:(\d+)", line)
                if not vstep_m and step_m:
                    vstep_m = step_m
                if vstep_m:
                    val_steps.append(int(vstep_m.group(1)))
                    val_accs.append(float(val_m.group(1)))
    return {"steps": steps, "score": scores, "entropy": entropies,
            "grad_norm": grad_norms, "resp_len": resp_lens,
            "val_steps": val_steps, "val_acc": val_accs}


def smooth(values, window=10):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_sft_curves(runs: dict, output_dir: Path):
    """Plot SFT training + validation loss curves for multiple datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"gsm8k": "#2ecc71", "math": "#e74c3c", "triviaqa": "#3498db"}

    # Training loss
    ax = axes[0]
    for name, data in runs.items():
        if data["steps"]:
            c = colors.get(name, "gray")
            ax.plot(data["steps"], data["loss"], alpha=0.3, color=c)
            sm = smooth(data["loss"], 20)
            offset = len(data["loss"]) - len(sm)
            ax.plot(data["steps"][offset:], sm, color=c,
                    linewidth=2, label=f"{name.upper()} (final={data['loss'][-1]:.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[1]
    for name, data in runs.items():
        if data.get("val_steps"):
            c = colors.get(name, "gray")
            ax.plot(data["val_steps"], data["val_loss"], marker="o", markersize=3,
                    color=c, linewidth=2,
                    label=f"{name.upper()} (final={data['val_loss'][-1]:.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LR schedule
    ax = axes[2]
    for name, data in runs.items():
        if data["steps"]:
            ax.plot(data["steps"], data["lr"], color=colors.get(name, "gray"),
                    linewidth=2, label=name.upper())
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("GT-SFT Training Curves (Qwen3-1.7B)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "sft_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'sft_training_curves.png'}")


def plot_grpo_curves(runs: dict, output_dir: Path):
    """Plot GRPO training curves for multiple datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"gsm8k": "#2ecc71", "math": "#e74c3c", "triviaqa": "#3498db"}

    # Validation accuracy (preferred over training batch score)
    ax = axes[0, 0]
    for name, data in runs.items():
        if data.get("val_steps"):
            c = colors.get(name, "gray")
            ax.plot(data["val_steps"], data["val_acc"], marker="o", markersize=3,
                    color=c, linewidth=2,
                    label=f"{name.upper()} (final={data['val_acc'][-1]:.3f})")
        elif data["steps"]:
            # Fallback to training batch score if no val data
            c = colors.get(name, "gray")
            ax.plot(data["steps"], data["score"], alpha=0.3, color=c)
            sm = smooth(data["score"], 10)
            offset = len(data["score"]) - len(sm)
            ax.plot(data["steps"][offset:], sm, color=c,
                    linewidth=2, label=f"{name.upper()} (train, final={data['score'][-1]:.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[0, 1]
    for name, data in runs.items():
        if data["steps"] and any(data["entropy"]):
            c = colors.get(name, "gray")
            ax.plot(data["steps"], data["entropy"], alpha=0.3, color=c)
            sm = smooth(data["entropy"], 10)
            offset = len(data["entropy"]) - len(sm)
            ax.plot(data["steps"][offset:], sm, color=c,
                    linewidth=2, label=name.upper())
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Response length
    ax = axes[1, 0]
    for name, data in runs.items():
        if data["steps"] and any(data["resp_len"]):
            c = colors.get(name, "gray")
            ax.plot(data["steps"], data["resp_len"], alpha=0.3, color=c)
            sm = smooth(data["resp_len"], 10)
            offset = len(data["resp_len"]) - len(sm)
            ax.plot(data["steps"][offset:], sm, color=c,
                    linewidth=2, label=name.upper())
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens")
    ax.set_title("Mean Response Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Grad norm
    ax = axes[1, 1]
    for name, data in runs.items():
        if data["steps"] and any(data["grad_norm"]):
            c = colors.get(name, "gray")
            ax.plot(data["steps"], data["grad_norm"], alpha=0.3, color=c)
            sm = smooth(data["grad_norm"], 10)
            offset = len(data["grad_norm"]) - len(sm)
            ax.plot(data["steps"][offset:], sm, color=c,
                    linewidth=2, label=name.upper())
    ax.set_xlabel("Step")
    ax.set_ylabel("Grad Norm")
    ax.set_title("Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("GRPO Training Curves (Qwen3-1.7B)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "grpo_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'grpo_training_curves.png'}")


def plot_combined_summary(sft_runs: dict, grpo_runs: dict, output_dir: Path):
    """Combined 2-panel: SFT val loss + GRPO val accuracy side by side.

    Same color per dataset across both panels. Labels include method name.
    Uses validation metrics (not training batch metrics).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    DATASET_COLORS = {"gsm8k": "#2ecc71", "math": "#e74c3c", "triviaqa": "#3498db"}

    for name, data in sft_runs.items():
        c = DATASET_COLORS.get(name, "gray")
        if data.get("val_steps"):
            ax1.plot(data["val_steps"], data["val_loss"], marker="o", markersize=3,
                     color=c, linewidth=2, label=f"GT-SFT {name.upper()}")
        elif data["steps"]:
            sm = smooth(data["loss"], 20)
            offset = len(data["loss"]) - len(sm)
            ax1.plot(data["steps"], data["loss"], alpha=0.15, color=c)
            ax1.plot(data["steps"][offset:], sm, color=c,
                     linewidth=2, label=f"GT-SFT {name.upper()} (train)")
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Validation Loss", fontsize=12)
    ax1.set_title("GT-SFT Validation Loss", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    for name, data in grpo_runs.items():
        c = DATASET_COLORS.get(name, "gray")
        if data.get("val_steps"):
            ax2.plot(data["val_steps"], data["val_acc"], marker="o", markersize=3,
                     color=c, linewidth=2, label=f"GRPO {name.upper()}")
        elif data["steps"]:
            sm = smooth(data["score"], 10)
            offset = len(data["score"]) - len(sm)
            ax2.plot(data["steps"], data["score"], alpha=0.15, color=c)
            ax2.plot(data["steps"][offset:], sm, color=c,
                     linewidth=2, label=f"GRPO {name.upper()} (train)")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Validation Accuracy", fontsize=12)
    ax2.set_title("GRPO Validation Accuracy", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training Dynamics: SFT vs GRPO across Datasets (Qwen3-1.7B)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'combined_training_summary.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sft_logs", nargs="+", default=None,
                        help="name=job_id pairs for SFT (e.g. gsm8k=8792473)")
    parser.add_argument("--grpo_logs", nargs="+", default=None,
                        help="name=job_id pairs for GRPO (e.g. gsm8k=8770013)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse SFT logs
    sft_runs = {}
    if args.sft_logs:
        for spec in args.sft_logs:
            name, job_id = spec.split("=")
            log_file = log_dir / f"{job_id}_sft-train.out"
            if log_file.exists():
                sft_runs[name] = parse_sft_log(str(log_file))
                print(f"SFT {name}: {len(sft_runs[name]['steps'])} steps")
            else:
                print(f"WARNING: {log_file} not found")

    # Parse GRPO logs
    grpo_runs = {}
    if args.grpo_logs:
        for spec in args.grpo_logs:
            name, rest = spec.split("=")
            job_id, log_suffix = rest, "grpo-seq"
            if ":" in rest:
                job_id, log_suffix = rest.split(":")
            # Try both naming patterns
            for suffix in [log_suffix, "grpo-full"]:
                log_file = log_dir / f"{job_id}_{suffix}.out"
                if log_file.exists():
                    grpo_runs[name] = parse_grpo_log(str(log_file))
                    print(f"GRPO {name}: {len(grpo_runs[name]['steps'])} steps")
                    break
            else:
                print(f"WARNING: No log found for GRPO {name} (job {job_id})")

    if sft_runs:
        plot_sft_curves(sft_runs, output_dir)
    if grpo_runs:
        plot_grpo_curves(grpo_runs, output_dir)
    if sft_runs and grpo_runs:
        plot_combined_summary(sft_runs, grpo_runs, output_dir)


if __name__ == "__main__":
    main()
