"""Plot training metrics from WandB for GRPO runs.

Usage:
    python scripts/plot_training.py --run grpo_full_qwen3_1.7b_gsm8k
    python scripts/plot_training.py --run grpo_full_qwen3_1.7b_gsm8k --output plots/
"""

import argparse
from pathlib import Path

import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


WANDB_PROJECT = "forgetting-llms"


def fetch_run(experiment_name: str) -> wandb.apis.public.Run:
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters={"config.experiment_name": experiment_name})
    if not runs:
        # Try matching by display name
        runs = api.runs(WANDB_PROJECT, filters={"display_name": experiment_name})
    if not runs:
        raise ValueError(f"No run found with experiment_name={experiment_name}")
    return runs[0]


def plot_metrics(run: wandb.apis.public.Run, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    history = run.scan_history()
    rows = list(history)
    if not rows:
        print("No data found in run history")
        return

    steps = [r["training/global_step"] for r in rows if "training/global_step" in r]

    # --- 1. Training reward (batch accuracy) ---
    reward_steps, rewards = [], []
    for r in rows:
        if "critic/score/mean" in r and "training/global_step" in r:
            reward_steps.append(r["training/global_step"])
            rewards.append(r["critic/score/mean"])

    if rewards:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(reward_steps, rewards, alpha=0.4, color="steelblue", label="Per-step")
        # Smoothed (rolling window)
        window = max(1, len(rewards) // 20)
        if window > 1:
            smoothed = [
                sum(rewards[max(0, i - window):i + 1]) / len(rewards[max(0, i - window):i + 1])
                for i in range(len(rewards))
            ]
            ax.plot(reward_steps, smoothed, color="darkblue", linewidth=2, label=f"Smoothed (w={window})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Reward (Batch Accuracy)")
        ax.set_title("GRPO Training — Batch Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "training_reward.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'training_reward.png'}")

    # --- 2. Validation accuracy ---
    val_steps, val_accs = [], []
    for r in rows:
        if "val-core/gsm8k/acc/mean@1" in r and r["val-core/gsm8k/acc/mean@1"] is not None:
            val_steps.append(r.get("training/global_step", 0))
            val_accs.append(r["val-core/gsm8k/acc/mean@1"])

    if val_accs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(val_steps, val_accs, "o-", color="green", markersize=6, linewidth=2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("GSM8K Accuracy (greedy)")
        ax.set_title("GRPO Training — Validation Accuracy")
        ax.grid(True, alpha=0.3)
        for i, (s, a) in enumerate(zip(val_steps, val_accs)):
            ax.annotate(f"{a:.1%}", (s, a), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "validation_accuracy.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'validation_accuracy.png'}")

    # --- 3. Policy gradient loss ---
    loss_steps, losses = [], []
    for r in rows:
        if "actor/pg_loss" in r and r["actor/pg_loss"] is not None:
            loss_steps.append(r.get("training/global_step", 0))
            losses.append(r["actor/pg_loss"])

    if losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(loss_steps, losses, alpha=0.4, color="coral", label="Per-step")
        window = max(1, len(losses) // 20)
        if window > 1:
            smoothed = [
                sum(losses[max(0, i - window):i + 1]) / len(losses[max(0, i - window):i + 1])
                for i in range(len(losses))
            ]
            ax.plot(loss_steps, smoothed, color="darkred", linewidth=2, label=f"Smoothed (w={window})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Policy Gradient Loss")
        ax.set_title("GRPO Training — PG Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "pg_loss.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'pg_loss.png'}")

    # --- 4. Response length and entropy ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    resp_steps, resp_lens, resp_clips = [], [], []
    for r in rows:
        if "response_length/mean" in r and r["response_length/mean"] is not None:
            resp_steps.append(r.get("training/global_step", 0))
            resp_lens.append(r["response_length/mean"])
            resp_clips.append(r.get("response_length/clip_ratio", 0))

    if resp_lens:
        ax1.plot(resp_steps, resp_lens, color="purple", alpha=0.6)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Mean Response Length (tokens)")
        ax1.set_title("Response Length")
        ax1.grid(True, alpha=0.3)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(resp_steps, resp_clips, color="orange", alpha=0.5)
        ax1_twin.set_ylabel("Clip Ratio", color="orange")

    ent_steps, entropies = [], []
    for r in rows:
        if "actor/entropy" in r and r["actor/entropy"] is not None:
            ent_steps.append(r.get("training/global_step", 0))
            entropies.append(r["actor/entropy"])

    if entropies:
        ax2.plot(ent_steps, entropies, color="teal")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Entropy")
        ax2.set_title("Policy Entropy")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "response_and_entropy.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'response_and_entropy.png'}")

    # --- 5. Summary stats ---
    print(f"\n{'='*50}")
    print(f"Run: {run.name}")
    print(f"Steps: {steps[-1] if steps else 'N/A'}")
    if val_accs:
        print(f"Validation accuracy: {val_accs[0]:.1%} → {val_accs[-1]:.1%} ({val_accs[-1]-val_accs[0]:+.1%})")
    if rewards:
        print(f"Final batch reward (last 10): {sum(rewards[-10:])/min(10,len(rewards)):.3f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training metrics from WandB")
    parser.add_argument("--run", required=True, help="WandB experiment name")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    print(f"Fetching run: {args.run}")
    run = fetch_run(args.run)
    print(f"Found: {run.name} ({run.id}) — {run.state}")

    plot_metrics(run, Path(args.output))


if __name__ == "__main__":
    main()
