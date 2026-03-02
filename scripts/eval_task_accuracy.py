"""Evaluate checkpoints on their IN-DISTRIBUTION training task.

Uses vLLM for fast generation and unified_reward.py for grading.
Evaluates base model + all checkpoints, outputs a JSON with per-step accuracy.

Usage:
    python scripts/eval_task_accuracy.py \
        --checkpoint_dir ~/scratch/forgetting-llms/checkpoints/gt_sft_qwen3_1.7b_gsm8k \
        --dataset gsm8k \
        --base_model Qwen/Qwen3-1.7B \
        --output_path ~/scratch/forgetting-llms/eval_results/gt_sft_qwen3_1.7b_gsm8k/task_accuracy.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def load_test_data(dataset: str) -> list[dict]:
    """Load test questions and ground truth from preprocessed parquet."""
    data_dir = Path(os.path.expanduser("~/scratch/forgetting-llms/data"))

    if dataset == "gsm8k":
        # GRPO format has prompt + ground_truth
        test_path = data_dir / "gsm8k" / "test.parquet"
    elif dataset == "math":
        test_path = data_dir / "math" / "test.parquet"
    elif dataset == "triviaqa":
        test_path = data_dir / "triviaqa" / "test.parquet"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df = pd.read_parquet(test_path)
    samples = []
    for _, row in df.iterrows():
        # GRPO format: prompt is list of chat messages, ground_truth in reward_model
        prompt = row["prompt"]
        if isinstance(prompt, list):
            # Extract the user message text
            user_msg = next((m["content"] for m in prompt if m["role"] == "user"), "")
        else:
            user_msg = str(prompt)

        gt = row.get("reward_model", {})
        if isinstance(gt, dict):
            ground_truth = gt.get("ground_truth", "")
        else:
            ground_truth = str(gt)

        samples.append({
            "prompt": prompt,
            "user_msg": user_msg,
            "ground_truth": ground_truth,
            "data_source": dataset,
        })

    return samples


def build_chat_prompts(samples: list[dict], tokenizer) -> list[str]:
    """Build formatted prompts using the model's chat template."""
    prompts = []
    for s in samples:
        if isinstance(s["prompt"], list):
            messages = s["prompt"]
        else:
            messages = [{"role": "user", "content": s["user_msg"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    return prompts


def evaluate_model(model_path: str, samples: list[dict], dataset: str,
                   max_tokens: int = 1024) -> dict:
    """Generate responses with vLLM and grade them."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Build prompts
    prompts = build_chat_prompts(samples, tokenizer)

    # Shorter generation for QA
    if dataset == "triviaqa":
        max_tokens = 256

    # vLLM generation
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    outputs = llm.generate(prompts, params)

    # Grade
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.rewards.unified_reward import compute_score

    correct = 0
    total = len(samples)
    details = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        gt = samples[i]["ground_truth"]
        score = compute_score(dataset, response, gt)
        correct += int(score > 0.5)
        details.append({
            "question": samples[i]["user_msg"][:200],
            "ground_truth": gt[:100],
            "response_snippet": response[:200],
            "correct": score > 0.5,
        })

    accuracy = correct / total if total > 0 else 0.0

    # Clean up vLLM to free GPU memory
    del llm
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details_sample": details[:10],  # First 10 for inspection
    }


def merge_checkpoint(ckpt_path: Path) -> Path:
    """Merge FSDP checkpoint to HF format, return merged path."""
    if (ckpt_path / "actor").is_dir():
        fsdp_dir = ckpt_path / "actor"
        merged_dir = ckpt_path / "actor_merged"
    else:
        fsdp_dir = ckpt_path
        merged_dir = ckpt_path / "merged"

    if merged_dir.is_dir() and any(merged_dir.iterdir()):
        print(f"  Using existing merged model: {merged_dir}")
        return merged_dir

    print(f"  Merging FSDP checkpoint: {fsdp_dir}")
    subprocess.run([
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(fsdp_dir),
        "--target_dir", str(merged_dir),
    ], check=True)
    return merged_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math", "triviaqa"])
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    ckpt_dir = Path(os.path.expanduser(args.checkpoint_dir))
    output_path = Path(os.path.expanduser(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load test data
    print(f"Loading {args.dataset} test data...")
    samples = load_test_data(args.dataset)
    print(f"  {len(samples)} test samples")

    results = {"dataset": args.dataset, "base_model": args.base_model, "steps": {}}

    # Check for existing results to enable resuming
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        results["steps"] = existing.get("steps", {})
        print(f"Resuming: {len(results['steps'])} steps already evaluated")

    # Evaluate base model
    if "0" not in results["steps"]:
        print(f"\n--- Evaluating base model: {args.base_model} ---")
        result = evaluate_model(args.base_model, samples, args.dataset)
        results["steps"]["0"] = result
        print(f"  Base model accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Find checkpoints
    ckpt_paths = sorted(ckpt_dir.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not ckpt_paths:
        print(f"No global_step_* checkpoints in {ckpt_dir}")
    else:
        print(f"\nFound {len(ckpt_paths)} checkpoints")

    for ckpt_path in ckpt_paths:
        step = ckpt_path.name.split("_")[-1]
        if step in results["steps"]:
            print(f"\nSkipping step {step} — already evaluated")
            continue

        print(f"\n--- Evaluating step {step} ---")

        # Merge FSDP checkpoint
        merged_dir = merge_checkpoint(ckpt_path)

        # Evaluate
        result = evaluate_model(str(merged_dir), samples, args.dataset)
        results["steps"][step] = result
        print(f"  Step {step} accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

        # Clean up merged model if we created it (has 'merged' or 'actor_merged' in name)
        if "merged" in merged_dir.name:
            import shutil
            print(f"  Cleaning up: {merged_dir}")
            shutil.rmtree(merged_dir, ignore_errors=True)

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Task Accuracy Summary: {args.dataset}")
    print(f"{'='*60}")
    for step in sorted(results["steps"].keys(), key=int):
        r = results["steps"][step]
        label = "base" if step == "0" else f"step {step}"
        print(f"  {label:>12}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
