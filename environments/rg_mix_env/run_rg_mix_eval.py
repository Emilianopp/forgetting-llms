#!/usr/bin/env python3
"""Evaluate the rg-mix-env with Qwen3-4B to validate scoring and get task-wise performance.

Usage (on compute node):
    python run_rg_mix_eval.py
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Force unbuffered output so we can tail the log
sys.stdout.reconfigure(line_buffering=True)

LOG_FILE = Path("/pscratch/sd/s/siddart2/logs/rg_mix_eval_live.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    """Print and write to log file (unbuffered)."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


def main():
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    num_eval = 2048
    max_tokens = 8192
    temperature = 1.0
    seed = 43
    tp = 4
    output_dir = Path("/pscratch/sd/s/siddart2/mech_interp/results/rg_mix_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/pscratch/sd/s/siddart2/hf_cache")

    # Clear log file
    LOG_FILE.write_text("")

    # ── STAGE 1: Load environment ──
    log("=" * 70)
    log("STAGE 1/4: Loading rg_mix_env")
    log("=" * 70)
    t_stage = time.time()

    import rg_mix_env
    env = rg_mix_env.RGMixEnv(
        num_train_examples=100,
        num_eval_examples=num_eval,
        seed=seed,
    )
    eval_ds = env.get_eval_dataset()
    log(f"  Eval dataset: {len(eval_ds)} examples")

    task_counts = defaultdict(int)
    for row in eval_ds:
        task_counts[row["task"]] += 1
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        log(f"  {task:25s}: {count:4d} ({count/len(eval_ds)*100:.1f}%)")

    log(f"STAGE 1 done in {time.time() - t_stage:.1f}s")

    # ── STAGE 2: Load model ──
    log("")
    log("=" * 70)
    log("STAGE 2/4: Loading model + tokenizer")
    log("=" * 70)
    t_stage = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log(f"  Tokenizer loaded")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        max_model_len=max_tokens + 2048,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    log(f"  Model loaded")
    log(f"STAGE 2 done in {time.time() - t_stage:.1f}s")

    # ── STAGE 3: Inference ──
    log("")
    log("=" * 70)
    log("STAGE 3/4: Running inference")
    log("=" * 70)
    t_stage = time.time()

    prompts = []
    for row in eval_ds:
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
    log(f"  Formatted {len(prompts)} prompts")

    sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)
    log(f"  Starting vLLM generate (max_tokens={max_tokens}, temp={temperature})...")

    t_infer = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_elapsed = time.time() - t_infer

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    avg_out_tokens = total_out_tokens / len(outputs)
    log(f"  Inference: {infer_elapsed:.0f}s ({infer_elapsed/60:.1f}min)")
    log(f"  Total output tokens: {total_out_tokens:,}")
    log(f"  Avg output tokens: {avg_out_tokens:.0f}")
    log(f"  Throughput: {total_out_tokens/infer_elapsed:.0f} tok/s")
    log(f"STAGE 3 done in {time.time() - t_stage:.1f}s")

    # ── STAGE 4: Scoring ──
    log("")
    log("=" * 70)
    log("STAGE 4/4: Scoring responses")
    log("=" * 70)
    t_stage = time.time()

    per_task_results = defaultdict(lambda: {"correct": 0, "total": 0, "tokens": []})
    per_task_score_time = defaultdict(float)
    total_correct = 0
    total = 0
    fallback_count = 0
    extract_fail_count = 0

    for i, (row, output) in enumerate(zip(eval_ds, outputs)):
        task = row["task"]
        answer_idx = int(row["answer"])
        vid, entry_idx = env._entry_map[answer_idx]
        ds = env._variant_datasets[vid]
        entry = ds[entry_idx]

        completion_text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)
        extracted = extract_answer(completion_text)

        t_score = time.time()
        try:
            score = ds.score_answer(answer=extracted, entry=entry)
        except Exception:
            score = 0.0

        if score < 0.5:
            fallback_count += 1
            try:
                score_full = ds.score_answer(answer=completion_text, entry=entry)
                score = max(score, score_full)
            except Exception:
                pass

        score_elapsed = time.time() - t_score
        per_task_score_time[task] += score_elapsed

        # Flag slow individual scores
        if score_elapsed > 1.0:
            log(f"  SLOW SCORE: item {i} task={task} took {score_elapsed:.2f}s (tokens={n_tokens})")

        correct = 1 if score >= 0.5 else 0
        total_correct += correct
        total += 1
        per_task_results[task]["correct"] += correct
        per_task_results[task]["total"] += 1
        per_task_results[task]["tokens"].append(n_tokens)

        # Progress every 128 items
        if (i + 1) % 128 == 0:
            elapsed_so_far = time.time() - t_stage
            rate = (i + 1) / elapsed_so_far
            eta = (len(eval_ds) - i - 1) / rate
            log(f"  Scored {i+1}/{len(eval_ds)} ({(i+1)/len(eval_ds)*100:.0f}%) "
                f"acc={total_correct/total:.4f} "
                f"fallbacks={fallback_count} "
                f"[{elapsed_so_far:.0f}s elapsed, ETA {eta:.0f}s]")

    scoring_elapsed = time.time() - t_stage
    log(f"  Scoring complete: {scoring_elapsed:.1f}s ({scoring_elapsed/60:.1f}min)")
    log(f"  Fallback (full-text) scores: {fallback_count}/{total}")

    # Per-task scoring time breakdown
    log(f"\n  Scoring time by task:")
    for task in sorted(per_task_score_time, key=per_task_score_time.get, reverse=True):
        t = per_task_score_time[task]
        n = per_task_results[task]["total"]
        log(f"    {task:25s}: {t:6.1f}s total, {t/n*1000:6.1f}ms/item ({n} items)")

    # ── RESULTS ──
    log("")
    log("=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(f"Overall pass@1: {total_correct/total:.4f} ({total_correct}/{total})")
    log(f"")
    log(f"{'Task':<25} {'pass@1':>8} {'correct':>8} {'total':>8} {'pct':>8} {'avg_tok':>8}")
    log(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    results_list = []
    for task in sorted(per_task_results.keys()):
        r = per_task_results[task]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        avg_tok = sum(r["tokens"]) / len(r["tokens"]) if r["tokens"] else 0
        pct = r["total"] / total * 100
        log(f"{task:<25} {acc:>8.4f} {r['correct']:>8d} {r['total']:>8d} {pct:>7.1f}% {avg_tok:>8.0f}")
        results_list.append({
            "task": task,
            "pass_at_1": acc,
            "correct": r["correct"],
            "total": r["total"],
            "pct_of_eval": pct,
            "avg_tokens": avg_tok,
        })

    log("=" * 70)

    # Save results
    result = {
        "model": model_name,
        "num_eval": num_eval,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "overall_pass_at_1": total_correct / total,
        "total_correct": total_correct,
        "total": total,
        "inference_seconds": infer_elapsed,
        "scoring_seconds": scoring_elapsed,
        "total_output_tokens": total_out_tokens,
        "avg_output_tokens": avg_out_tokens,
        "fallback_scores": fallback_count,
        "per_task_score_time": {k: round(v, 2) for k, v in per_task_score_time.items()},
        "per_task": results_list,
    }
    output_file = output_dir / "rg_mix_eval.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"\nResults saved to {output_file}")
    log(f"Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
