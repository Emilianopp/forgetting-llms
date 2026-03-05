# Experiment Status Tracker

**Last updated:** 2026-03-05
**WandB:** [forgetting-llms project](https://wandb.ai/laurent-charlin/forgetting-llms)
**Repo:** [github.com/Emilianopp/forgetting-llms](https://github.com/Emilianopp/forgetting-llms)

---

## Overview

**Forgetting**: How much does a model lose on general capabilities after being fine-tuned on a specific task?

- **Train** a base model (Qwen3-1.7B) on a target task using different methods
- **Evaluate** each checkpoint on 10 **unrelated** benchmarks the model was NOT trained on
- **Compare** scores vs the base model — any drop = forgetting

### Eval Benchmarks (10)

| Benchmark | Type | Metric |
|-----------|------|--------|
| ARC-Challenge | Science QA (hard) | acc_norm |
| ARC-Easy | Science QA (easy) | acc_norm |
| HellaSwag | Commonsense NLI | acc_norm |
| WinoGrande | Coreference | acc |
| PIQA | Physical intuition | acc_norm |
| BoolQ | Yes/No QA | acc |
| OpenbookQA | Science QA + retrieval | acc_norm |
| TruthfulQA MC2 | Truthfulness | acc |
| MMLU | Broad knowledge (57 subjects) | acc |
| IFEval | Instruction following | prompt_level_strict_acc |

All 0-shot via `lm-evaluation-harness`. ~1.5h per checkpoint on 1x A100/L40S.

### Training Datasets

| Dataset | Domain | Train Size | Task |
|---------|--------|-----------|------|
| GSM8K | Math | 7,473 | Grade-school math word problems |
| MATH | Math | 7,397 | Competition math (7 subjects, EleutherAI mirror) |
| TriviaQA | QA | 7,500 | Closed-book trivia questions |

### Methods

| Method | Description | Data Source | Status |
|--------|-------------|-------------|--------|
| **GT-SFT** | SFT on ground-truth solutions | Original dataset answers | Active |
| **SF-SFT** | SFT on same-family teacher trajectories | Qwen3-32B generated solutions | Data ready |
| **GRPO** | Online RL with verifiable rewards | Model generates, reward grades | Active |
| CF-SFT | SFT on cross-family teacher trajectories | Llama-3.1-70B | Not started |
| SELF/SPIN | Self-play / self-distillation | Model's own outputs | Not started |
| PI | Policy improvement distillation | — | Not started |

---

## Training Dynamics (All Runs Complete)

### GT-SFT: Training Loss + Validation Loss + LR

![GT-SFT Training Curves](figures/training_curves/sft_training_curves.png)

### GRPO: Validation Accuracy, Entropy, Response Length, Gradient Norm

![GRPO Training Curves](figures/training_curves/grpo_training_curves.png)

### Combined: SFT Val Loss vs GRPO Val Accuracy

![Combined Training Summary](figures/training_curves/combined_training_summary.png)

**Observations:**
- **GSM8K GRPO** reaches 87% val accuracy by step 200, plateaus at ~87%
- **TriviaQA GRPO** saturates at ~100% within 200 steps — too easy for RL
- **MATH GRPO** slower climb to ~60% — harder problems
- **GT-SFT** loss converges smoothly across all 3 datasets, cosine LR schedule visible
- GRPO entropy drops sharply early (policy becoming more deterministic)

---

## Phase 1: Qwen3-1.7B

### Training Runs

| Method | Dataset | Steps | Ckpts | WandB | Checkpoints Path |
|--------|---------|-------|-------|-------|-----------------|
| GT-SFT | GSM8K | 1,401 | 15 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/1hjen6hx) | `checkpoints/gt_sft_qwen3_1.7b_gsm8k/` |
| GT-SFT | MATH | 1,386 | 14 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/9pnqos4e) | `checkpoints/gt_sft_qwen3_1.7b_math/` |
| GT-SFT | TriviaQA | 1,404 | 15 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/jhufx5dt) | `checkpoints/gt_sft_qwen3_1.7b_triviaqa/` |
| GRPO | GSM8K | 1,200 | 6 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/4mv29sme) | `checkpoints/grpo_full_qwen3_1.7b_gsm8k/` |
| GRPO | MATH | 800 | 4 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/dz4fzki5) | `checkpoints/grpo_qwen3_1.7b_math/` |
| GRPO | TriviaQA | 1,000 | 5 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/os5k8l77) | `checkpoints/grpo_qwen3_1.7b_triviaqa/` |
| SF-SFT | GSM8K | — | — | — | Training not started |
| SF-SFT | MATH | — | — | — | Training not started |
| SF-SFT | TriviaQA | — | — | — | Training not started |

All checkpoint paths relative to `~/scratch/forgetting-llms/`.

### Eval Sweeps (10 benchmarks)

| Run | OOD Eval | Task Acc | Status | Notes |
|-----|----------|----------|--------|-------|
| GRPO GSM8K | **7/7** | **Done** | **COMPLETE** | 10 benchmarks |
| GT-SFT GSM8K | **6/6** | **Done** | **COMPLETE** | 8 benchmarks |
| GRPO MATH | **4/5** | **Done** | **COMPLETE** | step 800 missing JSON |
| GT-SFT MATH | 6/15 | **Done** | PARTIAL | 8 ckpts still need OOD eval |
| GRPO TriviaQA | 4/6 | **Done** | PARTIAL | step 600, 800 missing |
| GT-SFT TriviaQA | 4/16 | **Done** | PARTIAL | 11 ckpts still need OOD eval |
| SF-SFT GSM8K | — | — | **TRAINED** | 14 ckpts, needs eval |
| SF-SFT MATH | — | — | **TRAINED** | 6 ckpts, needs eval |
| SF-SFT TriviaQA | — | — | **TRAINED** | 13 ckpts, needs eval |

All eval/training jobs on L40S GPUs (46GB VRAM, validated). QOS: cpu=8, gpu=2 per user.

---

## Results

### Summary: OOD Forgetting (avg delta from baseline)

| Method | GSM8K | MATH | TriviaQA |
|--------|-------|------|----------|
| **GRPO** | **-0.0076** | +0.0028 | **-0.0026** |
| **GT-SFT** | +0.0188 | +0.0117 | +0.0120 |

### Summary: Task Accuracy (in-distribution)

![Task Accuracy All Datasets](figures/comparison/task_accuracy_comparison_all.png)

| Method | Dataset | Base | Best | Best Step | Trend |
|--------|---------|------|------|-----------|-------|
| GRPO | GSM8K | 59.0% | **87.0%** | 600 | +28pp |
| GT-SFT | GSM8K | 59.0% | 56.6% | 200 | -4pp, drops |
| GRPO | MATH | 11.1% | **65.1%** | 800 | +54pp |
| GT-SFT | MATH | 11.1% | 28.9% | 300 | +18pp, plateaus |
| GRPO | TriviaQA | 38.4% | **99.7%** | 1000 | +61pp, saturates |
| GT-SFT | TriviaQA | 38.1% | 37.4% | 600 | flat, no gain |

### Comparison: All Methods (6 runs × 3 datasets)

#### Forgetting (delta from baseline)

![Comparison Avg Forgetting](figures/comparison/avg_forgetting_comparison.png)

![Comparison Heatmap](figures/comparison/heatmap_comparison.png)

#### Benchmark Accuracy (absolute scores per step)

![Per-Benchmark Accuracy Comparison](figures/comparison/per_benchmark_accuracy_comparison.png)

![Average Benchmark Accuracy](figures/comparison/absolute_scores_comparison.png)

### Per-Experiment Details

<details>
<summary>GRPO GSM8K (7 checkpoints, 10 benchmarks)</summary>

![GRPO GSM8K Forgetting](figures/eval_results/grpo_gsm8k_forgetting_curves.png)
![GRPO GSM8K Heatmap](figures/eval_results/grpo_gsm8k_heatmap.png)

Avg forgetting: **-0.0076**. Worst: BoolQ (-0.022), TruthfulQA (-0.016). Task acc: 59% → 87%.
</details>

<details>
<summary>GT-SFT GSM8K (6 checkpoints, 8 benchmarks)</summary>

![GT-SFT GSM8K Forgetting](figures/eval_results/gt_sft_gsm8k_forgetting_curves.png)
![GT-SFT GSM8K Heatmap](figures/eval_results/gt_sft_gsm8k_heatmap.png)

Avg delta: **+0.0188**. All benchmarks improve. Task acc: 59% → 55% (format mismatch).
</details>

<details>
<summary>GRPO MATH (4 checkpoints, 10 benchmarks)</summary>

![GRPO MATH Forgetting](figures/eval_results/grpo_math/forgetting_curves.png)
![GRPO MATH Heatmap](figures/eval_results/grpo_math/heatmap.png)

Avg delta: **+0.0028**. No forgetting. IFEval improves +0.031. Task acc: 11% → 65%.
</details>

<details>
<summary>GT-SFT MATH (6/14 checkpoints, 10 benchmarks)</summary>

![GT-SFT MATH Forgetting](figures/eval_results/gt_sft_math/forgetting_curves.png)
![GT-SFT MATH Heatmap](figures/eval_results/gt_sft_math/heatmap.png)

Avg delta: **+0.0117**. 9/10 benchmarks improve, IFEval drops -0.035. Task acc: 11% → 27%.
</details>

<details>
<summary>GRPO TriviaQA (4/5 checkpoints, 10 benchmarks)</summary>

![GRPO TriviaQA Forgetting](figures/eval_results/grpo_triviaqa/forgetting_curves.png)
![GRPO TriviaQA Heatmap](figures/eval_results/grpo_triviaqa/heatmap.png)

Avg delta: **-0.0026**. Mild forgetting on TruthfulQA (-0.009), IFEval (-0.013). Task acc: 38% → 99.7%.
</details>

<details>
<summary>GT-SFT TriviaQA (4/15 checkpoints, 10 benchmarks)</summary>

![GT-SFT TriviaQA Forgetting](figures/eval_results/gt_sft_triviaqa/forgetting_curves.png)
![GT-SFT TriviaQA Heatmap](figures/eval_results/gt_sft_triviaqa/heatmap.png)

Avg delta: **+0.0120**. Most improve, but BoolQ (-0.013) and TruthfulQA (-0.029) drop. Task acc: 38% → 37% (no gain).
</details>

### Key Findings

1. **GRPO excels at task learning, GT-SFT does not.** GRPO improves task accuracy dramatically (GSM8K +28pp, MATH +54pp, TriviaQA +61pp). GT-SFT either drops or barely moves task accuracy because it overwrites the model's native `<think>` reasoning with short-form answers.

2. **GT-SFT improves OOD benchmarks, GRPO doesn't.** All 3 GT-SFT runs show +0.01–0.02 avg improvement on OOD benchmarks. GRPO shows mild forgetting on GSM8K (-0.008) and TriviaQA (-0.003), and no change on MATH (+0.003).

3. **Forgetting is method- and dataset-dependent.** GRPO forgets on GSM8K and TriviaQA but not MATH. GT-SFT consistently improves OOD regardless of dataset.

4. **The `<think>` format matters.** GT-SFT trains on short ground-truth solutions that don't use `<think>`, disrupting the model's chain-of-thought reasoning. SF-SFT (now trained, pending eval) uses Qwen3-32B teacher solutions that preserve `<think>` — expected to fix task accuracy while maintaining OOD performance.

---

## Data Inventory

All data in `~/scratch/forgetting-llms/data/`.

| Directory | Format | Samples | Source | Ready |
|-----------|--------|---------|--------|-------|
| `gsm8k/` | GRPO | 7,473 | openai/gsm8k | Yes |
| `gsm8k_sft/` | SFT | 7,473 | openai/gsm8k (reformatted) | Yes |
| `gsm8k_sf_sft/` | SFT | 6,946 | Qwen3-32B teacher (93% pass) | Yes |
| `math/` | GRPO | 7,397 | EleutherAI/hendrycks_math | Yes |
| `math_sft/` | SFT | 7,397 | EleutherAI/hendrycks_math | Yes |
| `math_sf_sft/` | SFT | 3,151 | Qwen3-32B teacher (43% pass) | Yes |
| `triviaqa/` | GRPO | 7,500 | mandarjoshi/trivia_qa | Yes |
| `triviaqa_sft/` | SFT | 7,500 | mandarjoshi/trivia_qa | Yes |
| `triviaqa_sf_sft/` | SFT | 6,568 | Qwen3-32B teacher (88% pass) | Yes |

---

## Cluster Notes

- **QOS limit (main)**: cpu=8, gpu=2, mem=48G per user — max 2 concurrent eval jobs
- **L40S validated**: Smoke test passed (CUDA, transformers, vLLM, lm_eval). 46GB VRAM, compute 8.9, batch_size=1 for evals
- **Eval speed**: ~1.5h per checkpoint (10 benchmarks, batch_size=1, 1x A100 or L40S)
- **Task accuracy eval**: ~20-30 min per experiment (all checkpoints, vLLM generation + grading)
- **Trajectory gen**: MATH ~6.5h, TriviaQA ~1.7h, GSM8K ~3.5h (Qwen3-32B, TP=2)
- **SFT training**: ~2-3h for 3 epochs (~1400 steps), 2x A100
- **GRPO training**: ~4-8h for 15 epochs, 2x A100
- Jobs chained via `--dependency=afterany` to stay within QOS
- All new eval jobs use L40S (`--gres=gpu:l40s:1 --cpus-per-task=4`)

---

## TODO

- [x] Validate L40S GPUs for eval pipeline
- [x] All 6 task accuracy evals (GRPO + GT-SFT × 3 datasets)
- [x] SF-SFT training (3 datasets) — all complete
- [ ] SF-SFT eval sweeps (OOD + task accuracy) — needs submission
- [ ] Finish remaining OOD eval sweeps (GT-SFT MATH 8/14, GT-SFT TriviaQA 11/15, GRPO TriviaQA 1/5)
- [ ] SF-SFT task accuracy evals
- [ ] Regenerate comparison plots with all 9 experiments (3 methods × 3 datasets)
- [ ] Start Phase 2: Qwen3-4B (repeat all experiments)
- [ ] CF-SFT: need Llama-3.1-70B trajectories
- [ ] SELF/SPIN: needs implementation
- [ ] PI: needs implementation

### SF-SFT Launch Commands (ready to run)

```bash
# Training (chain after eval sweeps finish to stay within QOS)
sbatch scripts/run_sft.sh gsm8k Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_gsm8k sf
sbatch scripts/run_sft.sh math Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_math sf
sbatch scripts/run_sft.sh triviaqa Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_triviaqa sf

# Eval sweeps (chain after training)
sbatch scripts/eval_sweep_resumable.sh ~/scratch/forgetting-llms/checkpoints/sf_sft_qwen3_1.7b_gsm8k sf_sft_qwen3_1.7b_gsm8k
sbatch scripts/eval_sweep_resumable.sh ~/scratch/forgetting-llms/checkpoints/sf_sft_qwen3_1.7b_math sf_sft_qwen3_1.7b_math
sbatch scripts/eval_sweep_resumable.sh ~/scratch/forgetting-llms/checkpoints/sf_sft_qwen3_1.7b_triviaqa sf_sft_qwen3_1.7b_triviaqa
```
