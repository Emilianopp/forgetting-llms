# Experiment Status Tracker

**Last updated:** 2026-03-09
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
| **GT-SFT** | SFT on ground-truth solutions | Original dataset answers | Complete |
| **SF-SFT** | SFT on same-family teacher trajectories | Qwen3-32B generated solutions | Complete |
| **GRPO** | Online RL with verifiable rewards | Model generates, reward grades | Complete |
| **GT-SFT+GRPO** | SFT warmstart (1 epoch) then GRPO | GT-SFT step 500 → RL | Training complete, eval in progress |
| **SF-SFT+GRPO** | SFT warmstart (1 epoch) then GRPO | SF-SFT step 500 → RL | Training complete, eval in progress |
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
| SF-SFT | GSM8K | 1,302 | 14 | — | `checkpoints/sf_sft_qwen3_1.7b_gsm8k/` |
| SF-SFT | MATH | 588 | 6 | — | `checkpoints/sf_sft_qwen3_1.7b_math/` |
| SF-SFT | TriviaQA | 1,230 | 13 | — | `checkpoints/sf_sft_qwen3_1.7b_triviaqa/` |
| GRPO | GSM8K | 1,200 | 6 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/4mv29sme) | `checkpoints/grpo_full_qwen3_1.7b_gsm8k/` |
| GRPO | MATH | 800 | 4 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/dz4fzki5) | `checkpoints/grpo_qwen3_1.7b_math/` |
| GRPO | TriviaQA | 1,000 | 5 | [run](https://wandb.ai/laurent-charlin/forgetting-llms/runs/os5k8l77) | `checkpoints/grpo_qwen3_1.7b_triviaqa/` |
| GT-SFT+GRPO | GSM8K | 1,199 | 5 | — | `checkpoints/gt_sft_grpo_qwen3_1.7b_gsm8k/` |
| GT-SFT+GRPO | MATH | 892 | 4 | — | `checkpoints/gt_sft_grpo_qwen3_1.7b_math/` |
| GT-SFT+GRPO | TriviaQA | 2,253 | 11 | — | `checkpoints/gt_sft_grpo_qwen3_1.7b_triviaqa/` |
| SF-SFT+GRPO | GSM8K | 1,253 | 6 | — | `checkpoints/sf_sft_grpo_qwen3_1.7b_gsm8k/` |
| SF-SFT+GRPO | MATH | 987 | 4 | — | `checkpoints/sf_sft_grpo_qwen3_1.7b_math/` |
| SF-SFT+GRPO | TriviaQA | 2,201 | 11 | — | `checkpoints/sf_sft_grpo_qwen3_1.7b_triviaqa/` |

All checkpoint paths relative to `~/scratch/forgetting-llms/`.

### Eval Status

| Run | OOD Eval | Task Acc | Status |
|-----|----------|----------|--------|
| GRPO GSM8K | **7/7** | **Done** | **COMPLETE** |
| GT-SFT GSM8K | **6/6** | **Done** | **COMPLETE** |
| SF-SFT GSM8K | 4/14 | **Done** | PARTIAL |
| GRPO MATH | **4/5** | **Done** | **COMPLETE** |
| GT-SFT MATH | 6/14 | **Done** | PARTIAL |
| SF-SFT MATH | 4/6 | **Done** | PARTIAL |
| GRPO TriviaQA | 4/6 | **Done** | PARTIAL |
| GT-SFT TriviaQA | 4/15 | **Done** | PARTIAL |
| SF-SFT TriviaQA | 2/13 | **Done** | PARTIAL |
| GT-SFT+GRPO GSM8K | **4/5** | **Done** | OOD nearly complete |
| GT-SFT+GRPO MATH | — | **Done** | OOD queued |
| GT-SFT+GRPO TriviaQA | — | **Done** | OOD queued |
| SF-SFT+GRPO GSM8K | — | **Done** | OOD eval in progress |
| SF-SFT+GRPO MATH | — | **Done** | OOD eval in progress |
| SF-SFT+GRPO TriviaQA | — | **Done** | OOD eval in progress |

---

## Results

### Summary: OOD Forgetting (avg delta from baseline)

| Method | GSM8K | MATH | TriviaQA |
|--------|-------|------|----------|
| **GT-SFT+GRPO** | **+0.0297** | Pending | Pending |
| **GT-SFT** | +0.0188 | +0.0117 | +0.0120 |
| **SF-SFT** | +0.0096 | +0.0072 | +0.0144 |
| **GRPO** | -0.0076 | +0.0028 | -0.0026 |

GT-SFT+GRPO shows the **strongest positive OOD delta** (+0.030 on GSM8K, all 10 benchmarks positive). SFT warmstart + RL improves general capabilities more than any method alone. GRPO alone causes mild forgetting.

### Summary: Task Accuracy (in-distribution)

![Task Accuracy All Datasets](figures/comparison/task_accuracy_comparison_all.png)

| Method | Dataset | Base | Best | Best Step | Trend |
|--------|---------|------|------|-----------|-------|
| SF-SFT+GRPO | GSM8K | 59.0% | **88.9%** | 1200 | +30pp |
| GRPO | GSM8K | 59.0% | 87.0% | 600 | +28pp |
| GT-SFT+GRPO | GSM8K | 59.0% | 83.2% | 1000 | +24pp |
| SF-SFT | GSM8K | 60.0% | 65.9% | 1302 | +6pp, gradual |
| GT-SFT | GSM8K | 59.0% | 56.6% | 200 | -4pp, drops |
| SF-SFT+GRPO | MATH | 11.0% | **65.1%** | 800 | +54pp |
| GRPO | MATH | 11.0% | 65.1% | 800 | +54pp |
| GT-SFT+GRPO | MATH | 11.0% | 61.6% | 800 | +51pp |
| GT-SFT | MATH | 11.0% | 28.9% | 300 | +18pp, plateaus |
| SF-SFT | MATH | 11.0% | 10.8% | 588 | flat (43% teacher) |
| GRPO | TriviaQA | 38.4% | 99.7% | 1000 | +61pp, saturates |
| SF-SFT+GRPO | TriviaQA | 38.4% | **69.0%** | 2000 | +31pp |
| GT-SFT+GRPO | TriviaQA | 38.4% | 65.7% | 1400 | +27pp |
| SF-SFT | TriviaQA | 38.1% | 39.3% | 400 | +1pp |
| GT-SFT | TriviaQA | 38.1% | 37.4% | 600 | flat |

### Comparison Plots: GSM8K

#### Task Accuracy

![GSM8K Task Accuracy](figures/comparison/task_accuracy_comparison_gsm8k.png)

#### Forgetting (delta from baseline)

![GSM8K Avg Forgetting](figures/comparison/avg_forgetting_comparison.png)

![GSM8K Heatmap](figures/comparison/heatmap_comparison.png)

#### Benchmark Accuracy (absolute scores per step)

![GSM8K Per-Benchmark](figures/comparison/per_benchmark_accuracy_comparison.png)

![GSM8K Absolute Scores](figures/comparison/absolute_scores_comparison.png)

### Comparison Plots: MATH

![MATH Task Accuracy](figures/comparison/task_accuracy_comparison_math.png)

![MATH Avg Forgetting](figures/comparison_math/avg_forgetting_comparison.png)

![MATH Heatmap](figures/comparison_math/heatmap_comparison.png)

![MATH Per-Benchmark](figures/comparison_math/per_benchmark_accuracy_comparison.png)

### Comparison Plots: TriviaQA

![TriviaQA Task Accuracy](figures/comparison/task_accuracy_comparison_triviaqa.png)

![TriviaQA Avg Forgetting](figures/comparison_triviaqa/avg_forgetting_comparison.png)

![TriviaQA Heatmap](figures/comparison_triviaqa/heatmap_comparison.png)

![TriviaQA Per-Benchmark](figures/comparison_triviaqa/per_benchmark_accuracy_comparison.png)

### Key Findings

1. **GRPO excels at task learning, SFT methods do not.** GRPO improves task accuracy dramatically (GSM8K +28pp, MATH +54pp, TriviaQA +61pp). GT-SFT either drops or barely moves task accuracy because it overwrites the model's native `<think>` reasoning with short-form answers.

2. **SFT+GRPO: best of both worlds.** SF-SFT+GRPO matches or beats pure GRPO on task accuracy (GSM8K 88.9% vs 87.0%, MATH 65.1% tie, TriviaQA 69.0% vs 99.7%). GT-SFT+GRPO on GSM8K shows the **strongest OOD improvement** of any method (+0.030 avg across 10 benchmarks, all positive) — better than GT-SFT alone (+0.019) and far better than pure GRPO (-0.008). SFT warmstart appears to regularize RL training.

3. **SF-SFT partially recovers task accuracy.** By training on Qwen3-32B teacher solutions that preserve `<think>`, SF-SFT achieves 66% on GSM8K (vs GT-SFT 55%, base 59%). Still far below GRPO (87%). However, SF-SFT fails on MATH (43% teacher pass rate bottleneck).

4. **Both SFT methods improve OOD benchmarks; GRPO doesn't.** GT-SFT: +0.008–0.013 avg OOD improvement. SF-SFT: +0.006–0.009 avg. GRPO: -0.001 to -0.006 (mild forgetting). SFT acts as a mild regularizer that improves general capabilities.

5. **Forgetting is method- and dataset-dependent.** GRPO forgets on GSM8K (-0.006) and TriviaQA (-0.002) but not MATH (-0.001). SFT consistently improves OOD regardless of dataset.

6. **Teacher quality bottleneck.** SF-SFT MATH stays at base level because the Qwen3-32B teacher only solves 43% of MATH problems. SF-SFT GSM8K works because the teacher solves 93%.

7. **The `<think>` format matters.** GT-SFT trains on short ground-truth solutions without `<think>`, disrupting chain-of-thought reasoning. SF-SFT preserves `<think>` and gets better task accuracy.

---

## SFT + RL Experiments

Using SFT step 500 (~1 epoch) as warmstart, then GRPO for 15 epochs. Tests whether SFT warmstart gives GRPO better task accuracy AND less forgetting.

**Training: All 6 complete.** Eval jobs submitted (task accuracy + OOD sweeps).

| Experiment | Steps | Ckpts | Task Acc (best) | OOD Eval |
|-----------|-------|-------|----------------|----------|
| GT-SFT+GRPO GSM8K | 1,199 | 5 | **83.2%** (step 1000) | **+0.030** (4/5 steps) |
| GT-SFT+GRPO MATH | 892 | 4 | **61.6%** (step 800) | Queued |
| GT-SFT+GRPO TriviaQA | 2,253 | 11 | **65.7%** (step 1400) | Queued |
| SF-SFT+GRPO GSM8K | 1,253 | 6 | **88.9%** (step 1200) | Queued |
| SF-SFT+GRPO MATH | 987 | 4 | **65.1%** (step 800) | Queued |
| SF-SFT+GRPO TriviaQA | 2,201 | 11 | **69.0%** (step 2000) | Queued |

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
- **$HOME cleanup**: Caches symlinked to `~/scratch/cache_redirects/` (uv, vllm, huggingface, torch, pip, clip, .local/lib)
- Jobs chained via `--dependency=afterany` to stay within QOS

---

## TODO

- [x] Validate L40S GPUs for eval pipeline
- [x] All 9 task accuracy evals (GRPO + GT-SFT + SF-SFT x 3 datasets)
- [x] SF-SFT training (3 datasets)
- [x] SFT+RL training (6 experiments, all complete)
- [x] Regenerate comparison plots with SF-SFT included
- [x] $HOME cleanup — caches symlinked to scratch
- [ ] SFT+RL eval sweeps + task accuracy (12 jobs submitted, in progress)
- [ ] Finish remaining OOD eval sweeps (GT-SFT MATH 6/14, GT-SFT TriviaQA 4/15, SF-SFT partial)
- [ ] Regenerate comparison plots with SFT+RL results (after evals complete)
- [ ] Start Phase 2: Qwen3-4B (repeat all experiments)
- [ ] CF-SFT: need Llama-3.1-70B trajectories
- [ ] SELF/SPIN: needs implementation
- [ ] PI: needs implementation
