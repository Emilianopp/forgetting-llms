# Experiment Status Tracker

**Last updated:** 2026-03-01

## What We're Measuring

**Forgetting**: How much does a model lose on general capabilities after being fine-tuned on a specific task?

- **Train** a base model (Qwen3-1.7B) on a target task using different methods (SFT, GRPO, etc.)
- **Evaluate** each checkpoint on 10 **unrelated** benchmarks that the model was NOT trained on
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

All 0-shot, run via `lm-evaluation-harness`. ~1.5h per checkpoint per 10 benchmarks on 1x A100.

### Training Datasets (3)

| Dataset | Domain | Train Size | Task |
|---------|--------|-----------|------|
| GSM8K | Math | 7,473 | Grade-school math word problems |
| MATH | Math | 7,397 | Competition math (7 subjects) |
| TriviaQA | QA | 7,500 | Closed-book trivia questions |

### Methods

| Method | Description | Data Source |
|--------|-------------|-------------|
| **GT-SFT** | SFT on ground-truth solutions | Original dataset answers |
| **SF-SFT** | SFT on same-family teacher trajectories | Qwen3-32B generated solutions |
| **GRPO** | Online RL with verifiable rewards | Model generates, reward grades |
| CF-SFT | SFT on cross-family teacher trajectories | Llama-3.1-70B (not started) |
| SELF/SPIN | Self-play / self-distillation | Model's own outputs (not started) |
| PI | Policy improvement distillation | (not started) |

---

## Phase 1: Qwen3-1.7B Experiments

### Training Runs

| Method | Dataset | Checkpoints | Steps | Status |
|--------|---------|-------------|-------|--------|
| GT-SFT | GSM8K | 15 (every 100) | 1,401 | DONE |
| GT-SFT | MATH | 14 (every 100) | 1,386 | DONE |
| GT-SFT | TriviaQA | 15 (every 100) | 1,404 | DONE |
| GRPO | GSM8K | 6 (every 200) | 1,200 | DONE |
| GRPO | MATH | 4 (every 200) | 800 | DONE |
| GRPO | TriviaQA | 5 (every 200) | 1,000 | DONE (saturated at ~100% acc) |
| SF-SFT | GSM8K | — | — | DATA READY (6,946 samples), training not started |
| SF-SFT | MATH | — | — | DATA READY (3,151 samples, 43% pass rate), training not started |
| SF-SFT | TriviaQA | — | — | DATA READY (6,568 samples), training not started |

### Eval Sweeps (10 benchmarks)

Each row = one eval sweep across all checkpoints of a training run.

| Run | Ckpts to Eval | Ckpts Done | Status | Job |
|-----|---------------|------------|--------|-----|
| GRPO GSM8K | 6 + base | **7/7** | DONE (old 8-bench run) | — |
| GRPO GSM8K (10-bench) | 6 + base | 3/7 | QUEUED (resumable) | 8847627 |
| GT-SFT GSM8K | 15 + base | 4/16 | QUEUED (resumable) | 8847628 |
| GT-SFT MATH | 14 + base | 1/15 | RUNNING → then resume | 8840042 → 8847629 |
| GT-SFT TriviaQA | 15 + base | 0/16 | QUEUED (resumable) | 8847630 |
| GRPO MATH | 4 + base | 0/5 | QUEUED (resumable) | 8847631 |
| GRPO TriviaQA | 5 + base | 0/6 | QUEUED (resumable) | 8847632 |

**Note:** Eval sweeps now auto-resubmit on timeout via `eval_sweep_resumable.sh`. Each 16h job evaluates ~10 checkpoints. Runs with 15 ckpts need 2 jobs.

### SF-SFT Training (not started)

Need to submit once we decide priority:

```bash
sbatch scripts/run_sft.sh gsm8k Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_gsm8k sf
sbatch scripts/run_sft.sh math Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_math sf
sbatch scripts/run_sft.sh triviaqa Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_triviaqa sf
```

Then eval sweeps for each.

---

## Data Inventory

All data lives in `~/scratch/forgetting-llms/data/`.

| Directory | Format | Train Samples | Purpose |
|-----------|--------|---------------|---------|
| `gsm8k/` | GRPO | 7,473 | GRPO training |
| `gsm8k_sft/` | SFT | 7,473 | GT-SFT training |
| `gsm8k_sf_sft/` | SFT | 6,946 | SF-SFT training (32B teacher) |
| `math/` | GRPO | 7,397 | GRPO training |
| `math_sft/` | SFT | 7,397 | GT-SFT training |
| `math_sf_sft/` | SFT | 3,151 | SF-SFT training (32B teacher, 43% pass) |
| `triviaqa/` | GRPO | 7,500 | GRPO training |
| `triviaqa_sft/` | SFT | 7,500 | GT-SFT training |
| `triviaqa_sf_sft/` | SFT | 6,568 | SF-SFT training (32B teacher) |

---

## Key Results So Far (Preliminary, 8-benchmark eval)

From the old GRPO GSM8K eval (complete, 8 benchmarks):

| Method | Dataset | Avg Forgetting | Worst Benchmark |
|--------|---------|---------------|-----------------|
| GRPO | GSM8K | **-0.0076** | BoolQ (-0.0223) |
| GT-SFT | GSM8K | **+0.0102** (no forgetting) | — |

*Full 10-benchmark results pending eval sweep completion.*

---

## Cluster Notes

- **QOS limit**: ~96G total memory across all running+pending jobs
- **Eval speed**: ~1.5h per checkpoint (10 benchmarks, batch_size=1, 1x A100)
- **Trajectory gen**: ~6h for MATH (7.4K questions), ~1.5h for TriviaQA (7.5K questions)
- **Resumable evals**: `eval_sweep_resumable.sh` auto-resubmits on timeout
- All jobs chained via `--dependency=afterany` to stay within QOS

---

## TODO

- [ ] Finish all 6 eval sweeps (10 benchmarks) — running now, auto-resume
- [ ] Submit SF-SFT training (3 datasets) — data ready
- [ ] Submit SF-SFT eval sweeps (3 runs)
- [ ] Generate comparison plots (once evals done)
- [ ] Start Phase 2: Qwen3-4B (same experiments, larger model)
- [ ] CF-SFT: need Llama-3.1-70B trajectory generation
- [ ] SELF/SPIN: needs implementation
- [ ] PI: needs implementation
