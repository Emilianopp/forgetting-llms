# Execution Plan

## Infrastructure

- **Primary cluster**: Mila (`login.server.mila.quebec`)
- **Backup clusters**: Narval, Cedar, Graham (Alliance Canada)
- **Job scheduler**: Slurm
- **Job config**: `--partition=main --gres=gpu:a100l:2 --time=8:00:00 --mem=48G`
- **Tracking**: Weights & Biases
- **Storage**: `~/scratch` for datasets/checkpoints, `$HOME` for code
- **RL framework**: VeRL (GRPO via `verl.trainer.main_ppo`) + GEM environments for rewards
- **SFT framework**: VeRL `fsdp_sft_trainer` (torchrun)

## Model Family: Qwen3

All experiments use the **Qwen3** family. GEM has native Qwen3 prompt templates (`qwen3_general`, `qwen3_game`).

| Role | Model | HuggingFace ID |
|------|-------|---------------|
| Student (small) | Qwen3-1.7B | `Qwen/Qwen3-1.7B` |
| Student (primary) | Qwen3-4B | `Qwen/Qwen3-4B` |
| Student (scale) | Qwen3-8B | `Qwen/Qwen3-8B` |
| Student (stretch) | Qwen3-14B | `Qwen/Qwen3-14B` |
| Teacher (same-family) | Qwen3-32B | `Qwen/Qwen3-32B` |
| Teacher (cross-family) | Llama-3.1-70B-Instruct | `meta-llama/Llama-3.1-70B-Instruct` |

## GEM Environments (Training Domains)

| Domain | Training env_id | Eval env_id | Reward |
|--------|----------------|-------------|--------|
| Math | `math:GSM8K` | `eval:MATH500` | Binary (boxed answer match) |
| Code | `code:CodeContest` | `eval:CodeContest` | Binary (test case execution) |
| QA | `qa:NaturalQuestions` | `eval:NaturalQuestions` | Binary (exact match) |

## Phase 0: Infrastructure & Data Pipeline (Week 1-2)

### 0.1 Environment Setup
- [x] Create venv on Mila: `bash scripts/setup_env.sh`
- [x] Install VeRL, GEM, vLLM, Ray, lm-eval, wandb
- [ ] Verify GPU access: `salloc --partition=main --gres=gpu:a100l:2 --time=1:00:00`
- [ ] Set up WandB project: `forgetting-llms`
- [ ] **Smoke test**: Run GRPO on Qwen3-1.7B + GSM8K via VeRL (`scripts/run_grpo_smoke_test.sh`)

### 0.2 GEM Trajectory Collector
Build a custom data collection layer on top of GEM environments since GEM only provides RL training, not SFT data generation.

```
src/data/collector.py
```

Needs to:
1. Wrap GEM environments (`math:GSM8K`, `code:CodeContest`, `qa:NaturalQuestions`, etc.)
2. Run any HF model through the environment via vLLM for fast inference
3. Record full trajectories: `(prompt, response, reward, terminated, metadata)`
4. Export in multiple formats:
   - **SFT format**: `{"prompt": ..., "completion": ...}` (filter for correct responses only)
   - **Raw trajectories**: Full interaction logs for analysis
5. Support batched/vectorized collection via `gem.make_vec()`

### 0.3 Data Generation Runs

For each domain, generate:

| Dataset | How | Size target |
|---------|-----|------------|
| Ground truth SFT data | Extract from GEM env ground-truth answers | ~10K per domain |
| Same-family teacher data | Run Qwen3-32B through GEM envs | ~10K per domain |
| Cross-family teacher data | Run Llama-3.1-70B-Instruct through GEM envs | ~10K per domain |

**Note**: Teacher inference on 32B/70B models requires multi-GPU or API access. Consider using vLLM with tensor parallelism or Together AI / Fireworks API.

### 0.4 Evaluation Pipeline
Build a unified evaluation script that runs all forgetting benchmarks in one pass.

```
src/evaluation/run_eval.py --model <path> --suite [forgetting|safety|policy|all]
```

Use `lm-evaluation-harness` for standardized benchmarks. Custom scripts for:
- Sample-level forgetting tracking (correct→incorrect transitions)
- Policy analysis (diversity, length, calibration)
- Safety evaluation (HarmBench, XSTest, etc.)

## Phase 1: Core Forgetting Study (Week 3-6)

### 1.1 Base Model Experiments (BASE starting point)

Run all 6 methods × 3 domains on Qwen3-4B (base):

| Method | Training framework | Estimated time (2x A100 80GB) |
|--------|-------------------|-------------------------------|
| `GT-SFT` | VeRL fsdp_sft_trainer | ~3h |
| `SF-SFT` | VeRL fsdp_sft_trainer | ~3h |
| `CF-SFT` | VeRL fsdp_sft_trainer | ~3h |
| `SELF` | Custom SPIN loop (VeRL) | ~6h |
| `ON-RL` | VeRL GRPO + GEM reward | ~8h |
| `PI` | Custom pi-distill loop | ~8h |

**Total Phase 1.1**: 18 runs × ~5h avg = ~180 GPU-hours (2x A100 = 2 GPU-hours per hour)

### 1.2 Evaluation
- Run full forgetting profile on all 18 checkpoints
- Run base model evaluation (no post-training baseline)
- Compute sample-level forgetting rates
- Generate forgetting heatmaps (method × domain × capability)

### 1.3 Intermediate Analysis
- Which methods forget the most/least overall?
- Is forgetting domain-specific? (e.g., math training helps reasoning but hurts factual?)
- Do SFT variants differ meaningfully?
- Does RL forget differently than SFT?

## Phase 2: Safety Study (Week 5-8)

### 2.1 Safety-Aligned Model Experiments (SAFE starting point)

Repeat all 6 methods × 3 domains on Qwen3-4B-Instruct (safety-aligned):

**Total Phase 2.1**: 18 runs × ~5h avg = ~180 GPU-hours

### 2.2 Safety Evaluation
- Run safety benchmark suite on all 18 SAFE checkpoints
- Compare against SAFE baseline (Qwen3-4B-Instruct before any post-training)
- Run adversarial attacks (GCG, AutoDAN) on selected checkpoints
- Extract refusal direction from SAFE baseline, measure attenuation per method

### 2.3 Safety-Capability Analysis
- Plot safety score vs target task performance (Pareto fronts)
- Is safety degradation correlated with capability forgetting?
- Which methods are Pareto-optimal (good task performance, minimal safety loss)?
- Category-specific breakdown: which safety categories break first?

## Phase 3: Scale Analysis (Week 7-10)

### 3.1 Select Top Findings
From Phase 1+2, identify the 3-4 most interesting methods/domains to scale.

### 3.2 Multi-Scale Runs
Run selected experiments at:
- Qwen3-1.7B (cheaper, tests whether small models forget more)
- Qwen3-8B (tests whether larger models are more robust)
- Qwen3-14B (stretch goal)

**Total Phase 3**: ~28 runs × ~6h avg = ~336 GPU-hours (varies by scale)

### 3.3 Scale Analysis
- Plot forgetting vs model scale for each method
- Does the forgetting-scale relationship differ by method?
- Are larger models more safety-robust during post-training?

## Phase 4: Deep Analysis & Paper (Week 9-12)

### 4.1 Policy Analysis
On selected checkpoints:
- Output diversity metrics (Self-BLEU, distinct n-grams)
- Length distribution shifts
- Calibration analysis (ECE)
- Qualitative examples of behavioral changes

### 4.2 Representation Analysis (if compute allows)
- CKA similarity at each layer (base vs post-trained)
- Linear probes for factual knowledge retention
- Refusal direction attenuation analysis
- Weight delta SVD analysis

### 4.3 Paper Writing
- Results compilation and visualization
- Framing and narrative
- Related work integration

## Compute Budget Summary

| Phase | Runs | Est. GPU-hours |
|-------|------|---------------|
| Phase 0 (data gen + test) | Teacher inference + test runs | ~80h |
| Phase 1 (core) | 18 | ~180h |
| Phase 2 (safety) | 18 | ~180h |
| Phase 3 (scale) | ~28 | ~336h |
| Phase 4 (analysis) | Eval only | ~40h |
| **Total** | | **~816 GPU-hours** |

At 2x A100 per job, this is ~408 wall-clock hours of jobs. Feasible on Mila's allocation.

## Slurm Job Template (Standard — all jobs)

```bash
#!/bin/bash
#SBATCH --job-name=forgetting-{method}-{domain}
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface

# For SFT methods (GT-SFT, SF-SFT, CF-SFT):
srun torchrun --nproc_per_node=2 src/training/{method}.py \
    --config configs/methods/{method}.yaml \
    --domain configs/domains/{domain}.yaml ...

# For ON-RL (GRPO):
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    custom_reward_function.path=src/rewards/math_reward.py \
    ...
```

## ON-RL Quick Start (VeRL GRPO)

The fastest path to a working run:

```bash
# 1. Preprocess data
python scripts/preprocess_data.py --dataset gsm8k

# 2. Run GRPO smoke test
sbatch scripts/run_grpo_smoke_test.sh
```

## Risk Mitigation

- **Teacher inference cost**: Use API access (Together AI, Fireworks) for 32B/70B if GPU allocation is tight
- **GEM environment issues**: Fall back to static datasets (GSM8K, MBPP, Alpaca) if GEM envs are unreliable
- **Pi-distill complexity**: This is the newest method — start with the other 5, add PI once infrastructure is stable
- **Evaluation bottleneck**: Run evals in parallel across multiple nodes; use `lm-evaluation-harness` batch mode
- **Preemption (long partition)**: Stick to `main` partition (not preemptable) with 8h jobs; checkpoint frequently
