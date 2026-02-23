# Execution Plan

## Infrastructure

- **Primary cluster**: Mila (`login.server.mila.quebec`)
- **Backup clusters**: Narval, Cedar, Graham (Alliance Canada)
- **Job scheduler**: Slurm
- **Tracking**: Weights & Biases
- **Storage**: `$SCRATCH` for datasets/checkpoints, `$HOME` for code
- **Modules**: `/cvmfs/config.mila.quebec/` for shared software

## Phase 0: Infrastructure & Data Pipeline (Week 1-2)

### 0.1 Environment Setup
- [ ] Create conda/venv environment on Mila with: `torch`, `transformers`, `trl`, `datasets`, `vllm`, `wandb`
- [ ] Install GEM: `pip install gem-llm`
- [ ] Install evaluation harness: `lm-evaluation-harness` (EleutherAI)
- [ ] Verify GPU access: request A100 (40GB or 80GB) nodes via Slurm
- [ ] Set up WandB project: `forgetting-llms`

### 0.2 GEM Trajectory Collector
Build a custom data collection layer on top of GEM environments since GEM only provides RL training, not SFT data generation.

```
src/data/collector.py
```

Needs to:
1. Wrap GEM environments (`Math12K`, `CodeContest`, `NaturalQuestions`, etc.)
2. Run any HF model through the environment via vLLM for fast inference
3. Record full trajectories: `(prompt, response, reward, terminated, metadata)`
4. Export in multiple formats:
   - **SFT format**: `{"prompt": ..., "completion": ...}` (filter for correct responses only)
   - **DPO format**: `{"prompt": ..., "chosen": ..., "rejected": ...}` (pair correct vs incorrect)
   - **Raw trajectories**: Full interaction logs for analysis
5. Support batched/vectorized collection via `gem.make_vec()`

### 0.3 Data Generation Runs

For each domain, generate:

| Dataset | How | Size target |
|---------|-----|------------|
| Ground truth SFT data | Extract from GEM env ground-truth answers | ~10K per domain |
| Same-family teacher data | Run Qwen2.5-72B-Instruct through GEM envs | ~10K per domain |
| Cross-family teacher data | Run Llama-3.1-70B-Instruct through GEM envs | ~10K per domain |
| Preference pairs (for DPO) | Collect correct + incorrect rollouts from base model | ~10K pairs per domain |

**Note**: Teacher inference on 72B/70B models requires multi-GPU (4x A100 80GB) or API access. Consider using vLLM with tensor parallelism.

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

Run all 7 methods × 3 domains on Qwen2.5-3B (base):

| Method | Training framework | Estimated GPU-hours (3B) |
|--------|-------------------|-------------------------|
| `GT-SFT` | `trl` SFTTrainer | ~4h on 1x A100 |
| `SF-SFT` | `trl` SFTTrainer | ~4h on 1x A100 |
| `CF-SFT` | `trl` SFTTrainer | ~4h on 1x A100 |
| `SELF` | Custom SPIN loop | ~8h on 1x A100 (iterative) |
| `ON-RL` | GEM + Oat/OpenRLHF (GRPO) | ~12h on 2x A100 |
| `OFF-RL` | `trl` DPOTrainer | ~6h on 1x A100 |
| `PI` | Custom pi-distill loop | ~12h on 2x A100 |

**Total Phase 1.1**: 21 runs × ~6h avg = ~126 GPU-hours

### 1.2 Evaluation
- Run full forgetting profile on all 21 checkpoints
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

Repeat all 7 methods × 3 domains on Qwen2.5-3B-Instruct:

**Total Phase 2.1**: 21 runs × ~6h avg = ~126 GPU-hours

### 2.2 Safety Evaluation
- Run safety benchmark suite on all 21 SAFE checkpoints
- Compare against SAFE baseline (Qwen2.5-3B-Instruct before any post-training)
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
- Qwen2.5-1.5B (cheaper, tests whether small models forget more)
- Qwen2.5-7B (tests whether larger models are more robust)
- Qwen2.5-14B (stretch goal)

**Total Phase 3**: ~28 runs × ~8h avg = ~224 GPU-hours (varies by scale)

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
| Phase 0 (data gen) | Teacher inference | ~50h (72B models) |
| Phase 1 (core) | 21 | ~126h |
| Phase 2 (safety) | 21 | ~126h |
| Phase 3 (scale) | ~28 | ~224h |
| Phase 4 (analysis) | Eval only | ~30h |
| **Total** | | **~556 GPU-hours** |

This is feasible on Mila's allocation. A100 nodes at Mila are available via `--gres=gpu:a100:1` (or `:2`, `:4`).

## Slurm Job Templates

### Single-GPU Training (SFT, DPO)
```bash
#!/bin/bash
#SBATCH --job-name=forgetting-{method}-{domain}
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j_%x.out

module load python/3.11
source $HOME/envs/forgetting/bin/activate

python src/training/{method}.py \
    --config configs/methods/{method}.yaml \
    --domain configs/domains/{domain}.yaml \
    --model configs/models/qwen_3b.yaml \
    --starting_point {base|safe} \
    --wandb_project forgetting-llms
```

### Multi-GPU Training (ON-RL, PI)
```bash
#!/bin/bash
#SBATCH --job-name=forgetting-{method}-{domain}
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_%x.out

module load python/3.11
source $HOME/envs/forgetting/bin/activate

torchrun --nproc_per_node=2 src/training/{method}.py \
    --config configs/methods/{method}.yaml \
    --domain configs/domains/{domain}.yaml \
    --model configs/models/qwen_3b.yaml \
    --starting_point {base|safe} \
    --wandb_project forgetting-llms
```

### Evaluation
```bash
#!/bin/bash
#SBATCH --job-name=eval-{run_id}
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%j_%x.out

module load python/3.11
source $HOME/envs/forgetting/bin/activate

python src/evaluation/run_eval.py \
    --model $SCRATCH/forgetting-llms/checkpoints/{run_id} \
    --suite all \
    --output results/{run_id}/
```

## Launcher Script

A master launcher to submit all Phase 1 jobs:

```bash
# scripts/launch_phase1.sh
for method in gt_sft sf_sft cf_sft self on_rl off_rl pi; do
    for domain in math code qa; do
        sbatch scripts/train.sh $method $domain base qwen_3b
    done
done
```

## Risk Mitigation

- **Teacher inference cost**: Use API access (Together AI, Fireworks) for 72B/70B if GPU allocation is tight
- **GEM environment issues**: Fall back to static datasets (GSM8K, MBPP, Alpaca) if GEM envs are unreliable
- **Pi-distill complexity**: This is the newest method — start with the other 6, add PI once infrastructure is stable
- **Evaluation bottleneck**: Run evals in parallel across multiple nodes; use `lm-evaluation-harness` batch mode
