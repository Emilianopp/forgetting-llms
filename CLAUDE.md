# forgetting-llms

## Project Overview
Systematic study of forgetting across 7 post-training methods (GT-SFT, SF-SFT, CF-SFT, Self-distillation, Online RL, Off-policy RL, Pi-distill) × 3 domains (math, code, QA) × 2 starting points (base, safety-aligned) on Qwen3 models.

## Key Files
- `PROJECT.md` — Full research design, evaluation framework, related work
- `PLAN.md` — Execution plan, compute budget, Slurm templates, phasing

## Workflow — Remote-First (Mila Cluster)

All development and execution happens on the Mila cluster. The local machine is only for editing docs and pushing code.

### Step 1: Connect to Mila
```bash
ssh mila
```
SSH config alias `mila` resolves to `login.server.mila.quebec` (port 2222, user `emiliano.penaloza`).

### Step 2: Clone repo (first time only)
```bash
cd $HOME
git clone https://github.com/Emilianopp/forgetting-llms.git
```

### Step 3: Navigate to project
```bash
cd $HOME/forgetting-llms
```

### Step 4: Activate environment
```bash
source $HOME/envs/forgetting/bin/activate
```

### Running experiments
- Submit training jobs: `sbatch scripts/train.sh <method> <domain> <starting_point> <model_scale>`
- Submit eval jobs: `sbatch scripts/eval.sh <run_name>`
- Launch full phase: `bash scripts/launch_phase1.sh`
- Monitor jobs: `squeue -u $USER`
- Check logs: `tail -f slurm_logs/<job_id>_*.out`

### Syncing code
Repo: https://github.com/Emilianopp/forgetting-llms (private)

```bash
# Local: push to remote
git push origin main

# Mila: pull latest
cd $HOME/forgetting-llms && git pull
```

## Cluster
- **Primary**: Mila (`login.server.mila.quebec`, port 2222, user: `emiliano.penaloza`)
- **SSH alias**: `mila` (configured in ~/.ssh/config)
- **Backup**: Narval, Cedar, Graham (Alliance Canada, user: `emiliano`)
- **Scheduler**: Slurm
- **GPUs**: A100 (40GB/80GB) via `--gres=gpu:a100:1`
- **Storage**:
  - `$HOME` — Code, configs, small files (quota limited)
  - `$SCRATCH` — Checkpoints, datasets, large files (no quota, auto-purged after 90 days)
  - `/cvmfs/` — Shared modules
- **Tracking**: Weights & Biases (project: `forgetting-llms`)

## Environment
- Python 3.11 (`module load python/3.11`)
- PyTorch, Transformers, VeRL, vLLM, datasets, wandb, Ray
- GEM (`gem-llm`) for RL environments
- TRL (fallback for offline DPO only)
- `lm-evaluation-harness` for benchmarks
- Setup script: `bash scripts/setup_env.sh`

## Training Frameworks
- **SFT methods** (GT-SFT, SF-SFT, CF-SFT): VeRL `fsdp_sft_trainer` (torchrun)
- **Online RL** (ON-RL): VeRL + GEM (GRPO via Ray)
- **Self-distillation** (SELF): VeRL SPIN recipe
- **Off-policy RL** (OFF-RL): TRL `DPOTrainer` (VeRL doesn't support offline DPO)
- **Pi-distill** (PI): Custom training loop

## Conventions
- Configs use YAML in `configs/{methods,domains,models}/`
- All training scripts in `src/training/`
- Slurm logs go to `slurm_logs/`
- Checkpoints go to `$SCRATCH/forgetting-llms/checkpoints/{method}_{domain}_{starting_point}_{scale}/`
- Results go to `results/{run_id}/`
- WandB run names: `{method}_{domain}_{starting_point}_{scale}`

## Slurm Partitions
- `main` — Short jobs (<12h), eval runs
- `long` — Training runs (up to 48h)
- Use `--gres=gpu:a100:1` for single-GPU (SFT, DPO, eval)
- Use `--gres=gpu:a100:2` for multi-GPU (ON-RL, PI)
