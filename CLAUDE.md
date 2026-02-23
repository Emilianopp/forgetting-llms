# forgetting-llms

## Project Overview
Systematic study of forgetting across 6 post-training methods (GT-SFT, SF-SFT, CF-SFT, Self-distillation, Online RL, Pi-distill) x 3 domains (math, code, QA) x 2 starting points (base, safety-aligned) on Qwen3 models.

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
module load python/3.10
source $HOME/envs/forgetting/bin/activate
```

### Running experiments
- Submit training jobs: `sbatch scripts/train.sh <method> <domain> <starting_point> <model_scale>`
- Submit eval jobs: `sbatch scripts/eval.sh <run_name>`
- Launch full phase: `bash scripts/launch_phase1.sh`
- Monitor jobs: `squeue -u $USER`
- Check logs: `tail -f slurm_logs/<job_id>_*.out`

### Syncing code
Repo: https://github.com/Emilianopp/forgetting-llms (public)

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
- **GPUs**: A100 80GB via `--gres=gpu:a100l:2`
- **Storage**:
  - `$HOME` — Code, configs, small files (quota limited)
  - `~/scratch` — **ALL** checkpoints, datasets, model weights, large files (auto-purged after 90 days)
  - `/cvmfs/` — Shared modules
- **Tracking**: Weights & Biases (project: `forgetting-llms`)

## STRICT RULE: Storage
**All data, checkpoints, datasets, and model weights MUST go to `~/scratch/`.** Never store large files in `$HOME`. Use paths like:
- Checkpoints: `~/scratch/forgetting-llms/checkpoints/{run_name}/`
- Datasets: `~/scratch/forgetting-llms/data/`
- Model cache: `~/scratch/huggingface/`
- Set `HF_HOME=~/scratch/huggingface` so transformers/datasets cache to scratch

## Environment
- Python 3.10 (`module load python/3.10`)
- PyTorch 2.9.1, Transformers 4.57.6, VeRL 0.7.0, vLLM 0.15.1, Ray 2.54.0
- GEM (`gem-llm`) for RL environments
- `lm-evaluation-harness` for benchmarks
- Setup script: `bash scripts/setup_env.sh`

## Training Frameworks
- **SFT methods** (GT-SFT, SF-SFT, CF-SFT): VeRL `fsdp_sft_trainer` (torchrun)
- **Online RL** (ON-RL): VeRL + GEM (GRPO via Ray)
- **Self-distillation** (SELF): VeRL SPIN recipe
- **Pi-distill** (PI): Custom training loop

## Conventions
- Configs use YAML in `configs/{methods,domains,models}/`
- All training scripts in `src/training/`
- Slurm logs go to `slurm_logs/`
- Checkpoints go to `~/scratch/forgetting-llms/checkpoints/{method}_{domain}_{starting_point}_{scale}/`
- Results go to `results/{run_id}/`
- WandB run names: `{method}_{domain}_{starting_point}_{scale}`

## Slurm Job Config (Standard — all jobs)
```
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
```
