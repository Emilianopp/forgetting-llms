# Cluster Skills — Mila

## Connection

```bash
# Connect (requires active SSH session for ControlMaster socket)
ssh mila

# Run a single command remotely
ssh mila "command here"

# Home directory on Mila
$HOME = /home/mila/e/emiliano.penaloza

# Scratch (large, fast, auto-purged after 90 days)
$SCRATCH = /network/scratch/e/emiliano.penaloza
```

## GPU Types Available

| Slurm GRES | GPU | VRAM | Nodes |
|------------|-----|------|-------|
| `gpu:a100l:N` | A100 | 80GB | 30 nodes (4 GPUs each) |
| `gpu:a100:N` | A100 | 40GB | 5 nodes (4-8 GPUs each) |
| `gpu:h100:N` | H100 | 80GB | 2 nodes (8 GPUs each) |
| `gpu:l40s:N` | L40S | 48GB | 92 nodes (4 GPUs each) |
| `gpu:rtx8000:N` | RTX 8000 | 48GB | ~50 nodes (8 GPUs each) |
| `gpu:v100:N` | V100 | 32GB | ~6 nodes (8 GPUs each) |

**For this project**: Use `gpu:a100l:2` (2x A100 80GB).

## Partitions

| Partition | Max Time | Max GPUs | Max CPUs | Max Mem | Preemptable? |
|-----------|----------|----------|----------|---------|-------------|
| `unkillable` | 2 days | 1 | 6 | 32G | No |
| `short-unkillable` | 3 hours | 4 | - | 1000G | No |
| `main` | 5 days | 2 | 8 | 48G | No (but can preempt `long`) |
| `main-grace` | 5 days | 2 | 8 | 48G | Yes (120s SIGTERM grace) |
| `long` | 7 days | unlimited | unlimited | unlimited | Yes (killed immediately) |
| `long-grace` | 7 days | unlimited | unlimited | unlimited | Yes (120s SIGTERM grace) |

**For this project**: Use `main` partition (5 day limit, 2 GPU max, not preemptable).

## Standard Job Template (8h, 2x A100 80GB)

This is our default job configuration for all training runs:

```bash
#!/bin/bash
#SBATCH --job-name=forgetting
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load python/3.11 2>/dev/null || true
source $HOME/envs/forgetting/bin/activate

# Copy data to fast local storage if needed
# cp -r $SCRATCH/forgetting-llms/data $SLURM_TMPDIR/data

# Run training
srun python src/training/train.py "$@"
```

## Multi-GPU Launch (2 GPUs)

For distributed training with 2 GPUs:

```bash
# With torchrun (recommended for DDP)
srun torchrun --nproc_per_node=2 src/training/train.py "$@"

# With accelerate
srun accelerate launch --num_processes=2 src/training/train.py "$@"
```

**Important**: For multi-GPU jobs, may need `--gres-flags=allow-task-sharing` to prevent NCCL errors:
```bash
srun --gres-flags=allow-task-sharing torchrun --nproc_per_node=2 ...
```

## Job Management

```bash
# Submit a job
sbatch scripts/train.sh gt_sft math base qwen_3b

# View your jobs
squeue -u $USER

# View job details
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Check which GPU was allocated
scontrol show -d job <JOB_ID> | grep 'GRES'

# Job history
sacct -u $USER --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,NodeList --start=2026-02-01

# Tail a running job's log
tail -f slurm_logs/<JOB_ID>_*.out
```

## Interactive Session

For debugging or quick experiments:

```bash
# Interactive session with 2x A100 80GB, 8 hours
salloc --partition=main --gres=gpu:a100l:2 --cpus-per-task=8 --mem=48G --time=8:00:00
```

## Storage Strategy

| What | Where | Why |
|------|-------|-----|
| Code | `$HOME/forgetting-llms/` | Backed up daily, persists |
| Model checkpoints | `$SCRATCH/forgetting-llms/checkpoints/` | Fast, large quota (5TB) |
| Datasets / generated data | `$SCRATCH/forgetting-llms/data/` | Fast, large |
| Temp data during job | `$SLURM_TMPDIR/` | Fastest (local SSD), deleted after job |
| Long-term archive | `$ARCHIVE/forgetting-llms/` | Persistent, login-node only |

**Key rule**: Copy frequently needed data to `$SLURM_TMPDIR` at job start for best I/O performance. Save results back to `$SCRATCH` before job ends.

```bash
# In job script: copy data to local fast storage
cp -r $SCRATCH/forgetting-llms/data/$DOMAIN $SLURM_TMPDIR/data

# At end: save checkpoints back
cp -r $SLURM_TMPDIR/checkpoints $SCRATCH/forgetting-llms/checkpoints/$RUN_NAME
```

## Environment Setup (First Time)

```bash
# On Mila login node
module load python/3.11
python -m venv $HOME/envs/forgetting
source $HOME/envs/forgetting/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets accelerate trl
pip install vllm wandb peft
pip install gem-llm
pip install lm-eval
pip install matplotlib seaborn pandas scipy
```

Or run: `bash scripts/setup_env.sh`

## Syncing Code

```bash
# From local machine: push changes
cd /Users/emilianopenaloza/git/forgetting-llms
git add . && git commit -m "message" && git push

# On Mila: pull latest
ssh mila "cd \$HOME/forgetting-llms && git pull"
```

## Common Patterns

### Submit and monitor a training run
```bash
# Submit
JOB_ID=$(sbatch --parsable scripts/train.sh gt_sft math base qwen_3b)
echo "Submitted job $JOB_ID"

# Monitor
squeue -u $USER
tail -f slurm_logs/${JOB_ID}_*.out
```

### Submit all runs for a phase
```bash
bash scripts/launch_phase1.sh
```

### Check GPU utilization on a running job
```bash
# SSH into the compute node
ssh <node_name>
nvidia-smi
```

### Recover from preemption (long partition)
Use `main-grace` or `long-grace` partitions and trap SIGTERM:
```bash
# In job script
trap 'echo "SIGTERM received, saving checkpoint..."; kill $PID; wait $PID' SIGTERM
```

## Constraints Reference

```bash
# Specific GPU memory
--constraint=80gb        # 80GB VRAM (A100L, H100)
--constraint=48gb        # 48GB VRAM (RTX8000, A6000, L40S)
--constraint=40gb        # 40GB VRAM (A100)

# GPU architecture
--constraint=ampere      # A100, A6000
--constraint=hopper      # H100
--constraint=lovelace    # L40S
--constraint=turing      # RTX8000
--constraint=volta       # V100

# Combine with &
--constraint="ampere&80gb"   # A100 80GB specifically
--constraint="ampere&nvlink" # A100 with NVLink
```

## Limits to Remember

- `main` partition: **max 2 GPUs**, 8 CPUs, 48G mem, 5 days
- `long` partition: no resource limit but **preemptable**
- Max 1000 submitted jobs per user
- `$SCRATCH` files unused for 90 days are auto-deleted
- `$SLURM_TMPDIR` is wiped after each job
- Login nodes: **no heavy compute** — use `salloc` for interactive work
- Internet access available on compute nodes (for pip install, wandb, HF downloads)
