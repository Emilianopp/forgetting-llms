# forgetting-llms

This repo currently has one reliable workflow:

- build a scratch-local Python env
- use a local model snapshot in `~/scratch`
- generate correct-only SFT traces
- keep PRIME-style tagged answers like `<answer>...</answer>`
- resume after preemption from question-level checkpoints
- run RL through PRIME-RL only

This README is intentionally focused on that path. It does not assume you are
using any legacy VeRL training scripts.

## Storage Layout

The repo can stay in home, for example:

```bash
~/forgetting-llms
```

Large artifacts should stay in scratch:

- Mila examples below use `~/scratch/...`
- Alliance / Compute Canada should use `$SCRATCH/...`
- env: `<scratch>/forgetting-llms/.venv`
- model snapshots: `<scratch>/forgetting-llms/models`
- datasets: `<scratch>/forgetting-llms/data`
- caches: `<scratch>/.cache`, `<scratch>/huggingface`
- temp files: `<scratch>/tmp`
- WandB local files: `<scratch>/forgetting-llms/wandb`

## Environment Setup

Package installation is defined in [scripts/setup_env.sh](/Users/danemalenfant/PycharmProjects/forgetting-llms/scripts/setup_env.sh). There is no separate `requirements.txt`.

Cluster-specific wrappers:

- Mila: [scripts/setup_mila.sh](/Users/danemalenfant/PycharmProjects/forgetting-llms/scripts/setup_mila.sh)
- Alliance / Compute Canada: [scripts/setup_alliance.sh](/Users/danemalenfant/PycharmProjects/forgetting-llms/scripts/setup_alliance.sh)

Those wrappers run the main env setup and PRIME-RL setup back to back and keep
their scratch roots separate.

Legacy VeRL training scripts in `scripts/run_grpo_*`, `scripts/run_sft.sh`, and
related wrappers are intentionally disabled by default. The supported RL path is
PRIME-RL only.

On Mila, the shortest path is:

```bash
bash scripts/setup_mila.sh
```

On Alliance / Compute Canada, use:

```bash
bash scripts/setup_alliance.sh
```

If you want the manual path instead, on Mila start from a clean shell and load the cluster Python first:

```bash
module purge
module load python/3.10
```

If your node requires OpenSSL 1.1 for `pyarrow.dataset`, load that before activating the env:

```bash
module spider libssl
```

Then load the matching module if available, for example one of:

```bash
module load libssl/1.1
```

or:

```bash
module load openssl/1.1
```

Create the scratch env:

```bash
bash scripts/setup_env.sh
source ~/scratch/forgetting-llms/.venv/bin/activate
```

If the env is broken or trapped in a symlink loop, rebuild it with copies:

```bash
REBUILD=1 USE_UV=0 VENV_COPIES=1 bash scripts/setup_env.sh
source ~/scratch/forgetting-llms/.venv/bin/activate
```

If the `transformers` install is incomplete, force a core reinstall:

```bash
REBUILD=1 USE_UV=0 VENV_COPIES=1 FORCE_REINSTALL_CORE=1 bash scripts/setup_env.sh
source ~/scratch/forgetting-llms/.venv/bin/activate
```

Verify the env:

```bash
python --version
echo "$VIRTUAL_ENV"
python -c "import ssl, pyarrow.dataset, transformers; print('ok')"
```

Expected:

- Python `3.10.x`
- `VIRTUAL_ENV` under `~/scratch/forgetting-llms/.venv`
- the import check prints `ok`

Most command examples below still show Mila-style `~/scratch/...` paths. On
Alliance, replace those with `$SCRATCH/...`.

## Model Snapshots

Download models into scratch with [scripts/download_models.py](/Users/danemalenfant/PycharmProjects/forgetting-llms/scripts/download_models.py).

Example:

```bash
source ~/scratch/forgetting-llms/.venv/bin/activate

python scripts/download_models.py \
  --model Qwen/Qwen3.5-27B
```

For Hugging Face ids, local snapshot paths are:

- `Qwen/Qwen3.5-27B` -> `~/scratch/forgetting-llms/models/Qwen__Qwen3.5-27B`
- `Qwen/Qwen3-32B` -> `~/scratch/forgetting-llms/models/Qwen__Qwen3-32B`
- `Qwen/Qwen3-1.7B` -> `~/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B`

The direct path for Qwen 3.5 27B is:

```bash
$HOME/scratch/forgetting-llms/models/Qwen__Qwen3.5-27B
```

## SFT Trace Generation

The trace generator is [src/data/generate_teacher_solutions.py](/Users/danemalenfant/PycharmProjects/forgetting-llms/src/data/generate_teacher_solutions.py).

What it does:

- generates multiple candidate solutions per question
- keeps only correct solutions
- tracks per-question progress in `status.parquet`
- stores positive solutions in `checkpoint.parquet`
- resumes from those files on rerun
- writes final `train.parquet`, `test.parquet`, and `summary.json`

For PRIME-style traces without RL, use:

- `--answer_format tagged`

That requests outputs like:

```text
<think>...</think><answer>...</answer>
```

The reward code has been updated so correctness checks read `<answer>...</answer>`.

## Recommended Mila Run Command

This is the direct non-RL command for tagged correct-only trace generation with Qwen 3.5 27B:

```bash
module purge
module load python/3.10
source ~/scratch/forgetting-llms/.venv/bin/activate

export VLLM_USE_STANDALONE_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1

python src/data/generate_teacher_solutions.py \
  --model "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3.5-27B" \
  --dataset gsm8k \
  --answer_format tagged \
  --samples_per_round 4 \
  --max_total_samples 16 \
  --target_correct_per_question 16 \
  --min_correct_per_question 8 \
  --solutions_per_question 8 \
  --tensor_parallel_size 4 \
  --max_model_len 8192 \
  --max_tokens 2048 \
  --gpu_memory_utilization 0.90 \
  --temperature 1.0 \
  --top_p 1.0 \
  --chunk_size 50 \
  --enforce_eager \
  --output_dir ~/scratch/forgetting-llms/data/gsm8k_sf_sft
```

Notes:

- `tensor_parallel_size=4` matches 4 GPUs
- `temperature=1.0` and `top_p=1.0` keep the fixed sampling policy
- `target_correct_per_question=16` prevents early stopping before 16 attempts
- `min_correct_per_question=8` keeps only questions with at least 8 correct traces
- `solutions_per_question=8` keeps exactly 8 positive traces per retained question
- `--enforce_eager` plus the two `VLLM_*` exports avoids the unstable standalone compile path

## Slurm Wrapper

There is also a Slurm wrapper at [scripts/generate_trajectories.sh](/Users/danemalenfant/PycharmProjects/forgetting-llms/scripts/generate_trajectories.sh).

Example:

```bash
module purge
module load python/3.10
source ~/scratch/forgetting-llms/.venv/bin/activate

export VLLM_USE_STANDALONE_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1

ANSWER_FORMAT=tagged \
MAX_TOTAL_SAMPLES=16 \
TARGET_CORRECT_PER_QUESTION=16 \
MIN_CORRECT_PER_QUESTION=8 \
SOLUTIONS_PER_QUESTION=8 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
TEMPERATURE=1.0 \
TOP_P=1.0 \
sbatch scripts/generate_trajectories.sh \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3.5-27B" \
  gsm8k \
  4
```

If you use the wrapper on 4 GPUs, make sure the script resource request matches your node allocation. The direct Python command is the clearer path when you need exact control.

## Resume Behavior

The generator is resumable as long as you keep the same output directory.

For the command above, progress is stored in:

- `~/scratch/forgetting-llms/data/gsm8k_sf_sft/checkpoint.parquet`
- `~/scratch/forgetting-llms/data/gsm8k_sf_sft/status.parquet`

If the job is preempted, rerun the same command. It will:

- reload existing positives
- reload question status
- skip completed questions
- continue only on remaining questions

You do not need a special resume flag.

## Final Outputs

When complete, the dataset directory contains:

- `checkpoint.parquet`
- `status.parquet`
- `train.parquet`
- `test.parquet`
- `summary.json`

`train.parquet` is the retained correct-only SFT dataset.

## Common Failures

- `ImportError: ... libssl.so.1.1`
  - `pyarrow` cannot find OpenSSL 1.1 on the node
  - fix: load the appropriate SSL module before activating the env

- `Too many levels of symbolic links`
  - broken scratch venv
  - fix: rebuild with `REBUILD=1 USE_UV=0 VENV_COPIES=1`

- missing files under `transformers/models/...`
  - incomplete `transformers` install
  - fix: `FORCE_REINSTALL_CORE=1 bash scripts/setup_env.sh`

- `standalone_compile ... FakeTensorMode`
  - unstable vLLM compile path
  - fix: use `VLLM_USE_STANDALONE_COMPILE=0`, `VLLM_DISABLE_COMPILE_CACHE=1`, and `--enforce_eager`

- long pause during vLLM startup
  - normal for 27B if weights and KV cache are still initializing
  - wait for the `init engine ... took ... seconds` line before assuming it is hung

## Other Datasets

Swap `--dataset gsm8k` and the output directory for:

- `math`
- `triviaqa`
- `polaris_math`

Examples:

```bash
--dataset math --output_dir ~/scratch/forgetting-llms/data/math_sf_sft
```

```bash
--dataset triviaqa --output_dir ~/scratch/forgetting-llms/data/triviaqa_sf_sft
```
