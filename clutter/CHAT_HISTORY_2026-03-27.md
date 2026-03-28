# Chat History - 2026-03-27

This file is the current end-to-end experiment guide for the repo.

It is focused on:

- how the repo is intended to be run now
- which entrypoints to use for each experiment family
- how asynchronous eval is supposed to work
- what is stable vs what still needs external benchmark setup

It is not a paper summary. It is an operator guide.

## Main Goal

Run a complete matrix of experiments where:

1. SFT traces are generated first
2. training is done on:
   - one dataset
   - a mixed dataset
   - a sequential chain of datasets
3. training modes can be:
   - `sft`
   - `rl`
   - `sft_rl`
   - self-distillation
4. training can be:
   - with LoRA
   - without LoRA
   - LoRA SFT continued with RL
5. checkpoints are evaluated on a separate GPU while training continues
6. sequential experiments follow:
   - train -> eval checkpoint(s) -> continue training -> eval newer checkpoint(s)
7. benchmark evals can be run as:
   - a stable working core
   - a broader suite once external dependencies are installed
8. local backward-task evals can be run after each sequential stage

## Current Repo Entry Points

### Data / Trace Generation

- `scripts/generate_trajectories.sh`
- `src/data/generate_teacher_solutions.py`
- `scripts/import_dolci_sft.py`

### Training

- `scripts/run_sft.sh`
- `scripts/run_training_plan.sh`
- `scripts/run_self_distill.sh`
- `src/training/plain_sft.py`
- `src/training/self_distill.py`

### PRIME-RL

- `scripts/prime_rl_runner.py`
- `scripts/setup_prime_rl.sh`

### Benchmark Eval

- `src/evaluation/run_eval.py`
- `scripts/run_eval_with_local_server.sh`
- `scripts/watch_checkpoints_eval.sh`
- `scripts/watch_prime_run_eval.sh`
- `scripts/eval_prime_checkpoint_sweep.py`

### Benchmark Dependency Setup

- `scripts/setup_tasks_md_benchmarks.sh`
- `scripts/setup_runner_venvs.sh`

## Recommended Baseline Setup

On Mila, the normal startup is:

```bash
module load python/3.10
source ~/scratch/forgetting-llms/.venv/bin/activate
source scripts/load_hf_auth.sh
```

If benchmark env vars are configured:

```bash
source ~/scratch/forgetting-llms/benchmark_env.sh
```

If PRIME-RL is needed:

```bash
source ~/scratch/forgetting-llms/prime_rl_env.sh
```

## Data Layout Convention

The repo expects parquet datasets under:

```bash
~/scratch/forgetting-llms/data/<dataset_name>_sft
~/scratch/forgetting-llms/data/<dataset_name>_sf_sft
~/scratch/forgetting-llms/data/<dataset_name>_cf_sft
```

Typical files:

- `train.parquet`
- `test.parquet`
- optionally `summary.json`

The common convention is:

- `gt` variant: existing gold/ground-truth style SFT data
- `sf` variant: self-filtered / sampled reasoning traces
- `cf` variant: counterfactual or custom variant

## 1. Generate SFT Traces First

All SFT-based experiments should start from trace generation.

### Verifiable datasets

Use:

- `gsm8k`
- `math`
- `triviaqa`
- `polaris_math`
- `openr1_math`

The default helper is:

```bash
bash scripts/generate_trajectories.sh <model> <dataset> <samples_per_round>
```

Example:

```bash
TARGET_CORRECT_PER_QUESTION=2 \
MIN_CORRECT_PER_QUESTION=2 \
SOLUTIONS_PER_QUESTION=2 \
MAX_TOTAL_SAMPLES=16 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
TEMPERATURE=1.0 \
TOP_P=1.0 \
bash scripts/generate_trajectories.sh \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  gsm8k \
  4
```

For these datasets, the correctness-gated knobs matter:

- `TARGET_CORRECT_PER_QUESTION`
- `MIN_CORRECT_PER_QUESTION`
- `SOLUTIONS_PER_QUESTION`
- `MAX_TOTAL_SAMPLES`

### Dolci / SYNTHETIC-2 prompt datasets

Supported prompt datasets in the wrapper include:

- `dolci_think_sft_7b`
- `dolci_think_sft_32b`
- `synthetic2_sft_verified`

Example:

```bash
MAX_SAMPLES=400 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
TEMPERATURE=1.0 \
TOP_P=1.0 \
MAX_TOKENS=2048 \
CHUNK_SIZE=128 \
bash scripts/generate_trajectories.sh \
  "$HOME/scratch/forgetting-llms/models/allenai__Olmo-3-7B-Instruct" \
  synthetic2_sft_verified \
  4
```

Important distinction:

- for `synthetic2_sft_verified` and Dolci prompt imports, `4` means generated samples per prompt
- not correctness-gated “sample until 2 correct”

## 2. Individual Training Experiments

The main unified launcher is:

```bash
bash scripts/run_training_plan.sh <training_mode> <schedule> <base_model> <run_prefix> <stage1> [stage2 ...]
```

For individual experiments:

- `schedule=individual`

### Individual SFT

```bash
DATA_VARIANT=sf \
SFT_MAX_STEPS=300 \
WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  sft individual \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_synth2_sft \
  synthetic2_sft_verified
```

### Individual RL

This is PRIME-only.

Example on a built-in RL environment:

```bash
PRIME_BATCH_SIZE=1 \
PRIME_SEQ_LEN=4096 \
PRIME_MAX_TOKENS=512 \
PRIME_ROLLOUTS_PER_PROMPT=1 \
PRIME_ENFORCE_EAGER=1 \
PRIME_WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  rl individual \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_gsm8k_rl \
  gsm8k
```

Important limitation:

- PRIME-RL on custom datasets like `synthetic2_sft_verified` requires a real PRIME env/config mapping via:
  - `STAGE_ENV_<DATASET>`
  - optional `STAGE_COMBINED_CONFIG_<DATASET>` or split configs

### Individual SFT + RL

```bash
DATA_VARIANT=sf \
SFT_MAX_STEPS=300 \
RL_MAX_STEPS=300 \
PRIME_BATCH_SIZE=1 \
PRIME_SEQ_LEN=4096 \
PRIME_MAX_TOKENS=512 \
PRIME_ROLLOUTS_PER_PROMPT=1 \
PRIME_ENFORCE_EAGER=1 \
bash scripts/run_training_plan.sh \
  sft_rl individual \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_synth2_sft_rl \
  synthetic2_sft_verified
```

## 3. Sequential Training Experiments

For sequential experiments:

- `schedule=sequential`

Example chain:

- `synthetic2_sft_verified -> triviaqa -> gsm8k`

### Sequential SFT

```bash
export STAGE_VARIANT_SYNTHETIC2_SFT_VERIFIED=sf
export STAGE_VARIANT_TRIVIAQA=gt
export STAGE_VARIANT_GSM8K=gt

SFT_MAX_STEPS=300 \
RUN_BENCHMARK_EVALS=1 \
RUN_TASK_EVALS=1 \
TASK_PASS_K=512 \
EVAL_SUITE=tasks_md_core \
WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  sft sequential \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_seq_synth2_triviaqa_gsm8k \
  synthetic2_sft_verified triviaqa gsm8k
```

### Sequential RL

Only practical when every stage maps to a real PRIME env.

```bash
RL_MAX_STEPS=300 \
PRIME_BATCH_SIZE=1 \
PRIME_SEQ_LEN=4096 \
PRIME_MAX_TOKENS=512 \
PRIME_ROLLOUTS_PER_PROMPT=1 \
PRIME_ENFORCE_EAGER=1 \
bash scripts/run_training_plan.sh \
  rl sequential \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_seq_rl \
  gsm8k triviaqa
```

### Sequential SFT + RL

```bash
export STAGE_VARIANT_SYNTHETIC2_SFT_VERIFIED=sf
export STAGE_VARIANT_TRIVIAQA=gt
export STAGE_VARIANT_GSM8K=gt

SFT_MAX_STEPS=300 \
RL_MAX_STEPS=300 \
RUN_BENCHMARK_EVALS=1 \
RUN_TASK_EVALS=1 \
TASK_PASS_K=512 \
EVAL_SUITE=tasks_md_core \
PRIME_BATCH_SIZE=1 \
PRIME_SEQ_LEN=4096 \
PRIME_MAX_TOKENS=512 \
PRIME_ROLLOUTS_PER_PROMPT=1 \
PRIME_ENFORCE_EAGER=1 \
WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  sft_rl sequential \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_seq_sft_rl \
  synthetic2_sft_verified triviaqa gsm8k
```

## 4. Mixed-Dataset Training

### Balanced mixed dataset from two sources

Helper:

- `scripts/build_mixed_dataset.py`

Current helper choices are built-in pairwise datasets:

- `gsm8k`
- `math`
- `triviaqa`
- `polaris_math`
- `openr1_math`

Example:

```bash
python scripts/build_mixed_dataset.py \
  --dataset-a gsm8k \
  --dataset-b triviaqa \
  --data-root ~/scratch/forgetting-llms/data \
  --output-dir ~/scratch/forgetting-llms/data/gsm8k_triviaqa_mixed
```

### Weighted mixed dataset

`build_mixed_dataset.py` now supports explicit weights.

Example 30/70 mix:

```bash
python scripts/build_mixed_dataset.py \
  --dataset-weight math=0.3 \
  --dataset-weight gsm8k=0.7 \
  --data-root ~/scratch/forgetting-llms/data \
  --output-dir ~/scratch/forgetting-llms/data/math30_gsm8k70_mixed
```

This downsamples both source splits to the largest mixture that satisfies:

- `math` = 30%
- `gsm8k` = 70%

The exact selected counts per split are written into:

- `manifest.json`

Then train on it by overriding the data dir:

```bash
DATA_VARIANT=sf \
DATA_DIR=~/scratch/forgetting-llms/data/gsm8k_triviaqa_mixed \
SFT_MAX_STEPS=300 \
WANDB_MODE=disabled \
bash scripts/run_sft.sh \
  gsm8k_triviaqa_mixed \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_mixed_gsm8k_triviaqa \
  sf
```

### IID control shards

Helper:

- `scripts/build_iid_control_dataset.py`

Example:

```bash
python scripts/build_iid_control_dataset.py \
  --dataset-a gsm8k \
  --dataset-b triviaqa \
  --data-root ~/scratch/forgetting-llms/data \
  --output-root ~/scratch/forgetting-llms/data/gsm8k_triviaqa_iid
```

## 5. LoRA vs Full Fine-Tune

The SFT and self-distill paths use:

- `LORA_RANK=0` for full fine-tune
- `LORA_RANK>0` for LoRA

Examples:

Full SFT:

```bash
LORA_RANK=0 \
MAX_STEPS=300 \
bash scripts/run_sft.sh \
  synthetic2_sft_verified \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_synth2_full \
  sf
```

LoRA SFT:

```bash
LORA_RANK=64 \
MAX_STEPS=300 \
bash scripts/run_sft.sh \
  synthetic2_sft_verified \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_synth2_lora \
  sf
```

Important current boundary:

- the PRIME-RL path does not expose a separate LoRA toggle in `run_training_plan.sh`
- RL continues from the checkpoint you give it
- so the practical comparison is:
  - LoRA SFT checkpoint -> RL
  - full SFT checkpoint -> RL

### LoRA + RL

The repo-level way to do "LoRA RL" is:

1. train an SFT checkpoint with LoRA
2. use that LoRA-adapted checkpoint as the input model for RL

That means the comparison family is:

- full SFT -> RL
- LoRA SFT -> RL

Example LoRA SFT stage:

```bash
LORA_RANK=64 \
MAX_STEPS=300 \
WANDB_MODE=disabled \
bash scripts/run_sft.sh \
  synthetic2_sft_verified \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_synth2_lora \
  sf
```

Then continue with RL from that checkpoint:

```bash
PRIME_BATCH_SIZE=1 \
PRIME_SEQ_LEN=4096 \
PRIME_MAX_TOKENS=512 \
PRIME_ROLLOUTS_PER_PROMPT=1 \
PRIME_ENFORCE_EAGER=1 \
PRIME_WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  rl individual \
  "$HOME/scratch/forgetting-llms/checkpoints/qwen17_synth2_lora" \
  qwen17_synth2_lora_rl \
  gsm8k
```

For sequential LoRA SFT -> RL, the same pattern applies:

- first run sequential LoRA SFT
- then use the resulting stage checkpoint as the RL base model

There is no separate "RL LoRA switch" in the PRIME launcher today. The LoRA part is inherited from the checkpoint you continue from.

## 6. Self-Distillation

Entry point:

- `scripts/run_self_distill.sh`

Current design:

- student and teacher can be the same base checkpoint on different GPUs
- teacher gets privileged information
- student is trained with:
  - CE
  - forward KL
  - reverse KL
  - interpolation between them

Important current behavior:

- default trace source is generation, not reuse of parquet SFT traces
- teacher can be conditioned on the answer / privileged info
- teacher and student can run on different GPUs in parallel
- forward KL, reverse KL, and interpolated KL are supported
- KL is chunked to reduce OOM risk

Example:

```bash
QUESTION_FIELD=extra_info.question \
GROUND_TRUTH_FIELD=extra_info.ground_truth \
TRACE_SOURCE=generate \
PRIVILEGED_SOURCE=auto \
STUDENT_DEVICE=cuda:0 \
TEACHER_DEVICE=cuda:1 \
TEACHER_SYNC_MODE=step \
PARALLEL_TRACE_GENERATION=1 \
KL_MODE=reverse \
LORA_RANK=0 \
WANDB_MODE=disabled \
bash scripts/run_self_distill.sh \
  ~/scratch/forgetting-llms/data/synthetic2_sft_verified_sf_sft/train.parquet \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_self_distill_synth2
```

### Self-Distillation as an Experiment Family

The intended family here is:

- no-distillation baseline
- self-distillation with forward KL
- self-distillation with reverse KL
- self-distillation with interpolated KL
- each of the above with or without LoRA on the student

Examples:

Forward KL:

```bash
KL_MODE=forward \
LORA_RANK=0 \
WANDB_MODE=disabled \
bash scripts/run_self_distill.sh \
  ~/scratch/forgetting-llms/data/synthetic2_sft_verified_sf_sft/train.parquet \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_self_distill_forward
```

Reverse KL:

```bash
KL_MODE=reverse \
LORA_RANK=0 \
WANDB_MODE=disabled \
bash scripts/run_self_distill.sh \
  ~/scratch/forgetting-llms/data/synthetic2_sft_verified_sf_sft/train.parquet \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_self_distill_reverse
```

Interpolated KL:

```bash
KL_MODE=interpolate \
KL_INTERP_ALPHA=0.5 \
LORA_RANK=0 \
WANDB_MODE=disabled \
bash scripts/run_self_distill.sh \
  ~/scratch/forgetting-llms/data/synthetic2_sft_verified_sf_sft/train.parquet \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_self_distill_interp
```

LoRA self-distillation:

```bash
KL_MODE=reverse \
LORA_RANK=64 \
WANDB_MODE=disabled \
bash scripts/run_self_distill.sh \
  ~/scratch/forgetting-llms/data/synthetic2_sft_verified_sf_sft/train.parquet \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_self_distill_reverse_lora
```

Important current boundary:

- self-distillation is implemented as a standalone trainer
- it is not yet wired as a native mode inside `run_training_plan.sh`
- so sequential self-distillation currently means chaining checkpoints manually rather than using the unified stage launcher

## 7. Async Eval on a Separate GPU

This is the intended pattern for long runs.

Training GPUs:

- `TRAIN_CUDA_VISIBLE_DEVICES=0,1,2`

Eval GPU:

- `ASYNC_EVAL_GPU=3`

The launcher will:

- train on GPUs `0,1,2`
- watch checkpoints
- evaluate checkpoints on GPU `3`

Example:

```bash
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
ASYNC_EVAL_GPU=3 \
AUTO_START_EVAL_SERVER=1 \
SFT_MAX_STEPS=300 \
SAVE_STEPS=300 \
EVAL_STEPS=300 \
RUN_BENCHMARK_EVALS=1 \
RUN_TASK_EVALS=1 \
TASK_PASS_K=512 \
EVAL_SUITE=tasks_md_core \
EVAL_CONTINUE_ON_ERROR=0 \
WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  sft sequential \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_seq_synth2_triviaqa_gsm8k \
  synthetic2_sft_verified triviaqa gsm8k
```

What happens:

- stage training runs
- `watch_checkpoints_eval.sh` evaluates new SFT checkpoints
- `watch_prime_run_eval.sh` evaluates new PRIME checkpoints
- stage-end eval runs one last time after training completes

## 8. Benchmark Suites

### Recommended default: `tasks_md_core`

This is the stable core suite intended for uninterrupted training runs.

Current members:

- `supergpqa`
- `aime`
- `livecodebench_v6`
- `simpleqa`
- `humaneval_plus`
- `mbpp_plus`
- `litqa2`

Use:

```bash
EVAL_SUITE=tasks_md_core
```

### Broader suite: `tasks_md`

Current members:

- `gpqa`
- `supergpqa`
- `rg_mix`
- `aime`
- `livecodebench_v6`
- `simpleqa`
- `humaneval_plus`
- `mbpp_plus`
- `litqa2`
- `bfcl`
- `safety`

Use this only when external dependencies are installed and verified.

## 9. Benchmark-Specific External Requirements

### GPQA

Requires:

- HF token
- approved access to gated `Idavidrein/gpqa`

### RG-mix

Requires one of:

- installed `rg-mix-env` package
- valid `RG_MIX_ROOT`

Recommended run path uses the served local endpoint.

### BFCL

Requires:

- `BFCL_ROOT`
- working `bfcl` CLI
- dedicated BFCL runner venv via:

```bash
bash scripts/setup_runner_venvs.sh bfcl
```

Current BFCL-specific gotcha:

- BFCL wants a canonical supported model name for its internal config lookup
- the local checkpoint directory basename is often rejected
- a BFCL model-name override is needed in the evaluation layer

### Safety

Requires:

- `SAFETY_EVAL_ROOT`
- dedicated safety runner env or working active env

Current safety-specific gotchas:

- `wildguardtest` and `harmbench` may require access to gated `allenai/wildguard`
- the wrapper now skips subtasks that fail specifically on this gated dependency
- native safety-eval vLLM is forced down to lower GPU memory utilization

## 10. Reverse / Backward Evaluation

### Current automatic behavior

`run_training_plan.sh` currently does local task eval after each sequential SFT stage with:

- completed stages first
- current stage last

That is backward-task coverage, but not reverse-order display by default.

### What is automatic now

For sequential SFT:

- after stage 2, local task eval is run on:
  - prior completed stages
  - current stage

Example:

- after `synthetic2 -> triviaqa`
- local task eval chain is:
  - `synthetic2`
  - `triviaqa`

### If strict reverse order is required

This is not the current default in `run_training_plan.sh`.

Current options:

1. patch `run_training_plan.sh` to reverse `COMPLETED_STAGES`
2. run local task evals manually in reverse order after each stage

So the repo supports backward-task evaluation now, but strict reverse-order task eval is still a launcher-policy choice.

## 11. What Is Stable vs What Is Still Fragile

### Stable core path

The intended stable end-to-end path is:

1. generate traces
2. train with `run_training_plan.sh`
3. use `tasks_md_core`
4. evaluate asynchronously on a separate GPU

### More fragile paths

These are wired but still depend on external benchmark repos, gated access, or benchmark-specific package issues:

- `gpqa`
- `rg_mix`
- `bfcl`
- `safety`
- custom PRIME-RL datasets without real env mappings

## 12. Recommended End-to-End Experiment Order

If the goal is to get a full matrix with minimum churn, the recommended order is:

1. generate SFT traces for each training dataset
2. run individual SFT on each dataset
3. run mixed SFT on selected dataset pairs
4. run sequential SFT chains
5. turn on async eval with `tasks_md_core`
6. compare LoRA vs full SFT
7. compare full SFT -> RL vs LoRA SFT -> RL
8. add self-distillation
9. add PRIME-RL on the built-in RL environments first
10. add PRIME-RL continuation from SFT checkpoints
11. expand from `tasks_md_core` to `tasks_md` only after each external benchmark is verified on the cluster

## 12A. Experiment Families Checklist

This is the intended matrix the repo should support operationally.

- Individual full SFT
- Individual LoRA SFT
- Individual RL
- Individual full SFT -> RL
- Individual LoRA SFT -> RL
- Mixed full SFT
- Mixed LoRA SFT
- Sequential full SFT
- Sequential LoRA SFT
- Sequential full SFT -> RL
- Sequential LoRA SFT -> RL
- Individual self-distillation
- Individual LoRA self-distillation
- Sequential self-distillation by manual checkpoint chaining
- Full benchmark eval on stable working benchmarks
- Backward-task eval after each sequential stage

## 13. One Good Default Command

This is the default command for the current main experiment shape:

```bash
module load python/3.10
source ~/scratch/forgetting-llms/.venv/bin/activate
source scripts/load_hf_auth.sh
source ~/scratch/forgetting-llms/benchmark_env.sh

export QWEN17_MODEL="$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B"
export STAGE_VARIANT_SYNTHETIC2_SFT_VERIFIED=sf
export STAGE_VARIANT_TRIVIAQA=gt
export STAGE_VARIANT_GSM8K=gt
export VLLM_USE_STANDALONE_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_SERVER_ENFORCE_EAGER=1
export VLLM_USE_FLASHINFER_SAMPLER=0

TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
ASYNC_EVAL_GPU=3 \
AUTO_START_EVAL_SERVER=1 \
SFT_MAX_STEPS=300 \
SAVE_STEPS=300 \
EVAL_STEPS=300 \
RUN_BENCHMARK_EVALS=1 \
RUN_TASK_EVALS=1 \
TASK_PASS_K=512 \
EVAL_SUITE=tasks_md_core \
EVAL_CONTINUE_ON_ERROR=0 \
WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  sft sequential \
  "$QWEN17_MODEL" \
  qwen17_seq_synth2_triviaqa_gsm8k \
  synthetic2_sft_verified triviaqa gsm8k
```

This is the current recommended baseline for an end-to-end run that is realistic on the current stack.
