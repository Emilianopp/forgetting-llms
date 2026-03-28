# Current Success

This file records the main workflows implemented in this chat, what they do,
and how to run them on Mila.

It is a practical runbook, not a full design document.

## Baseline Mila Setup

Repo:

```bash
cd ~/forget
```

Scratch-local env:

```bash
module load python/3.10
source ~/scratch/forgetting-llms/.venv/bin/activate
```

Optional auth and benchmark env:

```bash
source scripts/load_hf_auth.sh
source ~/scratch/forgetting-llms/benchmark_env.sh
```

Storage convention:

- repo and scripts: `~/forget`
- env: `~/scratch/forgetting-llms/.venv`
- data: `~/scratch/forgetting-llms/data`
- checkpoints: `~/scratch/forgetting-llms/checkpoints`
- benchmark outputs: `~/scratch/forgetting-llms/benchmark_plan_evals`

## Successful Workflow Shapes

The stable workflow shapes implemented and exercised in this repo are:

- individual native SFT
- sequential native SFT with async eval on another GPU
- LoRA SFT
- self-distillation with optional LoRA on the student
- PRIME-RL smoke-test style runs on built-in environments
- SFT then RL orchestration through `scripts/run_training_plan.sh`
- checkpoint mirroring during training to Google Drive via `rclone`
- checkpoint mirroring during training to Hugging Face Hub via `hf:` mirror roots

The main unresolved boundary is still mixed end-to-end SFT plus RL on built-in
task mixtures: for SFT, the mixed parquet must be built from SFT-formatted
source dirs like `gsm8k_sft` or `math_sft`, not from the raw RL parquet dirs.

## Dataset And Stage Glossary

This section is meant as a quick operational map for another agent. It tells
you what each dataset or stage name means, whether it is a built-in PRIME
environment or a parquet-backed local stage, and what to check on disk.

### Built-in verifier / benchmark-style datasets

- `gsm8k`
  - Grade-school math word problems with exact-answer grading.
  - Supported for native SFT, PRIME-RL, local task eval, and benchmark sweeps.
- `math`
  - Competition-style math reasoning with exact-answer grading.
  - Supported for native SFT, PRIME-RL, local task eval, and benchmark sweeps.
- `triviaqa`
  - Open-domain factoid QA with alias-aware answer matching.
  - Supported for native SFT, PRIME-RL, local task eval, and benchmark sweeps.
- `polaris_math`
  - Math-style exact-answer dataset wired through the same math reward path.
  - Supported for native SFT, PRIME-RL, local task eval, and benchmark sweeps.
- `openr1_math`
  - Math-style exact-answer dataset routed through the math reward path.
  - Supported for native SFT, PRIME-RL, local task eval, and benchmark sweeps.

### Parquet-backed prompt-style or imported core datasets

- `synthetic2_sft_verified`
  - Verified prompt/answer SFT parquet in repo-owned format.
  - SFT runs directly on `~/scratch/forgetting-llms/data/synthetic2_sft_verified_sf_sft`.
  - RL uses dataset-dir GRPO after auto-conversion into RL parquet if needed.
- `dolci_think_sft_7b`
  - Prompt-style SFT parquet intended for DOLCI 7B thinking traces.
  - Uses the dataset-dir RL backend after SFT-to-RL conversion.
- `dolci_think_sft_32b`
  - Prompt-style SFT parquet intended for DOLCI 32B thinking traces.
  - Uses the dataset-dir RL backend after SFT-to-RL conversion.
- `tau2bench`
  - Parquet-backed custom/core training stage for Tau2Bench-style task trajectories.
  - Treated as a local dataset-dir SFT/RL stage rather than a built-in PRIME environment.
  - Not the native Sierra `tau2-bench` environment.
  - Expects `~/scratch/forgetting-llms/data/tau2bench_sft` or `tau2bench_rl`.
  - The launcher can now auto-convert between `tau2bench_sft` and `tau2bench_rl`
    when one side exists and the other is missing.
- `olmo_rl_zero_math`
  - Imported OLMo RL-Zero math parquet routed through the math reward path.
  - Treated as an RL-first dataset-dir stage and can auto-convert into SFT parquet.
- `dolci_rl_zero_math`
  - DOLCI alias stage name for the imported RL-Zero math subset.
  - Resolves onto the same default paths and reward routing as `olmo_rl_zero_math`.
- `olmo_rl_zero_code`
  - Imported RL-Zero code subset with first-class stage names and default paths.
  - Default RL dir is `~/scratch/forgetting-llms/data/olmo_rl_code`.
  - Default SFT dir is `~/scratch/forgetting-llms/data/olmo_rl_zero_code_sft`.
- `dolci_rl_zero_code`
  - DOLCI alias stage name for the imported RL-Zero code subset.
  - Resolves onto the same default paths as `olmo_rl_zero_code`.
- `olmo_rl_zero_if`
  - Imported RL-Zero instruction-following subset with first-class stage names and default paths.
  - Default RL dir is `~/scratch/forgetting-llms/data/olmo_rl_if`.
  - Default SFT dir is `~/scratch/forgetting-llms/data/olmo_rl_zero_if_sft`.
- `dolci_rl_zero_if`
  - DOLCI alias stage name for the imported RL-Zero instruction-following subset.
  - Resolves onto the same default paths as `olmo_rl_zero_if`.
- `olmo_rl_zero_general`
  - Imported RL-Zero general-chat subset with first-class stage names and default paths.
  - Default RL dir is `~/scratch/forgetting-llms/data/olmo_rl_general`.
  - Default SFT dir is `~/scratch/forgetting-llms/data/olmo_rl_zero_general_sft`.
- `dolci_rl_zero_general`
  - DOLCI alias stage name for the imported RL-Zero general-chat subset.
  - Resolves onto the same default paths as `olmo_rl_zero_general`.
- `olmo_rl_zero_mix`
  - Imported RL-Zero mixed subset spanning multiple abilities.
  - Default RL dir is `~/scratch/forgetting-llms/data/olmo_rl_mix`.
  - Default SFT dir is `~/scratch/forgetting-llms/data/olmo_rl_zero_mix_sft`.
- `dolci_rl_zero_mix`
  - DOLCI alias stage name for the imported RL-Zero mixed subset.
  - Resolves onto the same default paths as `olmo_rl_zero_mix`.
- `olmo_instruct_rl`
  - Imported instruct-RL subset with first-class stage names and default paths.
  - Default RL dir is `~/scratch/forgetting-llms/data/olmo_rl_instruct`.
  - Default SFT dir is `~/scratch/forgetting-llms/data/olmo_instruct_rl_sft`.
- `dolci_instruct_rl`
  - DOLCI alias stage name for the imported instruct-RL subset.
  - Resolves onto the same default paths as `olmo_instruct_rl`.

Important reward caveat for imported OLMo / DOLCI RL subsets:

- only the math subset is currently wired to a full reward path in this repo
- the non-math subsets now have stage names and default paths, but the imported
  metadata still marks their reward support as missing
- dataset-dir RL on those non-math subsets should be treated as a wiring /
  data-path validation path unless you add the matching reward implementation
- `scripts/run_grpo_dataset_dir_local.sh` will refuse `reward_support=missing`
  unless you explicitly set `ALLOW_UNSUPPORTED_DATASET_RL=1`

### Built-in mixed or IID PRIME environments

- `mix_gsm8k_math`
  - Built-in PRIME mixed verifier environment spanning `gsm8k` and `math`.
  - Requires the matching PRIME config trio such as
    `mix_gsm8k_math.trainer.toml`, `orchestrator.toml`, and `inference.toml`.
- `iid_gsm8k_math`
  - Built-in IID control environment for the `gsm8k` / `math` pair.
  - Uses PRIME, not dataset-dir parquet RL.
- `iid_gsm8k_triviaqa`
  - Built-in IID control environment for the `gsm8k` / `triviaqa` pair.
  - Uses PRIME, not dataset-dir parquet RL.
- `iid_math_triviaqa`
  - Built-in IID control environment for the `math` / `triviaqa` pair.
  - Uses PRIME, not dataset-dir parquet RL.

### User-built mixed parquet datasets

- names like `math30_gsm8k70_mixed` or `gsm8k_math_mixed_sft`
  - These are not special built-in task identifiers.
  - They are output directories created by `scripts/build_mixed_dataset.py`.
  - SFT can train on them directly when you point `DATA_DIR` or
    `STAGE_DATA_DIR_<STAGE>` at the generated directory.
  - If you want stage-end local task eval on a custom mixed stage, set
    `STAGE_TASK_EVAL_DATASETS_<STAGE>` explicitly and, if needed,
    `TASK_EVAL_DATASET_PATH_<DATASET>` / `TASK_EVAL_DATA_SOURCE_<DATASET>`.

### Fast backend check

When another agent is validating the repo, the main backend split is:

- built-in verifier datasets and built-in mixed/IID names
  - use PRIME-RL
- parquet-backed prompt-style/core datasets
  - use dataset-dir GRPO

The authoritative switch is implemented in `scripts/run_training_plan.sh`.

## Training Datasets Vs Eval Datasets

This distinction is important. In this repo, training data selection and eval
data selection are separate control surfaces.

Training datasets:

- decide what the model trains on during SFT or RL
- usually live under `~/scratch/forgetting-llms/data/...`
- are chosen by the stage name plus optional explicit overrides such as:
  - `DATA_DIR` for direct `run_sft.sh`
  - `STAGE_DATA_DIR_<STAGE>` for `run_training_plan.sh`
  - `STAGE_RL_DATA_DIR_<STAGE>` for dataset-dir RL stages
  - PRIME environment names like `gsm8k`, `math`, or `mix_gsm8k_math` for
    built-in verifier RL stages

Eval datasets:

- decide what gets measured after or during training
- do not have to match the training dataset exactly
- split into two distinct eval families:
  - local task eval
  - benchmark-plan eval

Local task eval:

- controlled by `RUN_TASK_EVALS=1`
- uses repo-local dataset labels or explicit parquet paths
- for `run_training_plan.sh`, the dataset list is chosen by:
  - inferred stage names
  - or `STAGE_TASK_EVAL_DATASETS_<STAGE>` if you set it explicitly
- explicit parquet-backed local evals use:
  - `TASK_EVAL_DATASET_PATH_<DATASET>`
  - `TASK_EVAL_DATA_SOURCE_<DATASET>`

Benchmark-plan eval:

- controlled by `RUN_BENCHMARK_EVALS=1`
- uses `EVAL_SUITE`, typically `tasks_md` or `tasks_md_core`
- does not read the stage parquet directly
- instead it evaluates checkpoints against benchmark runners such as LightEval,
  EvalPlus, LAB-Bench, BFCL, SuperGPQA, RG-mix, and safety-eval
- those runners may depend on external repos, external APIs, or gated Hugging
  Face assets

Operational rule:

- `STAGE_DATA_DIR_<STAGE>` changes training data
- `STAGE_TASK_EVAL_DATASETS_<STAGE>` changes local task eval targets
- `EVAL_SUITE` changes benchmark eval targets

Example:

```bash
STAGE_DATA_DIR_MIX_GSM8K_MATH="$HOME/scratch/forgetting-llms/data/gsm8k_math_mixed_sft"
STAGE_TASK_EVAL_DATASETS_MIX_GSM8K_MATH="gsm8k math"
RUN_TASK_EVALS=1
RUN_BENCHMARK_EVALS=1
EVAL_SUITE=tasks_md_core
```

This means:

- train the mixed stage on the combined `gsm8k_math_mixed_sft` parquet
- run local task eval separately on `gsm8k` and `math`
- run the checkpoint through the `tasks_md_core` benchmark suite

Another example:

```bash
STAGE_DATA_DIR_CUSTOM_STAGE=/scratch/.../my_training_mix
STAGE_TASK_EVAL_DATASETS_CUSTOM_STAGE="heldout_mix"
TASK_EVAL_DATASET_PATH_HELDOUT_MIX=/scratch/.../heldout_eval/test.parquet
TASK_EVAL_DATA_SOURCE_HELDOUT_MIX=gsm8k
```

This means:

- train on `my_training_mix`
- evaluate locally on a different held-out parquet
- route local reward scoring through the `gsm8k` scorer

## Checkpoint Mirroring To Google Drive Or Hugging Face Hub

Training always writes primary outputs to scratch first. A background watcher
then mirrors the active run directory into a secondary destination.

Supported entry points:

- `scripts/run_sft.sh`
- `scripts/run_training_plan.sh`
- `scripts/run_self_distill.sh`

Supported destination forms:

- normal local path
- `gdrive:...` `rclone` remote
- `hf:model:<user_or_org>/<repo>`
- `hf:dataset:<user_or_org>/<repo>`

Behavior:

- training still writes to scratch first
- mirroring is poll-based, not event-triggered
- mirrored paths preserve scratch-relative layout like `checkpoints/...` or
  `prime_runs/...`
- the final sync runs when training exits
- uploads are best-effort; failures are logged, but training continues

### Google Drive Setup On Mila

If Drive is exposed through an `rclone` remote on the cluster, use
`CHECKPOINT_MIRROR_ROOT=gdrive:...`.

1. Install `rclone` into scratch if needed.

```bash
mkdir -p ~/scratch/forgetting-llms/bin
cd ~/scratch/forgetting-llms/tmp

ARCH=$(uname -m)
case "$ARCH" in
  x86_64) RCLONE_ARCH=linux-amd64 ;;
  aarch64|arm64) RCLONE_ARCH=linux-arm64 ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

curl -LO "https://downloads.rclone.org/rclone-current-${RCLONE_ARCH}.zip"
python -m zipfile -e "rclone-current-${RCLONE_ARCH}.zip" .
cp rclone-*/rclone ~/scratch/forgetting-llms/bin/
chmod +x ~/scratch/forgetting-llms/bin/rclone
export PATH="$HOME/scratch/forgetting-llms/bin:$PATH"
```

2. Configure the `gdrive` remote.

```bash
export PATH="$HOME/scratch/forgetting-llms/bin:$PATH"
rclone config
```

Headless flow:

- on Mila, answer `n` when asked to open a browser automatically
- `rclone` will print an `rclone authorize "drive" "..."` command
- run that command on a machine with a browser and the same `rclone` version
- paste the resulting token JSON back into Mila

3. Verify the remote.

```bash
export PATH="$HOME/scratch/forgetting-llms/bin:$PATH"
rclone lsd gdrive:
rclone mkdir gdrive:forgetting-llms-backups
```

4. Run training with checkpoint mirroring enabled.

```bash
export PATH="$HOME/scratch/forgetting-llms/bin:$PATH"
export CHECKPOINT_MIRROR_ROOT="gdrive:forgetting-llms-backups"
export CHECKPOINT_MIRROR_SOURCE_BASE="$HOME/scratch/forgetting-llms"
export CHECKPOINT_MIRROR_POLL_SECS=60
export CHECKPOINT_MIRROR_PRUNE_DEST=0
```

### Hugging Face Hub Setup

If you want checkpoints mirrored into a Hub repo instead of Drive, use an
`hf:` mirror root.

1. Create a Hub write token in your Hugging Face account.

2. Make the token available to repo launchers.

Recommended scratch-local auth file:

```bash
cat > ~/scratch/forgetting-llms/hf_auth.sh <<'EOF'
export HF_TOKEN=hf_your_write_token_here
EOF
chmod 600 ~/scratch/forgetting-llms/hf_auth.sh
```

3. Load auth.

```bash
cd ~/forget
source scripts/load_hf_auth.sh
```

This writes the token to `HF_TOKEN_PATH`, which defaults to
`~/scratch/huggingface/token`.

4. Choose the mirror destination.

Model repo:

```bash
export CHECKPOINT_MIRROR_ROOT="hf:model:<user_or_org>/<repo>"
```

Dataset repo:

```bash
export CHECKPOINT_MIRROR_ROOT="hf:dataset:<user_or_org>/<repo>"
```

Optional repo visibility on first upload:

```bash
export MIRROR_HF_PRIVATE=1
# or:
export MIRROR_HF_PUBLIC=1
```

5. Run training with Hub mirroring enabled.

```bash
export CHECKPOINT_MIRROR_ROOT="hf:model:<user_or_org>/forgetting-llms-checkpoints"
export CHECKPOINT_MIRROR_SOURCE_BASE="$HOME/scratch/forgetting-llms"
export CHECKPOINT_MIRROR_POLL_SECS=300
export CHECKPOINT_MIRROR_PRUNE_DEST=0
```

The watcher creates the repo if needed and uploads the current run directory
under the scratch-relative path inside the repo.

### Hugging Face Gated Access

There are two different Hugging Face auth cases in this repo:

- write access for mirroring checkpoints into your own Hub repo
- read access for gated benchmark datasets or models

The same `HF_TOKEN` can handle both if your account has the required approvals.

Recommended auth flow on Mila:

1. Request access on the relevant Hugging Face page while logged into the same
   account that created your token.
2. Put that token in `~/scratch/forgetting-llms/hf_auth.sh`.
3. Load it with `source scripts/load_hf_auth.sh`.
4. Verify the token file exists at `~/scratch/huggingface/token` or your
   configured `HF_TOKEN_PATH`.

Minimal scratch-local auth file:

```bash
cat > ~/scratch/forgetting-llms/hf_auth.sh <<'EOF'
export HF_TOKEN=hf_your_token_here
EOF
chmod 600 ~/scratch/forgetting-llms/hf_auth.sh
source scripts/load_hf_auth.sh
```

Known gated Hugging Face dependencies in repo-supported paths:

- dataset `Idavidrein/gpqa`
  - used by the `gpqa` LightEval benchmark in `tasks_md`
  - `run_eval.py` will complain if `HF_TOKEN` is missing or the account has not
    been approved
- model `allenai/wildguard`
  - used by safety-eval's `wildguardtest` subtask
  - the wrapper now skips that subtask instead of crashing if access is missing
- model `meta-llama/Llama-3.1-70B-Instruct`
  - used as a default teacher in some helper scripts such as
    `scripts/run_all_tasks_interactive.sh`
  - if you use those helpers unchanged, your account must also have accepted the
    Meta gating terms for that model

Practical rule:

- checkpoint mirroring only needs write access to your destination repo
- `tasks_md` may additionally need read access to gated benchmark assets
- benchmark auth and mirror auth can share the same token

### Example Mirror Configs

Google Drive via `rclone`:

```bash
export CHECKPOINT_MIRROR_ROOT="gdrive:forgetting-llms-backups"
export CHECKPOINT_MIRROR_SOURCE_BASE="$HOME/scratch/forgetting-llms"
export CHECKPOINT_MIRROR_POLL_SECS=60
export CHECKPOINT_MIRROR_PRUNE_DEST=0
```

Hugging Face Hub:

```bash
source scripts/load_hf_auth.sh
export CHECKPOINT_MIRROR_ROOT="hf:model:<user_or_org>/forgetting-llms-checkpoints"
export CHECKPOINT_MIRROR_SOURCE_BASE="$HOME/scratch/forgetting-llms"
export CHECKPOINT_MIRROR_POLL_SECS=300
export MIRROR_HF_PRIVATE=1
```

## 1. SFT Trace Generation

### Verifiable datasets

Supported through:

- `scripts/generate_trajectories.sh`
- `src/data/generate_teacher_solutions.py`

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

Typical supported datasets:

- `gsm8k`
- `math`
- `triviaqa`
- `polaris_math`
- `openr1_math`

### Prompt-style datasets

Supported in the wrapper:

- `synthetic2_sft_verified`
- `dolci_think_sft_7b`
- `dolci_think_sft_32b`
- `tau2bench` when you provide `~/scratch/forgetting-llms/data/tau2bench_sft`
- `olmo_rl_zero_math` / `dolci_rl_zero_math`
- `olmo_rl_zero_code` / `dolci_rl_zero_code`
- `olmo_rl_zero_if` / `dolci_rl_zero_if`
- `olmo_rl_zero_general` / `dolci_rl_zero_general`
- `olmo_rl_zero_mix` / `dolci_rl_zero_mix`
- `olmo_instruct_rl` / `dolci_instruct_rl`
  via RL-to-SFT conversion from imported OLMo / DOLCI RL parquet

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

## 2. Individual SFT

Entry points:

- `scripts/run_sft.sh`
- `scripts/run_training_plan.sh`

Direct SFT:

```bash
bash scripts/run_sft.sh \
  synthetic2_sft_verified \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_synth2_sft \
  sf
```

Unified launcher:

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

## 3. Sequential SFT With Async Eval On Another GPU

This is the main end-to-end experiment shape implemented in this chat.

Example:

```bash
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
EVAL_SUITE=tasks_md \
EVAL_CONTINUE_ON_ERROR=0 \
EVAL_EXTRA_ARGS="--lighteval-endpoint-max-concurrent-requests 1" \
WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  sft sequential \
  "$QWEN17_MODEL" \
  qwen17_seq_olmo_synth2_triviaqa_gsm8k \
  synthetic2_sft_verified triviaqa gsm8k
```

What this does:

- trains on stage 1
- checkpoints every stage boundary
- evaluates on GPU `3`
- continues training on later stages
- keeps benchmark outputs in scratch

## 4. Weighted Mixed Datasets

Implemented through:

- `scripts/build_mixed_dataset.py`

Weighted example:

```bash
python scripts/build_mixed_dataset.py \
  --dataset-weight math=0.3 \
  --dataset-weight gsm8k=0.7 \
  --data-root ~/scratch/forgetting-llms/data \
  --output-dir ~/scratch/forgetting-llms/data/math30_gsm8k70_mixed
```

Then train on it:

```bash
DATA_DIR=~/scratch/forgetting-llms/data/math30_gsm8k70_mixed \
MAX_STEPS=300 \
bash scripts/run_sft.sh \
  math30_gsm8k70_mixed \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_math30_gsm8k70 \
  sf
```

The builder writes:

- `train.parquet`
- `test.parquet`
- `train_selection.parquet`
- `test_selection.parquet`

The `*_selection.parquet` files record which source dataset and source row were
sampled into the final mixture. They are useful when you want to debug a bad
mix or reproduce the exact sampled composition.

### Important Mixed-SFT Detail

Native SFT expects rows with `extra_info.question` and `extra_info.answer`.
That means built-in mixed SFT datasets must be built from SFT-formatted source
dirs, not the raw RL parquet dirs.

Prepare built-in SFT sources:

```bash
python scripts/preprocess_data.py \
  --dataset gsm8k \
  --format sft \
  --output_dir ~/scratch/forgetting-llms/data/gsm8k_sft

python scripts/preprocess_data.py \
  --dataset math \
  --format sft \
  --output_dir ~/scratch/forgetting-llms/data/math_sft
```

Build the mixed SFT parquet from those sources:

```bash
python scripts/build_mixed_dataset.py \
  --dataset-a gsm8k \
  --dataset-b math \
  --data-root ~/scratch/forgetting-llms/data \
  --source-suffix _sft \
  --output-dir ~/scratch/forgetting-llms/data/gsm8k_math_mixed_sft
```

Then point SFT or `sft_rl` at the explicit mixed dir:

```bash
STAGE_DATA_DIR_MIX_GSM8K_MATH="$HOME/scratch/forgetting-llms/data/gsm8k_math_mixed_sft"
```

For mixed RL-style training there are two repo-supported shapes:

- built-in PRIME mixed or IID environments such as `mix_gsm8k_math` or
  `iid_gsm8k_math`
- parquet-backed dataset-dir RL for custom/core datasets

Built-in PRIME mixed example:

```bash
STAGE_ENV_MIX_GSM8K_MATH=mix_gsm8k_math \
STAGE_TRAINER_CONFIG_MIX_GSM8K_MATH="$HOME/scratch/forgetting-llms/prime-configs/mix_gsm8k_math.trainer.toml" \
STAGE_ORCHESTRATOR_CONFIG_MIX_GSM8K_MATH="$HOME/scratch/forgetting-llms/prime-configs/mix_gsm8k_math.orchestrator.toml" \
STAGE_INFERENCE_CONFIG_MIX_GSM8K_MATH="$HOME/scratch/forgetting-llms/prime-configs/mix_gsm8k_math.inference.toml" \
bash scripts/run_training_plan.sh \
  rl individual \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_mix_gsm8k_math_rl \
  mix_gsm8k_math
```

### Mixed Eval

For mixed stages, training data selection and eval data selection are separate
concerns.

Training:

- SFT or `sft_rl` uses `STAGE_DATA_DIR_<STAGE>` to point at the mixed parquet
- PRIME mixed RL uses the environment name plus PRIME config trio

Local task eval:

- use `STAGE_TASK_EVAL_DATASETS_<STAGE>` to decide which datasets should be
  evaluated after the stage
- for mixed stages, this is usually the original component datasets

Example:

```bash
STAGE_TASK_EVAL_DATASETS_MIX_GSM8K_MATH="gsm8k math"
```

That means the mixed stage trains on one combined dataset but local task eval
still runs separate `gsm8k` and `math` eval jobs. If you need a custom parquet
for eval rather than a built-in dataset label, set:

```bash
TASK_EVAL_DATASET_PATH_<DATASET>=/path/to/test.parquet
TASK_EVAL_DATA_SOURCE_<DATASET>=gsm8k
```

Benchmark-plan eval:

- benchmark sweeps are tied to checkpoints, not to the original mixed builder
- async PRIME checkpoint eval uses `scripts/eval_prime_checkpoint_sweep.py`
- benchmark results land under `~/scratch/forgetting-llms/benchmark_plan_evals`
  regardless of whether the training stage itself was mixed

## 5. LoRA-Capable SFT

The native SFT path supports LoRA through env vars passed into `run_sft.sh`.

Example:

```bash
LORA_RANK=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
bash scripts/run_sft.sh \
  synthetic2_sft_verified \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_synth2_lora \
  sf
```

This is the repo-supported path for LoRA SFT. The broader training-plan wrapper
can then continue from the resulting checkpoint.

## 6. Self-Distillation

Implemented through:

- `scripts/run_self_distill.sh`
- `src/training/self_distill.py`

Supported features:

- student and teacher on different GPUs
- generated student trace
- privileged teacher trace using the answer
- forward KL
- reverse KL
- interpolated KL
- optional LoRA on the student

Example:

```bash
TRACE_SOURCE=generate \
KL_MODE=interpolate \
KL_INTERP_ALPHA=0.5 \
TEACHER_SYNC_MODE=step \
PARALLEL_TRACE_GENERATION=1 \
STUDENT_DEVICE=cuda:0 \
TEACHER_DEVICE=cuda:1 \
LORA_RANK=16 \
bash scripts/run_self_distill.sh \
  ~/scratch/forgetting-llms/data/synthetic2_sft_verified_sf_sft/train.parquet \
  ~/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B \
  ~/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B \
  qwen17_self_distill_synth2
```

## 7. PRIME-RL And SFT+RL

Implemented orchestration exists through:

- `scripts/run_training_plan.sh`
- `scripts/prime_rl_runner.py`
- `scripts/setup_prime_rl.sh`

RL backends now split into two shapes:

- PRIME-RL for built-in verifier environments like `gsm8k`, `math`, `triviaqa`,
  `polaris_math`, `openr1_math`, and the mixed/iid PRIME environments
- dataset-dir GRPO for custom/core parquet datasets such as:
  - `synthetic2_sft_verified`
  - `dolci_think_sft_7b`
  - `dolci_think_sft_32b`
  - `olmo_rl_zero_math` / `dolci_rl_zero_math`
  - `olmo_rl_zero_code` / `dolci_rl_zero_code`
  - `olmo_rl_zero_if` / `dolci_rl_zero_if`
  - `olmo_rl_zero_general` / `dolci_rl_zero_general`
  - `olmo_rl_zero_mix` / `dolci_rl_zero_mix`
  - `olmo_instruct_rl` / `dolci_instruct_rl`
  - `tau2bench`

For the custom/core path, `run_training_plan.sh` now:

- auto-converts SFT parquet to RL parquet for `synthetic2_sft_verified`,
  `dolci_think_sft_*`, and `tau2bench` when the RL parquet dir is missing
- auto-converts `tau2bench_rl` back into `tau2bench_sft` when the SFT parquet
  dir is missing
- auto-converts imported OLMo RL math parquet into SFT parquet for
  `olmo_rl_zero_math`, `olmo_rl_zero_code`, `olmo_rl_zero_if`,
  `olmo_rl_zero_general`, `olmo_rl_zero_mix`, `olmo_instruct_rl`, and their
  `dolci_*` aliases when the SFT dir is missing
- routes RL stages through `scripts/run_grpo_dataset_dir_local.sh` instead of
  requiring a PRIME verifier environment

Important caveat:

- `tau2bench` in this repo is a parquet-backed training stage name. It is not
  the native Sierra `tau2-bench` environment. You still need a prepared
  `tau2bench_sft` or `tau2bench_rl` parquet dataset under scratch.

Built-in RL smoke test shape:

```bash
source ~/scratch/forgetting-llms/prime_rl_env.sh

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

SFT then RL:

```bash
source ~/scratch/forgetting-llms/prime_rl_env.sh

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

Custom/core dataset-dir RL example:

```bash
source ~/scratch/forgetting-llms/prime_rl_env.sh

RL_TOTAL_EPOCHS=1 \
RL_CKPT_INTERVAL=50 \
PRIME_BATCH_SIZE=1 \
PRIME_MAX_TOKENS=512 \
PRIME_ROLLOUTS_PER_PROMPT=1 \
WANDB_MODE=disabled \
PRIME_WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  rl individual \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_tau2bench_rl \
  tau2bench
```

LoRA SFT then RL uses the same launcher, with LoRA applied in the SFT stage and
the RL stage continuing from that checkpoint automatically:

```bash
source ~/scratch/forgetting-llms/prime_rl_env.sh

LORA_RANK=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
DATA_VARIANT=sf \
SFT_MAX_STEPS=300 \
RL_MAX_STEPS=100 \
RL_CKPT_INTERVAL=50 \
PRIME_BATCH_SIZE=1 \
PRIME_SEQ_LEN=4096 \
PRIME_MAX_TOKENS=512 \
PRIME_ROLLOUTS_PER_PROMPT=1 \
PRIME_ENFORCE_EAGER=1 \
PRIME_WANDB_MODE=disabled \
WANDB_MODE=disabled \
bash scripts/run_training_plan.sh \
  sft_rl individual \
  "$HOME/scratch/forgetting-llms/models/Qwen__Qwen3-1.7B" \
  qwen17_lora_synth2_sft_rl \
  synthetic2_sft_verified
```

## 8. Benchmark Eval Entry Points

Main files:

- `src/evaluation/run_eval.py`
- `scripts/run_eval_with_local_server.sh`
- `scripts/watch_checkpoints_eval.sh`
- `scripts/watch_prime_run_eval.sh`
- `scripts/eval_prime_checkpoint_sweep.py`

Stable local-server eval command pattern:

```bash
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_eval_with_local_server.sh \
  ~/scratch/forgetting-llms/checkpoints/<model_or_stage_dir> \
  ~/scratch/forgetting-llms/benchmark_plan_evals/<eval_name> \
  <eval_name>
```

### PRIME Checkpoint Sweep Outputs And Autogenerated Plots

`scripts/eval_prime_checkpoint_sweep.py` is the persistent bookkeeping layer
for PRIME checkpoint eval.

Default output artifacts:

- SQLite registry:
  `~/scratch/forgetting-llms/benchmark_plan_evals/eval_registry.sqlite`
- plots root:
  `~/scratch/forgetting-llms/benchmark_plan_evals/plots`

The SQLite registry stores:

- discovered PRIME runs
- discovered checkpoints or exported models
- local task eval status and metrics
- benchmark suite status per checkpoint
- benchmark primary metrics scraped from eval outputs

Plots are regenerated after each processed checkpoint. The current plot sets are:

- `plots/task_eval/<run_name>.png`
  - two-panel plot with `accuracy@1` and `pass@k` over checkpoint step
- `plots/benchmark_progress/<run_name>.png`
  - count of completed benchmark-plan tasks over checkpoint step
- `plots/benchmark_primary/<run_name>.png`
  - primary benchmark metrics over checkpoint step, one curve per benchmark

These plots are lightweight summaries, not the source of truth. If a plot looks
wrong, inspect the SQLite registry and the underlying per-checkpoint eval output
dirs first.

### WandB Behavior

WandB is used in three different places in this repo, and the mode knobs are
separate:

- native SFT and self-distillation use `WANDB_MODE`
- PRIME training bundles use `PRIME_WANDB_MODE`
- PRIME checkpoint eval sweep uses its own `--wandb-mode` flag inside
  `scripts/eval_prime_checkpoint_sweep.py`

Common modes:

- `disabled`
  - do not initialize WandB
- `offline`
  - write local WandB files without syncing immediately
- `online`
  - log to the remote project directly

Training behavior:

- `run_sft.sh` and native SFT respect `WANDB_MODE`
- PRIME bundles generated by `scripts/prime_rl_runner.py` respect
  `PRIME_WANDB_MODE`
- local task eval subprocesses launched from `run_training_plan.sh` force
  `--wandb-mode disabled`

Checkpoint-sweep WandB behavior:

- `scripts/eval_prime_checkpoint_sweep.py` creates one WandB run per PRIME run
- run names are prefixed as `eval_<run_name>`
- runs are grouped under `prime_checkpoint_eval`
- task eval metrics are logged under keys like
  `task_eval/<dataset>/accuracy_at_1` and `task_eval/<dataset>/pass_at_<k>`
- benchmark metrics are logged under keys like
  `benchmark/<benchmark>/<metric_name>`

If you do not want any WandB traffic during checkpoint sweeps, keep the sweep in
its default disabled mode or do not pass a sweep-specific WandB mode override.

## 9. Safety-Only Eval

This path was explicitly worked on in this chat.

Example:

```bash
export SAFETY_EVAL_ROOT="$HOME/scratch/safety-eval"

CUDA_VISIBLE_DEVICES=3 \
python src/evaluation/run_eval.py \
  --model_path ~/scratch/forgetting-llms/checkpoints/qwen17_seq_olmo_synth2_triviaqa_gsm8k_stage01_synthetic2_sft_verified_sft \
  --suite tasks_md \
  --include-benchmark safety \
  --benchmark-root safety="$SAFETY_EVAL_ROOT" \
  --safety-gpu-memory-utilization 0.7 \
  --output_dir ~/scratch/forgetting-llms/benchmark_plan_evals/safety_only_test \
  --run_name safety_only_test
```

Current behavior:

- lower vLLM GPU utilization for safety eval
- wrapper-based execution
- gated `wildguard`-dependent subtasks can be skipped instead of crashing the whole run

## 10. BFCL

BFCL support was wired through:

- isolated runner envs
- `run_eval.py`
- direct CLI smoke testing

Setup:

```bash
bash scripts/setup_runner_venvs.sh bfcl
source ~/scratch/forgetting-llms/benchmark_env.sh
```

Direct smoke test:

```bash
source ~/scratch/forgetting-llms/.venvs/bfcl/bin/activate
export CUDA_VISIBLE_DEVICES=3
cd ~/scratch/gorilla/berkeley-function-call-leaderboard

bfcl generate \
  --model "Qwen/Qwen3-1.7B" \
  --backend vllm \
  --local-model-path ~/scratch/forgetting-llms/checkpoints/qwen17_seq_olmo_synth2_triviaqa_gsm8k_stage01_synthetic2_sft_verified_sft \
  --skip-server-setup \
  --test-category simple_python \
  --result-dir /tmp/bfcl_smoke_result

bfcl evaluate \
  --model "Qwen/Qwen3-1.7B" \
  --test-category simple_python \
  --result-dir /tmp/bfcl_smoke_result \
  --score-dir /tmp/bfcl_smoke_score
```

The repo-side BFCL path also supports a canonical BFCL model-name override via
`run_eval.py`.

## 11. Safety / BFCL / RG-Mix Setup

External benchmark setup entry points:

- `scripts/setup_tasks_md_benchmarks.sh`
- `scripts/setup_runner_venvs.sh`

Run:

```bash
bash scripts/setup_tasks_md_benchmarks.sh
bash scripts/setup_runner_venvs.sh
```

These install or prepare:

- `safety-eval`
- BFCL
- isolated benchmark venvs
- benchmark env file updates

## 12. Current Benchmark Suites

Broader suite:

- `tasks_md`

Stable reduced suite:

- `tasks_md_core`

Use the reduced suite if external benchmark repos or runner deps are still in
progress.

### Benchmark Glossary

`tasks_md` currently includes these benchmark datasets or benchmark bundles:

- `gpqa`
  - GPQA Diamond, a hard graduate-level science QA benchmark.
  - Requires gated Hugging Face dataset access to `Idavidrein/gpqa`.
- `supergpqa`
  - SuperGPQA, a much larger GPQA-style benchmark run through the project’s
    dedicated `supergpqa` wrapper.
  - This is the benchmark that was missing from earlier doc descriptions.
- `rg_mix`
  - Project-local reasoning-gym mixed benchmark served through an
    OpenAI-compatible endpoint.
- `aime`
  - AIME 2024 math competition problems.
- `livecodebench_v6`
  - LiveCodeBench code-generation benchmark through LightEval.
- `simpleqa`
  - SimpleQA factoid QA benchmark.
- `humaneval_plus`
  - HumanEval+ code-generation benchmark through EvalPlus.
- `mbpp_plus`
  - MBPP+ code-generation benchmark through EvalPlus.
- `litqa2`
  - LitQA2 scientific literature QA benchmark through LAB-Bench.
- `bfcl`
  - Berkeley Function Calling Leaderboard bundle.
- `safety`
  - Composite safety bundle run through safety-eval.
  - It currently expands into these named safety datasets or subtasks:
  - `wildguardtest`: safety classification / refusal benchmark that depends on the gated `allenai/wildguard` model.
  - `harmbench`: harmful-behavior evaluation benchmark.
  - `xstest`: adversarial / unsafe prompt stress-test benchmark.
  - `toxigen:tiny`: small toxicity-generation evaluation slice.

`tasks_md_core` is the lighter subset of `tasks_md`. It currently includes:

- `supergpqa`
  - large GPQA-style benchmark through the dedicated wrapper
- `aime`
  - math competition benchmark
- `livecodebench_v6`
  - code-generation benchmark
- `simpleqa`
  - short-form factual QA benchmark
- `humaneval_plus`
  - code-generation benchmark
- `mbpp_plus`
  - code-generation benchmark
- `litqa2`
  - scientific literature QA benchmark

The older `forgetting` suite in `run_eval.py` currently contains:

- `arc_challenge`
  - ARC Challenge science QA
- `arc_easy`
  - ARC Easy science QA
- `hellaswag`
  - commonsense completion benchmark
- `winogrande`
  - pronoun resolution / commonsense benchmark
- `piqa`
  - physical commonsense QA
- `boolq`
  - boolean QA
- `openbookqa`
  - open-book science QA
- `truthfulqa_mc2`
  - truthful answering multiple-choice benchmark
- `mmlu`
  - broad multi-subject knowledge benchmark
- `ifeval`
  - instruction-following evaluation

## 13. What Is In Scope Now

These experiment families are implemented in the repo:

- individual SFT
- sequential SFT
- weighted mixed-dataset SFT
- LoRA SFT
- PRIME-RL orchestration
- SFT then RL orchestration
- self-distillation with privileged teacher
- async eval on another GPU
- benchmark-plan eval sweeps
- Google Drive checkpoint mirroring via `rclone`
- Hugging Face Hub checkpoint mirroring via `hf:` mirror roots
- safety-only and BFCL-focused one-off tests

## 14. Important Current Caveats

- PRIME-RL still depends on a real PRIME runtime/config setup.
- Custom RL datasets still need PRIME env mapping.
- Mixed SFT on built-in tasks must be built from SFT-formatted source dirs like
  `gsm8k_sft` and `math_sft`, not from the raw RL parquet dirs.
- Google Drive mirroring requires a working `rclone` install and authenticated
  `gdrive` remote on the machine.
- Hugging Face mirroring requires a valid Hub write token loaded through
  `scripts/load_hf_auth.sh` or equivalent env vars.
- Mirror uploads are best-effort and poll-based, so they may lag behind the
  exact checkpoint write time.
- Some external benchmark tasks still depend on external repos or approved gated access.
- Scratch is for env, caches, data, checkpoints, and benchmark outputs.
- Repo code and launchers are intended to be run from `~/forget`.
