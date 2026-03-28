# Chat History - 2026-03-28

This file captures the main implementation and debugging thread from the
current session so the next turn does not have to reconstruct it from scratch.

It is an operator guide for the repo as it exists after this session, not a
paper summary.

## Main Goal

Make the repo usable for:

1. benchmark verification on Mila
2. one-file end-to-end SFT, RL, and `sft_rl` runs
3. checkpoint mirroring during training
4. LoRA SFT followed by RL
5. mixed-dataset smoke tests with smaller parquet inputs
6. broader core training support beyond the original verifier tasks

## High-Level Conclusions

- `scripts/run_training_plan.sh` remains the main entrypoint for SFT, RL, and
  `sft_rl`.
- BFCL and RG-mix both work best through repo-owned wrappers, not by treating
  them as fully manual standalone flows.
- Google Drive checkpoint upload is feasible on Mila through `rclone`, but it
  still depends on an authenticated `rclone` remote.
- Hugging Face Hub checkpoint mirroring is now treated as a first-class mirror
  target alongside Google Drive.
- Mixed SFT on built-in tasks only works if the mixed parquet is built from
  SFT-formatted source dirs such as `gsm8k_sft` and `math_sft`.
- LoRA SFT needed a merged-model export step so post-SFT eval and later RL do
  not crash on adapter-only checkpoints.

## What Was Added Or Changed

### 1. Google Drive And Hugging Face Checkpoint Mirroring

The training launchers now support mirror targets of these forms:

- normal local path
- `gdrive:...`
- `hf:model:<user_or_org>/<repo>`
- `hf:dataset:<user_or_org>/<repo>`

Relevant behavior:

- training still writes to scratch first
- mirroring is poll-based, not event-triggered
- the final sync runs when training exits
- upload failures should not kill training

Relevant files:

- `scripts/watch_directory_mirror.sh`
- `scripts/run_training_plan.sh`
- `scripts/run_sft.sh`
- `Current_success.md`

### 2. Standalone Google Drive Checkpoint Uploader

A dedicated uploader was added:

- `scripts/upload_checkpoints_to_gdrive.py`

Purpose:

- watch a run directory
- detect ready checkpoints
- upload each checkpoint once through `rclone`

Important constraint:

- this script does not do Google auth itself
- it depends on a working `rclone` remote such as `gdrive`

### 3. BFCL Wrapper Fixes

BFCL smoke testing was clarified and the repo wrapper was fixed so BFCL uses
the explicit `--bfcl-model-name` rather than inferring a broken name from the
checkpoint directory.

Relevant file:

- `src/evaluation/run_eval.py`

Practical note:

- `simple_python` is the intended BFCL smoke category
- BFCL checkout and venv live under scratch by design

### 4. RG-mix Runner Hardening

The direct RG-mix runner was improved so it can:

- preflight the OpenAI-compatible endpoint
- print clearer connection failures
- show progress during generation
- optionally start its own local vLLM server from the same file
- use a separate Python for the server when the RG-mix client venv does not
  contain `vllm`

Relevant files:

- `scripts/run_rg_mix_benchmark.py`
- `src/evaluation/run_eval.py`

Practical note:

- the initial `localhost`/endpoint errors were not dataset errors
- they were server reachability and `vllm`-environment issues

### 5. LoRA SFT -> Merged HF Export

LoRA SFT checkpoints now need to produce a merged standalone model for eval and
RL continuation.

Relevant files:

- `src/training/plain_sft.py`
- `scripts/merge_lora_checkpoint.py`
- `src/evaluation/run_eval.py`
- `scripts/run_training_plan.sh`

Outcome:

- post-SFT eval no longer has to consume an adapter-only directory
- later RL stages can continue from `merged_hf/`

### 6. Mixed SFT Dataset Builder Improvements

The mixed-dataset builder now supports:

- choosing suffixed source dirs such as `_sft`
- building very small smoke-test splits

Relevant file:

- `scripts/build_mixed_dataset.py`

Important workflow:

1. preprocess built-in source datasets into SFT parquet
2. build the mixed parquet from those SFT dirs
3. point `STAGE_DATA_DIR_<STAGE>` at the mixed SFT output

The old failure mode was mixing the raw RL-format dirs and then feeding that
into SFT.

### 7. PRIME RL Log And Cache Hardening

PRIME runs were hardened after Mila failures caused by cache writes under the
home quota.

Relevant file:

- `scripts/prime_rl_runner.py`

What changed:

- failure reporting now tails the correct log files in combined PRIME mode
- common Hugging Face, Torch, Triton, and temp caches are redirected to scratch

This was added because the actual runtime failure was:

- `OSError: [Errno 122] Disk quota exceeded`

### 8. Expanded Core Dataset Support For SFT, RL, And SFT+RL

The training core now treats the following as first-class stage names:

- `synthetic2_sft_verified`
- `dolci_think_sft_7b`
- `dolci_think_sft_32b`
- `olmo_rl_zero_math`
- `dolci_rl_zero_math`
- `tau2bench`

Relevant files:

- `scripts/run_training_plan.sh`
- `scripts/run_sft.sh`
- `scripts/convert_sft_dataset_to_rl.py`
- `scripts/convert_rl_dataset_to_sft.py`
- `scripts/run_grpo_dataset_dir_local.sh`
- `scripts/build_mixed_dataset.py`

Behavior:

- built-in verifier tasks still use PRIME
- the new core datasets use a local dataset-dir GRPO path for RL
- SFT-only parquet can be auto-converted into RL parquet
- RL-first parquet can be auto-converted into SFT parquet

Important caveat:

- `tau2bench` support here is parquet-backed training support, not native Sierra
  `tau2-bench` environment integration

## What Was Verified Working

These paths were successfully exercised or clearly verified during the session:

- BFCL installation and basic invocation shape
- BFCL repo wrapper respecting explicit BFCL model name
- RG-mix dataset loading and single-file runner flow
- Google Drive auth via `rclone`
- one-file launcher support for checkpoint mirroring to `gdrive:...`
- one-file launcher support for checkpoint mirroring to `hf:...`
- tiny mixed-SFT parquet construction from SFT-formatted sources
- LoRA SFT progressing through save and merged-model handoff into RL

Important nuance:

- some of the later training-core expansions were syntax-checked and wired into
  the launcher, but not all of them were live-run end to end in this session

## Main Failure Modes And Their Causes

### Mixed SFT "No usable examples found"

Cause:

- the mixed parquet came from raw task parquet, not SFT-formatted source dirs

Fix:

- preprocess `gsm8k` and `math` into `gsm8k_sft` and `math_sft`
- rebuild the mixture with `--source-suffix _sft`

### LoRA SFT Post-Stage Eval Crash

Cause:

- stage eval was pointed at an adapter-only checkpoint

Fix:

- export or create `merged_hf/`
- make eval and later RL resolve `merged_hf/` automatically

### PRIME RL Inference Failure With Disk Quota Errors

Cause:

- runtime caches were being written under quota-limited home paths

Fix:

- redirect caches and temp dirs to scratch

### Confusion About "Missing Evals"

Cause:

- the terminal output being watched was training output, not eval output
- for PRIME RL, async checkpoint eval only starts if:
  - the backend is `prime`
  - `RUN_BENCHMARK_EVALS=1` or `RUN_TASK_EVALS=1`
  - `ASYNC_EVAL_GPU` is set
  - a checkpoint has actually been created

Important distinction:

- PRIME backend can run async checkpoint eval on another GPU
- dataset-dir RL backend does not start a checkpoint watcher; it only runs
  stage-end eval after RL finishes

## Current Recommended Smoke-Test Shapes

### Tiny Mixed LoRA `sft_rl` Smoke Test

Use:

- a very small mixed SFT parquet built from `gsm8k_sft` and `math_sft`
- low `SFT_MAX_STEPS`
- low `RL_MAX_STEPS`
- evals disabled first

Reason:

- this isolates the actual training path before adding async eval and benchmark
  complexity

### RL Checkpoint Mirroring Smoke Test

Use:

- `RL_MAX_STEPS=100`
- `RL_CKPT_INTERVAL=50`
- `CHECKPOINT_MIRROR_ROOT=gdrive:...` or `hf:...`

Reason:

- this verifies that checkpoints appear locally and are mirrored during the run

## Current Caveats

- Google Drive uploading still depends on a valid `rclone` remote and stored
  auth token.
- Personal Google Drive auth cannot be created from a Mila terminal alone; a
  browser-capable machine is still needed once for OAuth.
- Async RL checkpoint eval is only implemented for PRIME-backed RL stages.
- If trainer GPUs and inference GPUs already consume the whole allocation, there
  is no spare GPU for async eval.
- `tau2bench` currently means parquet-backed training support, not native
  benchmark-environment support.

## Files Changed In This Session

Main files touched:

- `Current_success.md`
- `src/evaluation/run_eval.py`
- `src/training/plain_sft.py`
- `scripts/run_training_plan.sh`
- `scripts/prime_rl_runner.py`
- `scripts/build_mixed_dataset.py`
- `scripts/run_rg_mix_benchmark.py`
- `scripts/watch_directory_mirror.sh`
- `scripts/upload_checkpoints_to_gdrive.py`
- `scripts/merge_lora_checkpoint.py`
- `scripts/convert_sft_dataset_to_rl.py`
- `scripts/convert_rl_dataset_to_sft.py`
- `scripts/run_grpo_dataset_dir_local.sh`

## Validation Completed

Static validation completed during the session included:

- `bash -n scripts/run_training_plan.sh`
- `bash -n scripts/run_sft.sh`
- `bash -n scripts/run_grpo_dataset_dir_local.sh`
- `python -m py_compile scripts/prime_rl_runner.py`
- `python -m py_compile scripts/build_mixed_dataset.py`
- `python -m py_compile scripts/convert_sft_dataset_to_rl.py`
- `python -m py_compile scripts/convert_rl_dataset_to_sft.py`

## Summary

This session moved the repo from ad hoc Mila debugging toward a more coherent
training operator flow:

- BFCL and RG-mix were brought under repo-owned launcher behavior
- checkpoint mirroring now covers both Google Drive and Hugging Face Hub
- LoRA SFT no longer blocks later eval/RL stages
- mixed SFT smoke tests have a correct small-data path
- PRIME RL failures now surface better logs and use scratch-backed caches
- the training core now includes prompt-style and parquet-backed datasets beyond
  the original verifier tasks
