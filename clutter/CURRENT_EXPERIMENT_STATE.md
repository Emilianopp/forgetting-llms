# Current Experiment State

This document captures:

- how the repo is currently intended to work
- what experiment workflow is being attempted
- what has been implemented
- what errors have been encountered and what they meant
- what is still unresolved

It is a working state document, not a paper-style overview.

## 1. Repo Architecture

The repo now has four main layers.

### A. Data / Trace Generation

Main files:

- `src/data/generate_teacher_solutions.py`
- `scripts/generate_trajectories.sh`
- `scripts/import_dolci_sft.py`

Supported modes:

- correctness-gated trace generation for verifiable datasets
- direct import / regeneration for Dolci-style SFT prompt datasets
- resumable question-level checkpoints via `checkpoint.parquet` and `status.parquet`

Important behavior:

- verifiable datasets can use:
  - `samples_per_round`
  - `max_total_samples`
  - `target_correct_per_question`
  - `min_correct_per_question`
  - `solutions_per_question`
- Dolci prompt datasets do not expose a uniform gold answer field, so they support prompt-based generation but not correctness-gated resampling in the same way

### B. Training

Main files:

- `src/training/plain_sft.py`
- `src/training/self_distill.py`
- `scripts/run_sft.sh`
- `scripts/run_self_distill.sh`
- `scripts/run_training_plan.sh`
- `scripts/run_sft_stage_chain.sh`

Supported modes:

- individual SFT
- sequential SFT
- self-distillation
- PRIME-RL launch orchestration
- SFT then RL orchestration

Important policy change:

- RL is intended to be PRIME-RL only
- legacy VeRL paths were intentionally blocked or de-emphasized

### C. PRIME-RL

Main files:

- `scripts/prime_rl_runner.py`
- `scripts/run_all_tasks_prime_interactive_session.sh`
- `scripts/setup_prime_rl.sh`
- `scripts/bootstrap_prime_configs.py`

Important distinction:

- this repo does not implement the PRIME-RL engine itself
- it orchestrates PrimeIntellect PRIME-RL and local config generation

### D. Evaluation

Main files:

- `src/evaluation/run_eval.py`
- `scripts/eval_prime_checkpoint_sweep.py`
- `scripts/run_eval_with_local_server.sh`
- `scripts/vllm_runner_common.sh`
- `scripts/watch_checkpoints_eval.sh`
- `scripts/watch_prime_run_eval.sh`

Supported evaluation shapes:

- evaluate one checkpoint
- sweep checkpoints
- benchmark-plan suites
- async evaluation on a separate GPU while training continues
- fail-fast behavior on benchmark errors

## 2. Experiment Workflow Being Attempted

The main intended workflow is:

1. Build or reuse SFT data.
2. Train a model on one dataset or in sequence across datasets.
3. Save checkpoints every fixed number of steps, usually `300`.
4. Evaluate each stage or checkpoint on a benchmark plan while training continues on other GPUs.
5. Compare:
   - individual vs sequential
   - SFT vs SFT+RL
   - with and without LoRA
   - backward-task retention after sequential stages

The main sequential experiment currently being attempted is:

- stage 1: `synthetic2_sft_verified`
- stage 2: `triviaqa`
- stage 3: `gsm8k`

using:

- `Qwen/Qwen3-1.7B`
- SFT first
- benchmark evaluation after each stage
- asynchronous eval on a separate GPU

The intended benchmark behavior is:

- train on GPUs `0,1,2`
- evaluate on GPU `3`
- automatically serve the current checkpoint with vLLM
- run the configured benchmark suite against that served model

## 3. Benchmark Plan

The original desired benchmark plan was:

- GPQA
- RG-mix
- AIME
- LiveCodeBench v6
- SimpleQA
- HumanEval+
- MBPP+
- LitQA2
- BFCL
- Safety

This is exposed in the repo as `tasks_md`.

Because several of those need external repos or extra environment setup, a reduced suite was added:

- `tasks_md_core`

which currently includes:

- GPQA
- AIME
- LiveCodeBench v6
- SimpleQA
- HumanEval+
- MBPP+
- LitQA2

and excludes:

- RG-mix
- BFCL
- Safety

## 4. What Has Been Implemented For This Workflow

### A. Async checkpoint eval on another GPU

Implemented.

Training can continue on one GPU set while benchmark eval serves the checkpoint on another GPU.

Relevant pieces:

- `TRAIN_CUDA_VISIBLE_DEVICES`
- `ASYNC_EVAL_GPU`
- `AUTO_START_EVAL_SERVER=1`

### B. Fail-fast evaluation

Implemented.

Default behavior is now to stop on benchmark failure unless `continue_on_error` is explicitly re-enabled.

### C. Benchmark banners and dataset counts

Implemented.

`run_eval.py` now prints benchmark name, runner, and expected example count before each benchmark starts.

### D. Hugging Face auth auto-loading

Implemented.

Launchers source:

- `scripts/load_hf_auth.sh`

which reads:

- `~/scratch/forgetting-llms/hf_auth.sh`

and writes the token to `HF_HOME/token`.

### E. Benchmark env auto-loading

Implemented.

Launchers source:

- `~/scratch/forgetting-llms/benchmark_env.sh`

for benchmark-specific roots and endpoint settings.

### F. vLLM eval-server stabilization

Implemented in the shared launcher layer.

The eval server path now defaults to:

- `VLLM_USE_STANDALONE_COMPILE=0`
- `VLLM_DISABLE_COMPILE_CACHE=1`
- `VLLM_SERVER_ENFORCE_EAGER=1`
- `VLLM_USE_FLASHINFER_SAMPLER=0`

and prints log tails when startup or runner execution fails.

### G. PRIME-style tagged traces

Implemented for non-RL trace generation.

The repo supports:

- `<think>...</think><answer>...</answer>`

and the reward parsing was updated to read tagged answers.

### H. Self-distillation scaffold

Implemented.

The current code supports:

- privileged teacher prompt
- generated student trace
- generated privileged teacher trace
- forward KL
- reverse KL
- interpolated KL
- student and teacher on separate GPUs

This was implemented for training, not for evaluation.

## 5. Major Errors Encountered And What They Meant

This section records the main failures that came up while trying to get the full workflow running.

### A. Broken scratch venv / symlink loop

Symptoms:

- `Too many levels of symbolic links`
- broken `python3` inside `.venv`

Meaning:

- the scratch env was corrupted

Fix direction:

- rebuild with copies instead of symlink-heavy creation

### B. Missing SSL / `libssl.so.1.1`

Symptoms:

- `SSL module is not available`
- `pyarrow.dataset` failing on `libssl.so.1.1`

Meaning:

- cluster Python / env linkage issue
- not a dataset or model logic issue

Fix direction:

- correct module load order
- cluster SSL module availability
- env rebuild if needed

### C. Incomplete `transformers` / `vllm` install

Symptoms:

- missing files like `llava_onevision`, `roformer`, `dinov2`

Meaning:

- broken or inconsistent Python package install

Fix direction:

- rebuild env
- reinstall core stack

### D. Qwen3.5-27B multimodal / environment incompatibility

Symptoms:

- model import failures under local `vllm.LLM(...)`

Meaning:

- the environment did not fully support `Qwen3.5-27B`
- not a prompt-format problem

Fix direction:

- newer compatible `vllm` and `transformers`
- or use a different model for the immediate task

### E. vLLM `FakeTensorMode` failures

Symptoms:

- standalone compile tracebacks mentioning `FakeTensorMode`

Meaning:

- unstable vLLM standalone-compile path

Fix direction:

- disable standalone compile
- use eager mode

### F. FlashInfer / `nvcc` failure

Symptoms:

- vLLM trying to JIT through FlashInfer
- missing `nvcc`

Meaning:

- eval server was still taking a sampler path that required local CUDA toolchain pieces

Fix direction:

- disable FlashInfer sampler in the eval-server launcher

### G. PRIME env/config mismatch

Symptoms:

- RL launch bundles start
- orchestrator or environment dies
- local model path mismatch across trainer / orchestrator / inference

Meaning:

- not every RL failure was a model failure
- many were environment/config/bootstrap issues

Fix direction:

- unify model name in configs
- ensure real PRIME env/config exists for the target task

### H. BFCL and Safety roots missing

Symptoms:

- eval preflight says configured roots do not exist

Meaning:

- placeholder paths were set but the repos were not installed there

Fix direction:

- clone/install benchmark repos or exclude them from the suite

### I. GPQA auth problems

Symptoms:

- GPQA preflight fails asking for `HF_TOKEN`

Meaning:

- GPQA is gated
- token and/or access approval were missing

Fix direction:

- load HF auth automatically
- accept token via `HF_HOME/token`

### J. Misleading mixed logs

Symptoms:

- screenshots appeared to show old and new code paths at the same time

Meaning:

- eval runner logs were being appended across runs

Fix direction:

- truncate server and runner logs at the start of each eval-server run

### K. LiteLLM / LightEval endpoint failures

Symptoms:

- red LiteLLM warnings
- `Hosted_vllmException`
- `Connection refused`
- `EngineCore encountered an issue`

Meaning:

- LiteLLM was usually the messenger, not the root cause
- the served vLLM engine was dying during benchmark requests

Important current diagnosis:

- GPQA could complete
- AIME was a case where the served engine died after benchmark start
- the endpoint-backed LightEval path was missing generation constraints that the native vLLM path already used

Repo fix already made:

- endpoint-backed LightEval now also sends:
  - `max_new_tokens`
  - `temperature=0.0`
  - `top_p=1.0`

This is intended to stop the endpoint path from using bad or uncontrolled defaults.

## 6. Current State Of The Sequential SFT Attempt

What is intended:

- `Qwen3-1.7B`
- train on `synthetic2_sft_verified`
- then `triviaqa`
- then `gsm8k`
- evaluate after each stage on a benchmark suite
- run eval on a separate GPU while training continues

What is currently true:

- the SFT training path exists
- stage-end and async eval launching exist
- HF auth loading exists
- reduced benchmark suite exists
- external benchmark roots can be installed via setup script
- the vLLM eval-server path has been hardened substantially

What is still the active area of instability:

- LightEval endpoint-backed execution against the auto-started local vLLM server
- especially on benchmarks after GPQA, such as AIME

## 7. Current Recommended Evaluation Shape

Until all external benchmark repos and endpoint-backed runners are fully stable together, the safest operating modes are:

### A. Core suite first

Use:

- `tasks_md_core`

instead of the full `tasks_md`

### B. Keep async eval on a separate GPU

Use:

- `TRAIN_CUDA_VISIBLE_DEVICES=0,1,2`
- `ASYNC_EVAL_GPU=3`
- `AUTO_START_EVAL_SERVER=1`

### C. Keep the hardened vLLM server flags

Use:

- `VLLM_USE_STANDALONE_COMPILE=0`
- `VLLM_DISABLE_COMPILE_CACHE=1`
- `VLLM_SERVER_ENFORCE_EAGER=1`
- `VLLM_USE_FLASHINFER_SAMPLER=0`

## 8. What The User Has Been Trying To Do

The intended scientific workflow has been:

- collect SFT traces from verifiable or OLMo-aligned datasets
- test individual and sequential training
- train on one dataset, evaluate, then continue to the next dataset
- compare retained performance across a fixed evaluation suite
- eventually support:
  - SFT only
  - RL only
  - SFT then RL
  - individual
  - sequential
  - LoRA vs full fine-tune
  - self-distillation

The immediate concrete target has been:

- get the sequential SFT path running robustly
- get asynchronous benchmark evaluation working reliably on a separate GPU
- keep all failures visible on screen instead of buried in logs

## 9. Current Open Issue

The main current unresolved issue is:

- endpoint-backed LightEval benchmarks can still kill the served vLLM engine during the run, producing LiteLLM retry noise

The repo-side diagnosis already made is:

- the red LiteLLM output is not itself the root cause
- the actual problem is when the served vLLM engine crashes during benchmark requests

The most recent repo-side fix for this was:

- make endpoint-backed LightEval use the same bounded generation settings as native vLLM

If failures persist after that, the next debugging target is:

- the served vLLM engine log for the current run only
- not historical appended logs

## 10. Practical Next Step

If continuing from the current state:

1. Keep the current hardened eval-server flags.
2. Use the updated `run_eval.py`.
3. Prefer `tasks_md_core` until the full suite is stable.
4. If the engine still dies during AIME or later LightEval benchmarks, inspect the current `vllm_server.log` for the exact engine-side traceback.

