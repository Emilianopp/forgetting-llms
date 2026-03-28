# Chat History - 2026-03-26

This file captures the debugging thread from the current session so the next turn
does not have to reconstruct it from scratch.

## Main Goal

Get a workflow where:

1. training runs on one GPU set
2. checkpoints can be evaluated on another GPU
3. all benchmark families eventually work
4. decoding uses temperature `1.0`
5. `pass@k` is configurable, especially `k=512`

## High-Level Conclusions

- The old `LightEval -> LiteLLM -> local OpenAI/vLLM server` path was the main source of the earlier API-call failures.
- Native LightEval with its own runner env works and is the correct path for LightEval benchmarks.
- One shared benchmark `.venv` is not reliable. The dependency conflicts are real and were repeatedly breaking the environment.
- The correct architecture is:
  - one lightweight orchestration env
  - one venv per heavy benchmark runner
  - native backends where possible
  - local OpenAI-compatible server only where truly needed

## What Was Verified Working

These benchmark paths were reported working during this session:

- `aime`
- `livecodebench_v6`
- `gpqa`
- `simpleqa`
- `supergpqa`

Important details:

- LightEval now runs natively rather than through LiteLLM.
- LightEval `pass@k` was fixed so it no longer collapses to `k=1`.
- The working runs showed:
  - `LightEval samples per prompt: 512 (pass@k=512)`
  - final metrics like `results/all/aime_pass@k:k=512`

## SuperGPQA

`supergpqa` was added as a repo-native benchmark runner.

Properties:

- uses local `vllm`
- multiple-choice benchmark
- supports batched generation
- computes:
  - `accuracy_at_1`
  - `pass@k`
- defaults now target:
  - `num_samples=512`
  - `pass_k=512`

Important note:

- SuperGPQA is not using LightEval.
- It is implemented in `scripts/run_supergpqa_eval.py`.

## Environment Architecture Change

New recommended setup:

- orchestration env:
  - `~/scratch/forgetting-llms/.venv`
- runner envs:
  - `~/scratch/forgetting-llms/.venvs/lighteval`
  - `~/scratch/forgetting-llms/.venvs/evalplus`
  - `~/scratch/forgetting-llms/.venvs/labbench`
  - later `safety`, `bfcl` as separate envs too

New script added:

- `scripts/setup_runner_venvs.sh`

Purpose:

- create one virtualenv per benchmark runner
- write `BENCHMARK_RUNNER_SHELL_PREFIX_*` into `~/scratch/forgetting-llms/benchmark_env.sh`

Important change:

- `scripts/setup_tasks_md_benchmarks.sh` no longer installs benchmark Python packages into the active env by default
- the old shared-env install path was causing real conflicts and is now disabled unless explicitly requested

## EvalPlus Status

### What worked before

- `humaneval_plus` had worked previously on the old endpoint/OpenAI path
- but it was extremely slow, around tens of tokens/sec

### Why it was slow

- the old EvalPlus path used:
  - `evalplus.evaluate`
  - backend `openai`
  - local OpenAI-compatible server
- this is problem-by-problem and throughput-poor

### What was changed

Repo defaults were changed so EvalPlus now prefers:

- backend `vllm`
- `evalplus_n_samples=512`
- `evalplus_batch_size=32`

Also:

- `run_eval_with_local_server.sh` no longer treats `evalplus` as an endpoint runner by default

### Current blocker

Native EvalPlus is currently failing in the dedicated `evalplus` env before HumanEval+ starts.

The failure is not HumanEval+ logic. It is a native `vllm` engine init crash:

- TorchDynamo / FakeTensor path
- vLLM V1 engine
- `EngineCore initialization failed`
- stack includes `site-packages/vllm/v1/...`

This is the same family of compile/init failures seen earlier on the local-server path.

## EvalPlus Hardening Added

`run_eval.py` was patched so native benchmark runners now inherit hardened vLLM defaults:

- `VLLM_USE_STANDALONE_COMPILE=0`
- `VLLM_DISABLE_COMPILE_CACHE=1`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `VLLM_USE_FLASHINFER_SAMPLER=0`

EvalPlus was also patched to prefer the legacy offline engine by default:

- for native EvalPlus runs, set `VLLM_USE_V1=0`

This was added specifically because the trace showed the crash was still happening in the V1 offline engine.

## LabBench Status

- dedicated `labbench` env was reported working
- import check succeeded
- `litqa2` runtime has not yet been fully re-verified in this session after the env split

## Current Verified Env State

Reported working:

- `lighteval` env: ok
- `evalplus` env: ok
- `labbench` env: ok

This means the remaining issue is runtime behavior, not venv creation.

## Files Changed In This Session

Main files touched:

- `src/evaluation/run_eval.py`
- `scripts/run_eval_with_local_server.sh`
- `scripts/run_supergpqa_eval.py`
- `scripts/setup_runner_venvs.sh`
- `scripts/setup_tasks_md_benchmarks.sh`
- `scripts/benchmark_env.sh.example`

Key changes:

- native LightEval pass@k fix
- SuperGPQA runner added
- EvalPlus moved off endpoint path by default
- per-runner venv setup added
- shared benchmark env installs disabled by default
- native benchmark runners now inherit hardened vLLM env vars
- EvalPlus native path now defaults `VLLM_USE_V1=0`

## Exact Next Step

The next thing to verify is whether the latest EvalPlus hardening fixes native HumanEval+.

After syncing the latest `run_eval.py` to Mila, rerun:

```bash
unset VLLM_USE_V1
export VLLM_USE_V1=0
export VLLM_USE_STANDALONE_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES=3 \
python3 src/evaluation/run_eval.py \
  --model_path /home/mila/d/dane.malenfant/scratch/qwen4b_instruct \
  --suite tasks_md_core \
  --output_dir "$HOME/scratch/forgetting-llms/benchmark_plan_evals/humaneval_plus_smoke_qwen4b" \
  --run_name humaneval_plus_smoke_qwen4b \
  --include-runner evalplus \
  --include-benchmark humaneval_plus \
  --evalplus-backend vllm \
  --evalplus-n-samples 8 \
  --evalplus-batch-size 32 \
  --sampling-temperature 1.0 \
  --sampling-top-p 1.0 \
  --force-rerun
```

If that still fails, the conclusion is:

- native EvalPlus `vllm` backend is still not stable enough on Mila in the current stack
- at that point the robust high-throughput path should be:
  - repo-native batched `vllm` code generation
  - EvalPlus grading only

## Remaining Benchmarks To Verify

Still not fully verified in the new architecture:

- `humaneval_plus`
- `mbpp_plus`
- `litqa2`
- later:
  - `safety`
  - `bfcl`
  - `rg_mix`

## Summary

The session did produce real progress:

- LightEval is fixed and working natively
- `pass@k=512` is working for LightEval
- SuperGPQA support was added
- the benchmark env architecture was corrected

The main unresolved issue is now narrow:

- native EvalPlus `vllm` startup on Mila is crashing in vLLM engine init

That is the current blocker, not the broader benchmark system.
