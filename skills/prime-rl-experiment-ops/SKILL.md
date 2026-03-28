---
name: prime-rl-experiment-ops
description: Use when working on the forgetting-llms experiment pipeline on Mila or PrimeRL, especially for preprocessing datasets, generating teacher data, launching Qwen/OLMo runs, sweeping checkpoints, updating the experiment board, or triaging scratch-storage and benchmark-coverage issues.
---

# PrimeRL Experiment Ops

Use this skill for repeated experiment operations in this repo. It is for execution and triage, not for inventing new methodology.

## Start Here

Read these files first:
- `tasks.md`
- `STATUS.md`
- `CLAUDE.md`

Then classify the request:
- Existing Qwen3 pipeline work: use the current repo scripts.
- Missing PrimeRL / OLMo / safety / external benchmark wiring: make the missing integration explicit and do not pretend it already exists.

## Hard Guardrails

- Keep all large artifacts under `~/scratch/forgetting-llms/`.
- Do not write checkpoints, datasets, eval outputs, or model caches under `$HOME`.
- Reuse existing scripts before adding new ones.
- If a benchmark needs an external runner, record it as a wiring gap instead of fabricating a command.

## Default Workflow

1. Data
- Ground-truth data: `python scripts/preprocess_data.py --dataset <gsm8k|math|triviaqa> --format sft`
- RL data: `python scripts/preprocess_data.py --dataset <gsm8k|math|triviaqa>`
- Same-family teacher data: `sbatch scripts/generate_trajectories.sh Qwen/Qwen3-32B <dataset> <n_samples>`
- Full model snapshots on scratch: `python scripts/download_models.py --model <hf_repo_id>` or `python scripts/download_models.py --config <yaml> --config-key teacher_model --config-key student_model`

2. Training
- SFT: `sbatch scripts/run_sft.sh <dataset> <model> <experiment_name> <gt|sf|cf>`
- GRPO: `sbatch scripts/run_grpo_full.sh`
- Sequential GRPO: `sbatch scripts/run_grpo_sequential.sh <dataset> <model_or_ckpt> <experiment_name>`

3. Evaluation
- OOD checkpoint sweeps: `sbatch scripts/eval_sweep_resumable.sh <checkpoint_dir> <results_name> [base_model]`
- In-distribution task accuracy: `sbatch scripts/eval_task_accuracy.sh <checkpoint_dir> <dataset> [base_model]`
- Unified eval entry point: `python src/evaluation/run_eval.py --model_path <run_or_model> --suite forgetting --output_dir <dir>`

4. Monitoring
- Update the board with `python scripts/update_experiment_board.py`
- Use `STATUS.md` for narrative status and `EXPERIMENT_BOARD.md` for the current matrix snapshot

## When The Request Mentions PrimeRL Or OLMo

- Check whether the repo already has the needed integration. Right now the repo does not contain PrimeRL wiring or OLMo-specific verifier support.
- Treat that as implementation work, not as a ready-to-run path.
- First deliver the smallest usable slice:
  1. config + path conventions
  2. verifier environment contract
  3. one smoke test
  4. only then launch the full matrix

## When The Request Mentions Tasks From `tasks.md`

Use this priority order:
1. Eval automation and scratch-safe outputs
2. Current Qwen3-1.7B backlog
3. Monitoring board
4. Mix vs sequential experiment matrix
5. OLMo / PrimeRL integration
6. LoRA, reverse-KL, and other low-priority branches
