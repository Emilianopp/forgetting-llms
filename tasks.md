# Two-Day Execution Plan

Updated repo status:
- checked items below mean the repo wiring/config support exists
- unchecked items are still execution, environment, or cluster-dependent

## Definition Of Done By End Of Tomorrow

- One eval entry point exists for checkpoint sweeps and benchmark suites.
- Eval outputs, checkpoints, and datasets all land in `~/scratch` on Mila.
- A lightweight board exists to monitor training and eval progress.
- The current Qwen3-1.7B backlog is drained or explicitly queued.
- The next experiment matrix is frozen: mix vs sequential, reversed vs forward order, IID control.
- PrimeRL / OLMo work has either a smoke test or an explicit blocker list.
- SFT data rules are written down tightly enough that data generation is unambiguous.

## Current Repo Reality

### Already Working

- Unified eval entry point for single checkpoints and checkpoint sweeps
- Scratch-safe eval outputs, checkpoints, datasets, and env setup
- Monitoring board generation from scratch outputs
- Repo-local automation skill scaffold
- Qwen3-1.7B GT-SFT, SF-SFT, GRPO, and SFT+GRPO training scripts
- Resumable OOD eval sweeps
- In-distribution task-accuracy sweeps
- Same-family teacher trajectory generation
- Sequential GRPO plus orchestrated sequential/mix/IID paths
- Mixed and IID dataset/control builders
- Cross-family SFT launchers
- OLMo baseline and sequential launchers
- LoRA vs full fine-tune launchers
- PRIME-style tagged correct-only SFT trace generation with question-level resume

### Partial / Missing

- OLMo-specific verifier environment support is still cluster-dependent
- Some benchmark runners are wired in the repo but still depend on external repos, endpoints, or installs
- Code evaluation in `src/rewards/unified_reward.py` is still a stub

### External Or Cluster-Dependent

- PrimeRL setup and job submission
- OLMo 3 7B environment validation
- BFCL, Safety, LiveCodeBench, LAB-Bench, and similar external benchmark environments
- Large-scale experiment execution on Mila

## Today

### Repo Work That Must Land

- [x] Add a unified eval entry point that can sweep checkpoints
- [x] Ensure the generic eval launcher writes to `~/scratch`
- [x] Add a board generator for experiment monitoring
- [x] Add a repo-local automation skill scaffold
- [x] Freeze the exact benchmark manifest for each tasks.md eval
- [x] Decide what stays manual vs what runs through `lm_eval`
- [x] Decide the canonical experiment naming scheme for mix / sequential / reversed / IID

### Cluster Work To Run Today

- [ ] Finish the remaining Qwen3-1.7B OOD eval sweeps already tracked in `STATUS.md`
- [ ] Finish SFT+RL eval sweeps
- [ ] Regenerate comparison plots after the eval backlog clears
- [ ] Update `EXPERIMENT_BOARD.md` from scratch paths

## Tomorrow

### Experiment Matrix To Freeze

- [x] `mix` vs `sequential`
- [x] `forward order` vs `reversed order`
- [x] `IID control`
- [x] `LoRA` vs `full fine-tune`
- [x] `Qwen3-1.7B` vs `OLMo 3 7B`

### Required Experiment Additions

- [ ] **Mix vs sequential comparison** for the three dataset pairs:
  `gsm8k <-> math`, `gsm8k <-> triviaqa`, `math <-> triviaqa`
- [ ] **IID control** for the same three dataset pairs
- [ ] **Cross-family SFT matrix** once Llama teacher trajectories exist:
  `CF-SFT` on `gsm8k`, `math`, `triviaqa`
- [ ] **OLMo 3 7B replication**:
  baseline on `gsm8k`, `math`, `triviaqa`
  plus the same six sequential order runs as Qwen3-1.7B
- [ ] **LoRA vs full fine-tune ablation**:
  start with `GT-SFT` and `SF-SFT` on `gsm8k`, `math`, `triviaqa`
  if stable, add `GT-SFT+GRPO` on the strongest pair

### PrimeRL / OLMo Minimum Viable Slice

- [x] Create the PrimeRL path contract: data locations, checkpoint locations, eval locations
- [ ] Build or adapt a verifier environment that works with the OLMo data path
- [ ] Run one OLMo 3 7B smoke test before committing to the full matrix
- [ ] If the smoke test fails, write the blocker list immediately instead of expanding scope

### SFT Dataset Rules

- [x] Filter impossible questions
- [x] Guarantee equal numbers of solutions per solvable question
- [x] Use the pass@16 heuristic: keep questions with at least 2 correct solutions
- [x] Sample synthetic solutions with the 32B teacher
- [x] Write down the balancing rule per domain so future generations are reproducible

## Concrete Run Matrix

### A. Mix vs Sequential vs IID Control

Run this for each pair:
- `gsm8k <-> math`
- `gsm8k <-> triviaqa`
- `math <-> triviaqa`

For each pair, queue:
- [ ] Sequential forward: `A -> B`
- [ ] Sequential reverse: `B -> A`
- [ ] Mixed training: pooled `A + B`
- [ ] IID staged control: matched pooled distribution split into two IID shards

Minimum total: `3 pairs x 4 run families = 12 runs`

### B. Cross-Family SFT

- [ ] Generate Llama teacher trajectories for `gsm8k`
- [ ] Generate Llama teacher trajectories for `math`
- [ ] Generate Llama teacher trajectories for `triviaqa`
- [ ] Run `CF-SFT` on `gsm8k`
- [ ] Run `CF-SFT` on `math`
- [ ] Run `CF-SFT` on `triviaqa`

### C. OLMo 3 7B Replication

Baseline:
- [ ] OLMo baseline on `gsm8k`
- [ ] OLMo baseline on `math`
- [ ] OLMo baseline on `triviaqa`

Sequential order runs:
- [ ] `gsm8k -> math`
- [ ] `math -> gsm8k`
- [ ] `gsm8k -> triviaqa`
- [ ] `triviaqa -> gsm8k`
- [ ] `math -> triviaqa`
- [ ] `triviaqa -> math`

Minimum total: `9 runs`

### D. LoRA vs Full Fine-Tune Ablation

Start here:
- [ ] `GT-SFT` LoRA on `gsm8k`
- [ ] `GT-SFT` full fine-tune on `gsm8k`
- [ ] `GT-SFT` LoRA on `math`
- [ ] `GT-SFT` full fine-tune on `math`
- [ ] `GT-SFT` LoRA on `triviaqa`
- [ ] `GT-SFT` full fine-tune on `triviaqa`
- [ ] `SF-SFT` LoRA on `gsm8k`
- [ ] `SF-SFT` full fine-tune on `gsm8k`
- [ ] `SF-SFT` LoRA on `math`
- [ ] `SF-SFT` full fine-tune on `math`
- [ ] `SF-SFT` LoRA on `triviaqa`
- [ ] `SF-SFT` full fine-tune on `triviaqa`

If stable:
- [ ] Add `GT-SFT+GRPO` LoRA vs full fine-tune on the strongest pair

### E. Evaluation Requirement For All New Runs

For every new run family above:
- [ ] Save checkpoints to scratch
- [ ] Save eval outputs to scratch
- [ ] Save metrics to scratch
- [ ] Log to WandB
- [ ] Run OOD forgetting eval
- [ ] Run in-distribution task eval
- [ ] For sequential runs, evaluate the backward task after each stage
- [ ] Update `EXPERIMENT_BOARD.md`

## Benchmark Plan

| Benchmark | Status | Runner plan |
| --- | --- | --- |
| GPQA | Wired | LightEval task in `run_eval.py` |
| RG-mix | Hooked | custom command hook, benchmark source remains project-local |
| AIME | Wired | LightEval task in `run_eval.py` |
| LiveCodeBenchv6 | Wired | LightEval task, or override with official runner |
| SimpleQA | Wired | LightEval task in `run_eval.py` |
| HumanEvalPlus | Wired | EvalPlus wrapper |
| MBPP+ | Wired | EvalPlus wrapper |
| LitQA2 | Wired | LAB-Bench wrapper |
| BFCL | Wired / external env | official runner, still needs benchmark root/env |
| Safety | Wired / external env | safety-eval runner, still needs benchmark root/env |

### Protocol Requirement

- [x] For sequential runs, always evaluate backward tasks after each stage

## Standard Hyperparameters To Keep Fixed Until A Smoke Test Breaks

- `temperature = 1.0`
- `batch size = 256`
- `group size k = 8`
- `sequence length = 8192`

Current orchestrator defaults are aligned to those values.

Do not branch on hyperparameters until the execution path is stable.

## Low Priority / Explicit Defers

- [ ] Reverse-KL vs forward-KL with teacher
- [ ] Self-distillation vs forward samples from the student

These should only move once the core infra, evals, and first OLMo smoke test are done.
