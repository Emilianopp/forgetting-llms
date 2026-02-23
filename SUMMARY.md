# Smoke Test Progress Log

## Goal
Run GRPO on Qwen3-1.7B + GSM8K via VeRL on Mila (2x A100 80GB) to verify the training pipeline works end-to-end.

## Timeline

### Session Start
- Researched VeRL+GEM integration (two approaches: standard VeRL GRPO vs GEM's train_verl.py)
- Chose standard VeRL GRPO — simpler, compatible with our VeRL 0.7.0 pip install, equivalent for single-turn tasks
- Wrote 3 new files:
  - `scripts/preprocess_data.py` — GSM8K → VeRL parquet format
  - `src/rewards/math_reward.py` — math reward function (GEM grading + fallback)
  - `scripts/run_grpo_smoke_test.sh` — Slurm launch script
- Updated `PLAN.md` — removed stale Oat/TRL references, corrected to 6 methods

### Step 1: Commit & Push
- Status: IN PROGRESS
