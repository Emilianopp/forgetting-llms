# Smoke Test Progress Log

## Goal
Run GRPO on Qwen3-1.7B + GSM8K via VeRL on Mila (2x A100 80GB) to verify the training pipeline works end-to-end.

## Timeline

### Session 1: Research & Initial Code
- Researched VeRL+GEM integration (two approaches: standard VeRL GRPO vs GEM's train_verl.py)
- Chose standard VeRL GRPO — simpler, compatible with our VeRL 0.7.0 pip install, equivalent for single-turn tasks
- Wrote 3 new files:
  - `scripts/preprocess_data.py` — GSM8K → VeRL parquet format
  - `src/rewards/math_reward.py` — math reward function
  - `scripts/run_grpo_smoke_test.sh` — Slurm launch script
- Updated `PLAN.md` — removed stale Oat/TRL references, corrected to 6 methods
- Committed and pushed to GitHub

### Session 2: Debug Loop (7 iterations)

**Error 1: Hydra config — `+data.custom_cls` (Job 8769353)**
- VeRL's default config already has `data.custom_cls` key
- Fix: Changed `+` to `++` prefix (then later removed entirely since default is already RLHFDataset)

**Error 2: Hydra config — `rollout_batch_size` (Job 8769356)**
- `rollout_batch_size` is not a valid key in VeRL's rollout config struct
- Fix: Removed it (train_batch_size already controls batch size)

**Error 3: `ROCR_VISIBLE_DEVICES` conflict (Job 8769408)**
- Mila cluster sets ROCR_VISIBLE_DEVICES (AMD ROCm) even on NVIDIA nodes
- VeRL's worker init raises ValueError when both ROCR and CUDA visible devices are set
- Fix: `unset ROCR_VISIBLE_DEVICES` in script

**Error 4: FlashAttention2 not installed (Job 8769418)**
- Qwen3 defaults to flash_attn_2 which is not installed
- Fix: Override to SDPA via `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`

**Error 5: vLLM 0.15.1 incompatible with VeRL 0.7.0 (Job 8769445)**
- VeRL 0.7.0 supports vLLM 0.8.5–0.12.0 (per setup.py)
- vLLM 0.15.1 changed WorkerWrapperBase API
- Fix: `pip install vllm==0.12.0`

**Error 6: GEM signal.SIGALRM in reward threads (Job 8769461)**
- GEM's math_grader uses signal.SIGALRM for timeouts, crashes in VeRL's worker threads
- Fix: Removed GEM grader from reward function, use simple number comparison

**Error 7: flash_attn.bert_padding missing (Job 8769522)**
- VeRL's `unpad_input` for remove_padding also requires flash_attn
- Fix: Set `use_remove_padding=False`

**Error 8: System OOM at 48GB (Jobs 8769522, 8769650)**
- Actor + ref model + optimizer offload + vLLM exceeded 48GB system RAM
- Could not request more memory due to QOS per-user limit (mila-code job using 128GB)
- Fix: Disabled KL loss to skip ref model entirely (smoke test only)
- Also reduced batch sizes: train_batch_size 32→16, response_length 2048→1024

### Current Status: SMOKE TEST PASSED (Job 8769777)
- **Pipeline verified end-to-end**: VeRL GRPO training with vLLM rollouts, custom reward function, WandB logging, periodic validation — all working
- **Steps 1-19+ completed** (~16 min elapsed, ~25s/step)
- **Initial GSM8K accuracy**: 37.5% (greedy, 1024 max tokens)
- **Step 10 validation accuracy**: 43.9% (+6.4pp improvement)
- **Training reward trend**: batch mean accuracy rising from ~0.38 (steps 1-5) → ~0.62 (steps 16-19)
- **GPU**: 50.5 GB / 80 GB per A100
- **CPU**: ~69 GB
- **Throughput**: ~1,370 tokens/sec, ~25s/step
- **WandB**: `grpo_smoke_test_qwen3_1.7b_gsm8k` in project `forgetting-llms`
- **Note**: Full 3-epoch run (1401 steps) would take ~9.7 hours, exceeding 4-hour wall limit. Job will be killed at ~step 550. This is fine for a smoke test — the goal was to verify the pipeline works.

### Observations for Production Runs
- High `response_length/clip_ratio` (70-85%) — many responses hit the 1024 token limit. Increase `max_response_length` to 2048 with more memory.
- Install `flash_attn` to enable flash attention and `use_remove_padding=True` for better throughput.
- Request 64-96GB system RAM to enable KL regularization (ref model).
- Consider increasing `max_response_length` to 2048 and `train_batch_size` to 32+ for real experiments.

## Environment Changes
- Downgraded vLLM: 0.15.1 → 0.12.0
- PyTorch: 2.9.1 → 2.9.0 (minor, came with vLLM downgrade)

## Key Learnings
1. VeRL 0.7.0 requires vLLM 0.8.5–0.12.0 (not the latest)
2. Mila cluster needs `unset ROCR_VISIBLE_DEVICES` for VeRL
3. flash_attn is needed for both model attention and VeRL's remove_padding — install it for production runs
4. GEM's math grader uses signals incompatible with VeRL's threaded reward — use simple grading instead
5. 48GB system RAM is tight for GRPO with ref model on 1.7B; need 64-96GB for real runs
6. VeRL's default dataset class is already RLHFDataset, no need to override `custom_cls`
