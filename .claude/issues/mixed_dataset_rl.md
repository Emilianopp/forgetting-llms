# Mixed-Dataset RL Training in VeRL

## Status: Research / Feasibility Analysis

## Question
Can we train a single GRPO run on multiple datasets simultaneously (e.g., GSM8K + MATH + NaturalQuestions) to study cross-domain forgetting within a single RL training run?

## What Works Today

### Multiple Parquet Files
VeRL's `data.train_files` accepts a list of parquet files. They are concatenated into a single dataset and shuffled together during training:
```yaml
data.train_files=[gsm8k/train.parquet, math/train.parquet, nq/train.parquet]
data.val_files=[gsm8k/test.parquet, math/test.parquet, nq/test.parquet]
```

### Data Source Field
Each example carries a `data_source` field (e.g., `"gsm8k"`, `"math"`, `"naturalquestions"`). This field is passed to the reward function, enabling per-domain grading.

### Unified Reward Function
VeRL only supports a single `custom_reward_function` config. We've built `src/rewards/unified_reward.py` which dispatches based on `data_source`:
- `gsm8k` / `math` → extract \boxed{} answer, normalize, compare
- `codecontest` → exact match stub (needs real test execution later)
- `naturalquestions` → normalized string match stub

This means a single reward function can handle mixed-domain batches.

## What Needs Building

### 1. Preprocessing for Each Domain
Currently we have preprocessors for:
- [x] GSM8K (`scripts/preprocess_data.py --dataset gsm8k`)
- [x] MATH (`scripts/preprocess_data.py --dataset math`)
- [ ] CodeContest — needs problem parsing, test case extraction
- [ ] NaturalQuestions — needs question/answer extraction

All must produce the same VeRL parquet schema: `prompt`, `data_source`, `ability`, `reward_model.ground_truth`, `extra_info`.

### 2. Domain-Specific Prompt Templates
Different domains may benefit from different system prompts:
- Math: "Please reason step by step, and put your final answer within \boxed{}."
- Code: "Write a solution in Python. Output only the code inside ```python``` blocks."
- QA: "Answer the following question concisely."

Currently, prompt templates are baked into the preprocessing step. This works — each parquet file already contains the domain-appropriate prompt format.

### 3. Response Length Differences
- Math: 512-1024 tokens is usually sufficient
- Code: May need 2048+ tokens for complex solutions
- QA: Often just 50-200 tokens

VeRL uses a single `data.max_response_length` for the entire run. Setting it to the max needed (e.g., 2048) wastes compute on short-response domains. No per-example response length control exists in VeRL currently.

### 4. Real Code Grading
The code reward stub uses exact match. Real code evaluation needs:
- Sandbox execution (subprocess with timeout)
- Test case comparison
- This is complex and may need a separate service

## Open Questions

### Batch Composition
- VeRL shuffles all data together. Should we control the ratio of domains per batch?
- If 80% of data is math and 20% is QA, the model may not see enough QA per batch.
- VeRL doesn't support stratified sampling natively. Options:
  - Oversample minority domains during preprocessing
  - Use equal-sized parquets per domain

### Curriculum Effects
- Does the order matter? (math first, then QA, vs. mixed)
- Sequential training (our current approach) naturally measures this
- Mixed training would provide the "no curriculum" baseline

### Evaluation During Mixed Training
- VeRL's `test_freq` evaluates on `data.val_files` using the reward function
- With mixed val files, the validation score is a blend across domains
- May want per-domain validation scores — would need a custom callback or post-hoc analysis of per-example scores

### KL Regularization
- With mixed domains, KL from the reference model penalizes deviation on *all* domains
- This could reduce forgetting but also slow learning on the target domain
- Worth comparing: mixed with KL vs. mixed without KL vs. sequential

## Recommendation

**For Phase 1: Use sequential training** (already implemented in `scripts/run_grpo_sequential.sh`). This is simpler, gives cleaner measurements of forgetting, and doesn't require solving the batch composition problem.

**For Phase 2: Try mixed-dataset training** as a comparison point. Steps:
1. Preprocess all domains into parquets with consistent schema
2. Use `unified_reward.py` for grading
3. Concatenate parquets via `data.train_files` list
4. Set `max_response_length` to the max across domains
5. Analyze per-domain performance from logged per-example scores

## Related Files
- `src/rewards/unified_reward.py` — Multi-domain reward dispatcher
- `scripts/preprocess_data.py` — Dataset preprocessor (gsm8k, math)
- `scripts/run_grpo_sequential.sh` — Sequential RL pipeline
