# How Post-Training Shapes and Breaks Language Models: A Systematic Study of Forgetting Across Methods, Domains, and Scale

## Motivation

Post-training (SFT, RLHF, DPO, distillation, etc.) is how base language models become useful. But every post-training method induces forgetting -- capabilities the base model had that the final model loses. Despite dozens of papers on individual methods, **no systematic study compares how different post-training approaches affect forgetting under controlled conditions**.

Key unanswered questions:
- Does SFT on distilled data forget differently than SFT on ground truth?
- Does RL-based post-training forget different capabilities than SFT-based?
- Does cross-family distillation cause more forgetting than same-family?
- Does self-distillation preserve more base model knowledge?
- How do these differences change across domains and model scale?
- **When we post-train an already safety-aligned model, which methods break safety and how badly?**

## Research Questions

1. **What does each post-training method forget?** Not just "how much" but "what kind" -- factual recall, reasoning, code, safety, multilingual ability?
2. **How does the source of training signal affect forgetting?** Ground truth vs. same-family teacher vs. cross-family teacher vs. self-generated data vs. RL reward.
3. **Does the domain of post-training determine what is forgotten?** Does math post-training preserve reasoning but damage factual recall? Does code training affect language generation?
4. **How does forgetting scale?** Do larger models forget less, more, or differently?
5. **How does each method shape the output policy?** Beyond benchmarks -- output diversity, length, calibration, style.
6. **Does post-training break safety alignment?** Starting from a safety-tuned model, which methods degrade safety the most? Is safety forgetting correlated with capability forgetting, or independent?

## Experimental Design

### Post-Training Methods (7 conditions)

| ID | Method | Description | Data Source | Online/Offline |
|----|--------|-------------|-------------|----------------|
| `GT-SFT` | SFT on ground truth | Standard supervised fine-tuning on verified data | Human-annotated / verified solutions | Offline |
| `SF-SFT` | SFT same-family distill | SFT on synthetic data from larger model, same architecture family | e.g., Qwen2.5-72B-Instruct outputs | Offline |
| `CF-SFT` | SFT cross-family distill | SFT on synthetic data from larger model, different architecture family | e.g., Llama-3.1-70B-Instruct outputs | Offline |
| `SELF` | Self-distillation | Model teaches itself via SPIN or SDFT | Own generations + ground truth signal | Online (iterative) |
| `ON-RL` | Online RL (from experience) | GRPO with verifiable rewards | Prompts + deterministic verifier | Online |
| `OFF-RL` | Off-policy RL (DPO) | Direct Preference Optimization on pre-collected pairs | Static preference dataset | Offline |
| `PI` | Pi-distill | Privileged information distillation | Task env + privileged signal (teacher actions) | Online (joint) |

### Domains (3 conditions) — via GEM Environments

We use the [GEM environment suite](https://github.com/axon-rl/gem) as the unified backbone for training and evaluation across domains. GEM provides a standardized Gym-like API with async vectorized execution, built-in reward signals, and integrations with RL training frameworks (OpenRLHF, Verl, ROLL).

Using GEM gives us:
- **Consistent environment interface** across all domains and methods
- **Built-in verifiers** for RL reward signals (answer correctness, unit test execution)
- **Native RL algorithm support** (GRPO, REINFORCE, PPO) for the ON-RL condition
- **Tool integration** (Python executor, search) for agentic tasks
- **Async vectorized execution** for efficient data collection and training

| Domain | GEM Environments | Verifier (for RL) | Held-out Evaluation |
|--------|-----------------|-------------------|---------------------|
| **Math Reasoning** | `Math12K`, `DeepScaleR40K` | Answer correctness check (built-in) | GSM8K test, MATH test, MINERVA |
| **Code Generation** | `CodeContest`, `Taco8k` | Unit test execution (Python executor tool) | HumanEval, HumanEval+, MBPP test, LiveCodeBench |
| **QA / Reasoning** | `NaturalQuestions`, `HotpotQA`, `ReasoningGym (ARC)` | Answer match / reward model | TriviaQA, MMLU (held-out), BBH |

**How GEM maps to each post-training method:**

| Method | How GEM is used |
|--------|----------------|
| `GT-SFT` | Use GEM environment ground-truth answers as SFT targets |
| `SF-SFT` | Generate synthetic data by running same-family teacher through GEM environments, SFT student on outputs |
| `CF-SFT` | Same as SF-SFT but with cross-family teacher |
| `SELF` | Student generates rollouts in GEM environments, self-distills using ground-truth signal |
| `ON-RL` | Train directly in GEM environments using GRPO/REINFORCE with built-in reward |
| `OFF-RL` | Collect preference pairs from GEM rollouts (correct vs. incorrect), then DPO offline |
| `PI` | Use teacher actions in GEM as privileged information, joint teacher-student training |

### Base Models

Primary experiments on **Qwen2.5** family for scale analysis:
- Qwen2.5-1.5B (low compute, fast iteration)
- Qwen2.5-3B (primary experimental scale)
- Qwen2.5-7B (validation scale)
- Qwen2.5-14B (stretch goal, if compute allows)

Teachers for distillation:
- Same-family: Qwen2.5-72B-Instruct
- Cross-family: Llama-3.1-70B-Instruct (or DeepSeek-V3)

### Starting Points (2 conditions)

Each experiment is run from **two starting points** to isolate the safety forgetting question:

| ID | Starting Point | Purpose |
|----|---------------|---------|
| `BASE` | Qwen2.5-{size} (base, no alignment) | Measures general capability forgetting from post-training |
| `SAFE` | Qwen2.5-{size}-Instruct (safety-aligned) | Measures whether post-training breaks existing safety alignment |

This doubles the experimental matrix but answers a critical question: **is safety forgetting an inherent cost of further post-training, or do some methods preserve safety while others destroy it?**

Key hypotheses (from literature):
- Safety alignment is shallow (~3% of parameters, first few tokens) -- so it may be especially fragile
- Even benign SFT on 10 examples can jailbreak aligned models (Qi et al., ICLR 2024)
- But nobody has compared which post-training *methods* are worst for safety
- RL methods might preserve safety better (they update the policy more gently) or worse (reward hacking can cause emergent misalignment)

### Matched Conditions

To ensure fair comparison:
- **Matched training compute**: Each method gets the same GPU-hour budget per experiment
- **Matched data volume**: Where possible, same number of training examples (for SFT variants and OFF-RL)
- **Matched prompts**: All methods train on the same prompt set per domain
- **Same base model checkpoint**: All experiments within a starting-point condition start from the same checkpoint

## Evaluation Framework

### A. Forgetting Profile (capabilities lost)

Measured on a fixed battery of benchmarks **before and after** post-training:

| Category | Benchmarks | What it measures |
|----------|-----------|-----------------|
| Factual Knowledge | MMLU, TriviaQA, NaturalQuestions | World knowledge retention |
| Reasoning | ARC-Challenge, BBH, GPQA | Logical and scientific reasoning |
| Math | GSM8K, MATH (held-out if domain = math) | Mathematical ability |
| Code | HumanEval, MBPP (held-out if domain = code) | Programming ability |
| Language Understanding | HellaSwag, WinoGrande, PIQA | Commonsense and language |
| Multilingual | MGSM, XWinograd | Cross-lingual transfer |
| Safety | TruthfulQA, XSTest, HarmBench, WildGuard | Safety alignment / honesty (see Section E for expanded safety eval) |
| Instruction Following | IFEval (held-out if domain = IF) | Format compliance |

For each benchmark, report:
- Absolute performance (pre and post)
- Delta (forgetting = negative, gain = positive)
- Sample-level forgetting rate (% of correct->incorrect transitions, per Harmon et al. 2025)

### B. Target Task Performance

Standard metrics for the training domain:
- Math: accuracy on test sets
- Code: pass@1, pass@10
- Instruction following: win rate, length-controlled win rate

### C. Policy Analysis (how the model changed)

Beyond benchmark scores, characterize the output distribution:
- **Diversity**: Self-BLEU, distinct n-grams across generations
- **Length distribution**: Mean/median/std of response length
- **Calibration**: ECE on multiple-choice tasks
- **Refusal rate**: % of prompts refused (safety-relevant)
- **Repetition**: n-gram repetition rate
- **Format compliance**: JSON accuracy, instruction adherence

### E. Safety Degradation Analysis (SAFE starting point only)

This section applies only to experiments starting from the safety-aligned model (Qwen2.5-{size}-Instruct).

**Safety Benchmarks (multi-dimensional):**

| Benchmark | What it measures | Metric |
|-----------|-----------------|--------|
| HarmBench | Resistance to adversarial attacks (standard + GCG + AutoDAN) | Attack Success Rate (ASR) |
| XSTest | Over-refusal vs. under-refusal balance | Safe completion rate + exaggerated safety rate |
| TruthfulQA | Honesty / resistance to common misconceptions | % truthful + informative |
| WildGuard | Safety on realistic in-the-wild prompts | Safety score |
| Do-Not-Answer | Refusal on clearly harmful prompts | Refusal rate |
| SaladBench | Multi-dimensional safety across attack types | Composite safety score |

**Safety Analysis Dimensions:**

1. **Safety erosion rate**: For each method x domain, how much does ASR increase from the SAFE baseline?
2. **Safety-capability trade-off**: Plot safety score vs. target task performance. Which methods give the best Pareto front?
3. **Category-specific safety breakdown**: Does math training break safety on bioweapons topics but not harassment? Does code training make the model more willing to generate malware?
4. **Refusal calibration shift**: Does the model become broadly less cautious (under-refusal) or does safety break only on specific categories?
5. **Safety vs. helpfulness**: Does the model become more helpful (less over-refusal) as a side effect, or does it just become unsafe?
6. **Adversarial robustness**: Test with GCG, AutoDAN, and jailbreak prompts. Does post-training make the model more or less susceptible to adversarial attacks?

**Key comparisons (SAFE starting point):**
- Which of the 7 methods breaks safety the least? The most?
- Does the training domain matter? (e.g., code post-training vs. math post-training vs. instruction following)
- Is safety degradation correlated with capability forgetting? Or can you have capability forgetting without safety loss (and vice versa)?
- Does scale help? Are larger safety-aligned models more robust to safety degradation from further post-training?

### F. Internal Representation Analysis (if compute allows)

- **CKA similarity** between base and post-trained model at each layer
- **Linear probes** for factual knowledge at each layer (before/after)
- **Singular value analysis** of weight deltas
- **Refusal direction analysis**: Extract the "refusal direction" (per Arditi et al., NeurIPS 2024) from the SAFE model and measure how much each post-training method attenuates it

## Experimental Matrix

Total experiments: 7 methods x 3 domains x 2 starting points x 2 scales (minimum) = **84 training runs**

Priority ordering:
1. **Phase 1** (core forgetting): 7 methods x 3 domains x BASE starting point x Qwen2.5-3B = 21 runs
2. **Phase 2** (safety): 7 methods x 3 domains x SAFE starting point x Qwen2.5-3B = 21 runs
3. **Phase 3** (scale): Repeat top findings from Phase 1+2 at 1.5B and 7B = ~28 runs
4. **Phase 4** (deep analysis): Representation analysis, refusal direction analysis, policy analysis on selected runs

## Expected Contributions

1. **First controlled comparison** of 7 post-training methods measuring forgetting on the same base model, benchmarks, and compute budget
2. **Forgetting profiles** showing what each method preserves and destroys, broken down by capability type
3. **Cross-domain analysis** revealing whether the training domain determines forgetting patterns
4. **Scale analysis** showing how forgetting dynamics change with model size
5. **Policy characterization** going beyond benchmarks to show how each method shapes generation behavior
6. **Safety fragility analysis**: First systematic study of how different post-training methods erode safety alignment, which methods are safest to apply on top of aligned models, and whether safety degradation is correlated with or independent from capability forgetting
7. **Practical recommendations** for practitioners choosing post-training methods, including safety-aware guidance

## Key Related Work

**Forgetting:**
- Kalajdzievski (2024) -- Scaling laws for forgetting during SFT (single model, LoRA only)
- Bethune et al. (ICML 2025) -- Scaling laws for forgetting with pretraining data injection
- Harmon et al. (2025) -- Mapping post-training forgetting at scale (30 models, but SFT/RL lumped together)
- Lin et al. (EMNLP 2024) -- Alignment tax of RLHF (PPO/DPO/RSF on 3B only)
- Chu et al. (ICML 2025) -- SFT memorizes, RL generalizes (learning behavior, not forgetting)
- Fernando et al. (2024) -- Sequential SFT+DPO is suboptimal (theoretical + practical)
- Li et al. (ICLR 2025) -- Spurious forgetting in continual learning

**Safety fragility:**
- Qi et al. (ICLR 2024) -- Fine-tuning aligned LLMs compromises safety with just 10 examples
- Qi et al. (ICLR 2025) -- Safety alignment is only a few tokens deep
- Wei et al. (ICML 2024) -- Safety-critical regions are ~3% of parameters, ~2.5% of rank
- Arditi et al. (NeurIPS 2024) -- Refusal is mediated by a single direction in activation space
- Anthropic (2025) -- Emergent misalignment from reward hacking in production RL
- Bach et al. (2025) -- Position-dependent gradient weakening explains shallow safety alignment

## Repo Structure (Planned)

```
forgetting-llms/
  PROJECT.md              # This file
  configs/
    methods/              # Training configs per method (gt_sft.yaml, on_rl.yaml, etc.)
    domains/              # Domain-specific configs (math.yaml, code.yaml, qa.yaml)
    models/               # Model configs per scale (qwen_1.5b.yaml, qwen_3b.yaml, etc.)
  src/
    environments/         # GEM environment wrappers and domain setup
    training/
      sft.py              # Unified SFT trainer (GT-SFT, SF-SFT, CF-SFT)
      online_rl.py        # GRPO/REINFORCE via GEM (ON-RL)
      offline_rl.py       # DPO trainer (OFF-RL)
      self_distill.py     # SPIN/SDFT (SELF)
      pi_distill.py       # Privileged information distillation (PI)
    data/
      collect_teacher.py  # Generate teacher rollouts through GEM environments
      collect_prefs.py    # Collect preference pairs from GEM rollouts
    evaluation/
      forgetting.py       # Forgetting profile evaluation pipeline
      safety.py           # Safety degradation evaluation
      policy_analysis.py  # Output distribution analysis
    analysis/
      forgetting_plots.py # Visualization and comparison
      safety_plots.py     # Safety analysis visualization
  results/                # Raw results (gitignored)
  notebooks/              # Analysis notebooks
  scripts/                # Slurm/launcher scripts
```

## Open Questions / Decisions Needed

- [ ] Exact compute budget per run (GPU-hours)
- [ ] Whether to use LoRA or full fine-tuning (or both as a variable)
- [ ] Which GEM environments to use per domain (exact subset and train/test splits)
- [ ] Whether to use GEM's built-in RL training integrations (Verl, OpenRLHF) or custom training loops
- [ ] Reward model choice for QA/reasoning RL (GEM has answer match, but may need something richer)
- [ ] Whether pi-distill is feasible at our scale (method is very new, Feb 2026)
- [ ] Whether to include a "no post-training" baseline (base model evaluated directly)
- [ ] Number of seeds per experiment for statistical significance
- [ ] How to generate OFF-RL preference pairs: from GEM rollouts (correct vs. incorrect) or external datasets?
