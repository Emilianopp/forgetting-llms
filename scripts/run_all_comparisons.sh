#!/bin/bash
# Orchestrate multi-dataset SFT vs GRPO forgetting comparison.
#
# Preprocesses all datasets, submits training + eval jobs, prints job IDs.
#
# Usage:
#   bash scripts/run_all_comparisons.sh [model]
#
# Default model: Qwen/Qwen3-1.7B

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-1.7B}"
MODEL_SHORT="qwen3_1.7b"

REPO_DIR=$HOME/forgetting-llms
DATA_BASE=~/scratch/forgetting-llms/data
CKPT_BASE=~/scratch/forgetting-llms/checkpoints

DATASETS=(gsm8k math triviaqa)

echo "========================================="
echo "  Multi-Dataset SFT vs GRPO Comparison"
echo "========================================="
echo "Model: $MODEL"
echo "Datasets: ${DATASETS[*]}"
echo ""

# --- Step 1: Preprocess all datasets (both formats) ---
echo "--- Preprocessing ---"
for DS in "${DATASETS[@]}"; do
    if [ ! -f "$DATA_BASE/$DS/train.parquet" ]; then
        echo "Preprocessing $DS (grpo)..."
        python "$REPO_DIR/scripts/preprocess_data.py" --dataset "$DS" --format grpo
    else
        echo "Skip $DS (grpo): already exists"
    fi

    if [ ! -f "$DATA_BASE/${DS}_sft/train.parquet" ]; then
        echo "Preprocessing $DS (sft)..."
        python "$REPO_DIR/scripts/preprocess_data.py" --dataset "$DS" --format sft
    else
        echo "Skip $DS (sft): already exists"
    fi
done
echo ""

# --- Step 2: Submit training jobs ---
echo "--- Submitting training jobs ---"

declare -A SFT_JOBS
declare -A GRPO_JOBS

for DS in "${DATASETS[@]}"; do
    EXP_SFT="gt_sft_${MODEL_SHORT}_${DS}"
    EXP_GRPO="grpo_${MODEL_SHORT}_${DS}"

    # Check if SFT checkpoints already exist
    if [ -d "$CKPT_BASE/$EXP_SFT" ] && ls "$CKPT_BASE/$EXP_SFT"/global_step_* >/dev/null 2>&1; then
        echo "Skip SFT $DS: checkpoints exist at $CKPT_BASE/$EXP_SFT"
        SFT_JOBS[$DS]=""
    else
        JOB=$(sbatch --parsable "$REPO_DIR/scripts/run_sft.sh" "$DS" "$MODEL" "$EXP_SFT" gt)
        echo "SFT $DS: job $JOB"
        SFT_JOBS[$DS]=$JOB
    fi

    # Check if GRPO checkpoints already exist
    if [ -d "$CKPT_BASE/$EXP_GRPO" ] && ls "$CKPT_BASE/$EXP_GRPO"/global_step_* >/dev/null 2>&1; then
        echo "Skip GRPO $DS: checkpoints exist at $CKPT_BASE/$EXP_GRPO"
        GRPO_JOBS[$DS]=""
    else
        JOB=$(sbatch --parsable "$REPO_DIR/scripts/run_grpo_sequential.sh" "$DS" "$MODEL" "$EXP_GRPO")
        echo "GRPO $DS: job $JOB"
        GRPO_JOBS[$DS]=$JOB
    fi
done
echo ""

# --- Step 3: Submit eval sweeps (chained after training) ---
echo "--- Submitting eval sweeps ---"

for DS in "${DATASETS[@]}"; do
    EXP_SFT="gt_sft_${MODEL_SHORT}_${DS}"
    EXP_GRPO="grpo_${MODEL_SHORT}_${DS}"

    # SFT eval
    DEP=""
    if [ -n "${SFT_JOBS[$DS]}" ]; then
        DEP="--dependency=afterok:${SFT_JOBS[$DS]}"
    fi
    EVAL_JOB=$(sbatch --parsable $DEP "$REPO_DIR/scripts/eval_sweep.sh" \
        "$CKPT_BASE/$EXP_SFT" "$EXP_SFT" "$MODEL")
    echo "Eval SFT $DS: job $EVAL_JOB ${DEP:+(after ${SFT_JOBS[$DS]})}"

    # GRPO eval
    DEP=""
    if [ -n "${GRPO_JOBS[$DS]}" ]; then
        DEP="--dependency=afterok:${GRPO_JOBS[$DS]}"
    fi
    EVAL_JOB=$(sbatch --parsable $DEP "$REPO_DIR/scripts/eval_sweep.sh" \
        "$CKPT_BASE/$EXP_GRPO" "$EXP_GRPO" "$MODEL")
    echo "Eval GRPO $DS: job $EVAL_JOB ${DEP:+(after ${GRPO_JOBS[$DS]})}"
done
echo ""

echo "========================================="
echo "  All jobs submitted. Monitor with:"
echo "    squeue -u \$USER"
echo "========================================="
echo ""
echo "When all evals complete, generate cross-dataset plots:"
echo "  python scripts/plot_comparison.py --datasets \\"
for DS in "${DATASETS[@]}"; do
    EXP_SFT="gt_sft_${MODEL_SHORT}_${DS}"
    EXP_GRPO="grpo_${MODEL_SHORT}_${DS}"
    EVAL_BASE=~/scratch/forgetting-llms/eval_results
    echo "    $DS=$EVAL_BASE/$EXP_GRPO,$EVAL_BASE/$EXP_SFT \\"
done
echo "    --output_dir ~/scratch/forgetting-llms/eval_results/cross_dataset_comparison"
