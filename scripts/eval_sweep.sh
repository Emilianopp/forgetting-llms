#!/bin/bash
# Eval Sweep: Evaluate training checkpoints on non-math benchmarks to measure forgetting.
#
# Evaluates the base model (step 0) plus all saved checkpoints on 8 benchmarks.
# FSDP checkpoints are merged to HF format before evaluation.
# Works with both GRPO and SFT checkpoints.
#
# Usage:
#   sbatch scripts/eval_sweep.sh <checkpoint_dir> <results_name> [base_model]
#   sbatch scripts/eval_sweep.sh  # defaults to GRPO smoke test dirs
#
# Arguments (all optional, with defaults):
#   $1 = checkpoint directory (default: grpo_full_qwen3_1.7b_gsm8k)
#   $2 = results directory name (default: same as checkpoint dir basename)
#   $3 = base model HF ID (default: Qwen/Qwen3-1.7B)
#
# Examples:
#   sbatch scripts/eval_sweep.sh ~/scratch/forgetting-llms/checkpoints/gt_sft_qwen3_1.7b_gsm8k gt_sft_qwen3_1.7b_gsm8k
#   sbatch --dependency=afterok:12345 scripts/eval_sweep.sh ~/scratch/forgetting-llms/checkpoints/sf_sft_qwen3_1.7b_gsm8k

#SBATCH --job-name=eval-sweep
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -uxo pipefail  # no -e: allow partial completion on timeout

# --- Environment ---
module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Configuration ---
CKPT_DIR=${1:-~/scratch/forgetting-llms/checkpoints/grpo_full_qwen3_1.7b_gsm8k}
RESULTS_NAME=${2:-$(basename "$CKPT_DIR")}
RESULTS_DIR=~/scratch/forgetting-llms/eval_results/$RESULTS_NAME
BASE_MODEL=${3:-"Qwen/Qwen3-1.7B"}
REPO_DIR=$HOME/forgetting-llms

# 10 benchmarks — mostly multiple-choice, 0-shot
BENCHMARKS="arc_challenge,arc_easy,hellaswag,winogrande,piqa,boolq,openbookqa,truthfulqa_mc2,mmlu,ifeval"

mkdir -p slurm_logs
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  Eval Sweep — Forgetting Benchmarks"
echo "========================================="
echo "Checkpoint dir: $CKPT_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Base model:     $BASE_MODEL"
echo "Benchmarks:     $BENCHMARKS"
echo "========================================="

# --- Helper: run lm_eval on a model path ---
run_eval() {
    local model_path="$1"
    local output_dir="$2"
    local label="$3"

    echo ""
    echo "--- Evaluating: $label ---"
    echo "Model: $model_path"
    echo "Output: $output_dir"

    mkdir -p "$output_dir"

    lm_eval \
        --model hf \
        --model_args "pretrained=$model_path,trust_remote_code=True,attn_implementation=sdpa" \
        --tasks "$BENCHMARKS" \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path "$output_dir" \
        --log_samples

    echo "--- Done: $label ---"
}

# --- Step 0: Evaluate base model ---
if [ ! -d "$RESULTS_DIR/base_model" ] || [ -z "$(ls -A "$RESULTS_DIR/base_model" 2>/dev/null)" ]; then
    run_eval "$BASE_MODEL" "$RESULTS_DIR/base_model" "Base Model (step 0)"
else
    echo "Skipping base model eval — results already exist"
fi

# --- Step 1: Find and evaluate all checkpoints ---
CKPT_DIRS=$(find "$CKPT_DIR" -maxdepth 1 -name "global_step_*" -type d | sort -t_ -k3 -n)

if [ -z "$CKPT_DIRS" ]; then
    echo "ERROR: No global_step_* checkpoints found in $CKPT_DIR"
    echo "Contents of checkpoint dir:"
    ls -la "$CKPT_DIR"
    exit 1
fi

echo ""
echo "Found checkpoints:"
echo "$CKPT_DIRS"

for ckpt in $CKPT_DIRS; do
    step_name=$(basename "$ckpt")
    result_dir="$RESULTS_DIR/$step_name"

    # Skip if already evaluated
    if [ -d "$result_dir" ] && [ -n "$(ls -A "$result_dir" 2>/dev/null)" ]; then
        echo "Skipping $step_name — results already exist"
        continue
    fi

    # Determine FSDP shard directory:
    #   GRPO checkpoints: global_step_X/actor/
    #   SFT checkpoints:  global_step_X/ (shards directly in step dir)
    if [ -d "$ckpt/actor" ]; then
        fsdp_dir="$ckpt/actor"
        merged_dir="$ckpt/actor_merged"
    else
        fsdp_dir="$ckpt"
        merged_dir="$ckpt/merged"
    fi

    if [ ! -d "$fsdp_dir" ]; then
        echo "WARNING: $fsdp_dir not found, skipping"
        continue
    fi

    # Merge FSDP checkpoint to HF format (skip if merged dir already exists)
    did_merge=false
    if [ ! -d "$merged_dir" ] || [ -z "$(ls -A "$merged_dir" 2>/dev/null)" ]; then
        echo "Merging FSDP checkpoint: $step_name"
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$fsdp_dir" \
            --target_dir "$merged_dir"
        did_merge=true
    else
        echo "Using existing merged model: $merged_dir"
    fi

    # Run evaluation
    run_eval "$merged_dir" "$result_dir" "$step_name"

    # Clean up merged model only if we created it (preserve pre-existing ones e.g. SDFT)
    if [ "$did_merge" = true ]; then
        echo "Cleaning up merged model: $merged_dir"
        rm -rf "$merged_dir"
    fi
done

# --- Step 2: Run plotting script ---
echo ""
echo "========================================="
echo "  Generating plots"
echo "========================================="

python "$REPO_DIR/scripts/plot_eval_sweep.py" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/plots"

echo ""
echo "========================================="
echo "  Eval Sweep Complete"
echo "========================================="
echo "Results: $RESULTS_DIR"
echo "Plots:   $RESULTS_DIR/plots/"
echo "========================================="
