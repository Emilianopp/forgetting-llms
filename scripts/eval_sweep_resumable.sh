#!/bin/bash
# Resumable Eval Sweep: auto-resubmits on timeout until all checkpoints done.
#
# Same interface as eval_sweep.sh. Uses Slurm --signal to catch approaching
# timeout, gracefully stop, and resubmit with remaining checkpoints.
# The eval_sweep.sh skip logic (non-empty result dirs) handles resume.
#
# Usage:
#   sbatch scripts/eval_sweep_resumable.sh <checkpoint_dir> <results_name> [base_model]

#SBATCH --job-name=eval-sweep
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# --- Arguments ---
CKPT_DIR=${1:-~/scratch/forgetting-llms/checkpoints/grpo_full_qwen3_1.7b_gsm8k}
RESULTS_NAME=${2:-$(basename "$CKPT_DIR")}
BASE_MODEL=${3:-"Qwen/Qwen3-1.7B"}
RESULTS_DIR=~/scratch/forgetting-llms/eval_results/$RESULTS_NAME
REPO_DIR=$HOME/forgetting-llms

# --- Trap: resubmit on approaching timeout (USR1 sent 5min before kill) ---
resubmit() {
    echo ""
    echo "========================================="
    echo "  TIMEOUT approaching — resubmitting to continue"
    echo "========================================="
    NEW_JOB=$(sbatch --parsable "$REPO_DIR/scripts/eval_sweep_resumable.sh" "$CKPT_DIR" "$RESULTS_NAME" "$BASE_MODEL")
    echo "  Resubmitted as job $NEW_JOB"
    echo "========================================="
    exit 0
}
trap 'resubmit' USR1

# --- Environment ---
module load python/3.10
source $HOME/envs/forgetting/bin/activate
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

mkdir -p slurm_logs

# --- Run the actual eval sweep (in background so trap works) ---
bash "$REPO_DIR/scripts/eval_sweep.sh" "$CKPT_DIR" "$RESULTS_NAME" "$BASE_MODEL" &
EVAL_PID=$!
wait $EVAL_PID
EXIT_CODE=$?

# --- If we get here, eval finished (no timeout) ---
# Check completeness
TOTAL_CKPTS=$(find "$CKPT_DIR" -maxdepth 1 -name "global_step_*" -type d 2>/dev/null | wc -l)
TOTAL_NEEDED=$((TOTAL_CKPTS + 1))  # +1 for base_model

DONE_COUNT=0
for d in "$RESULTS_DIR"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [ "$name" = "plots" ] && continue
    if [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
        DONE_COUNT=$((DONE_COUNT + 1))
    fi
done

echo ""
echo "========================================="
echo "  Progress: $DONE_COUNT / $TOTAL_NEEDED checkpoints evaluated"
echo "========================================="

if [ "$DONE_COUNT" -lt "$TOTAL_NEEDED" ] && [ "$EXIT_CODE" -ne 0 ]; then
    echo "  Eval exited with code $EXIT_CODE, $((TOTAL_NEEDED - DONE_COUNT)) remaining — resubmitting..."
    NEW_JOB=$(sbatch --parsable "$REPO_DIR/scripts/eval_sweep_resumable.sh" "$CKPT_DIR" "$RESULTS_NAME" "$BASE_MODEL")
    echo "  Resubmitted as job $NEW_JOB"
else
    echo "  All checkpoints evaluated!"
fi
echo "========================================="
