#!/bin/bash
# Launch all Phase 1 experiments (BASE starting point, Qwen2.5-3B)
# Usage: bash scripts/launch_phase1.sh

set -e

METHODS=(gt_sft sf_sft cf_sft self on_rl off_rl pi)
DOMAINS=(math code qa)
STARTING_POINT="base"
MODEL_SCALE="qwen_3b"

echo "=== Launching Phase 1: ${#METHODS[@]} methods × ${#DOMAINS[@]} domains ==="
echo "Starting point: ${STARTING_POINT}"
echo "Model scale: ${MODEL_SCALE}"
echo ""

for method in "${METHODS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        run_name="${method}_${domain}_${STARTING_POINT}_${MODEL_SCALE}"
        echo "Submitting: ${run_name}"
        sbatch --job-name="forgetting-${run_name}" \
            scripts/train.sh ${method} ${domain} ${STARTING_POINT} ${MODEL_SCALE}
    done
done

echo ""
echo "=== Submitted $((${#METHODS[@]} * ${#DOMAINS[@]})) jobs ==="
echo "Monitor with: squeue -u \$USER"
