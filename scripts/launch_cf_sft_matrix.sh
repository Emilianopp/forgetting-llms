#!/bin/bash
# Generate cross-family trajectories and submit CF-SFT runs for all datasets.
#
# Usage:
#   bash scripts/launch_cf_sft_matrix.sh [teacher_model] [student_model]

set -euo pipefail

TEACHER_MODEL="${1:-meta-llama/Llama-3.1-70B-Instruct}"
STUDENT_MODEL="${2:-Qwen/Qwen3-1.7B}"
DATASETS=(gsm8k math triviaqa)

for dataset in "${DATASETS[@]}"; do
    TRAJ_JOB=$(sbatch --parsable scripts/generate_cross_family_trajectories.sh "$TEACHER_MODEL" "$dataset" 4)
    echo "Trajectories $dataset: $TRAJ_JOB"
    SFT_JOB=$(sbatch --parsable --dependency=afterok:$TRAJ_JOB \
        scripts/run_sft.sh "$dataset" "$STUDENT_MODEL" "cf_sft_qwen3_1.7b_${dataset}" cf)
    echo "CF-SFT $dataset: $SFT_JOB"
done
