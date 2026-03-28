#!/bin/bash
# Submit LoRA vs full fine-tune SFT ablations for GT-SFT and SF-SFT.
#
# Usage:
#   bash scripts/launch_sft_lora_ablation.sh [model] [lora_rank]

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-1.7B}"
LORA_RANK_VALUE="${2:-64}"
DATASETS=(gsm8k math triviaqa)

for dataset in "${DATASETS[@]}"; do
    sbatch scripts/run_sft.sh "$dataset" "$MODEL" "gt_sft_full_${dataset}" gt
    LORA_RANK=$LORA_RANK_VALUE sbatch scripts/run_sft.sh "$dataset" "$MODEL" "gt_sft_lora_${dataset}" gt
    sbatch scripts/run_sft.sh "$dataset" "$MODEL" "sf_sft_full_${dataset}" sf
    LORA_RANK=$LORA_RANK_VALUE sbatch scripts/run_sft.sh "$dataset" "$MODEL" "sf_sft_lora_${dataset}" sf
done
