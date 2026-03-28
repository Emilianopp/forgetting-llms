#!/bin/bash
# Submit the six sequential order runs for OLMo 3 7B.
#
# This submits the first-stage jobs only. Submit the second stage from the chosen
# first-stage checkpoint once the first-stage runs finish.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/require_prime_only.sh"

MODEL_PATH="${MODEL_PATH:-$HOME/scratch/olmo3_7B-Instruct}"

sbatch scripts/run_grpo_sequential.sh gsm8k "$MODEL_PATH" olmo3_7b_seq_gsm8k
sbatch scripts/run_grpo_sequential.sh math "$MODEL_PATH" olmo3_7b_seq_math
sbatch scripts/run_grpo_sequential.sh triviaqa "$MODEL_PATH" olmo3_7b_seq_triviaqa

cat <<'EOF'
Next-stage run names to submit after the first-stage checkpoints are ready:
  olmo3_7b_seq_gsm8k_then_math
  olmo3_7b_seq_math_then_gsm8k
  olmo3_7b_seq_gsm8k_then_triviaqa
  olmo3_7b_seq_triviaqa_then_gsm8k
  olmo3_7b_seq_math_then_triviaqa
  olmo3_7b_seq_triviaqa_then_math

Example continuation:
  bash scripts/submit_grpo_continue_latest.sh olmo3_7b_seq_gsm8k dataset math olmo3_7b_seq_gsm8k_then_math
EOF
