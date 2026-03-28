#!/bin/bash
# Teacher Trajectory Generation: vLLM batch inference on 2x A100 80GB
#
# Generates SFT traces for training.
# For verifiable datasets, this uses adaptive correctness-gated rounds.
# For Dolci prompt datasets, this regenerates traces from the released prompt
# messages with the requested model.
#
# Usage:
#   sbatch scripts/generate_trajectories.sh [model] [dataset] [n_samples]
#
# Arguments (all optional):
#   $1 = teacher model (default: Qwen/Qwen3-32B)
#   $2 = dataset (default: gsm8k)
#   $3 = samples per round (default: 4)
#
# Examples:
#   sbatch scripts/generate_trajectories.sh
#   sbatch scripts/generate_trajectories.sh Qwen/Qwen3-32B gsm8k 8
#   sbatch scripts/generate_trajectories.sh ~/scratch/forgetting-llms/models/allenai__Olmo-3-7B-Instruct dolci_think_sft_7b 1
#   sbatch scripts/generate_trajectories.sh ~/scratch/forgetting-llms/models/allenai__Olmo-3-7B-Instruct synthetic2_sft_verified 1

#SBATCH --job-name=gen-trajectories
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
    set -x
fi

# --- Arguments ---
MODEL=${1:-"Qwen/Qwen3-32B"}
DATASET=${2:-"gsm8k"}
N_SAMPLES=${3:-4}

# --- Environment ---
module load python/3.10
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    :
elif [ -n "${VENV_DIR:-}" ] && [ -x "${VENV_DIR/#\~/$HOME}/bin/python" ]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR/#\~/$HOME}/bin/activate"
elif [ -f "$HOME/scratch/forgetting-llms/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/forgetting-llms/.venv/bin/activate"
elif [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$REPO_DIR/.venv/bin/activate"
else
    # shellcheck disable=SC1091
    source $HOME/envs/forgetting/bin/activate
fi
export HF_HOME=~/scratch/huggingface
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# --- Paths ---
# shellcheck disable=SC1090
source "$SCRIPT_DIR/load_hf_auth.sh"
OUTPUT_DIR=~/scratch/forgetting-llms/data/${DATASET}_sf_sft

mkdir -p slurm_logs

echo "========================================="
echo "  Teacher Trajectory Generation"
echo "========================================="
echo "Teacher:   $MODEL"
echo "Dataset:   $DATASET"
echo "Samples/round: $N_SAMPLES"
echo "Output:    $OUTPUT_DIR"
echo "GPUs:      2x A100 80GB (TP=2)"
echo "========================================="

case "$DATASET" in
    dolci_think_sft_7b)
        DOLCI_CMD=(
            python "$REPO_DIR/scripts/import_dolci_sft.py"
            --hf-dataset allenai/Dolci-Think-SFT-7B
            --generator-model "$MODEL"
            --output-dir "$OUTPUT_DIR"
            --tensor-parallel-size 2
            --max-model-len "${MAX_MODEL_LEN:-8192}"
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
            --max-tokens "${MAX_TOKENS:-2048}"
            --temperature "${TEMPERATURE:-1.0}"
            --top-p "${TOP_P:-1.0}"
            --samples-per-prompt "$N_SAMPLES"
            --chunk-size "${CHUNK_SIZE:-128}"
        )
        if [[ -n "${MAX_SAMPLES:-}" ]]; then
            DOLCI_CMD+=(--max-samples "$MAX_SAMPLES")
        fi
        "${DOLCI_CMD[@]}"
        ;;
    dolci_think_sft_32b)
        DOLCI_CMD=(
            python "$REPO_DIR/scripts/import_dolci_sft.py"
            --hf-dataset allenai/Dolci-Think-SFT-32B
            --generator-model "$MODEL"
            --output-dir "$OUTPUT_DIR"
            --tensor-parallel-size 2
            --max-model-len "${MAX_MODEL_LEN:-8192}"
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
            --max-tokens "${MAX_TOKENS:-2048}"
            --temperature "${TEMPERATURE:-1.0}"
            --top-p "${TOP_P:-1.0}"
            --samples-per-prompt "$N_SAMPLES"
            --chunk-size "${CHUNK_SIZE:-128}"
        )
        if [[ -n "${MAX_SAMPLES:-}" ]]; then
            DOLCI_CMD+=(--max-samples "$MAX_SAMPLES")
        fi
        "${DOLCI_CMD[@]}"
        ;;
    synthetic2_sft_verified)
        MESSAGE_DATASET_CMD=(
            python "$REPO_DIR/scripts/import_dolci_sft.py"
            --hf-dataset PrimeIntellect/SYNTHETIC-2-SFT-verified
            --generator-model "$MODEL"
            --output-dir "$OUTPUT_DIR"
            --tensor-parallel-size 2
            --max-model-len "${MAX_MODEL_LEN:-8192}"
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
            --max-tokens "${MAX_TOKENS:-2048}"
            --temperature "${TEMPERATURE:-1.0}"
            --top-p "${TOP_P:-1.0}"
            --samples-per-prompt "$N_SAMPLES"
            --chunk-size "${CHUNK_SIZE:-128}"
        )
        if [[ -n "${MAX_SAMPLES:-}" ]]; then
            MESSAGE_DATASET_CMD+=(--max-samples "$MAX_SAMPLES")
        fi
        "${MESSAGE_DATASET_CMD[@]}"
        ;;
    *)
        python "$REPO_DIR/src/data/generate_teacher_solutions.py" \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --samples_per_round "$N_SAMPLES" \
            --max_total_samples "${MAX_TOTAL_SAMPLES:-16}" \
            --target_correct_per_question "${TARGET_CORRECT_PER_QUESTION:-2}" \
            --min_correct_per_question "${MIN_CORRECT_PER_QUESTION:-2}" \
            --solutions_per_question "${SOLUTIONS_PER_QUESTION:-2}" \
            --tensor_parallel_size 2 \
            --answer_format "${ANSWER_FORMAT:-dataset_default}" \
            --max_model_len "${MAX_MODEL_LEN:-8192}" \
            --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
            --max_tokens "${MAX_TOKENS:-2048}" \
            --chunk_size "${CHUNK_SIZE:-500}" \
            --temperature "${TEMPERATURE:-0.7}" \
            --top_p "${TOP_P:-0.9}" \
            --output_dir "$OUTPUT_DIR"
        ;;
esac

echo "========================================="
echo "  Trajectory Generation Complete"
echo "========================================="
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next: submit SF-SFT training:"
echo "  sbatch scripts/run_sft.sh $DATASET Qwen/Qwen3-1.7B sf_sft_qwen3_1.7b_${DATASET} sf"
echo "========================================="
