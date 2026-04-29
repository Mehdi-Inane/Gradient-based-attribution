#!/bin/bash
# run_all.sh
#
# Submits the seeded gradient deviation job array, then automatically
# schedules the averaging step to run once ALL array tasks succeed.
#
# Usage:
#   bash run_all.sh [n_seeds] [dataset]
#
# Defaults:
#   n_seeds = 5
#   dataset = cifar100
#
# Examples:
#   bash run_all.sh           # 5 seeds, cifar100
#   bash run_all.sh 3         # 3 seeds, cifar100
#   bash run_all.sh 5 imagenet

set -euo pipefail

N_SEEDS=${1:-5}
DATASET=${2:-cifar100}
ARRAY_END=$((N_SEEDS - 1))   # 0-indexed, so 5 seeds → --array=0-4

SCRATCH_DIR="/network/scratch/a/ahmedm/attribution_training_runs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs

echo "========================================"
echo " Gradient Deviation Multi-Seed Pipeline"
echo "========================================"
echo " Dataset  : $DATASET"
echo " Seeds    : 0 .. $ARRAY_END  ($N_SEEDS total)"
echo " Scratch  : $SCRATCH_DIR"
echo "========================================"

# ── Step 1: Submit the training array ────────────────────────────────────────
TRAIN_JOB=$(sbatch \
    --array=0-${ARRAY_END} \
    --job-name=grad_dev_seeds \
    --output=logs/job_output_%A_%a.txt \
    --error=logs/job_error_%A_%a.txt \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --time=24:00:00 \
    --mem=48Gb \
    --export=ALL,DATASET=$DATASET \
    "$SCRIPT_DIR/train_multiple.sh" \
    | awk '{print $4}')

echo "Submitted training array job: $TRAIN_JOB"
echo "  → Tasks: ${TRAIN_JOB}_[0-${ARRAY_END}]"

# ── Step 2: Submit averaging job, depends on ALL array tasks succeeding ───────
# afterok on an array job ID waits for every element to finish successfully.
AVG_JOB=$(sbatch \
    --job-name=avg_grad_scores \
    --output=logs/avg_output_%j.txt \
    --error=logs/avg_error_%j.txt \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time=00:30:00 \
    --mem=16Gb \
    --dependency=afterok:${TRAIN_JOB} \
    --wrap="
        module --force purge
        module load miniconda/3
        conda activate unlearning
        echo 'All training jobs done. Running averaging...'
        python $SCRIPT_DIR/average_grad_scores.py \
            --dataset $DATASET \
            --n_seeds $N_SEEDS \
            --scratch_dir $SCRATCH_DIR
        echo 'Averaging complete.'
    ")

echo "Submitted averaging job   : $AVG_JOB (runs after $TRAIN_JOB completes)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/job_output_${TRAIN_JOB}_0.txt"
echo ""
echo "Results will appear in:"
echo "  $SCRATCH_DIR/batch_gradient_deviation_scores_${DATASET}_baseline_topk0_grad_avg.npy"
echo "  $SCRATCH_DIR/batch_gradient_deviation_scores_${DATASET}_baseline_topk0_grad_std.npy"