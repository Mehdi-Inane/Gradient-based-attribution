#!/bin/bash
#SBATCH --job-name=tracein_seeds
#SBATCH --output=logs/job_output_%A_%a.txt   # %A=job_id, %a=array_index
#SBATCH --error=logs/job_error_%A_%a.txt
#SBATCH --array=0-14                          # 15 seeds: 0 through 14
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48Gb

DATASET="cifar100"
SEED=$SLURM_ARRAY_TASK_ID

echo "================================================"
echo "Job Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Seed              : $SEED"
echo "Dataset           : $DATASET"
echo "================================================"

mkdir -p logs

# ── Data preparation ─────────────────────────────────────────────────────────
echo "Extracting $DATASET dataset to SLURM_TMPDIR..."

if [ "$DATASET" == "cifar100" ]; then
    tar -xzf /network/datasets/cifar100/cifar-100-python.tar.gz -C $SLURM_TMPDIR
elif [ "$DATASET" == "imagenet" ]; then
    tar -xf /network/datasets/imagenet/imagenet.tar -C $SLURM_TMPDIR
else
    echo "Error: Unknown dataset '$DATASET'"
    exit 1
fi

echo "Extraction complete."

# ── Environment ───────────────────────────────────────────────────────────────
module --force purge
module load miniconda/3
conda activate unlearning

# ── Training ─────────────────────────────────────────────────────────────────
echo "Starting TracIn (self-influence) tracking | dataset=$DATASET | seed=$SEED"

srun python train.py \
    --dataset $DATASET \
    --seed $SEED \
    --tracein_train

echo "Job $SLURM_ARRAY_TASK_ID finished."