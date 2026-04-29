#!/bin/bash
#SBATCH --job-name=resnet_train_k
#SBATCH --output=logs/job_output_%A_%a.txt  # %A=job_id, %a=array_index
#SBATCH --error=logs/job_error_%A_%a.txt
#SBATCH --array=0-11                    # 3 scores x 4 k-values = 12 parallel jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:1       
#SBATCH --time=24:00:00        
#SBATCH --mem=48Gb           

DATASET="cifar100"

SCORES_PATHS=(
    "/home/mila/a/ahmedm/Gradient-based-attribution/average_gradient_scores_15runs.npy"
    "/home/mila/a/ahmedm/Gradient-based-attribution/feldman_memorization_scores.npy"
    "/home/mila/a/ahmedm/Gradient-based-attribution/feldman_avg_influence.npy"
)

K_VALUES=(5000 10000 20000 30000)

# Map the 1D SLURM array ID to the 2D lists (Scores and K values)
SCORE_IDX=$((SLURM_ARRAY_TASK_ID % 3))
K_IDX=$((SLURM_ARRAY_TASK_ID / 3))

CURRENT_SCORE_PATH=${SCORES_PATHS[$SCORE_IDX]}
CURRENT_K=${K_VALUES[$K_IDX]}

echo "Starting Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using scores from: $CURRENT_SCORE_PATH"
echo "Dropping $CURRENT_K points."

echo "Preparing $DATASET dataset in SLURM_TMPDIR..."

if [ "$DATASET" == "cifar100" ]; then
    echo "Extracting CIFAR-100 to SLURM_TMPDIR..."
    tar -xzf /network/datasets/cifar100/cifar-100-python.tar.gz -C $SLURM_TMPDIR

elif [ "$DATASET" == "imagenet" ]; then
    echo "Extracting ImageNet to SLURM_TMPDIR..."
    tar -xf /network/datasets/imagenet/imagenet.tar -C $SLURM_TMPDIR
else
    echo "Error: Unknown dataset $DATASET"
    exit 1
fi

module --force purge
module load miniconda/3
conda activate unlearning

echo "Starting normal training for $DATASET with k=$CURRENT_K..."
srun python train.py \
    --dataset $DATASET \
    --normal_train \
    --scores_path "$CURRENT_SCORE_PATH" \
    --k $CURRENT_K