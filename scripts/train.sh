#!/bin/bash
#SBATCH --job-name=resnet_grad_dev
#SBATCH --output=logs/job_output_%A_%a.txt  # %A=job_id, %a=array_index
#SBATCH --error=logs/job_error_%A_%a.txt
#SBATCH --array=0-2                    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:1       
#SBATCH --time=24:00:00        
#SBATCH --mem=48Gb           

DATASET="cifar100"

SCORES_PATHS=(
    "/home/mila/a/ahmedm/Gradient-based-attribution/batch_gradient_deviation_scores_full_data.npy"
    "/home/mila/a/ahmedm/Gradient-based-attribution/feldman_memorization_scores.npy"
    "/home/mila/a/ahmedm/Gradient-based-attribution/feldman_avg_influence.npy"
)

K_VALUE=30000

CURRENT_SCORE_PATH=${SCORES_PATHS[$SLURM_ARRAY_TASK_ID]}

echo "Starting Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using scores from: $CURRENT_SCORE_PATH"
echo "Dropping $K_VALUE points."

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

echo "Starting normal training for $DATASET..."
srun python train.py \
    --dataset $DATASET \
    --normal_train \
    --scores_path "$CURRENT_SCORE_PATH" \
    --k $K_VALUE