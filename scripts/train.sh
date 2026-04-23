#!/bin/bash
#SBATCH --job-name=resnet_grad_dev
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Matches the num_workers in our dataloaders
#SBATCH --gres=gpu:a100l:1           # CRITICAL: Requests 1 GPU
#SBATCH --time=24:00:00        
#SBATCH --mem=80Gb           

DATASET="imagenet"

echo "Preparing $DATASET dataset in SLURM_TMPDIR..."

# 1. Extract the dataset directly to the node's local SSD (fastest I/O)
if [ "$DATASET" == "cifar100" ]; then
    echo "Extracting CIFAR-100 to SLURM_TMPDIR..."
    tar -xzf /network/datasets/cifar100/cifar-100-python.tar.gz -C $SLURM_TMPDIR

elif [ "$DATASET" == "imagenet" ]; then
    echo "Extracting ImageNet to SLURM_TMPDIR..."
    mkdir -p $SLURM_TMPDIR/imagenet

    tar -xf /network/datasets/imagenet/imagenet.tar -C $SLURM_TMPDIR/imagenet
else
    echo "Error: Unknown dataset $DATASET"
    exit 1
fi

# 2. Load modules and activate environment
module --force purge
module load miniconda/3
conda activate unlearning

# 3. Launch training
echo "Starting exact gradient deviation training for $DATASET..."
srun python train.py --dataset $DATASET