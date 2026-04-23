#!/bin/bash
#SBATCH --job-name=resnet_grad_dev
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Matches the num_workers=4 in our dataloaders
#SBATCH --gres=gpu:1           # CRITICAL: Requests 1 GPU
#SBATCH --time=24:00:00        
#SBATCH --mem=48Gb             

# 1. Extract the dataset directly to the node's local SSD (fastest I/O)
echo "Extracting CIFAR-100 to SLURM_TMPDIR..."
tar -xzf /network/datasets/cifar100/cifar-100-python.tar.gz -C $SLURM_TMPDIR


module --force purge
module load miniconda/3
conda activate unlearning
srun python train.py