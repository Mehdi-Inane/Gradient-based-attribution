#!/bin/bash
#SBATCH --job-name=eval_seeds
#SBATCH --output=logs/subset_trains/job_eval_%j.txt
#SBATCH --error=logs/subset_trains/job_eval_%j.txt
#SBATCH --gpus=1
#SBATCH --ntasks=5               # 5 seeds
#SBATCH --cpus-per-task=2        # 10 CPUs total for the job
#SBATCH --time=24:00:00
#SBATCH --mem=48Gb

# 1. Extract dataset to TMPDIR (Runs exactly ONCE per job)
echo "Extracting CIFAR-100 to SLURM_TMPDIR..."
tar -xzf /network/datasets/cifar100/cifar-100-python.tar.gz -C $SLURM_TMPDIR

# 2. Setup Environment (Runs exactly ONCE per job)
module --force purge
module load miniconda/3
conda activate unlearning

# 3. Launch the 5 tasks concurrently
# srun propagates "$@" (the arguments passed from master_eval.sh) to all 5 tasks
srun python train.py "$@"