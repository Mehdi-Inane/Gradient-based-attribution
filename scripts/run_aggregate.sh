#!/bin/bash
#SBATCH --job-name=ood_agg
#SBATCH --output=job_ood_agg_%j.txt
#SBATCH --error=job_ood_agg_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00        # Should take less than a minute
#SBATCH --mem=8Gb             

echo "Starting result aggregation..."

module --force purge
module load miniconda/3
conda activate unlearning

# Run the python script 
srun python aggregate_ood.py

echo "Aggregation finished successfully."