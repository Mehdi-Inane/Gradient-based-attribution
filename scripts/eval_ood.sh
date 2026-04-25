#!/bin/bash
#SBATCH --job-name=ood_eval
#SBATCH --output=logs/job_ood_eval_%A_%a.txt
#SBATCH --error=logs/job_ood_eval_%A_%a.txt
#SBATCH --array=0-2            # Launches 3 jobs (0, 1, and 2)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # Matches num_workers=4
#SBATCH --gres=gpu:1           # 1 GPU per job
#SBATCH --time=04:00:00        # Should be plenty of time for 4 checkpoints
#SBATCH --mem=32Gb             

# 1. Define the array of score types
SCORE_TYPES=("grad_dev" "influence" "memorization")

# Select the score type for THIS specific job in the array
CURRENT_SCORE=${SCORE_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "Starting evaluation job for score type: $CURRENT_SCORE"

export HF_DATASETS_CACHE=$SLURM_TMPDIR

# 3. Load environment
module --force purge
module load miniconda/3
conda activate unlearning

# 4. Run the python script for this specific score
srun python ood_eval.py --score_type $CURRENT_SCORE

echo "Job for $CURRENT_SCORE finished."