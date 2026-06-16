#!/bin/bash
#SBATCH --job-name=ood_eval
#SBATCH --output=logs/job_ood_eval_%A_%a.txt
#SBATCH --error=logs/job_ood_eval_%A_%a.txt
#SBATCH --array=0-3            # 4 jobs (one for each method)
#SBATCH --ntasks=1             # 1 task per node
#SBATCH --cpus-per-task=4      # Matches num_workers=4
#SBATCH --gres=gpu:a100l:1           # 1 GPU per job
#SBATCH --time=06:00:00        # 2 hours is plenty for 20 inference passes
#SBATCH --mem=48Gb             

METHODS=("our_method" "feldman_memorization" "tracein" "random_baseline")
CURRENT_METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

LOCAL_DATA="$SLURM_TMPDIR/cifar100-c"
echo "Copying dataset to local SSD..."
cp -r $SCRATCH/hf_datasets/cifar100-c $LOCAL_DATA

module load miniconda/3
conda activate unlearning

srun python ood_eval.py --method_name $CURRENT_METHOD