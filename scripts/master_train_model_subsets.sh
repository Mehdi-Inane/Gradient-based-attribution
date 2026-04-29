#!/bin/bash

# Define base paths for the scores
BASE_PATH="/home/mila/a/ahmedm/Gradient-based-attribution"

# Associate methods with their corresponding score arrays
METHODS=(
    "our_method|${BASE_PATH}/average_gradient_scores_15runs.npy"
    "feldman_memorization|${BASE_PATH}/feldman_memorization_scores.npy"
    "feldman_average_influence|${BASE_PATH}/feldman_avg_influence.npy"
)
# IMPORTANT: add traceIn scores once done
K_VALUES=(5000 10000 20000 30000)

echo "Submitting 16 evaluation jobs..."

for item in "${METHODS[@]}"; do
    METHOD_NAME="${item%%|*}"
    SCORE_PATH="${item##*|}"
    
    for K in "${K_VALUES[@]}"; do
        # Pass the arguments directly to the job script, which passes them to python
        sbatch scripts/train_seeds_subset.sh \
            --dataset cifar100 \
            --normal_train \
            --method_name "$METHOD_NAME" \
            --scores_path "$SCORE_PATH" \
            --k $K
            
        echo "Submitted job for $METHOD_NAME with K=$K"
    done
done

echo "All 16 jobs submitted!"