#!/bin/bash

echo "Submitting OOD Evaluation Job Array..."

# 1. Submit the array and capture all output
JOB_OUTPUT=$(sbatch --parsable scripts/eval_ood.sh 2>&1)

# 2. Extract ONLY the last line of the output to bypass cluster banners
LAST_LINE=$(echo "$JOB_OUTPUT" | tail -n 1)

# 3. Clean it up (handle the semicolon if your cluster uses 'JobID;ClusterName')
JOB_ID=$(echo "$LAST_LINE" | cut -d';' -f1)

# 4. Check if we finally have a clean number
if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Evaluation array submitted successfully with ID: $JOB_ID"
    
    # 5. Submit the aggregation script
    AGG_OUTPUT=$(sbatch --dependency=afterok:$JOB_ID scripts/run_aggregate.sh)
    
    echo "Aggregation job queued with dependency."
    echo "$AGG_OUTPUT"
    echo "All jobs submitted! Check status with 'squeue -u $USER'."
else
    echo "Error: Failed to capture Job ID from evaluation submission."
    echo "Raw output from sbatch was:"
    echo "----------------------------------------"
    echo "$JOB_OUTPUT"
    echo "----------------------------------------"
fi