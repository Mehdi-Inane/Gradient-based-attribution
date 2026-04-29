"""
average_grad_scores.py

Averages gradient deviation score files produced by multiple seeded runs.

Usage:
    python average_grad_scores.py \
        --dataset cifar100 \
        --n_seeds 5 \
        --scratch_dir /network/scratch/a/ahmedm/attribution_training_runs

Produces:
    batch_gradient_deviation_scores_<dataset>_baseline_topk0_grad_avg.npy
    batch_gradient_deviation_scores_<dataset>_baseline_topk0_grad_std.npy
"""

import os
import glob
import argparse
import numpy as np


def average_scores(dataset, n_seeds, scratch_dir, scores_path=None, k=0):
    # Reconstruct the run_name pattern (must match get_run_name in train.py)
    if scores_path:
        base = os.path.splitext(os.path.basename(scores_path))[0]
        prefix = f"batch_gradient_deviation_scores_{dataset}_{base}_topk{k}_grad"
    else:
        prefix = f"batch_gradient_deviation_scores_{dataset}_baseline_topk{k}_grad"

    # Collect files for seeds 0..n_seeds-1
    files = []
    missing = []
    for seed in range(n_seeds):
        path = os.path.join(scratch_dir, f"{prefix}_seed{seed}.npy")
        if os.path.exists(path):
            files.append((seed, path))
        else:
            missing.append(seed)

    if missing:
        print(f"WARNING: Missing score files for seeds: {missing}")

    if not files:
        raise FileNotFoundError(f"No score files found matching pattern: {prefix}_seed*.npy")

    print(f"Found {len(files)} score file(s):")
    for seed, f in files:
        print(f"  seed {seed}: {os.path.basename(f)}")

    # Stack and compute statistics
    arrays = np.stack([np.load(f) for _, f in files])   # shape: (n_seeds, n_samples)
    avg_scores = arrays.mean(axis=0)
    std_scores = arrays.std(axis=0)

    print(f"\nScore statistics across {len(files)} seeds:")
    print(f"  Mean of averages : {avg_scores.mean():.6f}")
    print(f"  Mean of std devs : {std_scores.mean():.6f}")
    print(f"  Max std dev      : {std_scores.max():.6f}  (sample index {std_scores.argmax()})")

    # Save
    avg_path = os.path.join(scratch_dir, f"{prefix}_avg.npy")
    std_path = os.path.join(scratch_dir, f"{prefix}_std.npy")
    np.save(avg_path, avg_scores)
    np.save(std_path, std_scores)

    print(f"\nSaved averaged scores  -> {avg_path}")
    print(f"Saved std-dev scores   -> {std_path}")

    return avg_scores, std_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Average gradient deviation scores across seeds.')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'])
    parser.add_argument('--n_seeds', type=int, default=5, help='Number of seeds used during training.')
    parser.add_argument('--scratch_dir', type=str,
                        default='/network/scratch/a/ahmedm/attribution_training_runs')
    parser.add_argument('--scores_path', type=str, default=None,
                        help='Scores path used during training (for run name reconstruction).')
    parser.add_argument('--k', type=int, default=0)
    args = parser.parse_args()

    average_scores(
        dataset=args.dataset,
        n_seeds=args.n_seeds,
        scratch_dir=args.scratch_dir,
        scores_path=args.scores_path,
        k=args.k,
    )