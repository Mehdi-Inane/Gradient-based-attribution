import os
import argparse
import numpy as np
from scipy.stats import spearmanr

# Force matplotlib to use a non-interactive backend for headless cluster environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH_DIR = '/network/scratch/a/ahmedm/attribution_training_runs'

def compute_vectorized_rbo(list1, list2, p=0.9995):
    """
    Computes Rank-Biased Overlap (RBO) between two ranked arrays of identifiers.
    Fully vectorized via NumPy for ultra-fast performance on large datasets.
    """
    n = len(list1)
    if n == 0:
        return 0.0
    
    pos2 = {x: i for i, x in enumerate(list2)}
    increments = np.zeros(n + 1, dtype=np.int32)
    for i, x in enumerate(list1):
        idx_in_2 = pos2.get(x, None)
        if idx_in_2 is not None:
            max_idx = max(i, idx_in_2)
            if max_idx < n:
                increments[max_idx + 1] += 1

    intersection_sizes = np.cumsum(increments)[1:]
    depths = np.arange(1, n + 1)
    agreements = intersection_sizes / depths
    weights = (1 - p) * (p ** (depths - 1))
    
    return float(np.sum(weights * agreements))

def save_evolution_plots(metrics, run_pattern):
    """
    Generates and saves professional diagnostic plots tracking score stability over time.
    """
    epochs = metrics['epochs']
    
    # Initialize a clean 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Ranking Stability Evolution Summary ({run_pattern})", fontsize=14, fontweight='bold', y=0.98)

    # ── Left Plot: Global Population Rank Similarity ─────────────────────────
    ax1.plot(epochs, metrics['spearman_rho'], label='Spearman Rho', color='#2ca02c', marker='o', linewidth=2, markersize=4)
    ax1.plot(epochs, metrics['rbo'], label='Rank-Biased Overlap (RBO)', color='#1f77b4', marker='s', linewidth=2, markersize=4)
    ax1.set_title("Global Ranking Similarity vs. Final Epoch", fontsize=11, fontweight='semibold')
    ax1.set_xlabel("Training Epoch", fontsize=10)
    ax1.set_ylabel("Metric Similarity Value (0 to 1)", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

    # ── Right Plot: High-Deviation Pruning Set Overlap ───────────────────────
    colors = ['#d62728', '#ff7f0e', '#9467bd', '#17becf', '#bcbd22']
    for idx, (k, overlaps) in enumerate(metrics['topk_overlap'].items()):
        color = colors[idx % len(colors)]
        ax2.plot(epochs, overlaps, label=f"Top-K (K={k})", color=color, marker='^', linewidth=2, markersize=4)
        
    ax2.set_title("Pruning Intersection Agreement with Final Epoch", fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Training Epoch", fontsize=10)
    ax2.set_ylabel("Set Overlap Percentage (%)", fontsize=10)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    
    # Save chart out directly into your scratch run repository
    plot_filename = f"ranking_stability_dashboard_{run_pattern}.png"
    plot_path = os.path.join(SCRATCH_DIR, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Visualization Dashboard exported successfully to: {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Average exact gradient deviation score trajectories and evaluate ranking stability.")
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--scores_path', type=str, default=None)
    parser.add_argument('--k_values', type=int, nargs='+', default=[5000, 10000, 20000, 30000], 
                        help="List of K values to compute top-K set overlaps for.")
    parser.add_argument('--max_epoch', type=int, default=160, help="Maximum possible training epochs to scan.")
    parser.add_argument('--rbo_p', type=float, default=0.9995, help="RBO parameter p (closer to 1 weights bottom ranks more).")
    args = parser.parse_args()

    if args.scores_path:
        if args.scores_path.lower() == 'random':
            base = 'random'
        else:
            base = os.path.splitext(os.path.basename(args.scores_path))[0]
        run_pattern = f"{args.dataset}_{base}_topk0_grad"
    else:
        run_pattern = f"{args.dataset}_baseline_grad"

    target_epochs = [ep for ep in range(1, args.max_epoch + 1) if ep <= 10 or ep % 10 == 0]
    seeds = list(range(15))

    # --- PASS 1: Average individual arrays across seeds ---
    print("=== Pass 1: Aggregating Score Configurations Across Seeds ===")
    averaged_score_map = {}
    final_epoch = None

    for epoch in target_epochs:
        collected_arrays = []
        for seed in seeds:
            filename = f"epoch_scores_{run_pattern}_seed{seed}_epoch{epoch}.npy"
            full_path = os.path.join(SCRATCH_DIR, filename)
            if os.path.exists(full_path):
                collected_arrays.append(np.load(full_path))
        
        if collected_arrays:
            avg_scores = np.mean(np.stack(collected_arrays, axis=0), axis=0)
            averaged_score_map[epoch] = avg_scores
            final_epoch = epoch
            
            output_name = f"averaged_scores_{run_pattern}_epoch{epoch}.npy"
            np.save(os.path.join(SCRATCH_DIR, output_name), avg_scores)
        else:
            if epoch <= args.max_epoch:
                print(f" -> Epoch {epoch:03d}: No seed files found.")

    if final_epoch is None:
        print("CRITICAL: No historical snapshot files were discovered! Terminating analysis.")
        return

    print(f"\nTarget ground truth finalized. Using Epoch {final_epoch} as Reference Final State.\n")

    final_scores = averaged_score_map[final_epoch]
    final_ranked_indices = np.argsort(-final_scores)

    evolution_metrics = {
        'epochs': [],
        'spearman_rho': [],
        'rbo': [],
        'topk_overlap': {k: [] for k in args.k_values}
    }

    # --- PASS 2: Compare each epoch to the final reference epoch ---
    print("=== Pass 2: Evaluating Structural Ranking Similarities ===")
    for epoch in sorted(averaged_score_map.keys()):
        current_scores = averaged_score_map[epoch]
        current_ranked_indices = np.argsort(-current_scores)
        
        # 1. Spearman Rank Correlation
        rho, _ = spearmanr(current_scores, final_scores)
        
        # 2. Rank-Biased Overlap (RBO)
        rbo_val = compute_vectorized_rbo(current_ranked_indices, final_ranked_indices, p=args.rbo_p)
        
        # 3. Top-K Set Overlap
        epoch_overlaps = {}
        for k in args.k_values:
            if k <= len(current_scores):
                set_current = set(current_ranked_indices[:k])
                set_final = set(final_ranked_indices[:k])
                overlap_pct = (len(set_current.intersection(set_final)) / k) * 100.0
                epoch_overlaps[k] = overlap_pct
            else:
                epoch_overlaps[k] = 0.0

        # Append data to the metrics structure
        evolution_metrics['epochs'].append(epoch)
        evolution_metrics['spearman_rho'].append(rho)
        evolution_metrics['rbo'].append(rbo_val)
        for k in args.k_values:
            evolution_metrics['topk_overlap'][k].append(epoch_overlaps[k])

        print(f"Epoch {epoch:03d} | Spearman: {rho:.4f} | RBO: {rbo_val:.4f} | Top-{args.k_values[0]} Overlap: {epoch_overlaps[args.k_values[0]]:.1f}%")

    # Save summary metrics structure
    summary_filename = f"ranking_evolution_metrics_{run_pattern}.npy"
    np.save(os.path.join(SCRATCH_DIR, summary_filename), evolution_metrics)
    print(f"\n=== Success! Evolution metrics compiled and written to: {summary_filename} ===")

    # --- PASS 3: Generate Plot Presentation Dashboard ---
    print("\n=== Pass 3: Constructing Visualization Assets ===")
    save_evolution_plots(evolution_metrics, run_pattern)

if __name__ == '__main__':
    main()