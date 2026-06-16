import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    scratch_dir = os.environ.get('SCRATCH', './')
    ckpt_dir = os.path.join(scratch_dir, 'attribution_training_runs')
    
    # Updated to include all your methods
    score_types = ['our_method', 'feldman_memorization', 'tracein', 'random_baseline']
    k_values = [5000, 10000, 20000, 30000]
    
    aggregated_data = {}
    
    print("Loading individual result files...")
    for stype in score_types:
        file_path = os.path.join(ckpt_dir, f'ood_results_{stype}.npy')
        if os.path.exists(file_path):
            # Load the data: ood_eval.py saves { 'k5000': stats_dict, ... }
            data = np.load(file_path, allow_pickle=True).item()
            
            # We prefix the keys with the method name to avoid collisions in the merged dict
            for k_key, stats in data.items():
                aggregated_data[f"{stype}_{k_key}"] = stats
            print(f"  -> Successfully loaded {stype}")
        else:
            print(f"  -> WARNING: {file_path} not found!")

    # Define plot styling
    styles = {
        'our_method': {'label': 'Our method', 'color': 'royalblue', 'marker': 'o'},
        'feldman_memorization': {'label': 'Feldman Memorization', 'color': 'forestgreen', 'marker': '^'},
        'tracein': {'label': 'TraceIn', 'color': 'crimson', 'marker': 'D'},
        'random_baseline': {'label': 'Random Baseline', 'color': 'gray', 'marker': 'x'}
    }

    # ====================================================================
    # PLOT 1: Overall Mean Accuracy vs K (Line Plot with Error Bars)
    # ====================================================================
    print("\nGenerating overall performance line plot with error bars...")
    plt.figure(figsize=(10, 6))
    
    for stype in score_types:
        x_vals = []
        y_means = []
        y_stds = []
        print(stype)
        for k in k_values:
            model_key = f"{stype}_k{k}"
            if model_key in aggregated_data:
                # Access the new nested structure: ['Overall_Mean']['mean']
                stats = aggregated_data[model_key].get('Overall_Mean', None)
                if stats is not None:
                    x_vals.append(k)
                    y_means.append(stats['mean'])
                    y_stds.append(stats['std'])
        
        if x_vals:
            # Ensure values are sorted by K for the line plot
            sorted_idx = np.argsort(x_vals)
            plt.errorbar(
                np.array(x_vals)[sorted_idx], 
                np.array(y_means)[sorted_idx], 
                yerr=np.array(y_stds)[sorted_idx],
                label=styles[stype]['label'], 
                color=styles[stype]['color'], 
                marker=styles[stype]['marker'], 
                linewidth=2, markersize=8, capsize=5
            )
            print(x_vals)
    plt.title('Overall OOD Generalization (Mean ± Std Dev)', fontsize=14)
    plt.xlabel('Number of Points Removed (k)', fontsize=12)
    plt.ylabel('Overall Mean Accuracy (%)', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='lower left')
    
    plot_output = os.path.join(ckpt_dir, 'ood_comparison_plot.png')
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"Line plot saved to {plot_output}")
    plt.close()

    # ====================================================================
    # PLOT 2: Individual Domains (Grouped Bar Charts per K)
    # ====================================================================
    print("\nGenerating domain-specific bar charts...")
    
    # Extract domains from any available entry (excluding Overall_Mean)
    valid_key = next(iter(aggregated_data.keys()), None)
    if not valid_key:
        print("No data available to plot bar charts.")
        return
        
    domains = [d for d in aggregated_data[valid_key].keys() if d != 'Overall_Mean']
    domains.sort()
    
    x = np.arange(len(domains))
    width = 0.15  # Narrower bars to fit more methods
    
    for k in k_values:
        plt.figure(figsize=(20, 8))
        plotted_any = False
        
        # Calculate offsets so the group is centered on the tick
        # (len(score_types) bars, centered at 0)
        total_width = width * len(score_types)
        start_offset = -total_width / 2 + width / 2

        for i, stype in enumerate(score_types):
            model_key = f"{stype}_k{k}"
            if model_key in aggregated_data:
                means = [aggregated_data[model_key][d]['mean'] for d in domains]
                stds = [aggregated_data[model_key][d]['std'] for d in domains]
                
                plt.bar(
                    x + start_offset + (i * width), 
                    means, width, yerr=stds,
                    label=styles[stype]['label'], color=styles[stype]['color'],
                    edgecolor='black', alpha=0.8, capsize=2
                )
                plotted_any = True
        
        if plotted_any:
            plt.title(f'Domain-Specific OOD Accuracy (k={k})', fontsize=16, fontweight='bold')
            plt.xticks(x, domains, rotation=45, ha='right', fontsize=11)
            plt.ylabel('Accuracy (%)', fontsize=13)
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.legend(loc='upper right', fontsize=12, ncol=2)
            plt.tight_layout()
            
            bar_output = os.path.join(ckpt_dir, f'ood_domain_histogram_k{k}.png')
            plt.savefig(bar_output, dpi=300)
            print(f"Domain bar chart (k={k}) saved to {bar_output}")
        plt.close()

if __name__ == '__main__':
    main()