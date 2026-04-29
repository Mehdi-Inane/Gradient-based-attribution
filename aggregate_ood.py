import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    scratch_dir = os.environ.get('SCRATCH', './')
    ckpt_dir = os.path.join(scratch_dir, 'attribution_training_runs')
    
    score_types = ['grad_dev', 'influence', 'memorization']
    k_values = [5000, 10000, 20000, 30000]
    
    aggregated_data = {}
    
    print("Loading individual result files...")
    for stype in score_types:
        file_path = os.path.join(ckpt_dir, f'ood_results_{stype}.npy')
        if os.path.exists(file_path):
            # Load the dictionary from the numpy object array
            data = np.load(file_path, allow_pickle=True).item()
            aggregated_data.update(data)
            print(f"  -> Successfully loaded {stype}")
        else:
            print(f"  -> WARNING: {file_path} not found!")

    # Save the merged dictionary
    merged_output = os.path.join(ckpt_dir, 'ood_results_ALL_MERGED.npy')
    np.save(merged_output, aggregated_data, allow_pickle=True)
    print(f"\nMerged results saved to {merged_output}")

    # Define plot styling for each score type
    styles = {
        'grad_dev': {'label': 'Gradient Deviation', 'color': 'royalblue', 'marker': 'o'},
        'influence': {'label': 'Feldman Influence', 'color': 'darkorange', 'marker': 's'},
        'memorization': {'label': 'Feldman Memorization', 'color': 'forestgreen', 'marker': '^'}
    }

    # ====================================================================
    # PLOT 1: Overall Mean Accuracy vs K (Line Plot)
    # ====================================================================
    print("\nGenerating overall performance line plot...")
    plt.figure(figsize=(10, 6))
    
    for stype in score_types:
        x_vals = []
        y_vals = []
        
        for k in k_values:
            model_key = f"{stype}_k{k}"
            if model_key in aggregated_data:
                acc = aggregated_data[model_key].get('Overall_Mean', None)
                if acc is not None:
                    x_vals.append(k)
                    y_vals.append(acc)
        
        if x_vals and y_vals:
            sorted_indices = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sorted_indices]
            y_vals = np.array(y_vals)[sorted_indices]
            
            plt.plot(x_vals, y_vals, label=styles[stype]['label'], 
                     color=styles[stype]['color'], marker=styles[stype]['marker'], 
                     linewidth=2, markersize=8)

    plt.title('Overall OOD Generalization vs. Number of Points (k)', fontsize=14)
    plt.xlabel('Number of Points (k)', fontsize=12)
    plt.ylabel('Overall Mean Accuracy (%)', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plot_output = os.path.join(ckpt_dir, 'ood_comparison_plot.png')
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"Line plot saved to {plot_output}")
    plt.close()

    # ====================================================================
    # PLOT 2: Individual Domains (Grouped Bar Charts per K)
    # ====================================================================
    print("\nGenerating domain-specific bar charts...")
    
    # Grab a valid key to extract the list of domains
    valid_model_key = next(iter(aggregated_data.keys()), None)
    if not valid_model_key:
        print("No data available to plot bar charts.")
        return
        
    # Extract domains and exclude 'Overall_Mean'
    domains = [key for key in aggregated_data[valid_model_key].keys() if key != 'Overall_Mean']
    domains.sort()  # Sort alphabetically for consistency
    
    x = np.arange(len(domains))
    width = 0.25  # Width of each bar
    offsets = [-width, 0, width] # Positions for the 3 bars inside each group
    
    # Create a separate histogram/bar chart for each value of k
    for k in k_values:
        plt.figure(figsize=(18, 8)) # Make it wide to fit all 19 domains
        plotted_any = False
        
        for i, stype in enumerate(score_types):
            model_key = f"{stype}_k{k}"
            if model_key in aggregated_data:
                # Extract the accuracies for this specific model in the exact order of `domains`
                accuracies = [aggregated_data[model_key].get(d, 0) for d in domains]
                
                plt.bar(x + offsets[i], accuracies, width, 
                        label=styles[stype]['label'], color=styles[stype]['color'],
                        edgecolor='black', alpha=0.85)
                plotted_any = True
                
        if plotted_any:
            plt.title(f'Domain-Specific OOD Accuracy (k={k})', fontsize=16, fontweight='bold')
            plt.xlabel('Corruption Type', fontsize=14)
            plt.ylabel('Accuracy (%)', fontsize=14)
            
            # Align the x-ticks exactly in the center of the 3 grouped bars
            plt.xticks(x, domains, rotation=45, ha='right', fontsize=12)
            
            # Optional: Lock y-axis to 0-100 for consistent comparing across k
            plt.ylim(0, 100) 
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout() # Ensures domain labels at the bottom aren't cut off
            
            bar_output = os.path.join(ckpt_dir, f'ood_domain_histogram_k{k}.png')
            plt.savefig(bar_output, dpi=300)
            print(f"Domain bar chart (k={k}) saved to {bar_output}")
        plt.close()

if __name__ == '__main__':
    main()