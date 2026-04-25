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

    # --- Plotting Overall Mean Accuracy vs K ---
    print("Generating performance plot...")
    
    plt.figure(figsize=(10, 6))
    
    # Define plot styling for each score type
    styles = {
        'grad_dev': {'label': 'Gradient Deviation', 'color': 'blue', 'marker': 'o'},
        'influence': {'label': 'Feldman Influence', 'color': 'orange', 'marker': 's'},
        'memorization': {'label': 'Feldman Memorization', 'color': 'green', 'marker': '^'}
    }

    for stype in score_types:
        x_vals = []
        y_vals = []
        
        for k in k_values:
            model_key = f"{stype}_k{k}"
            if model_key in aggregated_data:
                # Extract the overall mean
                acc = aggregated_data[model_key].get('Overall_Mean', None)
                if acc is not None:
                    x_vals.append(k)
                    y_vals.append(acc)
        
        if x_vals and y_vals:
            # Sort just in case they were loaded out of order
            sorted_indices = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sorted_indices]
            y_vals = np.array(y_vals)[sorted_indices]
            
            plt.plot(x_vals, y_vals, label=styles[stype]['label'], 
                     color=styles[stype]['color'], marker=styles[stype]['marker'], 
                     linewidth=2, markersize=8)

    plt.title('OOD Generalization (CIFAR-100-C) vs. Number of Points (k)', fontsize=14)
    plt.xlabel('Number of Points (k)', fontsize=12)
    plt.ylabel('Overall Mean Accuracy (%)', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plot_output = os.path.join(ckpt_dir, 'ood_comparison_plot.png')
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_output}")

if __name__ == '__main__':
    main()