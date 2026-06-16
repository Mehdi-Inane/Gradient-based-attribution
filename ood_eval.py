import os
import argparse
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets import load_from_disk
from resnet import get_resnet50

class HFCifar100CDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'] 
        label = item['label']
        corruption = item['corruption_name']
        if self.transform:
            img = self.transform(img)
        return img, label, corruption

def evaluate_model(model, dataloader, device):
    model.eval()
    stats_per_corruption = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with torch.no_grad():
        for inputs, targets, corruptions in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(targets.size(0)):
                corr = corruptions[i]
                stats_per_corruption[corr]['total'] += 1
                if predicted[i] == targets[i]:
                    stats_per_corruption[corr]['correct'] += 1
                    
    final_accs = {}
    overall_correct = 0
    overall_total = 0
    
    for corr, counts in stats_per_corruption.items():
        acc = 100. * counts['correct'] / counts['total']
        final_accs[corr] = acc
        overall_correct += counts['correct']
        overall_total += counts['total']
        
    final_accs['Overall_Mean'] = 100. * overall_correct / overall_total
    return final_accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scratch_dir = os.environ.get('SCRATCH', './')
    ckpt_dir = os.path.join(scratch_dir, 'attribution_training_runs')
    
    # Path where you saved the dataset using Step 1
    local_data_path = os.path.join(scratch_dir, 'hf_datasets', 'cifar100-c')
    
    k_values = [5000, 10000, 20000, 30000]
    seeds = [0, 1, 2, 3, 4]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    
    print(f"Loading CIFAR-100-C from local disk: {local_data_path}")
    hf_dataset = load_from_disk(local_data_path)
    dataset = HFCifar100CDataset(hf_dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    
    model = get_resnet50(dataset_name='cifar100', num_classes=100).to(device)
    
    all_results = {}

    for k in k_values:
        print(f"\n--- Evaluating K={k} ---")
        # seed_accs_per_corr[corruption_name] = [acc_seed0, acc_seed1, ...]
        seed_accs_per_corr = defaultdict(list)
        
        for seed in seeds:
            ckpt_path = os.path.join(ckpt_dir, args.method_name, str(k), f'final_checkpoint_seed{seed}.pth')
            
            if not os.path.exists(ckpt_path):
                print(f"  -> Skipping seed {seed}: Checkpoint not found.")
                continue
                
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                
            results = evaluate_model(model, dataloader, device)
            
            for key, acc in results.items():
                seed_accs_per_corr[key].append(acc)
                
        if seed_accs_per_corr:
            # We store a dictionary containing BOTH the mean and the raw seed list
            # to make plotting error bars very easy later.
            k_stats = {}
            for key, acc_list in seed_accs_per_corr.items():
                k_stats[key] = {
                    'mean': np.mean(acc_list),
                    'std': np.std(acc_list),
                    'raw': acc_list
                }
            
            all_results[f"k{k}"] = k_stats
            print(f" => Mean OOD Accuracy for K={k}: {k_stats['Overall_Mean']['mean']:.2f}%")

    output_file = os.path.join(ckpt_dir, f'ood_results_{args.method_name}.npy')
    np.save(output_file, all_results, allow_pickle=True)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()