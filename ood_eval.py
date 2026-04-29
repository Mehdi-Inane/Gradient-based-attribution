import os
import argparse
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
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
        for batch_idx, (inputs, targets, corruptions) in enumerate(dataloader):
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
    parser.add_argument('--score_type', type=str, required=True, 
                        choices=['grad_dev', 'influence', 'memorization'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Task: {args.score_type.upper()}")

    scratch_dir = os.environ.get('SCRATCH', './')
    ckpt_dir = os.path.join(scratch_dir, 'attribution_training_runs')
    
    # Map score_type to file name templates
    k_values = [5000, 10000, 20000, 30000]
    checkpoints = {}
    
    for k in k_values:
        if args.score_type == 'grad_dev':
            filename = f'checkpoint_cifar100_average_gradient_scores_15runs_topk{k}_normal_seed42.pth'
        elif args.score_type == 'influence':
            filename = f'checkpoint_cifar100_feldman_avg_influence_topk{k}_normal_seed42.pth'
        elif args.score_type == 'memorization':
            filename = f'checkpoint_cifar100_feldman_memorization_scores_topk{k}_normal_seed42.pth'
            
        checkpoints[f"{args.score_type}_k{k}"] = os.path.join(ckpt_dir, filename)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    
    # We rely on the bash script to export HF_DATASETS_CACHE=$SLURM_TMPDIR
    print("Loading CIFAR-100-C from Hugging Face...")
    hf_dataset = load_dataset("randall-lab/cifar100-c", split="test", trust_remote_code=True)
    dataset = HFCifar100CDataset(hf_dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    
    all_results = {}

    for model_name, ckpt_path in checkpoints.items():
        print(f"\nEvaluating {model_name}...")
        if not os.path.exists(ckpt_path):
            print(f"  -> Checkpoint not found at {ckpt_path}! Skipping.")
            continue
            
        model = get_resnet50(dataset_name='cifar100', num_classes=100).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        results = evaluate_model(model, dataloader, device)
        all_results[model_name] = results
        print(f"  -> Overall OOD Accuracy: {results['Overall_Mean']:.2f}%")

    output_file = os.path.join(ckpt_dir, f'ood_results_{args.score_type}.npy')
    np.save(output_file, all_results, allow_pickle=True)
    print(f"\nSaved {args.score_type} results to: {output_file}")

if __name__ == '__main__':
    main()