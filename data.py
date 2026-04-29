import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return index, data, target

class FilteredDataset(Dataset):
    """
    Wraps a PyTorch Dataset to only include specific indices based on score filtering.
    Maintains and yields the original dataset index (original_index, data, target).
    """
    def __init__(self, dataset, indices_to_keep):
        self.dataset = dataset
        self.indices_to_keep = indices_to_keep

    def __len__(self):
        return len(self.indices_to_keep)

    def __getitem__(self, index):
        original_idx = self.indices_to_keep[index]
        data, target = self.dataset[original_idx]
        return original_idx, data, target

def get_dataloaders(dataset_name, data_dir, batch_size, num_workers=8, scores_path=None, k=0):
    if dataset_name == 'cifar100':
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        raw_train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=train_transform
        )
        
        raw_test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=False, transform=test_transform
        )

    elif dataset_name == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Pad(32),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_dir = os.path.join(data_dir, 'imagenet', 'train')
        val_dir = os.path.join(data_dir, 'imagenet', 'val')

        raw_train_dataset = torchvision.datasets.ImageFolder(
            root=train_dir, transform=train_transform
        )
        
        raw_test_dataset = torchvision.datasets.ImageFolder(
            root=val_dir, transform=test_transform
        )
        
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    if scores_path is not None and k > 0:
        scores = np.load(scores_path)
        n = len(raw_train_dataset)
        k_to_drop = min(k, n) # Ensure we don't try to drop more points than exist
        
        # Sort indices by score (ascending), then reverse to get descending (highest first)
        descending_indices = np.argsort(scores)[::-1]
        
        # Keep everything AFTER the top k_to_drop indices
        indices_to_keep = descending_indices[k_to_drop:]
        
        train_dataset = FilteredDataset(raw_train_dataset, indices_to_keep)
        print(f"Filtered dataset: dropped top {k_to_drop} highest scoring points. Kept {len(indices_to_keep)} points out of {n}.")
    else:
        train_dataset = IndexedDataset(raw_train_dataset)

    test_dataset = IndexedDataset(raw_test_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_dataset