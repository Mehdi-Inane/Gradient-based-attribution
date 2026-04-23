import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class IndexedDataset(Dataset):
    """
    Wraps a PyTorch Dataset to yield (index, data, target).
    This allows us to track the original dataset index even when the DataLoader shuffles the data.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return index, data, target

def get_cifar100_dataloaders(data_dir, batch_size=512, num_workers=4):
    """
    Creates and returns the train and test dataloaders for CIFAR-100.
    Expects data_dir to contain the extracted 'cifar-100-python' folder.
    """
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

    # Load datasets locally, explicitly blocking downloads
    raw_train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    
    raw_test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=test_transform
    )

    # Wrap datasets to return indices
    train_dataset = IndexedDataset(raw_train_dataset)
    test_dataset = IndexedDataset(raw_test_dataset)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_dataset