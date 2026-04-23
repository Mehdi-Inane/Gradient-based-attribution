import os
import time
import csv
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.func import functional_call, vmap, grad

from data import get_cifar100_dataloaders
from resnet import get_modified_resnet50
from optim import get_optimizer_and_scheduler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_batch_deviations(model, batch_grads, inputs, targets, chunk_size=16):
    was_training = model.training
    model.eval() 
    
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def compute_loss(params, buffers, x, y):
        out = functional_call(model, (params, buffers), x.unsqueeze(0))
        return F.cross_entropy(out, y.unsqueeze(0))

    def compute_sq_dev(params, buffers, batch_grads, x, y):
        
        sample_grads = grad(compute_loss, argnums=0)(params, buffers, x, y)
        sq_dev = sum(torch.sum((g - bg) ** 2) for g, bg in zip(sample_grads.values(), batch_grads))
        return sq_dev

    vmap_fn = vmap(compute_sq_dev, in_dims=(None, None, None, 0, 0))
    dev_list = []
    
    for i in range(0, inputs.size(0), chunk_size):
        x_chunk = inputs[i:i+chunk_size]
        y_chunk = targets[i:i+chunk_size]
        
        # Compute the deviations for this chunk
        devs = vmap_fn(params, buffers, batch_grads, x_chunk, y_chunk)
        dev_list.append(devs.detach())
        
    if was_training:
        model.train()
        
    return torch.cat(dev_list)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for _, inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / total, 100. * correct / total

def train_with_exact_gradient_deviation():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = os.environ.get('SLURM_TMPDIR', './data')
    print(f"Loading data from: {data_dir}")
    
    batch_size = 512
    epochs = 160
    
    train_loader, test_loader, train_dataset = get_cifar100_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=4
    )
    
    model = get_modified_resnet50(num_classes=100).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, epochs=epochs, steps_per_epoch=len(train_loader), base_lr=0.4
    )
    
    G_scores = np.zeros(len(train_dataset), dtype=np.float32)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    best_test_acc = 0.0

    log_file = 'training_log.csv'
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'LR', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Time(s)'])

    print("Training begins")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (indices, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            train_loss_sum += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            batch_grads = tuple(p.grad.detach().clone() for p in model.parameters())
            
            deviations = get_batch_deviations(model, batch_grads, inputs, targets, chunk_size=64)
            G_scores[indices.cpu().numpy()] += deviations.cpu().numpy()
            
            optimizer.step()
            scheduler.step()
            
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        train_loss = train_loss_sum / train_total
        train_acc = 100. * train_correct / train_total
        
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
              
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, current_lr, train_loss, train_acc, test_loss, test_acc, epoch_time])
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'G_scores': G_scores,
            'best_test_acc': best_test_acc
        }
        torch.save(checkpoint, 'checkpoint.pth')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(checkpoint, 'checkpoint_best.pth')
        
    np.save('batch_gradient_deviation_scores.npy', G_scores)
    print("Training complete! .")

if __name__ == '__main__':
    train_with_exact_gradient_deviation()