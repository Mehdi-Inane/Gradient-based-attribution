import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from data import get_dataloaders
from optim import get_optimizer_and_scheduler
from resnet import get_resnet50


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_batch_deviations(
        model,
        batch_grads: tuple[torch.Tensor, ...],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int) -> torch.Tensor:
    was_training = model.training
    model.eval()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def compute_loss(params, buffers, x: torch.Tensor, y: torch.Tensor):
        out = functional_call(model, (params, buffers), x.unsqueeze(0))
        return F.cross_entropy(out, y.unsqueeze(0))

    # Pre-calculate to memory needed when mapping compute_sq_dev. Reduces ~35% of memory.
    batch_grads_norm_sq = sum(bg.pow(2).sum() for bg in batch_grads)

    def compute_sq_dev(params, buffers, x: torch.Tensor, y: torch.Tensor):
        sample_grads = grad(compute_loss, argnums=0)(params, buffers, x, y).values()
        # Re-written using the identity ||g - bg||² = ||g||² - 2(g·bg) + ||bg||²
        g_sq    = sum(g.pow(2).sum() for g in sample_grads)
        g_dot_b = sum((g * bg).sum() for g, bg in zip(sample_grads, batch_grads))
        return (g_sq - 2.0 * g_dot_b + batch_grads_norm_sq).detach()  # type: ignore[union-attr]

    vmap_fn = vmap(func=compute_sq_dev, in_dims=(None, None, 0, 0))

    # Pre-allocate to reduce in-memory accumulation. Reduces ~20% of memory.
    result = torch.empty(inputs.size(0), device=inputs.device)

    for i in range(0, inputs.size(0), chunk_size):
        result[i:i+chunk_size] = vmap_fn(params, buffers, inputs[i:i+chunk_size], targets[i:i+chunk_size])

    if was_training:
        model.train()

    return result


def get_batch_grad_norms(model, inputs, targets, chunk_size=16):
    was_training = model.training
    model.eval()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def compute_loss(params, buffers, x, y):
        out = functional_call(model, (params, buffers), x.unsqueeze(0))
        return F.cross_entropy(out, y.unsqueeze(0))

    def compute_sq_norm(params, buffers, x, y):
        sample_grads = grad(compute_loss, argnums=0)(params, buffers, x, y)
        sq_norm = sum(torch.sum(g ** 2) for g in sample_grads.values())
        return sq_norm

    vmap_fn = vmap(compute_sq_norm, in_dims=(None, None, 0, 0))
    norm_list = []

    for i in range(0, inputs.size(0), chunk_size):
        x_chunk = inputs[i:i+chunk_size]
        y_chunk = targets[i:i+chunk_size]
        norms = vmap_fn(params, buffers, x_chunk, y_chunk)
        norm_list.append(norms.detach())

    if was_training:
        model.train()

    return torch.cat(norm_list)

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

def get_run_name(args, mode):
    seed_tag = f"_seed{args.seed}"
    if args.scores_path:
        if args.scores_path.lower() == 'random':
            base = 'random'
        else:
            base = os.path.splitext(os.path.basename(args.scores_path))[0]
        return f"{args.dataset}_{base}_topk{args.k}_{mode}{seed_tag}"
    return f"{args.dataset}_baseline_{mode}{seed_tag}"

def train_normal(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task {os.environ.get('SLURM_PROCID', 0)} | Seed: {args.seed} | Method: {args.method_name} | K: {args.k} | Mode: Normal")

    save_dir = os.path.join(args.out, args.method_name, str(args.k))
    os.makedirs(save_dir, exist_ok=True)

    if args.dataset == 'imagenet':
        batch_size = 896
        epochs = 100
        base_lr = 0.7
        num_classes = 1000
    else:
        batch_size = 512
        epochs = 160
        base_lr = 0.4
        num_classes = 100

    train_loader, test_loader, _ = get_dataloaders(
        dataset_name=args.dataset, data_dir=args.data, batch_size=batch_size,
        num_workers=2, scores_path=args.scores_path, k=args.k, seed=args.seed
    )

    model = get_resnet50(dataset_name=args.dataset, num_classes=num_classes).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, dataset_name=args.dataset, epochs=epochs,
        steps_per_epoch=len(train_loader), base_lr=base_lr
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    best_test_acc = 0.0

    log_file = os.path.join(save_dir, f'training_log_seed{args.seed}.csv')
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'LR', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Time(s)'])

    print(f"Task {os.environ.get('SLURM_PROCID', 0)}: Training begins")
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

            optimizer.step()
            scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        train_loss = train_loss_sum / train_total
        train_acc = 100. * train_correct / train_total

        test_loss, test_acc = evaluate(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, current_lr, train_loss, train_acc, test_loss, test_acc, epoch_time])

        if epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'seed': args.seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_test_acc': best_test_acc
            }
            final_ckpt_name = f"final_checkpoint_seed{args.seed}.pth"
            torch.save(checkpoint, os.path.join(save_dir, final_ckpt_name))
            print(f"Task {os.environ.get('SLURM_PROCID', 0)} (Seed {args.seed}): Saved final checkpoint to {os.path.join(save_dir, final_ckpt_name)}")

    print(f"Task {os.environ.get('SLURM_PROCID', 0)}: Training complete!")


def train_with_tracein(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Dataset: {args.dataset.upper()} | Mode: Tracking TracIn | Seed: {args.seed}")

    tracein_dir = os.path.join(args.out, 'tracein_xps')
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(tracein_dir, exist_ok=True)

    run_name = get_run_name(args, "tracein")

    if args.dataset == 'imagenet':
        batch_size = 896
        epochs = 100
        base_lr = 0.7
        num_classes = 1000
    else:
        batch_size = 512
        epochs = 160
        base_lr = 0.4
        num_classes = 100

    train_loader, test_loader, _ = get_dataloaders(
        dataset_name=args.dataset, data_dir=args.data, batch_size=batch_size,
        num_workers=2, scores_path=args.scores_path, k=args.k, seed=args.seed
    )

    model = get_resnet50(dataset_name=args.dataset, num_classes=num_classes).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, dataset_name=args.dataset, epochs=epochs,
        steps_per_epoch=len(train_loader), base_lr=base_lr
    )

    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'dataset'):
        raw_dataset_len = len(train_dataset.dataset) if hasattr(train_dataset, 'indices_to_keep') else len(train_dataset)
    else:
        raw_dataset_len = len(train_dataset)

    TraceIn_scores = np.zeros(raw_dataset_len, dtype=np.float32)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    best_test_acc = 0.0

    log_file = os.path.join('checkpoints', f'training_log_{run_name}.csv')
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

            current_lr = optimizer.param_groups[0]['lr']
            sq_norms = get_batch_grad_norms(model, inputs, targets, chunk_size=16)
            TraceIn_scores[indices.cpu().numpy()] += (current_lr * sq_norms.cpu().numpy())

            optimizer.step()
            scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        train_loss = train_loss_sum / train_total
        train_acc = 100. * train_correct / train_total

        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, current_lr, train_loss, train_acc, test_loss, test_acc, epoch_time])

        checkpoint = {
            'epoch': epoch + 1,
            'seed': args.seed,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'TraceIn_scores': TraceIn_scores,
            'best_test_acc': best_test_acc
        }

        torch.save(checkpoint, os.path.join(args.out, f'checkpoint_{run_name}.pth'))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(checkpoint, os.path.join(args.out, f'checkpoint_best_{run_name}.pth'))

        if (epoch + 1) % 20 == 0:
            ckpt_name = f'checkpoint_{run_name}_epoch{epoch+1}.pth'
            torch.save(checkpoint, os.path.join(tracein_dir, ckpt_name))

    np.save(os.path.join(args.out, f'tracein_scores_{run_name}.npy'), TraceIn_scores)
    print(f"Training complete! TraceIn_scores saved for seed {args.seed}.")


def train_with_exact_gradient_deviation(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Dataset: {args.dataset.upper()} | Mode: Tracking Gradients | Seed: {args.seed}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    run_name = get_run_name(args, "grad")

    if args.dataset == 'imagenet':
        batch_size = 896
        epochs = 100
        base_lr = 0.7
        num_classes = 1000
    else:
        batch_size = 512
        epochs = 160
        base_lr = 0.4
        num_classes = 100

    train_loader, test_loader, _ = get_dataloaders(
        dataset_name=args.dataset, data_dir=args.data, batch_size=batch_size,
        num_workers=2, scores_path=args.scores_path, k=args.k, seed=args.seed
    )

    model = get_resnet50(dataset_name=args.dataset, num_classes=num_classes).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, dataset_name=args.dataset, epochs=epochs,
        steps_per_epoch=len(train_loader), base_lr=base_lr
    )

    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'dataset'):
        raw_dataset_len = len(train_dataset.dataset) if hasattr(train_dataset, 'indices_to_keep') else len(train_dataset)
    else:
        raw_dataset_len = len(train_dataset)

    G_scores = np.zeros(raw_dataset_len, dtype=np.float32)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    best_test_acc = 0.0

    log_file = os.path.join('checkpoints', f'training_log_{run_name}.csv')
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
            deviations = get_batch_deviations(model, batch_grads, inputs, targets, chunk_size=32)
            G_scores[indices.cpu().numpy()] += deviations.cpu().numpy()

            optimizer.step()
            scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        train_loss = train_loss_sum / train_total
        train_acc = 100. * train_correct / train_total

        test_loss, test_acc = evaluate(model, test_loader, device)
        epoch_num = epoch + 1
        print(f"Epoch [{epoch_num}/{epochs}] | LR: {current_lr:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch_num, current_lr, train_loss, train_acc, test_loss, test_acc, epoch_time])

        checkpoint = {
            'epoch': epoch_num,
            'seed': args.seed,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'G_scores': G_scores,
            'best_test_acc': best_test_acc
        }
        torch.save(checkpoint, os.path.join(args.out, f'checkpoint_{run_name}.pth'))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(checkpoint, os.path.join(args.out, f'checkpoint_best_{run_name}.pth'))

        # --- SAVE SCORES PROGRESSION TRAJECTORY ---
        if args.save_epoch_scores and (epoch_num <= 10 or epoch_num % 10 == 0):
            history_file_name = f"epoch_scores_{run_name}_epoch{epoch_num}.npy"
            np.save(os.path.join(args.out, history_file_name), G_scores)
            print(f" Saved score snapshot to {history_file_name}")

    np.save(os.path.join(args.out, f'batch_gradient_deviation_scores_{run_name}.npy'), G_scores)
    print(f"Training complete! Final G_scores saved for seed {args.seed}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exact Gradient Deviation Tracking')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--normal_train', action='store_true', help='If set, run normal training.')
    parser.add_argument('--tracein_train', action='store_true', help='If set, run training tracking TracIn.')
    parser.add_argument('--scores_path', type=str, default=None, help='Path to .npy file containing scores.')
    parser.add_argument('--k', type=int, default=0, help='Number of lowest scoring points to exclude.')
    parser.add_argument('--method_name', type=str, default='baseline', help='Name of the attribution method.')
    parser.add_argument('--out', type=str,
                        default=os.path.join(os.environ["SCRATCH"], "attribution_training_runs") if "SCRATCH" in os.environ else "./runs",
                        help='Output directory for saved files.')
    parser.add_argument('--data', type=str,
                        # May be copied from the cluster using:
                        # tar -xzf /network/datasets/cifar100/cifar-100-python.tar.gz -C ./data
                        default=os.environ.get('SLURM_TMPDIR', './data'),
                        help='Datasets directory.')
    parser.add_argument('--save_epoch_scores', action='store_true', help='Save score files sequentially over epochs.')
    args = parser.parse_args()

    if 'SLURM_PROCID' in os.environ:
        array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        num_tasks = int(os.environ.get('SLURM_NUM_TASKS', 1))
        proc_id = int(os.environ.get('SLURM_PROCID', 0))
        args.seed = (array_id * num_tasks) + proc_id

    if args.normal_train:
        train_normal(args)
    elif args.tracein_train:
        train_with_tracein(args)
    else:
        train_with_exact_gradient_deviation(args)
