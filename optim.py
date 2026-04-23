import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer_and_scheduler(model, dataset_name, epochs, steps_per_epoch, base_lr, momentum=0.9):
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum)
    
    total_steps = epochs * steps_per_epoch
    
    if dataset_name == 'cifar100':
        warmup_steps = int(0.15 * total_steps)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                decay_steps = total_steps - warmup_steps
                step_into_decay = current_step - warmup_steps
                return max(0.0, float(decay_steps - step_into_decay) / float(max(1, decay_steps)))
                
    elif dataset_name == 'imagenet':
        warmup_steps = 15 * steps_per_epoch
        
        def lr_lambda(current_step):
            current_epoch = current_step / float(max(1, steps_per_epoch))
            
            if current_epoch < 15.0:
                return float(current_step) / float(max(1, warmup_steps))
            elif current_epoch < 30.0:
                return 1.0
            elif current_epoch < 60.0:
                return 0.1
            elif current_epoch < 90.0:
                return 0.01
            else:
                return 0.001
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
        
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler