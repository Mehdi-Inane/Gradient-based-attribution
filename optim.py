import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer_and_scheduler(model, epochs, steps_per_epoch, base_lr=0.4, momentum=0.9):
    """
    Creates the SGD optimizer and the custom linear warmup/decay learning rate scheduler.
    Follows Feldman et al. (2020) for CIFAR-100:
    - SGD with momentum 0.9
    - Base learning rate: 0.4
    - Linear warmup from 0 to base_lr for the first 15% of iterations
    - Linear decay from base_lr to 0 for the remaining 85% of iterations
    """
    
    # Note: Feldman & Zhang do not explicitly mention weight decay in the snippet provided, 
    # but standard ResNet training often uses something like weight_decay=5e-4. 
    # We leave it at 0 to strictly follow the text provided, but you can adjust it here.
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum)
    
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.15 * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: multiplier goes from 0.0 to 1.0
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay: multiplier goes from 1.0 down to 0.0
            decay_steps = total_steps - warmup_steps
            step_into_decay = current_step - warmup_steps
            return max(0.0, float(decay_steps - step_into_decay) / float(max(1, decay_steps)))
            
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler