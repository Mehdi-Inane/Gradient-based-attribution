from datasets import load_dataset
import os

save_path = os.path.join(os.environ['SCRATCH'], 'hf_datasets', 'cifar100-c')

print(f"Downloading and saving CIFAR100-C to {save_path}...")
dataset = load_dataset("randall-lab/cifar100-c", split="test", trust_remote_code=True)
dataset.save_to_disk(save_path)
print("Done!")