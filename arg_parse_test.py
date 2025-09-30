'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l1")# "--opts",)
parser.add_argument("-l3")#, "--opts2",)

args = parser.parse_args()
l1 = float(args.l1)
l3 = float(args.l3)

print(type(l1), type(l3))
print(l1,l3)
'''
import torch
import os
import pytorch_lightning as pl

def check_devices():
    """
    Checks the number of visible CUDA devices from the perspective of PyTorch.
    """
    print(f"CUDA_VISIBLE_DEVICES environment variable is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    num_visible_gpus = torch.cuda.device_count()
    print(f"PyTorch can see {num_visible_gpus} GPU device(s).")
    return num_visible_gpus


if __name__ == "__main__":
    check_devices()


'''
import torch
def get_gpu_vram_info():
    """
    Checks and prints the VRAM information (free, total) for all available GPUs.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU devices found.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU device(s).")
    print("-" * 30)

    for i in range(num_gpus):
        # The torch.cuda.mem_get_info() function returns a tuple of
        # (free memory in bytes, total memory in bytes).
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=i)

        # Convert bytes to gigabytes for a more readable format.
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        
        # Get the name of the GPU for clarity
        gpu_name = torch.cuda.get_device_name(i)

        print(f"GPU {i}: {gpu_name}")
        print(f"  Total VRAM: {total_gb:.2f} GB")
        print(f"  Free VRAM:  {free_gb:.2f} GB")
        print("-" * 30)

if __name__ == "__main__":
    get_gpu_vram_info()
'''