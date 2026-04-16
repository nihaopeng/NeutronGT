import torch


def sync_device(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
