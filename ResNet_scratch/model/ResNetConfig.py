from dataclasses import dataclass
import torch

@dataclass
class ResNetConfig:
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2023, 0.1994, 0.2010)
    classes: tuple = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    batch_size_train: int = 32
    batch_size_val: int = 16
    num_workers: int = 2
    
    num_layers: int = 3
    num_channel: tuple = (3, 16, 32, 64)
    
    
    epochs: int = 32
    momentum: float = 0.9
    weight_decay : float = 0.0001
    gamma: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
