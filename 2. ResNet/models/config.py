from dataclasses import dataclass
import torch

@dataclass
class ResNetConfig:
    classes: tuple = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    batch_size_train: int = 32
    batch_size_val: int = 16
    
    num_layers: int = 3
    
    epochs: int = 32
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name: str = "resnet"
    dataset: str="cifar10"
    