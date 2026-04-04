from dataclasses import dataclass
import torch

@dataclass
class MLPConfig:
    # model architecture
    
    # train parameters
    batch_size_train: int = 32
    batch_size_val: int = 32
    epochs: int = 32
    beta1: int = 0.9
    beta2: int = 0.98
    eps: float = 1e-9
    weight_decay: float = 0.1
    lr_min: float = 1e-6
    
    mean: float = 0.3081
    std: float = 0.1307
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
