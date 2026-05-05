from dataclasses import dataclass
import torch
@dataclass
class DCGANConfig:
    batch_size_train: int = 1

    epochs: int = 128

    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.01
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"