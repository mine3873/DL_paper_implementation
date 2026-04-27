from dataclasses import dataclass
import torch

@dataclass
class VAEConfig:
    batch_size: int = 1
    
    epochs: int = 128

    lr: float = 0.0002
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    
    loss_beta: float = 1
    
    z_dim: int = 100
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"