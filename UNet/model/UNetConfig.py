from dataclasses import dataclass
import torch

@dataclass
class UNetConfig:
    batch_size_train: int = 32
    batch_size_val: int = 16
    
    epochs: int = 10
    
    image_size: int = 572
    
    lr: float = 0.1
    momentum: float = 0.99
    dropout: float = 0.1
    
    eps: float = 1e-6
    lr_min: float = 1e-6
    
    transfrom_alpha: float = 0.5
    transfrom_sigma: float = 5.0
    
    weight_w0: float = 10.0
    weight_sigma: float = 5.0
    
    weight_decay: float = 0.5
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"