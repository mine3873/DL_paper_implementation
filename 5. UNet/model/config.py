from dataclasses import dataclass
import torch

@dataclass
class UNetConfig:
    batch_size_train: int = 32
    batch_size_val: int = 16
    
    epochs: int = 32
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset: str="em"