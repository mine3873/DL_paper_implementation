from dataclasses import dataclass
import torch

@dataclass
class DiffusionConfig:
    batch_size_train: int = 64
    batch_size_val: int = 64

    total_steps: int = 200000
    num_timesteps: int = 1000
        
    log_interval: int = 500
    sample_interval: int = 1000
    
    
    num_resblock: int = 5

    betas: tuple = (1e-4, 2e-2)
    
    ema_decay: float = 0.9999
    
    lr: float = 2e-4
    dropout: float = 0.0
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"