from dataclasses import dataclass
import torch

@dataclass
class TransformerConfig:
    # model architecture
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers : int = 6
    dropout: float = 0.1
    pad_idx: int = 2
    
    # train parameters
    batch_size: int = 32
    epochs: int = 32
    warmup_steps: int = 4000
    beta1: int = 0.9
    beta2: int = 0.98
    epsilon: float = 1e-9
    label_smoothing: float = 0.1
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    
