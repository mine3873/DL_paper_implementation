from dataclasses import dataclass
from CLIP.models.CLIPScratch import BottleNeckBlock
import torch

@dataclass
class CLIPConfig:
    # trainer
    batch_size: int = 1
    batch_size_test: int = 8
    epochs: int = 128
    patience: int = 5
    
    # img encoder 
    img_n_layer: str = '50'
    block: type = BottleNeckBlock
    
    # text encoder
    seq_len: int = 32
    vocab_size: int = 49408
    d_model: int = 512
    max_len: int = 77
    n_heads: int = 8
    dropout: float = 0.1
    n_layers: int = 6
    
    pad_idx: int = 0
    
    text_enc_out_dim: int = 512

    # CLIP 
    d_e: int = 512
    
    
    # optimizer 
    
    # scheduler 
    
    #criterion 
    
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.01
    
    
    
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"