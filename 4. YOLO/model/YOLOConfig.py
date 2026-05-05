from dataclasses import dataclass
import torch

@dataclass
class YOLOConfig:
    batch_size_train: int = 32
    batch_size_val: int = 16
    batch_size_test: int = 16
    warmup_epochs: int = 1
    epochs: int = 32
    
    lambda_coord: float = 5.
    lambda_noobj: float = .5
    
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    dropout: float = 0.5
    
    leakyRelu_w: float = 0.1
    
    eps: float = 1e-6
    
    data_root_path: str = 'VOCdevkit'
    
    years: tuple = ('2007', '2012')
    classes: tuple = (
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    )
    
    image_size: int = 224
    
    S: int = 7
    B: int = 2
    C: int = 20
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    