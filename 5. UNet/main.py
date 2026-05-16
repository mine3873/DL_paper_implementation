import wandb
import torch
import torch.nn as nn
import albumentations as Album
import numpy as np
from torch.utils.data import DataLoader
from model.UNet_utils import ISBIDataset, Loss
from model.model import UNet
from model.config import UNetConfig
from utils.utils import initialize_weights
from model.trainer import UNetTrainer
from transformers import get_cosine_schedule_with_warmup

# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 1


NUM_LAYERS = 3

EPOCHS = 500

LR = 0.02
MONENTUM = 0.99

WARMUP = 0.1

WEIGHT_DECAY = 0.0001



DATA_ROOT_DIR = "./data/ISBI-2012"
IMG_SIZE = 572
# ==================================



def setup(ds_name = "em"):
    
    config = UNetConfig(
        epochs=EPOCHS,
    )
    
    assert ds_name == "em"
    if ds_name == "em":
        train_transform = Album.Compose([
            Album.Resize(IMG_SIZE, IMG_SIZE),
            Album.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.5),
            Album.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-180, 180), 
                p=0.5
            ),
            Album.RandomRotate90(),
            Album.HorizontalFlip(),
            Album.VerticalFlip(),
            
            Album.GaussNoise(std_range=(0.01, 0.1), p=0.3),
            Album.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            
            Album.Normalize(mean=(0.5,), std=(0.5,)),
        ], additional_targets={'weight_map': 'mask'})
        
        val_transform = Album.Compose([
            Album.Resize(IMG_SIZE, IMG_SIZE),
            Album.Normalize(mean=(0.5,), std=(0.5,)),
        ], additional_targets={'weight_map': 'mask'})
        
        train_dataset = ISBIDataset(
            data_root_dir=DATA_ROOT_DIR, train="train", transform=train_transform)
        
        val_dataset = ISBIDataset(
            data_root_dir=DATA_ROOT_DIR, train="test", transform=val_transform)
    
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL)

    else:
        NotImplementedError()
    
    model = UNet().to(config.device)
    initialize_weights(model, name="he")
    
    return config, model, train_loader, val_loader

def train(config, model, train_loader, val_loader):
    wandb.init(
        project="UNet", 
        name=f"train-UNet",
        config=config.__dict__ if hasattr(config, '__dict__') else config,
        )
    
    criterion = Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MONENTUM, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = total_steps * WARMUP
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    
    trainer = UNetTrainer(
        config=config, model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        log_period=2, wandb_log_period=1)
    
    trainer.train()
    
    wandb.finish()

if __name__ == "__main__":
    config, model, train_loader, val_loader = setup()
    
    train(config, model, train_loader, val_loader)
    