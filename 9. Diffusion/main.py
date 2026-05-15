import wandb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from models import UNet
from config import DiffusionConfig
from trainer import DiffusionTrainer
# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 16

TOTAL_STEPS = int(2e5)
NUM_TIMESTEPS = 1000

LOG_INTERVAL = 100
SAMPLE_INTERVAL = 1000

NUM_RESBLOCK = 2
CH_MULTS = (1, 2, 2, 2)

BETAS = (1e-4, 2e-2)

EMA_DECAY = 0.9999

# optimizer
LR = 2e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 0.01

# scheduler
WARMUP_STEPS = TOTAL_STEPS * 0.1

DROPOUT = 0.0

# ==================================


def setup():
    config = DiffusionConfig(
        batch_size_train=BATCH_SIZE_TRAIN, batch_size_val=BATCH_SIZE_VAL,
        total_steps=TOTAL_STEPS, num_timesteps=NUM_TIMESTEPS,
        log_interval=LOG_INTERVAL, sample_interval=SAMPLE_INTERVAL,
        betas=BETAS, ema_decay=EMA_DECAY,
        lr=LR, dropout=DROPOUT, 
    )
    
    
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_transform = transforms.Compose([
        transforms.Resize(80), 
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) 
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(80), 
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) 
    ])

    train_dataset = torchvision.datasets.CelebA(
        root='./data', 
        split='train', 
        target_type='attr', 
        download=False, 
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CelebA(
        root='./data', 
        split='valid', 
        target_type='attr', 
        download=False, 
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False)
    
    model = UNet(
        num_resBlock=NUM_RESBLOCK,
        ch_mults=CH_MULTS,
        dropout=DROPOUT
        ).to(config.device)
    
    return config, model, train_loader, val_loader,

def train(config, model, train_loader, val_loader, ):
    wandb.init(
        project="Diffusion", 
        name=f"train-ds_CelebA-numRes{NUM_RESBLOCK}-bs{BATCH_SIZE_TRAIN}-",
        config=config.__dict__ if hasattr(config, '__dict__') else config,
        )
    
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(ADAM_BETA1, ADAM_BETA2), weight_decay=ADAM_WEIGHT_DECAY)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS,)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS)
    
    trainer = DiffusionTrainer(
        config=config, model=model, betas=torch.linspace(BETAS[0], BETAS[1], steps=NUM_TIMESTEPS, device=config.device),
        use_ema=True, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion="mle",
    )
    
    trainer.train()
    
    wandb.finish()
    

if __name__ == "__main__":
    config, model, train_loader, val_loader, = setup()
    
    train(config, model, train_loader, val_loader)