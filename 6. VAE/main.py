import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader 
from models.VAEUtils import LSUNDataset, VAEUtils
from utils.utils import initialize_weights
from models.trainer import VAETrainer
import wandb

# ===========================
# PARAMETERS
# ===========================
BATCH_SIZE = 512
EPOCHS = 30

WARMUP = 0.2

# model architecture 
D_Z = 512
USE_LEAKY = True
LEAKYSLOPE = 0.2
L = 1

# Optimzier 
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
WEIGHT_DECAY = 0.01

LSUN_DATA_ROOT_DIR = f"./data/lsun/bedroom/"
device: str = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================

def setup(ds_name: str = "MNIST"):
    if ds_name == "MNIST":
        tranform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.MNIST(root='./data', train=True, transform=tranform)   
        
        chs = [1, 128, 256]
        
    elif ds_name == "LSUN":
        tranform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        dataset = LSUNDataset(root_dir=LSUN_DATA_ROOT_DIR, transform=tranform)
        
        chs = [3, 128, 256, 512, 1024]
    else:
        NotImplementedError()
        
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)  
    
    if ds_name == "MNIST":
        p_dist = "bernoulli"
    elif ds_name == "LSUN":
        p_dist = "gaussian"
    else:
        NotImplementedError()
    
    models = VAEUtils(
        d_z=D_Z,
        chs=chs,
        L=L,
        ds_name=ds_name,
        use_leaky=USE_LEAKY,
        leakySlope=LEAKYSLOPE,
        p_dist=p_dist
    ).to(device)
    
    return train_loader, models

def train(train_loader, models, ds_name):
    
    initialize_weights(models.encoder, name="normal", mean=0, std=0.01)
    initialize_weights(models.decoder, name="normal", mean=0, std=0.01)
    
    if ds_name == "LSUN":
        criterion = nn.MSELoss(reduction='sum')
    elif ds_name == "MNIST":
        criterion = nn.BCELoss(reduction='sum')
    else:
        NotImplementedError()
        
    optimizer = torch.optim.AdamW(models.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    
    total_steps = EPOCHS * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    num_sampling = 25
    
    fixed_noise = torch.randn((num_sampling, D_Z), device=device)
    
    trainer = VAETrainer(
        epochs=EPOCHS, models=models, 
        train_loader=train_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        log_period=50, wandb_log_period=10, save_period=5, fixed_noise=fixed_noise, num_sampling=num_sampling, ds_name=ds_name, total_steps=total_steps,
    )
    
    wandb.init(
        project="VAE", 
        name=f"train-VAE-ds_{ds_name}-dz{D_Z}",
        )
    
    trainer.train()
    
    wandb.finish()

def test(train_loader, models, ds_name):
    checkpoint = torch.load("VAE-LSUN-epoch40-d_z512.pth")

    models.load_state_dict(checkpoint['models_state_dict'])
    
    wandb.init(
        project="VAE", 
        name=f"test-VAE-ds_{ds_name}-dz{D_Z}",
        )
    models.test_reconstruction(train_loader, num_img=4)
    
    models.interpolate(num_img=16, steps=10)
    
    models.interpolate_some_dimension(dim=200, num_img=16, steps=10, val_range=(-5.0, 5.0))
    
    wandb.finish()
    

if __name__ == "__main__":
    ds_name = "LSUN"
    train_loader, models = setup(ds_name)
    
    #train(train_loader, models, ds_name)
    test(train_loader, models, ds_name)