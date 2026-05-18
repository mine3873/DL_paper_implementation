import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.utils import initialize_weights
from models.DCGANutils import LSUNDataset, ModelsUtils
from transformers import get_cosine_schedule_with_warmup
from models.trainer import DCGANTrainer

# ==================================
# PARAMETERS & PATHS
# ==================================
EPOCHS = 40

BATCH_SIZE_TRAIN = 512
BATCH_SIZE_VAL = 32

#optimzier
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

#scheduler
#WARMUP_RATIO = 0.1

D_Z = 100

LEAKYSLOPE = 0.2

LSUN_ROOT_DIR = f"./data/lsun/bedroom/"

device: str = "cuda" if torch.cuda.is_available() else "cpu"
# ==================================


def setup(ds_name: str = "MNIST"):
    if ds_name == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        
        
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False)
        
        chs = [1,64,128]
        
    elif ds_name == "LSUN":
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        dataset = LSUNDataset(root_dir=LSUN_ROOT_DIR, transform=transform)
        
        total_size = len(dataset)
        train_size = int(0.99 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(30))
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False)
        
        chs = [3, 128, 256, 512, 1024]
    
    models = ModelsUtils(d_z=D_Z, chs=chs, ds_name=ds_name, leakySlope=LEAKYSLOPE)
    models.to(device)
    
    return models, train_loader, val_loader, ds_name

def train(models, train_loader, val_loader, ds_name):
    wandb.init(
        project="DCGAN", 
        name=f"train-DCGAN-ds_{ds_name}",
        )
    
    initialize_weights(models.D, name="normal", mean=0, std=0.02)
    initialize_weights(models.G, name="normal", mean=0, std=0.02)
    
    criterion = nn.BCELoss()
    
    optimzier_D = torch.optim.AdamW(params=models.D.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=0.01)
    optimzier_G = torch.optim.AdamW(params=models.G.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimzier_D, T_max=total_steps)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimzier_G, T_max=total_steps)
    
    
    num_sampling = 25
    trainer = DCGANTrainer(
        epochs=EPOCHS, models=models, 
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizers=(optimzier_D, optimzier_G), schedulers=(scheduler_D, scheduler_G),
        log_period=50, wandb_log_period=10, device=device, num_sampling=num_sampling, save_model_period=5, fixed_noise=torch.randn(num_sampling, D_Z, device=device)
    )
    
    trainer.train()
    
    wandb.finish()
    
def test(models, ds_name):
    wandb.init(
        project="DCGAN", 
        name=f"test-DCGAN-ds_{ds_name}",
        )
    checkpoint = torch.load("DCGAN-epoch40.pth")

    models.G.load_state_dict(checkpoint['model_G_state_dict'])
    
    models.interpolate(steps=10, use_wandb=True)
    models.interpolate_some_dimension(dim=20)
    wandb.finish()
    

if __name__ == "__main__":
    models, train_loader, val_loader, ds_name = setup(ds_name="LSUN")
    
    #train(models, train_loader, val_loader, ds_name)
    test(models, ds_name)
    
