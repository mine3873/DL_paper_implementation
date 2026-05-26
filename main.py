import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader 
from diffussionUtils import LSUNDataset, DiffusionUtils, EMA
from models import UNet

# =============================
# PARAMETERS
# =============================
BATCH_SIZE = 64

TOTAL_STEPS = int(2e5)

# model architecture
CH = 128
NUM_RESBLOCK = 2
ATTN_IMG_SIZE = 16

# Optimzier
LR = 2e-4

# Scheduler 

LSUN_DATA_ROOT_DIR = f"./data/lsun/bedroom/"
device: str = "cuda" if torch.cuda.is_available() else "cpu"
# =============================

def setup(ds_name):
    if ds_name == "LSUN":
        tranform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        dataset = LSUNDataset(root_dir=LSUN_DATA_ROOT_DIR, transform=tranform)
        # 64 - 32 - 16 - 8 - 4
        chs = [1, 1, 2, 2]
        dropout = 0.0
    elif ds_name == "CIFAR10":
        tranform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset = datasets.CIFAR10(root='./data', train=True, transform=tranform)
        # 32 - 16 - 8 - 4
        chs = [1, 2, 2, 2]
        dropout = 0.1
    else:
        NotImplementedError(ds_name)
        
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True) 

    data, _ = next(iter(train_loader))
    
    model = UNet(
        in_ch=data.shape[1], img_size=data.shape[-1],
        ch=CH, chs=chs,
        num_resBlock=2, dropout=dropout, attn_img_size=ATTN_IMG_SIZE).to(device)
    
    criterion = nn.MSELoss(reduction="sum")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
    
    return train_loader, model, criterion, optimizer, scheduler

def train(
    train_loader, model, ds_name,
    criterion, optimizer, scheduler,
    total_timesteps = 1000, scheduler_betas_start=1e-4, scheduler_betas_end=2e-2,
    log_period = 10, wandb_log_period=10, save_model_period = 1e4, save_sample_period=10
    ):
    
    betas = torch.linspace(scheduler_betas_start, scheduler_betas_end, steps=total_timesteps, device=device)
    
    diffusion_utils = DiffusionUtils(betas=betas)
    ema = EMA(model, decay=0.9999)
    
    wandb.init(
        project="Diffusion", 
        name=f"train-ds_{ds_name}")
    
    data_iter = iter(train_loader)
    model.train()
    
    for step in range(TOTAL_STEPS):
        try: 
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x0, _ = next(data_iter)
            
        x0 = x0.to(device)
        shape = x0.shape
        
        t = torch.randint(0, total_timesteps, (shape[0],), dtype=torch.long)
        x_t = diffusion_utils.q_sample(x0, t, noise=torch.randn(shape, device=x0.device))
        
        model_output = model(x_t, t)
        
        loss = criterion(model_output, torch.randn(shape, device=x0.device))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % log_period == 0:
            print(f"Step {step}/{TOTAL_STEPS}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if step % wandb_log_period == 0 and step > 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }, step=step)
        
        if step % save_sample_period == 0 and step > 0:
            ema.apply_shadow()
            
            samples = diffusion_utils.p_sample_loop(total_timesteps, x0.shape)
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)
            
            save_image(samples, f"sample_{step}.png", nrow=int(x0.shape[0]**0.5))
            
            wandb.log({
                "train/samples": [wandb.Image(samples)]
            },step=step)
            
            ema.restore()
        
        if (step % save_model_period == 0 or step == TOTAL_STEPS - 1) and step > 0:
            torch.save({
                'step': step,
                'models_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"Diffusion-{ds_name}-step_{step}.pth")
            
    wandb.finish()

if __name__ == "__main__":
    ds_name="LSUN"
    train_loader, model, criterion, optimizer, scheduler = setup(ds_name)
    
    train(
        train_loader, model, ds_name, criterion, optimizer, scheduler, 
        total_timesteps=1000, scheduler_betas_start=1e-4, scheduler_betas_end=2e-2,
        log_period=10, wandb_log_period=10, save_model_period=1e4, save_sample_period=1e4
    )