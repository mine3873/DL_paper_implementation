import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader 
from models.diffussionUtils import LSUNDataset, DiffusionUtils, EMA
from models.models import UNet
from PIL import Image

# =============================
# PARAMETERS
# =============================
BATCH_SIZE = 32

TOTAL_STEPS = int(1.5e5)

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
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        dataset = LSUNDataset(root_dir=LSUN_DATA_ROOT_DIR, transform=transform)
        # 64 - 32 - 16 - 8 - 4
        chs = [1, 1, 2, 2]
        dropout = 0.0
    elif ds_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset = datasets.CIFAR10(root='./data', train=True, transform=transform)
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
    
    criterion = nn.MSELoss(reduction="mean")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
    
    return train_loader, model, criterion, optimizer, scheduler

def forward_Process_With_certain_img():
    image_path = f"Others/DORO.png"
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img = transform(img).unsqueeze(0).to(device)
    
    total_timesteps = 1000
    scheduler_betas_start=1e-4
    scheduler_betas_end=2e-2
    
    betas = torch.linspace(scheduler_betas_start, scheduler_betas_end, steps=total_timesteps, device=device)
    
    diffusion_utils = DiffusionUtils(betas=betas)
    
    diffusion_utils.forward_process_steps(img)

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
        noise = torch.randn(shape, device=x0.device)
        
        t = torch.randint(0, total_timesteps, (shape[0],), dtype=torch.long, device=x0.device)
        x_t = diffusion_utils.q_sample(x0, t, noise)
        
        model_output = model(x_t, t)
        
        loss = criterion(model_output, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        ema.update()
        
        if step % log_period == 0:
            print(f"Step {step}/{TOTAL_STEPS}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if step % wandb_log_period == 0 and step > 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }, step=step)
        
        if step % save_sample_period == 0 and step > 0:
            ema.apply_shadow()
            
            samples = diffusion_utils.p_sample_loop(model, total_timesteps, shape)
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)
            
            save_image(samples, f"sample_{step}.png", nrow=int(shape[0]**0.5))
            
            grid = make_grid(samples, nrow=int(shape[0]**0.5))
            
            wandb.log({
                "train/samples": [wandb.Image(grid)]
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

def test(model, total_timesteps = 1000, scheduler_betas_start=1e-4, scheduler_betas_end=2e-2,):
    checkpoint = torch.load("Diffusion-LSUN-step_149999.pth")

    model.load_state_dict(checkpoint['models_state_dict'])
    
    betas = torch.linspace(scheduler_betas_start, scheduler_betas_end, steps=total_timesteps, device=device)
    
    diffusion_utils = DiffusionUtils(betas=betas)
    
    diffusion_utils.p_sample_loop(model, total_timesteps, shape=[1, 3, 64, 64], save_imgs=True)
    
def test_interpolation(model, total_timesteps = 1000, scheduler_betas_start=1e-4, scheduler_betas_end=2e-2, ):
    img1 = Image.open("inter_img1.jpg").convert("RGB")
    img2 = Image.open("inter_img2.jpg").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    
    checkpoint = torch.load("Diffusion-LSUN-step_149999.pth")

    model.load_state_dict(checkpoint['models_state_dict'])
    
    betas = torch.linspace(scheduler_betas_start, scheduler_betas_end, steps=total_timesteps, device=device)
    
    diffusion_utils = DiffusionUtils(betas=betas)
    
    diffusion_utils.interpolate(
        model, img1, img2, t_idx=500, num_lambdas=7
    )

if __name__ == "__main__":
    ds_name="LSUN"
    train_loader, model, criterion, optimizer, scheduler = setup(ds_name)
    
    #test_interpolation(model)
    
    """
    train(
        train_loader, model, ds_name, criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
        total_timesteps=1000, scheduler_betas_start=1e-4, scheduler_betas_end=2e-2,
        log_period=100, wandb_log_period=50, save_model_period=3e4, save_sample_period=5e3
    )
    """