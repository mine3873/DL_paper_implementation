import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from .utils.AnimeDS import AnimeFaceDataSet
import wandb
from .models.Gan import Discriminator
from .models.Auto_Encoder import Vae
from .models.UNet import UNet, EMA
from .utils.loss import AutoEncoderLoss, LDM_With_FM_Loss
from .utils.utils import AdaptiveWeight, get_scaling_factor

IMG_SIZE = 96
BS = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = "./data/faces"

def load_dataset():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = AnimeFaceDataSet(data_dir, transform=transform)
    
    train_loader = DataLoader(
        dataset, batch_size=BS, shuffle=True, pin_memory=True, num_workers=4
    )
    
    return train_loader

def train_AutoEncoder(
    train_loader,
    in_channel, ch, img_size,
    train_warmup_step, gan_warm_step, total_steps,
    kl_weight,
    log_period, wandb_log_period, save_period, sample_period,
    ):
    model = Vae(
        in_channels=in_channel, ch=ch, img_size=img_size,
        ch_mults=(1,2,4,), 
        z_channel=4, num_resblock=2,
        attn_resolution=(24,), dropout=0.1,
        with_asymmetric_pad=False, with_conv=False, bias=False,
        regularizer='KL'
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.99)
        )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer, start_factor=1e-3, end_factor=1.0, total_iters=train_warmup_step
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=total_steps - train_warmup_step
            )
        ],
        milestones=[train_warmup_step]
    )
    
    disc = Discriminator(
        in_channels=in_channel, ch=ch, leLU_slope=0.2
    ).to(device)
    
    optimizer_disc = torch.optim.AdamW(
        disc.parameters(), lr=5e-5, betas=(0.9, 0.99)
    )
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, T_max=total_steps)
    
    criterion = AutoEncoderLoss().to(device)
    
    adaptive_weight_fn = AdaptiveWeight(disc_start_step=gan_warm_step)
    last_layer_params = model.decoder.out_conv.weight
    
    data_iter = iter(train_loader)
    
    wandb.init(
        project="LDM", 
        name=f"train-AutoEncoder"
    )
    
    model.train()
    disc.train()
    for i in range(total_steps):
        try: 
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)
        x = x.to(device)
        
        recon_x, mu, logvar = model(x)
        
        recon_loss, gan_loss, kl_loss = criterion(
            x=x, recon_x=recon_x, mu=mu, logvar=logvar, disc=disc, for_AE=True
        )
        
        adaptive_weight = adaptive_weight_fn.get_weight(
            recon_loss=recon_loss, gan_loss=gan_loss, last_layer_params=last_layer_params, step=i
        )
        
        total_vae_loss = recon_loss + adaptive_weight * gan_loss + kl_loss * kl_weight
        
        optimizer.zero_grad()
        total_vae_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i >= gan_warm_step:
            disc_loss = criterion(
                x=x, recon_x=recon_x.detach(), disc=disc, for_AE=False
            )
            
            optimizer_disc.zero_grad()
            disc_loss.backward()
            optimizer_disc.step()
            scheduler_disc.step()
        else:
            disc_loss = torch.tensor(0.0).to(device)
        
        if i % log_period == 0 and i > 0:
            print(
                f"step {i}, total vae loss: {total_vae_loss.item():.4f}, dics_loss: {disc_loss.item():.4f}, lr:{scheduler.get_last_lr()[0]:.6f}"
            )
        
        if i % wandb_log_period == 0 and i > 0:
            wandb.log({
                'train/vae loss': total_vae_loss.item(),
                'train/disc loss': disc_loss.item(),
                'train/lr': scheduler.get_last_lr()[0]
            }, step=i)
        
        if i % sample_period == 0 and i > 0:
            grid = model.sample(x=x, recon_x=recon_x)
            wandb.log({
                f'sample/recon': wandb.Image(
                    grid.clamp(0, 1), caption=f"top: original / bottom: reconstruction "
                )
            }, step=i)
        
        if (i % save_period == 0 or i == total_steps - 1) and i > 0:
            torch.save({
                'step': i+1,
                'models_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"VAE-step_{i+1}.pth")
        
    wandb.finish()

def train_LDM(
    train_loader,
    in_channel, z_channel, vae_ch, ch, img_size, time_dim,
    train_warmup_step, total_steps,
    log_period, wandb_log_period, save_period, sample_period,
):
    vae = Vae(
        in_channels=in_channel, ch=vae_ch, img_size=img_size,
        ch_mults=(1,2,4,), 
        z_channel=4, num_resblock=2,
        attn_resolution=(24,), dropout=0.1,
        with_asymmetric_pad=False, with_conv=False, bias=False,
        regularizer='KL'
    ).to(device)
    
    vae_checkpoint = torch.load("VAE-step_100000.pth")

    vae.load_state_dict(vae_checkpoint['models_state_dict'])
    
    vae.eval()
    
    scaling_factor = get_scaling_factor(vae=vae, train_loader=train_loader, device=device, num_batches=30)
    
    model = UNet(
        in_channels=z_channel, ch=ch, img_size=img_size,
        time_dim=time_dim, ch_mults=(1, 2, 4), num_resblock=4,
        attn_resolutions=(24, 12), dropout=0.1
    ).to(device)
    
    ema = EMA(model=model, decay=0.999)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.99), 
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer, start_factor=1e-3, end_factor=1.0, total_iters=train_warmup_step
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=total_steps - train_warmup_step
            )
        ],
        milestones=[train_warmup_step]
    )
    
    criterion = LDM_With_FM_Loss()
    
    sigma_min = 1e-5
    
    def get_psi(x1, x0, sigma_min, t_exp):
        return t_exp * x1 + (1 - (1-sigma_min) * t_exp) * x0
    
    data_iter = iter(train_loader)
    
    wandb.init(
        project="LDM", 
        name=f"train-LDM"
    )
    
    wandb.log({
        "train/params": scaling_factor
    }, step=0)
    
    for i in range(total_steps):
        try: 
            x1, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x1, _ = next(data_iter)
        x1 = x1.to(device)
        
        with torch.no_grad():
            x1 = vae.encode_z_DDPM(x1, scaling_factor=scaling_factor)
        
        t = torch.rand(x1.shape[0], device=x1.device)
        t_exp = t.view(-1, 1, 1, 1)
        
        x0 = torch.randn_like(x1)
        
        psi_t = get_psi(x1, x0, sigma_min, t_exp)
        
        model_output = model(psi_t, t)
        
        loss = criterion(model_output, x0, x1, sigma_min)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        scheduler.step()
        
        if i % log_period == 0 and i > 0:
            print(
                f"step {i}, total loss: {loss.item():.4f}, lr:{scheduler.get_last_lr()[0]:.6f}"
            )
        
        if i % wandb_log_period == 0 and i > 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': scheduler.get_last_lr()[0]
            }, step=i)
        
        if i % sample_period == 0 and i > 0:
            z = model.sample(x1, steps=50, ema=ema)
            z = vae.decode_z_DDPM(z, scaling_factor=scaling_factor)
            grid = vae.sample(z)
            wandb.log({
                f'sample/generated': wandb.Image(
                    grid.clamp(0, 1), caption=f"step{i} gerneration"
                )
            }, step=i)
            
            
        if (i % save_period == 0 or i == total_steps - 1) and i > 0:
            torch.save({
                'step': i+1,
                'models_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaling_factor': scaling_factor,
            }, f"LDM-step_{i+1}.pth")
            
    wandb.finish()
        

if __name__ == "__main__":
    train_loader = load_dataset()
    x, _ = next(iter(train_loader))
    
    """
    train_AutoEncoder(
        train_loader=train_loader,
        in_channel=x.size(1),
        ch=128, img_size=x.size(-1), 
        train_warmup_step=1500, gan_warm_step=8000, total_steps=100000, 
        kl_weight=1e-5,
        log_period=100, wandb_log_period=50, save_period=10000, sample_period=1000
    )
    """
    
    
    train_LDM(
        train_loader=train_loader, 
        in_channel=x.size(1), z_channel=4, vae_ch=128, ch=256, img_size=x.size(-1),
        time_dim=256, train_warmup_step=2000, total_steps=150000, 
        log_period=100, wandb_log_period=50, save_period=10000, sample_period=5000
    )
    