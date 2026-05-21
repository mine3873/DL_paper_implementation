import wandb
import torch
from utils import DiffusionUtils, EMA
from torchvision.utils import save_image

class DiffusionTrainer:
    def __init__(
        self,
        config, model, betas, use_ema: bool = True,
        
        train_loader=None, val_loader=None,
        optimizer=None, scheduler=None, criterion=None
        ):
        self.config = config
        self.model = model
        
        self.use_ema = use_ema
        self.ema = EMA(self.model, decay=config.ema_decay) if use_ema else None
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        self.utils = DiffusionUtils(betas=betas, criterion=criterion)
        
    def train(self):
        device = self.config.device
        self.model.train()

        data_iter = iter(self.train_loader)
         
        num_timesteps = self.config.num_timesteps
        
        for step in range(self.config.total_steps):
            try: 
                x0, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x0, _ = next(data_iter)
                
            x0 = x0.to(device)
            B = x0.shape[0]
            
            t = torch.randint(0, num_timesteps, (B,), device=device).long()
            
            self.optimizer.zero_grad()
            
            loss = self.utils.loss(self.model, x0, t)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if self.use_ema:
                self.ema.update()
                
            if step % self.config.log_interval == 0:
                print(f"Step {step}/{self.config.total_steps}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                }, step=step)
            
            if step > 0 and step % self.config.sample_interval == 0:
                self.save_samples(step)
                
    
    @torch.no_grad()
    def save_samples(self, step, shape): 
        
        if self.use_ema:
            self.ema.apply_shadow()
            
        samples = self.utils.p_sample_loop(self.model, shape)
        
        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)
        
        save_image(samples, f"sample_{step}.png", nrow=int(shape[0]**0.5))
        
        wandb.log({
            "train/samples": [wandb.Image(samples)]
        },step=step)
        
        if self.use_ema:
            self.ema.restore()
            
        self.model.train()
            
            