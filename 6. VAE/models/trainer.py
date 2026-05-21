import wandb
import torch

class VAETrainer:
    def __init__(
        self, epochs, models, ds_name,
        train_loader, criterion, optimizer, scheduler, 
        log_period=10, wandb_log_period=10, save_period=5, fixed_noise=None, num_sampling=25,
        total_steps=None,
    ):
        self.epochs = epochs
        self.models = models
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_period = log_period
        self.wandb_log_period = wandb_log_period
        self.save_period = save_period
        self.fixed_noise = fixed_noise
        self.num_sampling = num_sampling
        self.ds_name = ds_name
        self.global_steps = 0
        self.total_steps = total_steps
        
        
        self.get_loss = models.get_loss
        self.generate = models.generate
        
    def train(self):
        device = self.models.device
        
        for epoch in range(self.epochs):
            self.models.train()
            total_train_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            for i, (img, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                img = img.to(device)
                
                model_outputs, mu, logvar = self.models(img)
                
                #beta = min(1.0, self.global_steps / anneal_steps)
                beta=1.0
                
                loss, recon_loss, kl_loss = self.get_loss(self.criterion, model_outputs, img, mu, logvar, beta)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.models.encoder.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.models.decoder.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_train_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                
                if i % self.log_period == 0 and i > 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                    
                if i % self.wandb_log_period == 0 and i > 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/beta": beta
                    },step=self.global_steps)
                
                self.global_steps += 1
                    
            self.generate(num_img=self.num_sampling, noise=self.fixed_noise, train=True, step=self.global_steps)
            
            wandb.log({
                "train/avg_loss": total_train_loss / len(self.train_loader),
                "train/avg_recon_loss": total_recon_loss / len(self.train_loader),
                "train/avg_kl_loss": total_kl_loss / len(self.train_loader),
            },step=self.global_steps)
            
            if epoch > 0 and ((epoch + 1) % self.save_period == 0 or (epoch + 1) == self.epochs):
                torch.save({
                    'epoch': epoch,
                    'models_state_dict': self.models.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, f"VAE-{self.ds_name}-epoch{epoch + 1}-d_z{mu.size(1)}.pth")
            
                    
                
                
            