import wandb
import torch

class DCGANTrainer:
    def __init__(
        self, epochs, models, 
        train_loader=None, val_loader=None,
        criterion=None, optimizers=None, schedulers=None,
        log_period=5, wandb_log_period=5, device = "cuda",
        num_sampling=9, save_model_period=5, fixed_noise=None
        ):
        
        self.epochs = epochs
        self.models = models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizers = optimizers
        self.schedulers = schedulers
        
        self.log_period = log_period
        self.wandb_log_period = wandb_log_period
        self.global_steps = 0
        
        self.device = device
        
        self.num_sampling = num_sampling
        
        self.save_model_period = save_model_period
        self.fixed_noise = fixed_noise
        
    def train(self):
        device = self.device
        D, G = self.models.D, self.models.G
        opt_D, opt_G = self.optimizers
        sch_D, sch_G = self.schedulers
        
        get_loss = self.models.get_loss
        get_noise = self.models.get_noise
        
        for epoch in range(self.epochs):
            D.train()
            G.train()
            total_train_loss_D = 0
            total_train_loss_G = 0
            
            for i, (img, _) in enumerate(self.train_loader):
                img = img.to(device)
                
                # maximize log D(x) + log(1 - D(G(z)))
                opt_D.zero_grad()
                
                true_label_for_D = torch.full((img.size(0),), 0.9, device=device)
                output_D_from_data = D(img)
                loss_D_from_data = get_loss(self.criterion, output_D_from_data, true_label_for_D)
                loss_D_from_data.backward()
                
                z = get_noise(img.size(0)).to(device)
                G_z_output = G(z)
                
                output_D_from_fake = D(G_z_output.detach())
                
                false_label = torch.full((img.size(0),), 0.0, device=device)
                loss_D_from_fake = get_loss(self.criterion, output_D_from_fake, false_label)
                loss_D_from_fake.backward()
                
                # minimize log(1 - D(G(z)))
                opt_G.zero_grad()
                
                output_D_from_fake_for_G = D(G_z_output)
                
                true_label_for_G = torch.full((img.size(0),), 1.0, device=device)
                loss_G = get_loss(self.criterion, output_D_from_fake_for_G, true_label_for_G)
                loss_G.backward()
                
                opt_D.step()
                opt_G.step()
                
                sch_D.step()
                sch_G.step()
                
                loss_D = loss_D_from_fake.item() + loss_D_from_data.item()
                total_train_loss_D += loss_D
                total_train_loss_G += loss_G.item()
                
                if i % self.log_period == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Step {i}/{len(self.train_loader)}, Loss_D: {loss_D:.4f}, Loss_G: {loss_G.item():.4f}")
                    
                if i % self.wandb_log_period == 0:
                    wandb.log({
                        "train/loss_D": loss_D,
                        "train/loss_G": loss_G.item(),
                        "train/lr_D": sch_D.get_last_lr()[0],
                        "train/lr_G": sch_G.get_last_lr()[0]
                    }, step=self.global_steps)
                
                self.global_steps += 1
            
            avg_train_loss_D = total_train_loss_D / len(self.train_loader)
            avg_train_loss_G = total_train_loss_G / len(self.train_loader)
            
            wandb.log({
                "train/avg_train_loss_D": avg_train_loss_D,
                "train/avg_train_loss_G": avg_train_loss_G,
            }, step=self.global_steps)
            
            # sampling images with current G 
            self.models.generate(num_img=self.num_sampling, noise=self.fixed_noise, use_wnadb=True, train=True, step=self.global_steps)
            
            if (epoch % self.save_model_period == 0 or epoch == self.epochs - 1) and epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'model_D_state_dict': D.state_dict(),
                    'model_G_state_dict': G.state_dict(),
                    'optimizer_D_state_dict': opt_D.state_dict(),
                    'optimizer_G_state_dict': opt_G.state_dict(),
                    'scheduler_D_state_dict': sch_D.state_dict(),
                    'scheduler_G_state_dict': sch_G.state_dict()
                }, f"DCGAN-epoch{epoch + 1}.pth")
            
            
            
              
                       
    
        
        
    
        
        
                    


