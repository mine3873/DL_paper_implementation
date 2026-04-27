import torch
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np

class DCGANTrainer:
    def __init__(self, models, config, train_laoder=None, criterion=None, optimizers=None, schedulers=None, fixed_noise=None):
        self.models = models
        self.config = config
        self.train_loader = train_laoder
        self.criterion = criterion
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.fixed_noise = fixed_noise
        
        self.history = {
            'loss_D': [],
            'loss_G': [],
            'lr': [],
            'time_per_epoch': []
        }
        
    def train(self):
        G, D = self.models
        optD, optG = self.optimizers
        scheD, scheG = self.schedulers
        device = self.config.device
        
        for epoch in range(self.config.epochs):
            D.train()
            G.train()
            
            total_loss_D = 0
            total_loss_G = 0
            start_t = time.time()
            
            for i, (img, _) in enumerate(self.train_loader):
                batch_size = img.size(0)
                img = img.to(device)
                
                label = torch.full((batch_size,), 0.9, device=device)
                fake_label = torch.full((batch_size,), 0.0, device=device)
                
                D.zero_grad()
                
                output_D = D(img)
                loss_D = self.criterion(output_D, label)
                loss_D.backward()
                
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake_img = G(noise)
                
                output_D_fake = D(fake_img.detach())
                loss_D_fake = self.criterion(output_D_fake, fake_label)
                loss_D_fake.backward()
                
                
                
                G.zero_grad()
                
                output_G = D(fake_img)
                loss_G = self.criterion(output_G, label)
                loss_G.backward()
                
                optD.step()
                loss_D = loss_D + loss_D_fake
                total_loss_D += loss_D.item()
                optG.step()
                total_loss_G += loss_G.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, LR_D: {scheD.get_last_lr()[0]:.6f}, LR_G: {scheG.get_last_lr()[0]:.6f}")
                    
                
            scheD.step()
            scheG.step()
            
            avg_loss_D = total_loss_D / len(self.train_loader)
            avg_loss_G = total_loss_G / len(self.train_loader)
            current_lr = optG.param_groups[0]['lr']
            time_per_epoch = time.time() - start_t
            
            self.history['loss_D'].append(avg_loss_D)
            self.history['loss_G'].append(avg_loss_G)
            self.history['lr'].append(current_lr)
            self.history['time_per_epoch'].append(time_per_epoch)
            
            print(f"Epoch [{epoch+1}/{self.config.epochs}]")
            print(f" - Loss_D: {avg_loss_D:.4f}")
            print(f" - Loss_G: {avg_loss_G:.4f}")
            print(f" - Lr:     {current_lr:.6f}")
            print(f" - Time:   {time_per_epoch:.2f} seconds")
            print("-" * 20)
            
            with torch.no_grad():
                G.eval()
                
                fake_samples = G(self.fixed_noise).detach().cpu()
                torchvision.utils.save_image(
                    fake_samples,
                    f"DCGAN_output_epoch_{epoch+1}.png",
                    normalize=True,
                    nrow=8
                    )  
                
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config.epochs:
                        
                torch.save({
                    'epoch': epoch,
                    'model_G_state_dict': G.state_dict(),
                    'model_D_state_dict': D.state_dict(),
                    'optimizerG_state_dict': optG.state_dict(),
                    'optimizerD_state_dict': optD.state_dict(),
                }, f"DCGAN_epoch_{epoch+1}.pth")
                
    def test(self, noise_start, noise_end=None, steps=10, image_name=None):
        G, _ = self.models
        G.eval()
        
        with torch.no_grad():
            if noise_end is not None:
                alpha = torch.linspace(0, 1, steps, device=self.config.device)
                alpha = alpha.view(steps, 1, 1, 1)
                
                interpolated_noise = (1 - alpha) * noise_start + alpha * noise_end
                fake_samples = G(interpolated_noise).detach().cpu()
            else:
                fake_samples = G(noise_start).detach().cpu()
                steps = fake_samples.size(0)
            
            plt.figure(figsize=(steps * 2, 2))
            for i in range(steps):
                plt.subplot(1, steps, i + 1)
                
                img = fake_samples[i].permute(1, 2, 0).numpy()
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                
                plt.imshow(img)
                plt.axis('off')
                if noise_end is not None:
                    plt.title(f"{int((i / (steps - 1.0)) * 100)}%")
            
            plt.tight_layout()
            
            plt.savefig(f"test_{image_name}")
            plt.show()
        