import torch
import torchvision
import time

class VAETrainer:
    def __init__(self, model, config, train_laoder=None, criterion=None, optimizer=None, scheduler=None, fixed_noise=None):
        self.model = model
        self.config = config
        self.train_loader = train_laoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fixed_noise = fixed_noise
        
        self.history = {
            'loss': [],
            'lr': [],
            'time_per_epoch': []
        }

    def train(self):
        device = self.config.device
        
        test_batch, _ = next(iter(self.train_loader))
        test_batch = test_batch[:8].to(device)
        
        total_steps = len(self.train_loader) * self.config.epochs
        current_step = 0
        
        for epoch in range(self.config.epochs):
            self.model.train()
            
            total_loss = 0
            start_t = time.time()
            
            for i, (img, _) in enumerate(self.train_loader):
                img = img.to(device)
                
                self.optimizer.zero_grad()
                
                model_output, mu, logvar = self.model(img)
                
                anneal_steps = total_steps // 2
                beta = min(1.0, current_step / anneal_steps)
                
                self.criterion.beta = beta
                loss = self.criterion(img, model_output, mu, logvar)
                
                loss.backward()
                
                self.optimizer.step()
                
                current_step += 1
                total_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

            self.scheduler.step()
            
            time_per_epoch = time.time() - start_t
            avg_loss = total_loss / len(self.train_loader)
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.history['loss'].append(avg_loss)
            self.history['lr'].append(current_lr)
            self.history['time_per_epoch'].append(time_per_epoch)
            
            print(f"Epoch [{epoch+1}/{self.config.epochs}]")
            print(f" - Loss: {avg_loss:.4f}")
            print(f" - Lr:     {current_lr:.6f}")
            print(f" - Time:   {time_per_epoch:.2f} seconds")
            print("-" * 20)
            
            if self.fixed_noise is not None:
                with torch.no_grad():
                    self.model.eval()
                    
                    fake_samples = self.model.decoder(self.fixed_noise).detach().cpu()
                    torchvision.utils.save_image(
                        fake_samples,
                        f"VAE_output_with_fixed_noise_epoch_{epoch+1}.png",
                        normalize=True,
                        nrow=8
                        )  
            
            with torch.no_grad():
                self.model.eval()
                
                model_output, _, _ = self.model(test_batch)
                model_output = model_output.detach().cpu()
                
                comparison = torch.cat([test_batch.cpu(), model_output], dim=0)
                torchvision.utils.save_image(
                    comparison,
                    f"VAE_compare_reconstruction_epoch_{epoch+1}.png",
                    normalize=True,
                    nrow=8
                )
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.config.epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f"VAE_epoch_{epoch+1}.pth")
            
            
            