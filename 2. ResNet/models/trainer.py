import torch
import wandb

class ResnetTrainer:
    def __init__(
        self, config, model, model_name, num_layers, train_loader=None, val_loader=None,
        criterion=None, scheduler=None, optimizer=None, log_period=100, wandb_log_period=10
        ):
        
        self.config = config
        self.model = model
        self.model_name = model_name
        self.num_layers = num_layers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.log_period = log_period
        self.wandb_log_period = wandb_log_period
        
        self.global_steps = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 3
        
    def train(self):
        device = self.config.device
        
        for epoch in range(self.config.epochs):
            total_train_loss = 0
            self.model.train()
            
            for i, (x, target) in enumerate(self.train_loader):
                x, target = x.to(device), target.to(device)
                
                self.optimizer.zero_grad()
                
                model_output = self.model(x)
    
                loss = self.criterion(model_output, target)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                
                if i > 0 and i % self.log_period == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
                if i % self.wandb_log_period == 0:
                    wandb.log({
                        f"{self.model_name}-{self.num_layers}/loss": loss.item(),
                        f"{self.model_name}-{self.num_layers}/lr": self.scheduler.get_last_lr()[0],
                    }, step=self.global_steps)
                    
                self.global_steps += 1
                
            self.scheduler.step()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_val_loss = self.validate()
            
            wandb.log({
                f"{self.model_name}-{self.num_layers}/avg_train_loss": avg_train_loss,
                f"{self.model_name}-{self.num_layers}/avg_val_loss": avg_val_loss
            }, step=self.global_steps)
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, f"{self.model_name}-n{self.num_layers}.pth")
            
    def validate(self):
        device = self.config.device
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for x, target in self.val_loader:
                x, target = x.to(device), target.to(device)
                model_output = self.model(x)
                loss = self.criterion(model_output, target)
                total_val_loss += loss.item()
                
        return total_val_loss / len(self.val_loader)
                
        
        