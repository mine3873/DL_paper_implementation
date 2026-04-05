import torch
import time

class ResNetTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, config):
        super(ResNetTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
            
        self.best_val_loss = float('inf')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'time_per_epoch': []
        }
            
            
    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            start_t = time.time()
            total_train_loss = 0
            
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                model_output = self.model(inputs)
                
                loss = self.criterion(model_output, targets)
                
                loss.backward()
                
                self.optimizer.step()
                
                
                total_train_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            self.scheduler.step()
            
            time_per_epoch = time.time() - start_t
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            
            torch.cuda.empty_cache()
            
            val_loss = self.validate()    
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            self.history['time_per_epoch'].append(time_per_epoch)
            
            print(f"Epoch [{epoch+1}]/{self.config.epochs}")
            print(f" - Train Loss: {avg_train_loss:.4f}")
            print(f" - Val Loss:   {val_loss:.4f}")
            print(f" - Lr:         {current_lr:.6f}")
            print(f" - time:       {time_per_epoch} seconds")
            print("-"*20)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"ResNet_model_best_n{self.config.num_layers}.pth")
                
    def validate(self):
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
        
                model_output = self.model(inputs)
                
                loss = self.criterion(model_output, targets)
                
                total_val_loss += loss.item()
                
        return total_val_loss / len(self.val_loader)
            
        