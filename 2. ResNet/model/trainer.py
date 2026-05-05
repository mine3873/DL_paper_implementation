import torch
import time
import matplotlib.pyplot as plt
import numpy as np

class ResNetTrainer:
    def __init__(self, model, config, train_loader=None, val_loader=None, test_loader=None, optimizer=None, criterion=None, scheduler=None):
        super(ResNetTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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
            
    def test(self, num_images=10):
        self.model.eval()
        images_shown = 0
        
        cols = 5
        rows = (num_images + cols - 1) // cols
        plt.figure(figsize=(12, 3 * rows))
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.config.device)
                
                outputs = self.model.forward(inputs)
                prediction = torch.argmax(outputs, dim=-1)
                
                for i in range(inputs.size(0)):
                    if images_shown >= num_images:
                        break
                    
                    img = inputs[i].permute(1,2,0).cpu().numpy()
                    
                    img = img * self.config.std + self.config.mean
                    img = img.clip(0, 1)
                    
                    plt.subplot(rows, cols, images_shown + 1)
                    plt.imshow(img)
                    
                    is_correct = (prediction[i] == targets[i])
                    color = 'blue' if is_correct else 'red'
                    
                    pred_class = self.config.classes[prediction[i]]
                    true_class = self.config.classes[targets[i]]
                    
                    plt.title(f"Pred: {pred_class}\nTrue_Label: {true_class}", color=color, fontsize=10)
                    plt.axis('off')
                    
                    images_shown += 1
                    
                if images_shown >= num_images:
                    break
                
        plt.tight_layout()
        
        plt.savefig(f"test_bs{self.config.batch_size_train}_ep{self.config.epochs}_n{self.config.num_layers}.png")
        plt.show()
        