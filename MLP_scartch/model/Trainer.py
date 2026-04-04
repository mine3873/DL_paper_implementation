import torch
import time
import matplotlib.pyplot as plt

class MLPTrainer:
    def __init__(
        self,
        model, 
        config,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        optimizer=None,
        criterion=None,
        scheduler=None
        ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader =test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        self.best_val_loss = float('inf')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'time_per_epoch': []
        }
    
    def train(self):
        self.model.train()
        for epoch in range(self.config.epochs):
            total_train_loss = 0
            start_t = time.time()
            
            for i, (inputs, targets) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                model_outputs = self.model.forward(inputs)
                loss = self.criterion.forward(model_outputs, targets)
                
                grad_output = self.criterion.backward()
                
                self.model.backward(grad_output)
                
                self.optimizer.step()
                
                self.scheduler.step()
                
                total_train_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.optimizer.lr:.6f}")
                    
            avg_train_loss = total_train_loss / len(self.train_loader)
            
            torch.cuda.empty_cache()
            
            val_loss = self.validate()
            
            current_lr = self.optimizer.lr
            time_per_epoch = time.time() - start_t
            
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
                checkpoint = {
                    'W1': self.model.W1,
                    'b1': self.model.b1,
                    'W2': self.model.W2,
                    'b2': self.model.b2
                }
                torch.save(checkpoint, "mlp_model_best.pth")
                
    def validate(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad(): 
            for inputs, targets in self.val_loader:       
                model_outputs = self.model.forward(inputs)
                loss = self.criterion.forward(model_outputs, targets)
                
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
                outputs = self.model.forward(inputs)
                prediction = torch.argmax(outputs, dim=-1)
                
                for i in range(inputs.size(0)):
                    if images_shown >= num_images:
                        break
                    
                    img = inputs[i].squeeze().numpy()
                    
                    img = img * self.config.mean + self.config.std
                    
                    plt.subplot(rows, cols, images_shown + 1)
                    plt.imshow(img, cmap='gray')
                    
                    is_correct = (prediction[i] == targets[i])
                    color = 'blue' if is_correct else 'red'
                    
                    plt.title(f"Pred: {prediction[i]}\nTrue_Label: {targets[i]}", color=color, fontsize=10)
                    
                    plt.axis('off')
                    
                    images_shown += 1
                    
                if images_shown >= num_images:
                    break
                
        plt.tight_layout()
        
        plt.savefig(f"test_result_{self.config.batch_size_train}_ep{self.config.epochs}.png")
        plt.show()
                    
            