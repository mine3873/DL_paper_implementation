import torch
import time
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class UnetTrainer:
    def __init__(self, model, config, train_loader=None, val_loader=None, optimizer=None, scheduler=None, criterion=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
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
            total_train_loss = 0
            start_t = time.time()
            
            for i, (inputs, targets, weight_maps) in enumerate(self.train_loader):
                inputs, targets, weight_maps = inputs.to(self.config.device), targets.to(self.config.device), weight_maps.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                model_outputs = self.model(inputs)
                
                loss = self.criterion(model_outputs, targets, weight_maps)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_train_loss += loss.item()
                
                if i % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

            
            time_per_epoch = time.time() - start_t
            avg_train_loss = total_train_loss / len(self.train_loader)
            
            #torch.cuda.empty_cache()
            
            val_loss, dice_loss = self.validate()  
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
                torch.save(self.model.state_dict(), f"UNet_model_data_augm_best.pth")
                
    def validate(self):
        self.model.eval()
        total_val_loss = 0
        total_dice = 0
        with torch.no_grad():
            for inputs, targets, weight_maps in self.val_loader:
                inputs, targets, weight_maps = inputs.to(self.config.device), targets.to(self.config.device), weight_maps.to(self.config.device)
                
                model_outputs = self.model(inputs)
                
                loss = self.criterion(model_outputs, targets, weight_maps)
                
                total_val_loss += loss.item()
                
                probs = torch.softmax(model_outputs, dim=1)
                batch_dice = self.calculate_dice(probs[:, 1], targets)
                total_dice += batch_dice
        
        avg_loss = total_val_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        
        return avg_loss, avg_dice
    
    
    def test(self, num_images=5):
        self.model.eval()
        images_shown = 0
        total_dice = 0
        cols = 3
        rows = num_images
        
        plt.figure(figsize=(9, 3 * rows))
        
        with torch.no_grad():
            for i, (inputs, targets, weight_maps) in enumerate(self.val_loader):
                if i >= num_images: break
                inputs, targets, weight_maps = inputs.to(self.config.device), targets.to(self.config.device), weight_maps.to(self.config.device)
                model_outputs = self.model(inputs)
                
                probs = torch.softmax(model_outputs, dim=1)
                prediction = torch.argmax(probs, dim=1).cpu().numpy()
                
                batch_dice = self.calculate_dice(probs[:, 1], targets)
                total_dice += batch_dice
                
                input_img = inputs[0, 0]
                input_img = F.center_crop(input_img, [388, 388])
                input_img = input_img.cpu().numpy()
                # 이런게 있었네;;
                
                target_img = targets[0]
                target_img = F.center_crop(target_img, [388, 388])
                target_img = target_img.cpu().numpy()
                
                plt.subplot(rows, cols, i * 3 + 1)
                plt.imshow(input_img, cmap='gray')
                plt.title("Input Image")
                plt.axis('off')
                
                plt.subplot(rows, cols, i * 3 + 2)
                plt.imshow(target_img, cmap='gray')
                plt.title("Ground Truth")
                plt.axis('off')
                
                plt.subplot(rows, cols, i * 3 + 3)
                plt.imshow(prediction[0], cmap='gray')
                plt.title("UNet Prediction")
                plt.axis('off')
                
                images_shown += 1
        
        avg_dice = total_dice / images_shown
        print(f"avg_dice : {avg_dice:.4f}")
        
        plt.tight_layout()
        plt.savefig(f"test_augm_bs{self.config.batch_size_train}_ep{self.config.epochs}.png")
        plt.show()
                
    def calculate_dice(self, pred, true, threshold=0.5, smooth=1e-6):
        pred = (pred > threshold).float()
        true = true.float()
        
        if pred.shape != true.shape:
            true = F.center_crop(true, pred.shape[-2:])
        
        inter = (pred * true).sum()
        
        dice = (2.0 * inter + smooth) / (pred.sum() + true.sum() + smooth)
        return dice.item()