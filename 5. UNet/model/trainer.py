import torch
import wandb
import numpy as np
import torchvision.transforms.functional as F

class UNetTrainer:
    def __init__(
        self, config, model, 
        train_loader=None, val_loader=None,
        criterion=None, optimizer=None, scheduler=None, log_period=5, wandb_log_period=5
        ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.log_period = log_period
        self.wandb_log_period = 5
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train(self):
        device = self.config.device
        
        for epoch in range(self.config.epochs):
            self.model.train()
            total_train_loss = 0
            
            for step, (img, target, w_map) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                img, target, w_map = img.to(device), target.to(device), w_map.to(device)
                
                model_output = self.model(img)
                
                loss = self.criterion(model_output, target, w_map)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                total_train_loss += loss.item()
                
                if step % self.log_period == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {step}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

                if step % self.wandb_log_period == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                    }, step=self.global_step)
                    
                self.global_step += 1
                
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_val_loss = self.validate()
            
            wandb.log({
                "train/avg_train_loss": avg_train_loss,
                "train/avg_val_loss": avg_val_loss
            }, step=self.global_step)
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, f"UNet.pth")
    
    def validate(self):
        device = self.config.device
        self.model.eval()
        total_val_loss = 0
        
        val_log_list = []
        
        with torch.no_grad():
            for step, (img, target, w_map) in enumerate(self.val_loader):
                self.optimizer.zero_grad()
                
                img, target, w_map = img.to(device), target.to(device), w_map.to(device)
                
                model_output = self.model(img)
                
                loss = self.criterion(model_output, target, w_map)
                
                total_val_loss += loss.item()
                
                if step == 1:
                    pred_mask = torch.argmax(model_output, dim=1)
                    cur_size = model_output.size(-1)
                    
                    img_c = F.center_crop(img[0], [cur_size, cur_size])
                    img_c = (img_c * 0.5 + 0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    img_c = (img_c * 255).astype(np.uint8)
                    
                    target_c = F.center_crop(target[0], [cur_size, cur_size]).squeeze().cpu().numpy().astype(np.uint8)
                    pred_c = pred_mask[0].squeeze().cpu().numpy().astype(np.uint8)

                    overlay_image = wandb.Image(img_c, masks={
                        "predictions": {"mask_data": pred_c, "class_labels": {1: "cell"}},
                        "ground_truth": {"mask_data": target_c, "class_labels": {1: "cell"}}
                    }, caption="Overlay Analysis")


                    comparison_images = [
                        wandb.Image(img_c, caption="Raw Image"),
                        wandb.Image(target_c * 255, caption="GT Mask (Side)"),
                        wandb.Image(pred_c * 255, caption="Pred Mask (Side)")
                    ]

                    val_log_list.append(overlay_image)
                    val_log_list.extend(comparison_images)
                    
        if val_log_list:
            wandb.log({"val/visuals": val_log_list}, step=self.global_step)
                
        return total_val_loss / len(self.val_loader)
        