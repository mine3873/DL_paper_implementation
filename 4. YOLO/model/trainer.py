import torch
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YOLOTrainer:
    def __init__(self,
                 model, config, train_loader=None, val_loader=None, test_loader=None,
                 optimizer=None, scheduler=None, criterion=None, 
                 ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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
            
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                model_outputs = self.model(inputs)
                
                loss = self.criterion(model_outputs, targets)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_train_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
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
                torch.save(self.model.state_dict(), f"YOLO_model_best.pth")
    
    def validate(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                model_outputs = self.model(inputs)
                
                loss = self.criterion(model_outputs, targets)
                
                total_val_loss += loss.item()
                
        return total_val_loss / len(self.val_loader)
        
        
    def nonMaximalSuppression(self, bboxes, iou_trheshold=0.4):
        """
        bboxes: [[x, y, w, h, conf, class_idx], ...]
        """
        if not bboxes: return []
        
        bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
        bboxes_after_nms = []
        
        while bboxes:
            chosen_box = bboxes.pop(0)
            
            remaining_bboxes = []
            for box in bboxes:
                if box[5] != chosen_box[5]:
                    remaining_bboxes.append(box)
                    continue
                
                b1 = torch.tensor(chosen_box[:4]).view(1, 1, 1, 4)
                b2 = torch.tensor(box[:4]).view(1, 1, 1, 4)
                
                iou = self.criterion.returnIoU(b1, b2).item()
                
                if iou < iou_trheshold:
                    remaining_bboxes.append(box)
                    
            bboxes = remaining_bboxes
            bboxes_after_nms.append(chosen_box)
        
        return bboxes_after_nms
        
        
    def test(self, num_images=10, thresh=0.1, iou_thresh=0.4):
        self.model.eval()
        images_shown = 0
        
        cols = 5
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()
        
        S, B, C = self.config.S, self.config.B, self.config.C
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                if images_shown >= num_images:
                    break
                
                inputs = inputs.to(self.config.device)
                outputs = self.model(inputs)
                
                for i in range(inputs.size(0)):
                    if images_shown >= num_images:
                        break
                    
                    img = inputs[i].permute(1,2,0).cpu().numpy()
                    img = img.clip(0, 1)
                    
                    ax = axes[images_shown]
                    ax.imshow(img)
                    
                    output = outputs[i].view(S, S, B * 5 + C)
                    
                    all_bboxes = []
                    mask = output[:, :, 4] > thresh
                    indices = torch.nonzero(mask) 
                    
                    
                    for idx in indices:
                        row, col = idx[0].item(), idx[1].item()
                        x_c, y_c, w, h = output[row, col, 0:4].tolist()
                        conf = output[row, col, 4].item()
                        class_idx = torch.argmax(output[row, col, B * 5:]).item()
                        
                        x_img = (x_c + col) * (self.config.image_size / S)
                        y_img = (y_c + row) * (self.config.image_size / S)
                        w_img = w * self.config.image_size
                        h_img = h * self.config.image_size
                        
                        all_bboxes.append([x_img, y_img, w_img, h_img, conf, class_idx])
                        
                    nms_boxes = self.nonMaximalSuppression(all_bboxes, iou_trheshold=iou_thresh)
                    
                    if not nms_boxes:
                        ax.set_title("No detection", fontsize=8)
                    else:
                        for box in nms_boxes:
                            x_img, y_img, w_img, h_img, conf, class_idx = box
                        
                            rect = patches.Rectangle(
                                (x_img - w_img/2, y_img - h_img/2), w_img, h_img,
                                linewidth=1.5, edgecolor='r', facecolor='none'
                            )
                            ax.add_patch(rect)
                            
                            pred_class = self.config.classes[class_idx]
                            ax.text(x_img - w_img/2, y_img - h_img/2, pred_class, 
                                    color='white', fontsize=8, backgroundcolor='red')

                        ax.set_title(f"Detections: {len(indices)}", color='red', fontsize=10)
                    
                    ax.axis('off')
                    images_shown += 1
                    
        plt.tight_layout()
        plt.savefig(f"test_bs{self.config.batch_size_train}_ep{self.config.epochs}.png")
        plt.show()