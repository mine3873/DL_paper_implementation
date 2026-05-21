import torch
import time
import wandb
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

class CLIPTrainer:
    def __init__(self, config, model, train_loader=None, val_loader=None, criterion=None, optimizer=None, scheduler=None, tokenizer=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        
        self.global_step = 0
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.patience
        
        self.templates = [
            'a photo of a {}.',
            'a photo of the {}.',
            'a rendering of a {}.',
            'a cropped photo of the {}.',
            'the photo of a {}.',
            'a photo of a clean {}.',
            'a photo of a dirty {}.',
            'a dark photo of the {}.',
            'a photo of my {}.',
            'a photo of the cool {}.',
            'a close-up photo of a {}.',
            'a bright photo of the {}.',
            'a good photo of a {}.',
            'a photo of one {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the weird {}.',
            'a photo of the large {}.',
            'a photo of a nice {}.',
            'a photo of a strange {}.',
            'a blurry photo of a {}.',
            'a pixelated photo of a {}.',
            'a black and white photo of the {}.',
            'a painting of a {}.',
            'a sketch of a {}.',
            'a doodle of a {}.',
            'a origami {}.',
            'a low resolution photo of a {}.',
            'a cartoon {}.',
            'art of a {}.',
            'a drawing of a {}.',
            'a photo of the plush {}.',
            'a photo of the sharp {}.',
            'a photo of the fuzzy {}.',
            'a photo of a cool {}.',
            'a photo of a small {}.',
            'a tattooed {}.',
            'a embroidered {}.',
            'a photo of a hard to see {}.',
            'a photo of many {}.',
        ]
        
        
    def train(self):
        wandb.watch(self.model, log="all", log_freq=10, log_graph=True)
        
        device = self.config.device
        
        for param in self.model.img_encoder.parameters():
            param.requires_grad = False

        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

        for param in self.model.Wi.parameters():
            param.requires_grad = True

        for param in self.model.Wt.parameters():
            param.requires_grad = True

        self.model.t.requires_grad = True
        
        for epoch in range(self.config.epochs):
            if epoch == 1:
                for param in self.model.parameters():
                    param.requires_grad = True
                
                steps_per_epoch = len(self.train_loader)
                remaining_epochs = self.config.epochs - epoch
                total_remaining_steps = steps_per_epoch * remaining_epochs
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=int(total_remaining_steps * 0.1),
                    num_training_steps=total_remaining_steps
                )
            
            self.model.train()
            total_loss = 0
            start_t = time.time()
            
            for i, (image, text, eos_idx, _) in enumerate(self.train_loader):
                image, text, eos_idx = image.to(device), text.to(device), eos_idx.to(device)
                
                model_output = self.model((image, text), eos_idx, self.config.pad_idx)
                
                loss = self.criterion(model_output)
                
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                self.global_step += 1
                
                if i % 10 == 0:
                    wandb.log({
                            "loss": loss.item(),
                            "lr/backbone": self.optimizer.param_groups[0]['lr'],
                            "lr/heads": self.optimizer.param_groups[2]['lr'],
                        }, step=self.global_step)
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                    
            time_per_epoch = time.time() - start_t
            avg_loss = total_loss / len(self.train_loader)
            current_lr = self.scheduler.get_last_lr()[0]
            
            avg_val_loss = self.validate()
            
            wandb.log({
                "Loss/train": avg_loss,
                "Loss/Val": avg_val_loss,
                "time_per_epoch": time_per_epoch,
            }, step=self.global_step)
            
            print(f"Epoch [{epoch+1}/{self.config.epochs}]")
            print(f" - Train Loss: {avg_loss:.4f}")
            print(f" - Val Loss:   {avg_val_loss:.4f}")
            print(f" - Lr:         {current_lr:.6f}")
            print(f" - Time:       {time_per_epoch:.2f} seconds")
            print("-" * 20)
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f"CLIP_best_val_ResNet{self.config.img_n_layer}.pth")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    print("Early stopping.")
                    return
                
    def validate(self):
        self.model.eval()
        val_loss = 0
        device = self.config.device
        
        val_table = wandb.Table(columns=["Image", "Prediction", "Confidence", "Ground Truth"])
        
        with torch.no_grad():
            for i, (image, text, eos_idx, _) in enumerate(self.val_loader):
                image, text, eos_idx = image.to(device), text.to(device), eos_idx.to(device)
                model_output = self.model((image, text), eos_idx, self.config.pad_idx)
                loss = self.criterion(model_output)
                val_loss += loss.item()
                
                if i == 0:
                    probs = torch.softmax(model_output, dim=-1)
                    top_prob, top_idx = torch.max(probs, dim=-1)
                    
                    for idx in range(min(8, image.size(0))):
                        pred_text = self.tokenizer.decode(text[top_idx[idx]].tolist(), skip_special_tokens=True)
                        true_text = self.tokenizer.decode(text[idx].tolist(), skip_special_tokens=True)
                        
                        img_vis = image[idx].cpu().permute(1, 2, 0).numpy()
                        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-5)
                        img_vis = (img_vis * 255).astype('uint8')
                        
                        val_table.add_data(
                            wandb.Image(img_vis), 
                            pred_text, 
                            round(top_prob[idx].item(), 4),
                            true_text
                        )
        avg_val_loss = val_loss / len(self.val_loader)
        
        wandb.log({
            "Val_Visual_Results": val_table
        }, step=self.global_step)
        
        return avg_val_loss
           
    def zero_shot_test(self, class_names, test_loader, topK=5):
        self.model.eval()
        device = self.config.device 
         
        with torch.no_grad():
            all_prompts = []
            for name in class_names:
                for template in self.templates:
                    all_prompts.append(template.format(name.replace('_', ' ')))
                
            inputs = self.tokenizer(all_prompts, padding=True, return_tensors="pt").to(device)
            text_tokens = inputs['input_ids']
            eos_idx = (text_tokens != self.tokenizer.pad_token_id).sum(dim=-1) - 1
            
            features = self.model.text_encoder(text_tokens, eos_idx, self.config.pad_idx)
            features = self.model.Wt(features)
            features = F.normalize(features, p=2, dim=-1) 
            
            features = features.view(len(class_names), len(self.templates), -1)
            class_features = features.mean(dim=1)
            class_features = F.normalize(class_features, p=2, dim=-1)
            
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                image, _, _, captions = batch 
                image = image.to(device)
                
                labels = []
                valid_indices = []
                
                for i, cap in enumerate(captions):
                    found_label = -1
                    
                    for idx, name in enumerate(class_names):
                        if name.lower() in cap.lower():
                            found_label = idx
                            break 
                    
                    if found_label != -1:
                        labels.append(found_label)
                        valid_indices.append(i)
                
                if not labels: 
                    continue
                
                label = torch.tensor(labels).to(device)
                image = image[valid_indices]
                
                image_features = self.model.img_encoder(image)
                image_features = self.model.Wi(image_features)
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                model_output = (image_features @ class_features.t()) * torch.exp(self.model.t)
                
                preds = model_output.argmax(dim=-1)
                top1_correct += (preds == label).sum().item()
                
                _, top5_preds = model_output.topk(min(topK, len(class_names)), dim=-1)
                top5_correct += (top5_preds == label.view(-1, 1)).any(dim=1).sum().item()
        
                total += label.size(0)
        if total == 0:
            return 0, 0
                
        top1_accuracy = (top1_correct / total) * 100
        top5_accuracy = (top5_correct / total) * 100
        
        print(f"Zero shot Top 1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Zero shot Top 5 Accuracy: {top5_accuracy:.2f}%")
        
        wandb.log({
            "zero_shot_top1": top1_accuracy,
            "zero_shot_top5": top5_accuracy
        })
        
        return top1_accuracy, top5_accuracy
    
    def imageNet_zero_shot_test(self, imagenet_loader, class_names, topK=5):
        self.model.eval()
        device = self.config.device
        
        with torch.no_grad():
            all_class_embeddings = []
            
            for name in class_names:
                texts = [temp.format(name.replace('_', ' ')) for temp in self.templates]
                inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
                
                text_tokens = inputs['input_ids']
                eos_idx = (text_tokens != self.tokenizer.pad_token_id).sum(dim=-1) - 1
                
                features = self.model.text_encoder(text_tokens, eos_idx, self.config.pad_idx)
                features = self.model.Wt(features)
                features = F.normalize(features, p=2, dim=-1)
                
                all_class_embeddings.append(features.mean(dim=0, keepdim=True))
                
            classifier_weights = torch.cat(all_class_embeddings, dim=0)
            classifier_weights = F.normalize(classifier_weights, p=2, dim=-1)
            
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in imagenet_loader:
                images, targets = images.to(device), targets.to(device)
                
                image_features = self.model.img_encoder(images)
                image_features = self.model.Wi(image_features)
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                model_output = (image_features @ classifier_weights.t()) * torch.exp(self.model.t)
                
                preds = model_output.argmax(dim=-1)
                top1_correct += (preds == targets).sum().item()
                
                _, top5_preds = model_output.topk(min(topK, len(class_names)), dim=-1)
                top5_correct += (top5_preds == targets.view(-1, 1)).any(dim=1).sum().item()
        
                total += targets.size(0)
                
        if total == 0:
            return 0, 0
                
        top1_accuracy = (top1_correct / total) * 100
        top5_accuracy = (top5_correct / total) * 100
        
        print(f"Zero shot Top 1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Zero shot Top 5 Accuracy: {top5_accuracy:.2f}%")
        
        wandb.log({
            "ImageNet/zero_shot_top1": top1_accuracy,
            "ImageNet/zero_shot_top5": top5_accuracy
        })
        
        return top1_accuracy, top5_accuracy
    
    
    def retrieval_test(self, test_loader, topK=[1, 5, 10]):
        self.model.eval()
        device = self.config.device
        
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for images, input_ids, eos_idx, _ in test_loader: 
                images = images.to(device)
                input_ids = input_ids.to(device)
                eos_idx = eos_idx.to(device)
                
                img_feat = self.model.img_encoder(images)
                img_feat = self.model.Wi(img_feat)
                img_feat = F.normalize(img_feat, p=2, dim=-1)
                all_image_features.append(img_feat)
                
                text_feat = self.model.text_encoder(input_ids, eos_idx, self.config.pad_idx)
                text_feat = self.model.Wt(text_feat)
                text_feat = F.normalize(text_feat, p=2, dim=-1)
                all_text_features.append(text_feat)
        
        # tensor(batch_size, d_e)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        
        model_output = all_image_features @ all_text_features.t()
        
        num_images = model_output.size(0)
        labels = torch.arange(num_images).to(device)
        
        log_dict = {}
        for k in topK:
            _, topk_indices = model_output.topk(k, dim=1)
            correct = (topk_indices == labels.view(-1, 1)).any(dim=1).sum().item()
            recall = (correct / num_images) * 100
            
            print(f"Recall@{k}: {recall:.2f}%")
            log_dict[f"retrieval/R@{k}"] = recall
            
        _, top5_indices = model_output[0].topk(5)
        
        def get_data(idx):
            image, _, _, cap = test_loader.dataset[idx]
            return image, cap
        
        image, real_cap = get_data(0)
        
        img_vis = image.cpu().permute(1, 2, 0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-5)
        img_vis = (img_vis * 255).astype('uint8')
        
        top5_caps = [get_data(i.item())[1] for i in top5_indices]
        
        log_dict["retrieval/query_image"] = wandb.Image(
            img_vis, 
            caption=f"Real: {real_cap}"
        )
        
        log_dict["retrieval/top5_search_example"] = wandb.Html(
            f"<b>Query Image's Real Caption (Randomly Picked):</b> {real_cap} <br><br>" +
            "<b>Model's Top 5 Predicted Captions:</b><ol>" + 
            "".join([f"<li>{cap}</li>" for cap in top5_caps]) + "</ol>"
        )
        wandb.log(log_dict)
        
    
        