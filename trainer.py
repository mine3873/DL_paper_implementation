import time
from utils import create_masks
import torch

class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        
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
            
            """
            src: tensor(batch, seq_len)
            tgt: tensor(batch, seq_len)
            """
            for i, (src, tgt) in enumerate(self.train_loader):
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)
                
                tgt_in = tgt[:,:-1]
                tgt_out = tgt[:,1:]
                
                src_mask, tgt_mask = create_masks(
                    src=src,
                    tgt=tgt_in,
                    pad_idx=self.config.pad_idx,
                    device=self.config.device
                )
                
                self.optimizer.zero_grad()
                
                # model_output: tensor(batch, seq_len, vocab_size)
                model_output = self.model(
                    src=src,
                    tgt=tgt_in,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                
                
                loss = self.criterion(
                    model_output.reshape(-1, self.config.vocab_size), # tensor(batch * seq_len, vocab_size)
                    tgt_out.reshape(-1) # tensor(batch * seq_len)
                    )
                
                
                loss.backward()
                
                # prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_train_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
                    
                
                
                
            end_t = time.time()
            time_per_epoch = end_t - start_t
            
            avg_train_loss = total_train_loss / len(self.train_loader)
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
            
        torch.save(self.model.state_dict(), "transformer_model.pth")
            
            
                
    def validate(self):
        """
        - return:
        avg_val_loss : total_val_loss / (len(val_data) / batch_size)
        """
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)
                
                tgt_in = tgt[:,:-1]
                tgt_out = tgt[:,1:]
                
                src_mask, tgt_mask = create_masks(
                    src=src,
                    tgt=tgt_in,
                    pad_idx=self.config.pad_idx,
                    device=self.config.device
                )
                
                model_output = self.model(
                    src=src,
                    tgt=tgt_in,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                
                loss = self.criterion(
                    model_output.reshape(-1, self.config.vocab_size), # tensor(batch * seq_len, vocab_size)
                    tgt_out.reshape(-1) # tensor(batch * seq_len)
                    )
                
                total_val_loss += loss.item()
        
        return total_val_loss / len(self.val_loader)