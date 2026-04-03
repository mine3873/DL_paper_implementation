import time
from transformer_scratch.model.utils import create_masks
import torch

class TransformerTrainer:
    def __init__(self, model, config, train_loader=None, val_loader=None, optimizer=None, scheduler=None, criterion=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.best_val_loss = float('inf')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'time_per_epoch': []
        }
         
    def train(self):
        scaler = torch.amp.GradScaler("cuda")
        
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
                
                
                with torch.amp.autocast("cuda"):
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
                    
                scaler.scale(loss).backward()
                
                scaler.unscale_(self.optimizer)
                
                #loss.backward()
                
                # prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                #self.optimizer.step()
                #self.scheduler.step()
                scaler.step(self.optimizer)
                scaler.update()
                
                self.scheduler.step()
                
                total_train_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs}, Step {i}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
                   
                    
            end_t = time.time()
            time_per_epoch = end_t - start_t
            
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
                torch.save(self.model.state_dict(), f"transformer_model_best.pth")
            
    def validate(self):
        """
        - return:
        avg_val_loss : total_val_loss / (len(val_data) / batch_size)
        """
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
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
    
    def translate(self, sentence, tokenizer, max_len=50, beam_size = 5):
        self.model.eval()
        
        src_tokens = [tokenizer.bos_id()] + tokenizer.encode_as_ids(sentence) + [tokenizer.eos_id()]
        src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(self.config.device)
        
        # Greedy serach
        """
        outputs = [tokenizer.bos_id()]
        
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(outputs).unsqueeze(0).to(self.config.device)
            
            src_mask, tgt_mask = create_masks(
                src=src_tensor,
                tgt=tgt_tensor,
                pad_idx=self.config.pad_idx,
                device=self.config.device
            )
            
            with torch.no_grad():
                output = self.model(src_tensor, tgt_tensor, src_mask, tgt_mask)
                
                next_token_logits = output[0, -1, :]
                next_word = next_token_logits.argmax().item()

            outputs.append(next_word)
            
            if next_word == tokenizer.eos_id():
                break
                
        return tokenizer.decode_ids(outputs)
        """
        # Beam search
        beams = [([tokenizer.bos_id()], 0.0)]
        completed_beas = []
        
        for i in range(max_len):
            candidate = []
            
            for tokens, score in beams:
                if tokens[-1] == tokenizer.eos_id():
                    completed_beas.append((tokens, score))
                    continue
                
                tgt_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.config.device)
                
                src_mask, tgt_mask = create_masks(
                    src=src_tensor,
                    tgt=tgt_tensor,
                    pad_idx=self.config.pad_idx,
                    device=self.config.device
                )
                
                with torch.no_grad():
                    output = self.model(src_tensor, tgt_tensor, src_mask, tgt_mask)
                    log_probs = torch.log_softmax(output[0, -1, :], dim=-1)
                    
                    if len(tokens) >= 1:
                        last_token = tokens[-1]
                        
                        for m in range(len(tokens) - 1):
                            if tokens[m] == last_token:
                                repeated_token = tokens[m+1]
                                log_probs[repeated_token] = -float('inf')
                    
                    topK_log_probs, topK_indices = log_probs.topk(beam_size)
                    
                    for j in range(beam_size):
                        next_token = topK_indices[j].item()
                        next_score = score + topK_log_probs[j].item()
                        candidate.append((tokens + [next_token], next_score))
                        
            beams = sorted(candidate, key=lambda x: x[1], reverse=True)[:beam_size]
                    
            if not beams:
                break
        
        all_candidate = completed_beas + beams
        best_beam = max(all_candidate, key=lambda x: x[1] / (len(x[0])**0.7))
        
        
        return tokenizer.decode_ids(best_beam[0])
        