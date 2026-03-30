from transformer import Transformer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import time
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

en_tokenizer = Tokenizer.from_file("tokenizer/Multi30k/en_tokenizer.json")
de_tokenizer = Tokenizer.from_file("tokenizer/Multi30k/de_tokenizer.json")

d_model = 512
num_heads = 8
num_encoder_blocks = 6
num_decoder_blocks = 6
vocab_size = en_tokenizer.get_vocab_size()
batch_size = 32
epochs = 30

def load_data(en_path, de_path):
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = f.read().splitlines()
    
    with open(de_path, 'r', encoding='utf-8') as f:
        de_lines = f.read().splitlines()
    
    tokenized_data = []
    for en, de in zip(en_lines, de_lines):
        en_id = [2] + en_tokenizer.encode(en).ids + [3]  # <sos> + tokens + <eos>
        de_id = [2] + de_tokenizer.encode(de).ids + [3]
        tokenized_data.append({"src": en_id, "tgt": de_id})
    return tokenized_data

def collate_fn(batch):
    src_batch = [torch.tensor(item["src"]) for item in batch]
    tgt_batch = [torch.tensor(item["tgt"]) for item in batch]
    
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=en_tokenizer.token_to_id("<pad>"))
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=de_tokenizer.token_to_id("<pad>"))
    
    return src_padded, tgt_padded

train_data = load_data("data/Multi30k/train.en", "data/Multi30k/train.de")
val_data = load_data("data/Multi30k/val.en", "data/Multi30k/val.de")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)

model = Transformer(vocab_size, d_model, num_heads, num_encoder_blocks, num_decoder_blocks)
device = torch.device("cuda")
model.to(device)
pad_idx = en_tokenizer.token_to_id("<pad>")  

optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)

def lr_lambda(step):
    step += 1 
    warmup_steps = 4000
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=pad_idx)

def create_masks(src, tgt):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len_src]
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len_tgt]
    
    size = tgt.size(1)
    mask = torch.tril(torch.ones((size, size), device=device)).bool()  # [seq_len_tgt, seq_len_tgt]
    tgt_mask = tgt_mask & mask  # [batch, 1, seq_len_tgt, seq_len_tgt]
    
    return src_mask, tgt_mask
    

def train(model, train_loader, optimizer, scheduler, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        start_t = time.time()
        
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:,1:]
            
            src_mask, tgt_mask = create_masks(src, tgt_input)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            loss = criterion(output.reshape(-1, vocab_size), tgt_expected.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            if i % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
   
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"- Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_t:.2f} seconds")
        
    torch.save(model.state_dict(), "transformer_model.pth")
   
   
def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:,:-1]
            tgt_expected = tgt[:,1:]
            
            src_mask, tgt_mask = create_masks(src, tgt_input)
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, vocab_size), tgt_expected.reshape(-1))
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_loader)




train(model, train_loader, optimizer, scheduler, criterion, epochs, device)