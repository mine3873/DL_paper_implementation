import torch
from tokenizers import Tokenizer
from transformer import Transformer
import math

def translate(model, sentence, en_tokenizer, de_tokenizer, device, max_length=50):
    model.eval()
    
    tokens = en_tokenizer.encode(sentence).ids
    src_tensor = torch.LongTensor([tokens]).to(device)
    
    src_mask = (src_tensor != en_tokenizer.token_to_id("<pad>")).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        src_emb = model.sharedEmbedding(src_tensor) * math.sqrt(model.d_model)
        src_emb += model.positional_encoding(src_tensor.size(1)).to(device)
        encoder_output = model.dropout(src_emb)
        
        src_mask = (src_tensor != en_tokenizer.token_to_id("<pad>")).unsqueeze(1).unsqueeze(2)
        
        for encoder in model.encoder_blocks:
            encoder_output = encoder(encoder_output, mask=src_mask)
    
    tgt_indices = [de_tokenizer.token_to_id("<sos>")]
    
    for i in range(max_length):
        tgt_tensor = torch.LongTensor([tgt_indices]).to(device)
        
        with torch.no_grad():
            tgt_emb = model.sharedEmbedding(tgt_tensor) * math.sqrt(model.d_model)
            tgt_emb += model.positional_encoding(tgt_emb.size(1)).to(device)
            decoder_output = model.dropout(tgt_emb)
            
            size = tgt_tensor.size(1)
            tgt_mask = (tgt_tensor != de_tokenizer.token_to_id("<pad>")).unsqueeze(1).unsqueeze(2)
            mask = torch.tril(torch.ones((size, size), device=device)).bool()
            tgt_mask = tgt_mask & mask
            
            for decoder in model.decoder_blocks:
                decoder_output = decoder(decoder_output, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
            
            output = model.final_linear(decoder_output[:, -1, :])
            predicted_id = output.argmax(dim=-1).item()
        
        if predicted_id == de_tokenizer.token_to_id("<eos>"):
            break
        
        tgt_indices.append(predicted_id)
        
    return de_tokenizer.decode(tgt_indices, skip_special_tokens=True)

en_tokenizer = Tokenizer.from_file("tokenizer/Multi30k/en_tokenizer.json")
de_tokenizer = Tokenizer.from_file("tokenizer/Multi30k/de_tokenizer.json")

device = torch.device("cuda")

d_model = 512
num_heads = 8
num_encoder_blocks = 6
num_decoder_blocks = 6
vocab_size = en_tokenizer.get_vocab_size()

model = Transformer(vocab_size, d_model, num_heads, num_encoder_blocks, num_decoder_blocks).to(device)

model_path = "transformer_model.pth" 
model.load_state_dict(torch.load(model_path, map_location=device))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_sentence = "There's a man walking in the park."
result = translate(model, test_sentence, en_tokenizer, de_tokenizer, device)

print(f"en: {test_sentence}")
print(f"de: {result}")