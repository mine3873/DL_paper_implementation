import torch
import torch.nn as nn
import math

dropout = 0.1

class Softmax(nn.Module):
    def __init__(self, dim):
        super(Softmax, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        exp_x = torch.exp(x - torch.max(x, dim=self.dim, keepdim=True)[0]) 
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = Softmax(dim=-1)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, seq_len_q, d_k)
            K: (batch_size, seq_len_k, d_k)
            V: (batch_size, seq_len_v, d_v)
            mask: (batch_size, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # QK^T / sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = self.softmax(scores)
        return torch.matmul(attention_weights, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
        self.scaled_dot_product = ScaledDotProductAttention(self.d_k)
        
    def forward(self, Q, K, V, num_heads = 8, mask=None):
        """
        Args:
            Q: (batch_size, seq_len_q, d_model)
            K: (batch_size, seq_len_k, d_model)
            V: (batch_size, seq_len_v, d_model)
            num_heads: number of attention heads
            src_mask: (batch_size, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
            tgt_mask: (batch_size, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
        """
        d_model = Q.size(-1)
        d_k = d_model // num_heads

        # Linear projections for Q, K, V and then reshape
        Q_proj = self.W_q(Q).view(Q.size(0), -1, num_heads, d_k).transpose(1, 2)  # Q @ W_Q (batch_size, num_heads, seq_len_q, d_k)
        K_proj = self.W_k(K).view(K.size(0), -1, num_heads, d_k).transpose(1, 2)  # K @ W_K (batch_size, num_heads, seq_len_k, d_k)
        V_proj = self.W_v(V).view(V.size(0), -1, num_heads, d_k).transpose(1, 2)  # V @ W_V (batch_size, num_heads, seq_len_v, d_k)

        if mask is not None:
            attention_output = self.scaled_dot_product(Q_proj, K_proj, V_proj, mask=mask)  # (batch_size, num_heads, seq_len_q, d_k)
        else:
            attention_output = self.scaled_dot_product(Q_proj, K_proj, V_proj)  # (batch_size, num_heads, seq_len_q, d_k)

        # Concatenate heads and apply final linear projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(Q.size(0), -1, d_model)  # (batch_size, seq_len_q, d_model)
        output = self.W_o(attention_output) # Concat(head1, head2, ... , headh) @ W_o

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.ffLayer1 = torch.nn.Linear(d_model, 4 * d_model)
        self.ffLayer2 = torch.nn.Linear(4 * d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        ff_output = self.ffLayer1(x)
        ff_output = torch.nn.ReLU()(ff_output)
        ff_output = self.ffLayer2(ff_output)
        return ff_output


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.multi_head_attention(x, x, x, mask=mask)  
        x = self.layer_norm1(x + self.dropout(attn_output))  
        
        ff_output = self.feed_forward(x)  
        x = self.layer_norm2(x + self.dropout(ff_output))  
        
        return x
            
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.layer_norm3 = torch.nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attention(x, x, x, mask=tgt_mask) 
        x = self.layer_norm1(x + self.dropout(attn_output))  
        
        attn_output = self.encoder_decoder_attention(x, encoder_output, encoder_output, mask=src_mask)  
        x = self.layer_norm2(x + self.dropout(attn_output))  
        
        ff_output = self.feed_forward(x)   
        x = self.layer_norm3(x + self.dropout(ff_output))  
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_blocks, num_decoder_blocks, dropout=0.1):
        super(Transformer, self).__init__()
        self.sharedEmbedding = nn.Embedding(vocab_size, d_model)
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout=dropout) for _ in range(num_encoder_blocks)])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout=dropout) for _ in range(num_decoder_blocks)])
        
        self.softmax = Softmax(dim=-1)
        self.final_linear = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def positional_encoding(self, seq_len):
        c = 10000
        pos = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        
        # 1/ C^{2j/d_model} = exp(-1 *Log(C) * (2j/d_model)) 
        i = torch.arange(0, self.d_model, 2)  # (d_model/2,)
        factor = torch.exp(i * -(math.log(10000.0) / self.d_model))  # (d_model/2,)
        
        PE = torch.zeros(seq_len, self.d_model)
        PE[:, 0::2] = torch.sin(pos * factor)  
        PE[:, 1::2] = torch.cos(pos * factor)  
        
        return PE.unsqueeze(0)  # (1, seq_len, d_model)
        
    def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None):
        src_seq = self.sharedEmbedding(src_seq) * math.sqrt(self.d_model)
        tgt_seq = self.sharedEmbedding(tgt_seq) * math.sqrt(self.d_model)
        
        seq_len_src = src_seq.size(1)
        seq_len_tgt = tgt_seq.size(1)
        
        src_seq += self.positional_encoding(seq_len_src).to(src_seq.device)  
        src_seq = self.dropout(src_seq)
        tgt_seq += self.positional_encoding(seq_len_tgt).to(tgt_seq.device)  
        tgt_seq = self.dropout(tgt_seq)
        
        encoder_output = src_seq
        for encoder in self.encoder_blocks:
            encoder_output = encoder(encoder_output, mask=src_mask)  
        
        decoder_output = tgt_seq
        for decoder in self.decoder_blocks:
            decoder_output = decoder(decoder_output, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)  
        
        return self.final_linear(decoder_output)  # (batch_size, seq_len_tgt, vocab_size)
        
        
        
