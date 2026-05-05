import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self,):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, Q, K, V, mask=None):
        """
        - Tensor dimensions (Batch, seq_len, d):
        Batch : the number of sequences trained at each step
        seq_len : the number of tokens in each sequence
        d : the number of features explaining a token (word)
        
        - Args:
        Q : tensor(Batch, h, seq_len_q, d_k)
        K : tensor(Batch, h, seq_len_k, d_k)
        V : tensor(Batch, h, seq_len_k, d_v)
        
        - return:
        tensor(Batch, h, seq_len_q, d_v)
        """
        d_k = Q.size(-1)
        
        # QK^T / (d_k ** 2)
        # scores : tensor(Batch, h, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, torch.transpose(K, -1, -2)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e4)
        
        # attention_weights : tensor(Batch, h, seq_len_q, seq_len_k)
        attention_weights = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
        
        return torch.matmul(attention_weights, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
        self.scaledDotAttention = ScaledDotProductAttention()
        
    def forward(self, Q, K, V, mask=None):
        """
        
        Args:
        Q : tensor(Batch, seq_len_q, d_model)
        K : tensor(Batch, seq_len_k, d_model)
        V : tensor(Batch, seq_len_v, d_model)
        
        - return:
        tensor(Batch, seq_len_q, d_model)
        """
        batch_size = Q.size(0)
        d_model = Q.size(-1)
        d_k = d_model // self.n_heads
        
        Q_proj = self.W_q(Q).view(batch_size, Q.size(1), self.n_heads, d_k).transpose(1, 2)
        K_proj = self.W_k(K).view(batch_size, K.size(1), self.n_heads, d_k).transpose(1, 2)
        V_proj = self.W_v(V).view(batch_size, V.size(1), self.n_heads, d_k).transpose(1, 2)
        
        attention_output = self.scaledDotAttention(Q_proj, K_proj, V_proj, mask)
        attention_output = torch.transpose(attention_output, 1, 2).contiguous().view(attention_output.size(0), -1, d_model)
        """
        - contiguous():
        after performing transpose(), the data in tensor, stored in memory, is not changed in position, only the data access is changed.
        So with performing contiguous(), change the location of the memory stored in memory in fact and then can perform view()
        """
        
        output = self.W_o(attention_output)
        
        return output
        
class FeedForward(nn.Module):
    def __init__(self, d_model,):
        super(FeedForward, self).__init__()
        self.d_ff = d_model * 4
        self.ffLinear1 = torch.nn.Linear(d_model, self.d_ff)
        self.ffLinear2 = torch.nn.Linear(self.d_ff, d_model)
        
    def forward(self, X):
        output = self.ffLinear2(torch.nn.ReLU()(self.ffLinear1(X)))
        
        return output

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(EncoderBlock, self).__init__()
        
        self.multiHeadAttention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.normLayer1 = torch.nn.LayerNorm(d_model)
        
        self.feedForward = FeedForward(d_model)
        self.normLayer2 = torch.nn.LayerNorm(d_model)
        
        self.dropOut = torch.nn.Dropout(dropout)
        
    
    def forward(self, X, mask=None):
        residual = X
        X = self.normLayer1(X)
        X = residual + self.dropOut(self.multiHeadAttention(Q=X, K=X, V=X, mask=mask))
        
        residual = X
        X = self.normLayer2(X)
        X = residual + self.dropOut(self.feedForward(X=X))
        
        return X
        
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(DecoderBlock, self).__init__()
        
        self.self_masked_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.normLayer1 = torch.nn.LayerNorm(d_model)
        
        self.encoder_decoder_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.normLayer2 = torch.nn.LayerNorm(d_model)
        
        self.feedForward = FeedForward(d_model)
        self.normLayer3 = torch.nn.LayerNorm(d_model)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, X, encoder_output, src_mask=None, tgt_mask=None):
        residual = X
        X = self.normLayer1(X)
        X = residual + self.dropout(self.self_masked_attention(Q=X, K=X, V=X, mask=tgt_mask))
        
        residual = X
        X = self.normLayer2(X)
        X = residual + self.dropout(self.encoder_decoder_attention(Q=X, K=encoder_output, V=encoder_output, mask=src_mask))
        
        residual = X
        X = self.normLayer3(X)
        X = residual + self.dropout(self.feedForward(X=X))
        
        return X

class PositionalEncoding(nn.Module):
    def __init__(self,  d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        C = 10000.0
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        factor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(C) / d_model))
        
        self.pe[:, 0::2] = torch.sin(pos * factor)
        self.pe[:, 1::2] = torch.cos(pos * factor)
        
        self.register_buffer('pe_buffer', self.pe.unsqueeze(0))
        
    def forward(self, seq_len):
        """
        PE(pos, 2i) = sin(pos / 10000^{2i / d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})
        
        - return:
        tensor(Batch, seq_len, d_model)
        """
        return self.pe_buffer[:, :seq_len, :]
        
class Transformer_scatch(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_heads, dropout=0.1):
        super(Transformer_scatch, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dropout = dropout
        
        self.embedding = torch.nn.Embedding(self.vocab_size, d_model)
        
        self.dropOut = torch.nn.Dropout(dropout)
        
        self.pos_encoding1 = PositionalEncoding(self.d_model)
        self.pos_encoding2 = PositionalEncoding(self.d_model)
        
        self.encoders = torch.nn.ModuleList([
            EncoderBlock(d_model=self.d_model, n_heads=n_heads, dropout=self.dropout) for _ in range(n_layer)
        ])
        self.decoders = torch.nn.ModuleList([
            DecoderBlock(d_model=self.d_model, n_heads=n_heads, dropout=self.dropout) for _ in range(n_layer)
        ])
        
        self.output_nrom = torch.nn.LayerNorm(d_model)
        self.linear = torch.nn.Linear(self.d_model, self.vocab_size)
        
        
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        - Args:
        inpusts : tensor(batch, seq_len)
        outputs : tensor(batch, seq_len)
        
        - return:
        tensor(batch, seq_len, vocab_size)
        """
        
        # after embbding : tensor(batch, seq_len, d_model)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        src_emb = self.dropOut(src_emb + self.pos_encoding1(src_seq_len).to(src_emb.device))
        tgt_emb = self.dropOut(tgt_emb + self.pos_encoding2(tgt_seq_len).to(tgt_emb.device))
        
        encoder_output = src_emb
        for encoder in self.encoders:
            encoder_output = encoder(X=encoder_output, mask=src_mask)
            
        decoder_output = tgt_emb
        for decoder in self.decoders:
            decoder_output = decoder(X=decoder_output, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
            
        output = self.linear(self.output_nrom(decoder_output))
        
        return output
    
    
    
    
    
    