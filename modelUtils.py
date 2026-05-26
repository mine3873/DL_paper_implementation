import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self,):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        scores = torch.matmul(Q, torch.transpose(K, -1, -2)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e4)
        
        attention_weights = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
        
        return torch.matmul(attention_weights, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.scaledDotAttention = ScaledDotProductAttention()
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        d_model = Q.size(-1)
        d_k = d_model // self.n_heads
        
        Q_proj = self.W_q(Q).view(batch_size, Q.size(1), self.n_heads, d_k).transpose(1, 2)
        K_proj = self.W_k(K).view(batch_size, K.size(1), self.n_heads, d_k).transpose(1, 2)
        V_proj = self.W_v(V).view(batch_size, V.size(1), self.n_heads, d_k).transpose(1, 2)
        
        attention_output = self.scaledDotAttention(Q_proj, K_proj, V_proj, mask)
        attention_output = torch.transpose(attention_output, 1, 2).contiguous().view(attention_output.size(0), -1, d_model)
        
        output = self.W_o(attention_output)
        
        return output
    
class TimeEmbbedding(nn.Module):
    def __init__(self, d_emb=128):
        super(TimeEmbbedding, self).__init__()
        self.d_emb = d_emb
        
        self.mle = nn.Sequential(
            nn.Linear(d_emb, d_emb * 4),
            nn.SiLU(),
            nn.Linear(d_emb * 4, d_emb * 4)
        )
        
    def forward(self, t):
        """
        t: tensor(B, 1000)
        """

        half_dim = self.d_emb // 2
        
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=t.device) * -emb) # (half_d, )
        emb = t[:, None].float() * emb[None, :] # (T, 1) * (1, half_d)
        emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=1) 
        
        return self.mle(emb)