import torch.nn as nn
import torch
import math
from utils import MultiHeadAttention

NUM_GROUPS = 32

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
        half_dim = self.d_emb // 2
        
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=t.device) * -emb) # (half_d, )
        emb = t[:, None].float() * emb[None, :] # (T, 1) * (1, half_d)
        emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=1) 
        
        return self.mle(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d_t, dropout=0.1):
        super(ResBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=in_ch)
        self.nonLinear = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
        
        self.proj_t = nn.Linear(d_t, out_ch)
        
        self.norm2 = nn.GroupNorm(num_groups=NUM_GROUPS,num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
        
    def forward(self, X, temb):
        h = X
        h = self.nonLinear(self.norm1(h))
        h = self.conv1(h)
        
        h = h + self.proj_t(self.nonLinear(temb))[:, :, None, None]
        
        h = self.dropout(self.nonLinear(self.norm2(h)))
        h = self.conv2(h)
            
        return h + self.shortcut(X)

class DownSampling(nn.Module):
    def __init__(self, in_ch):
        super(DownSampling, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, stride=2, bias=False)
        
    def forward(self, X):
        return self.conv(X)
    

class UpSampling(nn.Module):
    def __init__(self, in_ch):
        super(UpSampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, bias=False)
        
    def forward(self, X):
        h = self.upsample(X)
        return self.conv(h)



class AttBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=in_ch)
        self.mha = MultiHeadAttention(d_model=in_ch, n_heads=1)
        
    def forward(self, X):
        B, C, H, W = X.shape
        
        h = self.norm(X)
        h = h.view(B, C, -1).transpose(1, 2)
        h = self.mha(h, h, h)
        
        return h.transpose(1, 2).view(B, C, H, W)
        


class UNet(nn.Module):
    def __init__(self, num_resBlock, ch_mults, in_ch=3, ch=128, dropout=0.1, attn_resolution=16, skip_connection: bool = True):
        super(UNet, self).__init__()
        
        self.num_resBlock = num_resBlock
        
        self.dropout = dropout
        
        self.attn_resolution = attn_resolution
        self.skip_connection = skip_connection
        
        self.temb_layer = TimeEmbbedding(ch)
        d_t = ch * 4
        
        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False)
        
        self.downs = nn.ModuleList([])
        cur_ch = ch
        hs_ch = [ch]
        num_resolution = len(ch_mults)
        
        res = 64
        for level, ch_mult in enumerate(ch_mults):
            out_ch = ch * ch_mult
            for _ in range(num_resBlock):
                self.downs.append(ResBlock(cur_ch, out_ch, d_t, dropout))
                cur_ch = out_ch
                
                if res == attn_resolution:
                    self.downs.append(AttBlock(cur_ch))
                
                if skip_connection == True:
                    hs_ch.append(cur_ch)
                
            if level != num_resolution - 1:
                self.downs.append(DownSampling(cur_ch))
                
                if skip_connection == True:
                    hs_ch.append(cur_ch)
                res //= 2
        
        
        self.middle = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, d_t, dropout),
            AttBlock(cur_ch),
            ResBlock(cur_ch, cur_ch, d_t, dropout),
        ])
        
        self.ups = nn.ModuleList([])
        
        
        for level, ch_mult in enumerate(reversed(ch_mults)):
            out_ch = ch * ch_mult
            for _ in range(num_resBlock + 1):
                if skip_connection == True:
                    cur_ch += hs_ch.pop()
                
                self.ups.append(ResBlock(cur_ch, out_ch, d_t, dropout))
                
                cur_ch = out_ch
                
                if res == attn_resolution:
                    self.ups.append(AttBlock(cur_ch))
                
            if level != num_resolution - 1:
                self.ups.append(UpSampling(cur_ch))
                res *= 2
            
        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=cur_ch),
            nn.SiLU(),
            nn.Conv2d(cur_ch, in_ch, kernel_size=3, padding=1)
        )
        
    def forward(self, X, t):
        
        # time embedding
        temb = self.temb_layer(t)
        
        h = self.head(X)
        hs = [h]
        
        for block in self.downs:
            if isinstance(block, ResBlock):
                h = block(h, temb)
                if h.size(-1) == self.attn_resolution:
                    continue
            else:
                h = block(h)
            
            hs.append(h)
        
        for block in self.middle:
            if isinstance(block, ResBlock):
                h = block(h, temb)
            else:
                h = block(h)
        
        for block in self.ups:
            if isinstance(block, ResBlock):
                h = block(torch.concat([h, hs.pop()], dim=1), temb)
            else:
                h = block(h)
                
        h = self.tail(h)
        
        return h
        

