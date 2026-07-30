import torch
import torch.nn as nn
import math

def normalize(in_channels):
    return nn.GroupNorm(
        num_groups=32, num_channels=in_channels, affine=True
    )
    
def nonlinear():
    return nn.SiLU()

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
    def forward(self, t):
        dim_half = self.dim // 2
        
        h = torch.log(torch.tensor(10000.0)) / (dim_half - 1)
        h = torch.exp(torch.arange(dim_half, device=t.device) * -h)
        h = t[:, None] * h[None, :]
        h = torch.concat([torch.sin(h), torch.cos(h)], dim=-1)
        return self.mlp(h)

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super(ContinuousTimeEmbedding, self).__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
    def forward(self, t):
        dim_half = self.dim // 2
        
        h = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=dim_half, dtype=torch.float32) / dim_half
        ).to(t.device)
        
        h = t[:, None] * h[None, :] * 2 * math.pi
        h = torch.concat([torch.cos(h), torch.sin(h)], dim=-1)
        return self.mlp(h)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: int = 3, padding: int = 1,
        bias: bool = False, dropout: float = 0.0,
        time_emb_dim: int = None
        ):
        super(ResBlock, self).__init__()
        
        self.gn1 = normalize(in_channels)
        self.nl1 = nonlinear()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=1, padding=padding, bias=bias, 
        )
        
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim is not None:
            self.t_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.gn2 = normalize(out_channels)
        self.nl2 = nonlinear()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=1, padding=padding, bias=bias, 
        )
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, padding=0, bias=bias
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t=None):
        h = x
        
        h = self.conv1(self.nl1(self.gn1(h)))
        
        if self.time_emb_dim is not None:
            h = h + self.t_proj(t)[:, :, None, None]
        
        h = self.conv2(self.dropout(self.nl2(self.gn2(h)))) 
        
        return h + self.shortcut(x)
    

class AttnBlock(nn.Module):
    def __init__(self, in_channel):
        super(AttnBlock, self).__init__()
        self.gn = normalize(in_channel)
        self.Wq = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, bias=False)
        self.Wk = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, bias=False)
        self.Wv = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, bias=False)
        
        self.Wo = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        h = self.gn(x)
        Q = self.Wq(h)
        K = self.Wk(h)
        V = self.Wv(h)
        
        B, C, H, W = Q.shape
        Q = Q.view(B, C, -1).permute(0, 2, 1)
        K = K.view(B, C, -1)
        
        weight = torch.bmm(Q, K) * (int(C) ** (-0.5))
        weight = nn.functional.softmax(weight, dim=-1)
        weight = weight.permute(0, 2, 1)
        
        V = V.view(B, C, -1)
        h = torch.bmm(V, weight)
        h = h.view(B, C, H, W)
        
        h = self.Wo(h)
        
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_asymmetric_pad: bool =True, bias: bool= False):
        super(Downsample, self).__init__()
        self.with_asymmetric_pad = with_asymmetric_pad
        if with_asymmetric_pad:
            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=0, stride=2, bias=bias
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=2, bias=bias
            )    
        
    def forward(self, x):
        if self.with_asymmetric_pad:
            pad = (0, 1, 0, 1)
            h = nn.functional.pad(x, pad=pad, mode="constant", value=0)
        else:
            h = x
        h = self.conv(h)
        
        return h

class Upsample(nn.Module):
    def __init__(self, in_channel, with_conv=True, bias: bool=False):
        super(Upsample, self).__init__()
        
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(
                in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, bias=bias
                )
 
    def forward(self, x):
        h = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            h = self.conv(h)
            
        return h