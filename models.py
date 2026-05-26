import torch
import torch.nn as nn
from modelUtils import MultiHeadAttention, TimeEmbbedding


"""
img_size: 64 - 64 - 32  - 16  - 8   - 4
channels: 3  - 64 - 128 - 256 - 512 - 1024
"""

NUM_GROUP = 32
    
class AttnBlock(nn.Module):
    def __init__(self, ch):
        super(AttnBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups=NUM_GROUP, num_channels=ch)
        self.mha = MultiHeadAttention(d_model=ch, n_heads=1)
        
    def forward(self, X):
        B, C, H, W = X.shape
        
        h = self.norm(X)
        h = h.view(B, C, -1).transpose(1, 2)
        h = self.mha(h, h, h)
        
        return h.transpose(1, 2).view(B, C, H, W)

class UpSampling(nn.Module):
    def __init__(self, ch):
        super(UpSampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=ch, out_channels=ch, kernel_size=3, padding=1, bias=False)
        
    def forward(self, X):
        h = self.upsample(X)
        return self.conv(h)
        
class DownSampling(nn.Module):
    def __init__(self, ch,):
        super(DownSampling, self).__init__()
        
        self.down_conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.down_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d_t, dropout=0.):
        super(ResBlock, self).__init__()
        
        self.gn1 = nn.GroupNorm(num_groups=NUM_GROUP, num_channels=in_ch)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
        
        self.w_t = nn.Linear(d_t, out_ch)
        self.silu2 = nn.SiLU()        
        
        self.gn2 = nn.GroupNorm(num_groups=NUM_GROUP, num_channels=out_ch)
        self.silu3 = nn.SiLU()
        
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x, temb):
        h = x
        
        h = self.silu1(self.gn1(h))
        h = self.conv1(h)
        
        h = h  + self.w_t(self.silu2(temb))[:, :, None, None]
        
        h = self.silu3(self.gn2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h
   
class UNet(nn.Module):
    def __init__(self, in_ch, ch=128, chs=[1, 2, 4, 8], img_size=64, num_resBlock=2, dropout=0.0, attn_img_size=16):
        super(UNet, self).__init__()

        d_t = ch * 4
        num_resolution = len(chs)
        self.attn_img_size = attn_img_size
        
        self.temb_layer = TimeEmbbedding(ch)
        
        self.downs = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch, out_channels=ch, kernel_size=3, padding=1, bias=False)
        ])
        
        hs = [ch]
        in_ch = ch
        for resolution in range(num_resolution):
            out_ch = ch * chs[resolution]
            for _ in range(num_resBlock):
                self.downs.append(ResBlock(in_ch=in_ch, out_ch=out_ch, d_t=d_t, dropout=dropout))
                in_ch = out_ch
                
                if img_size == attn_img_size:
                    self.downs.append(AttnBlock(ch=out_ch))
                hs.append(out_ch)
            if resolution != num_resolution - 1:
                self.downs.append(DownSampling(out_ch))
                hs.append(out_ch)
                img_size /= 2
        
        self.middle = nn.ModuleList([
            ResBlock(in_ch=in_ch, out_ch=out_ch, d_t=d_t, dropout=dropout),
            AttnBlock(ch=out_ch),
            ResBlock(in_ch=out_ch, out_ch=out_ch, d_t=d_t, dropout=dropout)
        ])
        
        self.ups = nn.ModuleList([])
        for resolution in reversed(range(num_resolution)):
            out_ch = ch * chs[resolution]
            for _ in range(num_resBlock + 1):
                skip_ch = hs.pop()
                self.ups.append(ResBlock(in_ch=skip_ch + in_ch, out_ch=out_ch, d_t=d_t, dropout=dropout))
                in_ch = out_ch
                
                if img_size == attn_img_size:
                    self.ups.append(AttnBlock(ch=out_ch))
            if resolution != 0:
                self.ups.append(UpSampling(out_ch))
                img_size *= 2
            
        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups=NUM_GROUP, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=3, padding=1, bias=False)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x, t):
        temb = self.temb_layer(t)
        
        h = x
        hs = []
        
        h = self.downs[0](h)
        hs.append(h)
        
        for block in self.downs[1:]:
            if isinstance(block, ResBlock):
                h = block(h, temb)
                if h.size(-1) == self.attn_img_size:
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
                
        return self.tail(h)
                
if __name__ == "__main__":
    B = 8
    x = torch.randn((B, 3, 64, 64))
    
    total_timesteps = 1000
    t = torch.randint(0, total_timesteps, (B,), dtype=torch.long)
    
    model = UNet(in_ch=3)
    output = model(x, t)
    
    print(output.shape)
    
    