import torch
import torch.nn as nn
from ..utils.Modules import TimeEmbedding, ContinuousTimeEmbedding, Downsample, Upsample, AttnBlock, ResBlock, nonlinear, normalize

# 4 x 12 x 12

class UNet(nn.Module):
    def __init__(
        self, 
        in_channels: int, ch: int, img_size: int, time_dim: int,
        ch_mults: tuple = (1, 2),
        num_resblock: int = 2,
        attn_resolutions: tuple = (),
        dropout: float = 0.0,
        with_asymmetric_pad: bool = False, with_conv: bool = False,
        bias: bool = False
        ):
        super(UNet, self).__init__()
        
        #self.time_mlp = TimeEmbedding(dim=time_dim)
        self.time_mlp = ContinuousTimeEmbedding(dim=time_dim)
        time_emb_dim = time_dim * 4
        self.start = nn.Conv2d(
            in_channels=in_channels, out_channels=ch, kernel_size=3, padding=1, bias=bias
        )
        
        cur_resolution = img_size
        num_resolutions = len(ch_mults)
        
        self.downs = nn.ModuleList([])
        
        hs = [ch]
        in_ch = ch
        for resolution_idx in range(num_resolutions):
            out_ch = ch * ch_mults[resolution_idx]
            for _ in range(num_resblock):
                self.downs.append(
                    ResBlock(
                        in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=bias, dropout=dropout, time_emb_dim=time_emb_dim
                    )
                )
                in_ch = out_ch
                if cur_resolution in attn_resolutions:
                    self.downs.append(
                        AttnBlock(in_channel=out_ch)
                    )
                hs.append(out_ch)
                    
            if resolution_idx != num_resolutions - 1:
                self.downs.append(
                    Downsample(in_channels=in_ch, with_asymmetric_pad=with_asymmetric_pad, bias=bias)
                )
                hs.append(out_ch)
                cur_resolution = cur_resolution // 2
                
        self.middles = nn.ModuleList([
            ResBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=bias, dropout=dropout, time_emb_dim=time_emb_dim),
            AttnBlock(in_channel=out_ch),
            ResBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=bias, dropout=dropout, time_emb_dim=time_emb_dim)
        ])
        
        self.ups = nn.ModuleList([])
        for resolution_idx in reversed(range(num_resolutions)):
            out_ch = ch * ch_mults[resolution_idx]
            for _ in range(num_resblock + 1):
                skip_ch = hs.pop()
                self.ups.append(
                    ResBlock(
                        in_channels=skip_ch + in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=bias, dropout=dropout, time_emb_dim=time_emb_dim
                    )
                )
                in_ch = out_ch
                if cur_resolution in attn_resolutions:
                    self.ups.append(
                        AttnBlock(in_channel=out_ch)
                    )
            
            if resolution_idx != 0:
                self.ups.append(
                    Upsample(in_channel=out_ch, with_conv=with_conv, bias=bias)
                    )
                cur_resolution = cur_resolution * 2
        
        self.end = nn.Sequential(
            normalize(in_channels=out_ch),
            nonlinear(),
            nn.Conv2d(in_channels=out_ch, out_channels=in_channels, kernel_size=3, padding=1, bias=bias)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
        
        
    def forward(self, x, t=None):
        temb = self.time_mlp(t)
        
        h = self.start(x)
        hs = [h]
        
        for block in self.downs:
            if isinstance(block, ResBlock):
                h = block(h, temb)
            else:
                h = block(h)
                
            if isinstance(block, AttnBlock):
                continue
            hs.append(h)
        
        for block in self.middles:
            if isinstance(block, ResBlock):
                h = block(h, temb)
            else:
                h = block(h)
        
        for block in self.ups:
            if isinstance(block, ResBlock):
                h = block(torch.concat([h, hs.pop()], dim=1), temb)
            else:
                h = block(h)
                
        out = self.end(h)
        
        return out
    
    @torch.no_grad()
    def sample(self, x1, steps: int, ema=None):
        self.eval()
        
        if ema is not None:
            ema.apply_shadow()
        
        num_samples = min(4, x1.size(0))
        z = torch.randn_like(x1[:num_samples])
        
        dt = 1.0 / steps
        
        for step in range(steps):
            t = torch.ones(num_samples, device=self.device) * (step/steps)
            model_output = self(z, t)
            z = z + model_output * dt
        
        if ema is not None:
            ema.restore()
        
        self.train()
        
        return z
        
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters() if param.requires_grad}
        
        self.backup = {}
        self.is_applied = False
        
    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)
                        
                    new_avg = (1.0 - self.decay) * param + self.decay * self.shadow[name]
                    self.shadow[name].copy_(new_avg)
    
    def apply_shadow(self):
        if self.is_applied:
            return
        
        self.backup = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters() if param.requires_grad}
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name])
        
        self.is_applied = True
        
    def restore(self):
        if not self.is_applied:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.backup:
                    param.copy_(self.backup[name])
        
        self.backup = {}
        self.is_applied = False

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    B = 8
    C = 4
    H, W = 12, 12
    
    x = torch.randn((B,C,H,W), device=device)
    model = UNet(
        in_channels=C, ch=128, img_size=H, time_dim=256, 
        ch_mults=(1, 2), num_resblock=2, attn_resolutions=(12,),
        dropout=0.0
    ).to(device)
    
    t = torch.rand(x.shape[0], device=device)

    print(f"input shape: {x.shape}")
    out = model(x, t)
    print(f"out shape: {out.shape}")
    