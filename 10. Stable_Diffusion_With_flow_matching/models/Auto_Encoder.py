import torch
import torch.nn as nn
import torchvision
from ..utils.Modules import ResBlock, AttnBlock, Downsample, Upsample, normalize, nonlinear
# 96 48 24 12  

class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, ch: int, img_size: int,
        ch_mults: tuple = (1, 2, 4, 8),
        z_channel: int = 4,
        regularizer: str = 'KL',
        num_resblock: int = 2,
        attn_resolution: tuple = (),
        dropout: float = 0.0,
        with_asymmetric_pad: bool = False,
        bias: bool = False
        ):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([])
        
        self.start = nn.Conv2d(
            in_channels=in_channels, out_channels=ch, kernel_size=3, padding=1, stride=1, bias=bias
        )
        cur_resolution = img_size
        num_resolution = len(ch_mults)
        
        in_ch = ch
        for resolution_idx in range(num_resolution):
            out_ch = ch * ch_mults[resolution_idx]
            for _ in range(num_resblock):
                self.layers.append(
                    ResBlock(in_channels=in_ch, out_channels=out_ch, bias=bias, dropout=dropout)
                )
                in_ch = out_ch
                if cur_resolution in attn_resolution:
                    self.layers.append(
                        AttnBlock(in_channel=in_ch)
                    )
                    
            if resolution_idx != num_resolution - 1:
                self.layers.append(
                    Downsample(in_channels=in_ch, with_asymmetric_pad=with_asymmetric_pad, bias=bias)
                )
                cur_resolution = cur_resolution // 2
            
        self.layers.append(ResBlock(out_ch, out_ch, dropout=dropout))
        self.layers.append(AttnBlock(out_ch))
        self.layers.append(ResBlock(out_ch, out_ch, dropout=dropout))
        
        self.gn = normalize(out_ch)
        self.nl = nonlinear()
        self.out_conv = nn.Conv2d(
            in_channels=out_ch, out_channels= z_channel * 2 if regularizer == 'KL' else z_channel, kernel_size=3, padding=1, bias=bias
        )
    
    def forward(self, x):
        h = self.start(x)
        
        for block in self.layers:
            h = block(h)
        
        h = self.nl(self.gn(h))
        h = self.out_conv(h)
        
        return h

class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int, ch: int, img_size: int,
        ch_mults: tuple = (1, 2, 4, 8),
        z_channel: int = 4,
        num_resblock: int = 2,
        attn_resolution: tuple = (),
        dropout: float = 0.0,
        with_conv: bool = False,
        bias: bool = False
        ):
        super(Decoder, self).__init__()

        in_ch = ch * ch_mults[-1]
        
        self.start = nn.Conv2d(
            in_channels=z_channel, out_channels=in_ch, kernel_size=3, padding=1, bias=bias
        )
        
        self.layers = nn.ModuleList([
            ResBlock(in_ch, in_ch, dropout=dropout),
            AttnBlock(in_ch),
            ResBlock(in_ch, in_ch, dropout=dropout)
        ])
        num_resolution = len(ch_mults)
        cur_resolution = img_size / (2**(num_resolution-1))
        for resolution_idx in reversed(range(num_resolution)):
            out_ch = ch * ch_mults[resolution_idx]
            for _ in range(num_resblock):
                self.layers.append(
                    ResBlock(in_channels=in_ch, out_channels=out_ch, bias=bias, dropout=dropout)
                )
                in_ch = out_ch
                if cur_resolution in attn_resolution:
                    self.layers.append(
                        AttnBlock(out_ch)
                    )
            
            if resolution_idx > 0:
                self.layers.append(
                    Upsample(in_ch, with_conv=with_conv, bias=bias)
                )
                cur_resolution = cur_resolution * 2
        
        self.gn = normalize(out_ch)
        self.nl = nonlinear()
        self.out_conv = nn.Conv2d(
            in_channels=out_ch, out_channels=in_channels, kernel_size=3, padding=1, bias=bias, 
        )
    
    def forward(self, x):
        h = self.start(x)
        
        for block in self.layers:
            h = block(h)
            
        h = self.nl(self.gn(h))
        h = self.out_conv(h)
        
        return h
        

class Vae(nn.Module):
    def __init__(
        self,
        in_channels: int, ch: int, img_size: int,
        ch_mults: tuple, 
        z_channel: int,
        num_resblock: int,
        attn_resolution: tuple,
        dropout: float = 0.0,
        with_asymmetric_pad: bool = False,
        with_conv: bool = False,
        bias: bool = False,
        regularizer: str = 'KL',
        ):
        super(Vae, self).__init__()
        self.regularizer = regularizer
        
        self.encoder = Encoder(
            in_channels=in_channels, ch=ch, img_size=img_size,
            ch_mults=ch_mults, z_channel=z_channel,
            regularizer=regularizer,
            num_resblock=num_resblock, attn_resolution=attn_resolution,
            dropout=dropout, with_asymmetric_pad=with_asymmetric_pad, bias=bias
        )
        
        if self.regularizer == 'KL':
            self.mu_layer = nn.Conv2d(
                in_channels=z_channel * 2, out_channels= z_channel,
                kernel_size=1, bias=bias
            )
            self.logvar_layer = nn.Conv2d(
                in_channels=z_channel * 2, out_channels= z_channel,
                kernel_size=1, bias=bias
            )
        elif self.regularizer == 'VQ':
            pass
        
        
        self.decoder = Decoder(
            in_channels=in_channels, ch=ch, img_size=img_size,
            ch_mults=ch_mults, z_channel=z_channel,
            num_resblock=num_resblock, attn_resolution=attn_resolution,
            dropout=dropout, with_conv=with_conv, bias=bias
        )
          
    def forward(self, x):
        h = self.encoder(x)
        
        if self.regularizer == 'KL':
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            
            h = mu + torch.exp(logvar / 2.0) * torch.randn_like(mu)
        else:
            pass
        
        out = self.decoder(h)
        
        return out, mu, logvar
    
    @torch.no_grad()
    def sample(self, x, recon_x=None):
        self.eval()
        num_samples = min(4, x.size(0))
        
        x = (x[:num_samples] + 1.0) / 2.0
        
        if recon_x is None:
            grid = x
        else:
            recon_x = (recon_x[:num_samples] + 1.0) / 2.0
            grid = torch.cat([x, recon_x], dim=0)
            
        grid = torchvision.utils.make_grid(
            grid, nrow=num_samples, padding=2
        )
        
        self.train()
        return grid
    
    @torch.no_grad()
    def encode_z_DDPM(self, x, scaling_factor=0.18215):
        h = self.encoder(x)
        
        if self.regularizer == 'KL':
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            
            h = mu + torch.exp(logvar / 2.0) * torch.randn_like(mu)
        else:
            pass
        
        return h * scaling_factor
    
    @torch.no_grad()
    def decode_z_DDPM(self, z, scaling_factor=0.18215):
        z = z / scaling_factor
        
        return self.decoder(z)
        

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x = torch.randn(2, 3, 96, 96).to(device)
    
    vae = Vae(
        in_channels=3,
        ch=64,
        img_size=96,
        ch_mults=(1, 2, 4, 8),
        z_channel=4,
        num_resblock=2,
        attn_resolution=(12,),
        regularizer='KL'
    ).to(device)
    
    out, mu, logvar = vae(x)
    
    print(f"input img shape : {x.shape}")
    print(f"mu shape : {mu.shape}")
    print(f"out shape : {out.shape}")
    
