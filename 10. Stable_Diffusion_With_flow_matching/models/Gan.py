import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int, ch: int = 64,
        leLU_slope: float = 0.2,
        ):
        super(Discriminator, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leLU_slope, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(leLU_slope, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=ch * 2, out_channels=ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(leLU_slope, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=ch * 4, out_channels=ch * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(leLU_slope, inplace=True)
        )
        
        self.out_conv = nn.Conv2d(
            in_channels=ch * 8, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        
        
    def forward(self, x):
        # x : (B, 3, 96, 96)
        h = self.layer1(x) # (B, 64, 48, 48)
        h = self.layer2(h) # (B, 128, 24, 24)
        h = self.layer3(h) # (B, 256, 12, 12)
        h = self.layer4(h) # (B, 512, 6, 6)
        
        return self.out_conv(h) # (B, 1, 6, 6)
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    disc = Discriminator(in_channels=3, ch=64).to(device)
    
    x = torch.randn(2, 3, 96, 96).to(device)
    
    out = disc(x)
    
    print(f"out shape {out.shape}")