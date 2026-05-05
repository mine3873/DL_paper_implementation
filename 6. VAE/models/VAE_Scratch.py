import torch.nn as nn
import torch

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)    

class Encoder(nn.Module):
    def __init__(self, rgb_dim=3, alpha=0.2, z_dim=100):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=rgb_dim, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(alpha, True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha, True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha, True),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(alpha, True),
        )     
        self.fc_input_dim = 1024 * 4 * 4
        
        self.fc_mu = nn.Linear(self.fc_input_dim, z_dim)
        self.fc_logvar = nn.Linear(self.fc_input_dim, z_dim)
            
    def forward(self, X):
        X = self.convs(X)
        X = X.view(X.size(0), -1)
        
        mu = self.fc_mu(X)
        logvar = self.fc_logvar(X)
        
        return mu, logvar
    
        
class Decoder(nn.Module):
    def __init__(self, z_dim=100):
        super(Decoder, self).__init__()
        
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, Z):
        # Z = (batch_size, z_dim)
        Z = Z.view(Z.size(0), Z.size(1), 1, 1)
        return self.convs(Z)
        
class VAEScratch(nn.Module):
    def __init__(self, z_dim=100):
        super(VAEScratch, self).__init__()
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, X):
        mu, logvar = self.encoder(X)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
        
        
if __name__ == "__main__":
    pass
