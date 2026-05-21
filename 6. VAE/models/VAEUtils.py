from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
import torch.nn as nn
from models import Encoder, Decoder
import torch
import torchvision
import wandb
import math

LOCATION_SCALE_FAM = ('gaussian', 'laplace', )

class LSUNDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        
        search_path = os.path.join(root_dir, "**", "*.jpg")
        self.image_path = glob(search_path, recursive=True)
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, i):
        img_path = self.image_path[i]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0
    
class VAEUtils(nn.Module):
    def __init__(self, d_z, chs, L=1, ds_name="LSUN", use_leaky=False, leakySlope=0.2, p_dist="gaussian"):
        super(VAEUtils, self).__init__()
        self.d_z = d_z
        self.p_dist = p_dist
        self.L = L
        
        if chs is None:
            if ds_name == "LSUN":
                chs = [3, 128, 256, 512, 1024]
            elif ds_name == "MNIST":
                chs = [1, 128, 256]
        
        self.encoder = Encoder(d_z=d_z, chs=chs, leaky=use_leaky, leakySlope=leakySlope, ds_name=ds_name)
        self.decoder = Decoder(d_z=d_z, chs=list(reversed(chs.copy())), ds_name=ds_name)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def train(self):
        self.decoder.train()
        self.encoder.train()
        
    def eval(self):
        self.decoder.eval()
        self.encoder.eval()
    
    def get_noise(self, batch_size):
        return torch.randn(batch_size, self.d_z, device=self.device)
    
    def reparameterization(self, mu, logvar):
        eps = self.get_noise(mu.size(0))
        std = torch.exp(0.5 * logvar)
        return mu + std * eps
        
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization(mu, logvar)
        
        outputs = []
        for _ in range(self.L):
            outputs.append(self.decoder(z))
            
        return outputs, mu, logvar
    
    def get_loss(self, criterion, model_outputs, target, mu, logvar, beta=1.0):
        kl_loss = (1 + logvar - mu**2 - torch.exp(logvar)) * -0.5
        kl_loss = kl_loss.sum(dim=1).mean()
        
        assert criterion is not None
        
        recon_loss = 0
        for output in model_outputs:
            recon_loss += criterion(output, target) / target.size(0)
            
        recon_loss = recon_loss / self.L
        
        return (kl_loss * beta) + recon_loss, recon_loss, kl_loss
       
    def send_wandb(self, generated, title, caption, nrow, step=None, normalize=True):
        grid = torchvision.utils.make_grid(generated, nrow=nrow, normalize=normalize)
        
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        wandb.log({
            f"{title}": wandb.Image(ndarr, caption=caption)
        }, step=(step if step is not None else None))
    
    @torch.no_grad()
    def generate(self, num_img=1, noise=None, train=False, step=None):
        self.decoder.eval()
        
        if noise is None:
            noise = self.get_noise(num_img)
        
        generated = self.decoder(noise)
        
        self.send_wandb(
            generated=generated,
            title=f"{'train' if train else 'test'}/samples",
            caption=(f"{step}" if step is not None else None),
            nrow=int(math.sqrt(generated.size(0))),
            step=step,
            )

    @torch.no_grad()
    def interpolate(self, num_img=1, steps=10):
        self.decoder.eval()
        
        z1 = self.get_noise(num_img)
        z2 = self.get_noise(num_img)
        
        # z_inner = (1-alpha) * z1 + alpha * z2
        alphas = torch.linspace(0, 1, steps, device=self.device)
        
        for i, alpha in enumerate(alphas):
            noise = (1.0 - alpha) * z1 + alpha * z2
            generated = self.decoder(noise)
            
            self.send_wandb(
                generated=generated,
                title=f"latent/interpolation_frame",
                caption=f"Frame {i}",
                nrow=int(math.sqrt(generated.size(0))),
            )
                
    @torch.no_grad()
    def interpolate_some_dimension(self, dim, num_img=1, steps=10, val_range=(-3.0, 3.0)):
        self.decoder.eval()
        
        z1 = self.get_noise(num_img)
        
        target_values = torch.linspace(val_range[0], val_range[1], steps, device=self.device)
        
        for i, val in enumerate(target_values):
            noise = z1.clone()
            noise[:, dim] = val
            
            generated = self.decoder(noise)
            self.send_wandb(
                generated=generated,
                title=f"latent/interpolation_dim_{dim}",
                caption=f"Frame {i}",
                nrow=int(math.sqrt(generated.size(0)))
            )
    
    @torch.no_grad() 
    def test_reconstruction(self, train_loader, num_img):
        self.eval()
        
        x, _ = next(iter(train_loader))
        x = x[:num_img].to(self.device)
        
        outputs, _, _ = self.forward(x)
        
        recon_x = outputs[0]
        
        compares = torch.cat([x, recon_x], dim=0)
        
        self.send_wandb(
            compares,
            title="test/reconstruction",
            caption="Top: Original / Bottom: Reconstruction",
            nrow=num_img,
            normalize=(self.p_dist == "gaussian")
            )

if __name__=="__main__":
    B = 8
    d_z = 100
    
    #chs = [3, 128, 256, 512, 1024]
    chs = [1, 64, 128]
    models = VAEUtils(d_z=d_z, chs=chs, ds_name="MNIST")
    
    LSUN_IMG = (B, 3, 64, 64)
    MNIST_IMG = (B, 1, 28, 28)
    
    temp_img = torch.randn(MNIST_IMG)
    
    model_outputs, _, _ = models(temp_img)
    
    print(model_outputs[0].shape)