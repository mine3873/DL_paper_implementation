import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.nn as nn
from models import Discriminator, Generator
import torchvision
import math
import wandb

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

class ModelsUtils(nn.Module):
    def __init__(
        self, d_z, chs, ds_name,
        leakySlope=0.2
        ):
        super(ModelsUtils, self).__init__()
        
        self.d_z = d_z
        
        D_chs = chs
        G_chs = list(reversed(chs))
        
        self.D = Discriminator(d_z=d_z, chs=D_chs, ds_name=ds_name, leakySlope=leakySlope)
        self.G = Generator(d_z=d_z, chs=G_chs, ds_name=ds_name)
    
    def _save_models(self, D, G):
        self.D = D
        self.G = G
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def train(self):
        self.D.train()
        self.G.train()
        
    def eval(self):
        self.D.eval()
        self.G.eval()
    
    def get_noise(self, batch_size):
        return torch.randn(batch_size, self.d_z, device=self.device)
    
    def get_loss(self, criterion, model_output, labels):
        
        labels = labels.view_as(model_output)
        return criterion(model_output, labels)
    
    @torch.no_grad()
    def generate(self, num_img=1, noise=None, use_wnadb=False, train=False, step=None):
        self.G.eval()
        if noise is None:
            noise = self.get_noise(num_img)
            
        generated = self.G(noise)
        
        grid = torchvision.utils.make_grid(generated, nrow=int(math.sqrt(num_img)), normalize=True)
        
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        if use_wnadb or train:
            wandb.log({
                f"{'train' if train else 'test'}/samples": wandb.Image(ndarr, caption=(f"{step}" if step is not None else None))
            }, step=(step if step is not None else None))
    
    @torch.no_grad()
    def interpolate(self, steps=10, use_wandb=True):
        self.G.eval()
        
        z1 = self.get_noise(1)
        z2 = self.get_noise(1)
        
        # z_inner = (1-alpha) * z1 + alpha * z2
        alphas = torch.linspace(0, 1, steps, device=self.device)
        
        for i, alpha in enumerate(alphas):
            noise = (1.0 - alpha) * z1 + alpha * z2
            model_output = self.G(noise)
            grid = torchvision.utils.make_grid(model_output, nrow=1, normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            
            if use_wandb:
                wandb.log({
                    f"latent/interpolation_frame": wandb.Image(ndarr, caption=f"Frame {i}")
                })
                
    @torch.no_grad()
    def interpolate_some_dimension(self, dim, steps=10, val_range=(-3.0, 3.0), use_wandb=True):
        self.G.eval()
        
        z1 = self.get_noise(1)
        
        target_values = torch.linspace(val_range[0], val_range[1], steps, device=self.device)
        
        for i, val in enumerate(target_values):
            z = z1.clone()
            z[0, dim] = val
            
            model_output = self.G(z)
            grid = torchvision.utils.make_grid(model_output, nrow=1, normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            
            if use_wandb:
                wandb.log({
                    f"latent/interpolation_dim_{dim}": wandb.Image(ndarr, caption=f"Frame {i}")
                })
        
        
        
        
        
        
        
if __name__ == "__main__":
    B = 8
    d_z = 100   
    z = torch.randn([B, d_z])
    models = ModelsUtils(d_z=d_z, chs=[1,64,128],ds_name="MNIST")
    
    model_output = models.G(z)
    print(model_output.shape)
    
    
    x = torch.randn((B, 1, 28, 28))
    model_output = models.D(x)
    print(model_output)
    
    true_label = torch.full((B, ), 1.0)
    loss = models.get_loss(criterion=torch.nn.BCELoss(), model_output=model_output, labels=true_label)
    
    print(loss.item())
        
    
        
        
        
        
        
    
