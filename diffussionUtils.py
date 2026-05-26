import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os

class DiffusionUtils:
    def __init__(self, betas):
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        
        self.one_minus_alpha_bars = 1.0 - self.alpha_bars
        self.sqrt_one_minus_alpha_bars = torch.sqrt(self.one_minus_alpha_bars)
        
        self.prev_alpha_bars = torch.cat([torch.tensor([1.0], device=betas.device), self.alpha_bars[:-1]])
        self.sqrt_prev_alpha_bars = torch.sqrt(self.prev_alpha_bars)
        self.one_minus_prev_alpha_bars = 1.0 - self.prev_alpha_bars
        
        self.posterior_mean_coeff1 = self.sqrt_prev_alpha_bars * betas / self.one_minus_alpha_bars
        self.posterior_mean_coeff2 = self.sqrt_alphas * self.one_minus_prev_alpha_bars / self.one_minus_alpha_bars
        
        self.posterior_var = (self.one_minus_prev_alpha_bars * betas) / self.one_minus_alpha_bars
        self.posterior_log_var = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        
        self.factor_sqrt_alpha_bars = torch.sqrt(1.0 / self.alpha_bars)
        self.factor_sqrt_alpha_bars_minus_one = torch.sqrt((1.0 / self.alpha_bars) - 1)
    
    @staticmethod
    def _extract(a, t, x_shape):
        B, = t.shape
        
        out = a.to(t.device).gather(0, t).float()
        
        return out.view(B, *((len(x_shape) - 1) * (1,)))
    
    def q_sample(self, x0, t, noise=None):
        """
        x_t ~ q(x_t|x0)
        """
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)
        assert noise.shape == x0.shape
        
        return (self._extract(self.sqrt_alpha_bars, t, x0.shape) * x0 
                + self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape) * noise)

    def q_posterior(self, x_t, x0, t):
        posterior_mean = (self._extract(self.posterior_mean_coeff1, t, x0.shape) * x0 +
                          self._extract(self.posterior_mean_coeff2, t, x0.shape) * x_t)
        
        posterior_var = self._extract(self.posterior_var, t, x0.shape)
        posterior_log_var = self._extract(self.posterior_log_var, t, x0.shape)
        
        return posterior_mean, posterior_var, posterior_log_var
    
    def predict_x0_from_noise(self, x_t, t, noise):
        return (self._extract(self.factor_sqrt_alpha_bars, t, x_t.shape) * x_t -
                self._extract(self.factor_sqrt_alpha_bars_minus_one, t, x_t.shape) * noise)    
     
    def p_sample(self, model, x_t, t, t_idx):
        model_output = model(x_t, t)
        
        x0_recon = self.predict_x0_from_noise(x_t, t, noise=model_output)
        #x0_recon = torch.clamp(x0_recon, -1.0, 1.0)
        
        posterior_mean, _, posterior_log_var = self.q_posterior(x_t, x0_recon, t)
        
        if t_idx == 0:
            return posterior_mean
        else:
            z = torch.randn_like(x_t)
            return posterior_mean + torch.exp(0.5 * posterior_log_var) * z
    
    @torch.no_grad()
    def p_sample_loop(self, model, total_timesteps, shape):
        model.eval()
        x_t = torch.randn(shape, device=model.device)
        batch_size = shape[0]
        
        for t_idx in reversed(range(total_timesteps)):
            t = torch.randint(0, total_timesteps, (batch_size,), dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, t_idx)
        
        model.train()
        return x_t

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
        
        
        
    