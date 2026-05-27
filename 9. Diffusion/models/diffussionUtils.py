import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os
from torchvision.utils import save_image, make_grid

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
        self.posterior_log_var = torch.log(torch.clamp(self.posterior_var, min=1e-20))
        
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
        """
        x_{t-1} ~ q(x_{t-1}| x_t, x0) ~= p(x_{t-1}|x_t)
        """
        model_output = model(x_t, t)
        
        x0_recon = self.predict_x0_from_noise(x_t, t, noise=model_output)
        x0_recon = torch.clamp(x0_recon, -1.0, 1.0)
        
        posterior_mean, _, posterior_log_var = self.q_posterior(x_t, x0_recon, t)
        
        if t_idx == 0:
            return posterior_mean
        else:
            z = torch.randn_like(x_t)
            return posterior_mean + torch.exp(0.5 * posterior_log_var) * z
    
    @torch.no_grad()
    def p_sample_loop(self, model, total_timesteps, shape, save_imgs=False):
        model.eval()
        x_t = torch.randn(shape, device=model.device)
        batch_size = shape[0]
        
        for t_idx in reversed(range(total_timesteps)):
            t = torch.full((batch_size,), t_idx, dtype=torch.long, device=model.device)
            x_t = self.p_sample(model, x_t, t, t_idx)
            
            if save_imgs:
                if t_idx % 100 == 0:
                    x_t_ = (x_t + 1.0) / 2.0
                    x_t_ = torch.clamp(x_t_, 0.0, 1.0)
                    
                    save_image(x_t_[0], f"step_{t_idx:01d}.png")
        
        model.train()
        return x_t
    
    @torch.no_grad()
    def interpolate(self, model, x0_1, x0_2, t_idx=500, num_lambdas=5, save_path="interpolation_grid.png"):
        model.eval()
        
        if len(x0_1.shape) == 3: x0_1 = x0_1.unsqueeze(0)
        if len(x0_2.shape) == 3: x0_2 = x0_2.unsqueeze(0)
        
        device = x0_1.device
        batch_size = x0_1.shape[0]
        
        noise = torch.randn_like(x0_1, device=device)
        t_tensor = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)
        
        z_t_1 = self.q_sample(x0_1, t_tensor, noise=noise)
        x_t_2 = self.q_sample(x0_2, t_tensor, noise=noise)
        
        lambdas = torch.linspace(0.0, 1.0, steps=num_lambdas, device=device)
        interp_images = []
        
        for lam in lambdas:
            xt_lam = (1.0 - lam.item()) * z_t_1 + lam.item() * x_t_2
            
            curr_x_t = xt_lam.clone()
            
            for step in reversed(range(t_idx + 1)):
                t_step = torch.full((batch_size,), step, dtype=torch.long, device=device)
                curr_x_t = self.p_sample(model, curr_x_t, t_step, step)
                
            curr_x_t = (curr_x_t + 1.0) / 2.0
            curr_x_t = torch.clamp(curr_x_t, 0.0, 1.0)
            interp_images.append(curr_x_t[0]) 
            
        grid = make_grid(torch.stack(interp_images), nrow=num_lambdas, padding=2)
        save_image(grid, save_path)
        
        model.train()
    
    @torch.no_grad()
    def forward_process_steps(self, x0, num_steps=10):
        if len(x0.shape) == 3:
            x0 = x0.unsqueeze(0)
            
        device = x0.device
        total_T = len(self.betas)
        
        step_indices = torch.linspace(0, total_T - 1, steps=num_steps, dtype=torch.long, device=device)
        
        for t_ in step_indices:
            t_idx = t_.item()
            t = torch.full((x0.shape[0],), t_idx, dtype=torch.long, device=device)
            
            x_t = self.q_sample(x0, t)
            x_t = (x_t + 1.0) / 2.0
            x_t = torch.clamp(x_t, 0.0, 1.0)
            
            save_image(x_t[0], f"step_{t_idx:01d}.png")
            
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
        
        
        
    