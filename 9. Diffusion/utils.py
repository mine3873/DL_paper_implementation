import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self,):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        scores = torch.matmul(Q, torch.transpose(K, -1, -2)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e4)
        
        attention_weights = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
        
        return torch.matmul(attention_weights, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.scaledDotAttention = ScaledDotProductAttention()
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        d_model = Q.size(-1)
        d_k = d_model // self.n_heads
        
        Q_proj = self.W_q(Q).view(batch_size, Q.size(1), self.n_heads, d_k).transpose(1, 2)
        K_proj = self.W_k(K).view(batch_size, K.size(1), self.n_heads, d_k).transpose(1, 2)
        V_proj = self.W_v(V).view(batch_size, V.size(1), self.n_heads, d_k).transpose(1, 2)
        
        attention_output = self.scaledDotAttention(Q_proj, K_proj, V_proj, mask)
        attention_output = torch.transpose(attention_output, 1, 2).contiguous().view(attention_output.size(0), -1, d_model)
        
        output = self.W_o(attention_output)
        
        return output
    
class DiffusionUtils(nn.Module):
    def __init__(self, betas, criterion):
        super(DiffusionUtils, self).__init__()
        
        assert isinstance(betas, torch.Tensor)
        self.betas = betas
        self.alphas = 1.0 - betas
        
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod)
        
        self.one_minus_alpha_cumprod = 1.0 - self.alphas_cumprod
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(self.one_minus_alpha_cumprod)
        
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        
        self.one_minus_alpha_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        
        
        self.q_posterior_mean_coefficient1 = self.sqrt_alphas_cumprod_prev * betas / self.one_minus_alpha_cumprod
        self.q_posterior_mean_coefficient2 = self.sqrt_alpha_cumprod * self.one_minus_alpha_cumprod_prev / self.one_minus_alpha_cumprod
        
        self.posterior_variance = betas * self.one_minus_alpha_cumprod_prev / self.one_minus_alpha_cumprod 
        self.posterior_log_variance = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        
        if criterion == "mse":
            self.criterion = torch.nn.MSELoss(reduction="mean")
        else:
            self.criterion = torch.nn.MSELoss(reduction="mean") 
    
    @staticmethod
    def _extract(a, t, x_shape):
        B = t.size(0)
        assert x_shape[0] == B
        
        out = a.to(t.device).gather(0, t).float()
        assert out.shape[0] == B
        
        return out.view(B, *((len(x_shape) - 1) * (1,)))
        
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        assert noise.shape == x0.shape
        
        return self._extract(self.sqrt_alpha_cumprod, t, x0.shape) * x0 + self._extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape) * noise
    
    def q_posterior(self, x0, x_t, t):
        assert x0.shape == x_t.shape
        posterior_mean = (
            self._extract(self.q_posterior_mean_coefficient1, t, x0.shape) * x0
            + self._extract(self.q_posterior_mean_coefficient2, t, x0.shape) * x_t)
        
        posterior_variance = self._extract(self.posterior_variance, t, x0.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance, t, x0.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance.shape[0] ==x0.shape[0])
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def loss(self, model, x0, t, noise=None):
        assert t.shape[0] == x0.shape[0]
        
        if noise is None:
            noise = torch.randn_like(x0)
        assert noise.shape == x0.shape and noise.dtype == x0.dtype
        
        x_t = self.q_sample(x0, t, noise)
        assert x_t.shape == x0.shape
        
        x_recon = model(x_t, t)
        
        return self.criterion(x_recon, noise)
    
    @torch.no_grad()
    def p_mean_variance(self, model, x, t):
        pred_noise = model(x, t)
        
        x_recon = (
            x - self._extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape)* pred_noise
            ) / self._extract(self.sqrt_alpha_cumprod, t, x.shape)
        
        x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior(x0=x_recon, x_t=x, t=t)
        assert posterior_mean.shape == x_recon.shape == x.shape
        assert posterior_variance.shape == posterior_log_variance.shape
        
        return posterior_mean, posterior_variance, posterior_log_variance
            
    @torch.no_grad()
    def p_sample(self, model, x, t):
        model_mean, _, model_log_variance = self.q_posterior(model, x, t)
        
        noise = torch.randn_like(x)
        assert noise.shape == x.shape
        
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, num_timesteps):
        device = next(model.parameters()).device
        B = shape[0]
        
        assert isinstance(shape, (tuple, list))
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
            
        assert img.shape == shape
        return img
    
    @torch.no_grad()    
    def p_sample_loop_trajectory(self, model, shape, num_timesteps, per_step=100):
        device = next(model.parameters()).device
        B = shape[0]
        
        assert isinstance(shape, (tuple, list))
        img = torch.randn(shape, device=device)
        
        imgs = [img]
        
        for i in reversed(range(0, num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
            
            if i % per_step == 0 or i == 0:
                imgs.append(img)
        
        assert imgs[-1].shape == shape
        return torch.stack(imgs)
    
    @torch.no_grad()
    def interpolate(self, model, x1, x2, t, lam=0.5):
        t_batched = torch.full((x1.shape[0],), t, device=x1.device)
        xt1 = self.q_sample(x1, t=t_batched)
        xt2 = self.q_sample(x2, t=t_batched)
        
        xt_interp = (1-lam) * xt1 + lam * xt2
        img = xt_interp
        for i in reversed(range(0, t + 1)):
            step = torch.full((x1.shape[0],), i, device=x1.device)
            img = self.p_sample(model, img, step)
            
        return img
    
      
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }        
        
    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name].copy_(new_avg)
        
    def apply_shadow(self):
        self.backup = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
            
        