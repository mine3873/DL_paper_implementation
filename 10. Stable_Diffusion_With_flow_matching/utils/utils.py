import torch
import torch.nn as nn

class AdaptiveWeight:
    def __init__(self, disc_start_step: int = 1e4, disc_weight: float = 1.0):
        self.disc_start_step = disc_start_step
        self.disc_weight = disc_weight
        
    def get_weight(self, recon_loss, gan_loss, last_layer_params, step):
        if step < self.disc_start_step:
            return 0.0
        
        nll_grads = torch.autograd.grad(recon_loss, last_layer_params, retain_graph=True)[0]
        g_grads = torch.autograd.grad(gan_loss, last_layer_params, retain_graph=True)[0]
        
        weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        weight = torch.clamp(weight, 0.0, 1e4).detach()
        
        return weight * self.disc_weight

@torch.no_grad()
def get_scaling_factor(vae, train_loader, device, num_batches=30):
    vae.eval()
    latents = []
    
    for i, (x, _) in enumerate(train_loader):
        if i >= num_batches:
            break
        
        x = x.to(device)
        h = vae.encoder(x)
        
        if vae.regularizer == 'KL':
            mu = vae.mu_layer(h)
        else:
            mu = h
            
        latents.append(mu.cpu())
        
    all_latents = torch.concat(latents, dim=0)
    std = all_latents.std().item()
    scaling_factor = 1.0 / std
    
    return scaling_factor


