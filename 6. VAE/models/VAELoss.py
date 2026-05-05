import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta=1):
        super(VAELoss, self).__init__()
        self.beta = beta
        
    def forward(self, X, X_hat, mu, logvar):
        # L = Reconstruction Loss + KL divergence
        
        # Reconstruction Loss
        # p(x|z) = 1 / ((2 * pi * (std ** 2)) ** 0.5) * exp(-((x - x_hat) ** 2) / (2 * (std**2)))
        # logp(x|z) = ln( 1 / ((2 * pi * (std**2)) ** 0.5)) - ((x - x_hat) ** 2) / (2 * (std**2))
        # logp(x|z) \propto -((x - x_hat) ** 2)
        # Maximizing logp(x|z) equals minimizing mse(x,x_hat)
        
        reconstruction_loss = F.mse_loss(X, X_hat, reduction='sum')
        
        # KL divergence
        # 0.5 * SUM( std ** 2 + mu ** 2 -1 - ln(std ** 2))
        # = -0.5 * sum(1 + logvar - mu ** 2 - exp(logvar))
        
        KL_loss = -0.5 * torch.sum(1 + logvar - (mu**2) - logvar.exp())
        
        batch_size = X.size(0)
        
        return (reconstruction_loss + self.beta * KL_loss) / batch_size
    