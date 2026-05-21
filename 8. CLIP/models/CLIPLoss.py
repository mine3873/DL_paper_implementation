import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        
        
    def forward(self, logits):
        device = logits.device
        batch_size = logits.size(0)
        
        labels = torch.arange(batch_size, device=device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        
        return (loss_i + loss_t) / 2
        