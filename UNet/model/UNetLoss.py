import torch.nn as nn
import torch

class UNetLoss(nn.Module):
    """
    p_k(x) = exp(a_k(x)) / sum exp(a_k'(x))
    a_k(x) = the activation with (c', h', w') 
    K: num of classes
    p_k(x): approximated maximum-function
        -> if k make activation a_k(x) maximum, p_k(x) is around 1
        -> otherwise, p_k(x) is around 0
        
    crossEntropy: 
        E = sum(w(x) * log(p_{ell(x)}(x)))
    ell: Z^2 -> {1,...,k} 
        the true label of each pixel
    
    w: Z^2 -> R
        weight map, which gives some pixel more importance 
        
    w(x) = w_c(x) + w_0 * exp( -(d_1(x) + d_2(x))^2 / 2 * sigma^2)
    
    w_c: Z^2 -> R
        the weight map to balance the class frequencies 
    
    d_1: Z^2 -> R
        the distance to the border of the nearest cell
        
    d_2: Z^2 -> R
        the distance to the border of the second nearest cell
        
    w_0 = 10
    sigma is about 5 pixels
    
    """
    
    def __init__(self, eps=1e-6):
        super(UNetLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.eps = eps
        
    def forward(self, outputs, targets, weight_maps):
        """
        outputs : tensor(batch_size, 2, 388, 388)
        targets : tensor(batch_size, 572, 572)
        weight_maps : tensor(batch_size, 572, 572)
        """
        diff = (targets.size(-1) - outputs.size(-1)) // 2
        targets = targets[:, diff:diff + outputs.size(-2), diff:diff + outputs.size(-1)]
        weight_maps = weight_maps[:, diff:diff + outputs.size(-2), diff:diff + outputs.size(-1)]
        
        log_probs = torch.log_softmax(outputs, dim=1)
        log_p_target = torch.gather(log_probs, dim=1, index=targets.unsqueeze(1))
        # p_target : tensor(batch_size, 1, 388, 388) 
        
        log_p_target = log_p_target.squeeze(1)
        
        E = weight_maps * log_p_target
        loss = -E
        return loss.mean()
        
