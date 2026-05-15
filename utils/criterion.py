import torch
 
class CrossEntropy:
    def __init__(self):
        self.outputs = None
        self.targets = None
        self.eps = 1e-9
        pass
    
    def forward(self, outputs, targets, eps=1e-9):
        
        self.outputs = outputs
        self.targets = targets
        self.eps = eps
        
        num_class = outputs.size(-1)
        reshaped_outputs = outputs.view(-1, num_class)
        reshaped_targets = targets.view(-1)
        
        batch_size = reshaped_outputs.size(0)
        
        # tensor(batch_size)
        target_probs = reshaped_outputs[range(batch_size), reshaped_targets]
        
        loss = -torch.mean(torch.log(target_probs + eps))
        
        return loss  
    
    def backward(self):
        batch_size = self.outputs.size(0)
        num_class = self.outputs.size(-1)
        
        grad = torch.zeros_like(self.outputs)
        
        target_probs = self.outputs[range(batch_size), self.targets]
        
        grad[range(batch_size), self.targets] = -1.0 / (batch_size * (target_probs + self.eps))
        
        return grad