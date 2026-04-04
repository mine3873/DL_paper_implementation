import torch

class ReLU:
    """
    ReLU(X) = max(X,0) = X * (X>0)
    """
    
    def __init__(self):
        self.mask = None
    
    def forward(self, X):
        self.mask = (X > 0)
        return torch.where(self.mask, X, torch.tensor(0.0))
    
    def backward(self, grad_output):
        """
        if x > 0: 
            dx = 1
        else:
            dx = 0    
        """
        
        dx = grad_output.clone()
        
        dx[~self.mask] = 0
        
        return dx
    
class Softmax:
    """
    Softmax(x) = exp(X - C) / sum(exp(X-C))
    """
    
    def __init__(self, dim=-1):
        self.dim = dim
        self.output = None
    
    def forward(self, X):
        C = torch.max(X, dim=self.dim, keepdim=True)[0]
        exp_x = torch.exp(X - C)
        sum_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        
        self.output = exp_x / sum_x
        return self.output

    def backward(self, grad_output):
        """
        grad_output: tensor(batch, ... , output_size)
        
        - calcuate example:
        x = [x1, x2, x3]
        y = softmax(x) = [y1, y2, y3]
        grad_output = [g1, g2, g3]
        
        dx = dy/dx @ grad_output
        dx = [dx1, dx2, dx3]
        
        dy/dx = [
            [y1(1 - y1), -y1y2, -y1y3],
            [-y2y1, y2(1 - y2), -y2y3],
            [-y3y1, -y3y2, y3(1  -y3)]
        ]
        
        dx1 = y1(1 - y1)g1 + -y1y2g2 + -y1y3g3
            = y1g1 - y1(y1g1 + y2g2 + y3g3)
            = y1(g1 - sum(y * grad_output))
            = y1(g1 - v)
        ...
        dx = y(g - v)    
        """
        
        v = torch.sum(self.output * grad_output, dim=self.dim, keepdim=True)
        
        dx = self.output * (grad_output - v)
        
        return dx
        
        
        
        
    