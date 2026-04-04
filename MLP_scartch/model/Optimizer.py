import torch

class Optimizer:
    def __init__(self, params):
        self.params = [p for p in params if p.requires_grad]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
        

class AdamW(Optimizer):
    """
    m_t = beta1 * m_{t-1} + (1-beta1) * g_t
    m_t = m_t / 1 - beta1**t
    
    s_t = beta2 * s_{t-1} + (1-beta2) * g_t ** 2
    s_t = s_t / 1 - beta2**t
    
    theta_t = theta_t - lr_t * (m_t / (math.sqrt(s_t) + epsilon) + lambda * theta_t)
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps=eps
        self.weight_decay=weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.s = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g_t = p.grad.data
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t
                self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * (g_t**2)
                
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                s_hat = self.s[i] / (1 - self.beta2**self.t)
                
                if self.weight_decay != 0:
                    p.data -= self.lr * self.weight_decay * p.data
                
                p.data -= self.lr * (m_hat / (torch.sqrt(s_hat) + self.eps))
                
                
        