import math

class CosineAnnealing:
    """
    lr_t = lr_min + 1/2 * (lr_max - lr_min) * (1 + cos(T_cur / T_max * pi))
    """
    def __init__(self, optimizer, T_max, lr_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.lr_min = lr_min
        self.lr_max = optimizer.lr
        self.T_cur = 0
        
    def step(self):
        self.T_cur += 1
        
        cos_val = math.cos(math.pi * self.T_cur / self.T_max)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + cos_val)
        
        self.optimizer.lr = lr
        
class MultiStepLR:
    
    def __init__(self, optimizer, milestones, gamma):
        self.optimizer = optimizer
        self.milestones = set(milestones)
        self.gamma = gamma
        
        self.T_cur = 0
    
    def step(self):
        self.T_cur += 1
        
        if self.T_cur in self.milestones:
            self.optimizer.lr = self.optimizer.lr * self.gamma
    