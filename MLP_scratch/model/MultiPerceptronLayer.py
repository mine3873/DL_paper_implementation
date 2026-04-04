import torch
import math
from model.utils import ReLU, Softmax

class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        #h_layer = Activation( Input @ W1 + b1 )
        # W1 : tensor(28 * 28, hidden_size)
        # b1 : tensor(hidden_size)
        
        #o_layer = Activation( hidden_output @ W2 + b2 )
        # W2 : tensor(hidden_size, output_size=10)
        # b2 : tensor(output_size=10)
        
        self.is_training = True
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W1 = torch.randn(self.input_size, self.hidden_size) * math.sqrt(1.0 / self.input_size)
        self.b1 = torch.zeros(self.hidden_size)
        
        self.W2 = torch.randn(self.hidden_size, self.output_size) * math.sqrt(1.0 / self.hidden_size)
        self.b2 = torch.zeros(self.output_size)
        
        for p in [self.W1, self.b1, self.W2, self.b2]:
            p.requires_grad = True
        
        self.ReLU = ReLU()
        self.softmax = Softmax()
        
        self.input_flat = None
        self.hidden_output = None
        self.output = None
        
        self.dW1 = None
        self.db1 = None
        
        self.dW2 = None
        self.db2 = None
        
        self.params = []
        
    def train(self):
        self.is_training = True
        
    def eval(self):
        self.is_training = False
    
    def forward(self, input):
        self.input_flat = input.view(-1, self.input_size)
        
        pre_act1 = self.input_flat @ self.W1 + self.b1
        self.hidden_output = self.ReLU.forward(X=pre_act1)
        
        pre_act2 = self.hidden_output @ self.W2 + self.b2
        self.output = self.softmax.forward(pre_act2)
        
        return self.output
            
    def backward(self, grad_output):
        """
        self.output = o
        self.hidden_output = x3
        L = o
        o = softmax(x5)
        x5 = x4 + b2
        x4 = x3 @ W2
        x3 = ReLU(x2)
        x2 = x1 + b1
        x1 = x @ W1
        
        """
        
        d_pre_act2 = self.softmax.backward(grad_output)
        
        """
        dL/dW2 = dL/do * do/dx5 * dx5/dW2
               = d_pre_act2 * hidden_output
               = (batch, output_size) * (batch, hidden_size)
        to make dW2 to be tensor(hidden_size, output_size), dW2 = hidden_output^T @ d_pre_act2
        """
        self.dW2 = self.hidden_output.t() @ d_pre_act2
        
        """
        pre_act: tensor(batch_size, ..., output_size)
        
        dpre_act2/db2 = 1
        dL/db2 = dL/dx5_{1} * 1 + dL/dx5_{2} * 1 + ... + dL/dx5_{batch_size} * 1
               = sum(d_pre_act2, dim=0)
        """
        self.db2 = torch.sum(d_pre_act2, dim=0)
        
        """
        d_hidden = dL/dx3 = dL/do * do/dx5 * dx5/dx4 * dx4/dx3
                 = pre_act2 * 1 * W2
        (batch_size, hidden_size) = (batch_size, output_size) @ (hidden_size, output_size)^T
        """
        d_hidden = d_pre_act2 @ self.W2.t()
        
        d_pre_act1 = self.ReLU.backward(d_hidden)
        
        self.dW1 = self.input_flat.t() @ d_pre_act1
        self.db1 = torch.sum(d_pre_act1, dim=0)
        
        self.W1.grad = self.dW1
        self.W2.grad = self.dW2
        self.b1.grad = self.db1
        self.b2.grad = self.db2
        
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
        
        
            
if __name__ == "__main__":
    mlp = MLP()

