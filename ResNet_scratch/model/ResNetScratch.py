import torch
import math

class Conv2d:
    """
    (batch_size, in_planes, in_H, in_W) -> (batch_size, planes, out_h, out_w)
    
    X: tensor(xc, xh, xw)
    Y: tensor(yc, yh, yw)
    W: tensor(yc, xc, k, k)
    b: tensor(yc)
    
    Y[yc, yh, yw] = b + sum(X[:, stride * yh:stride * yh + k, stride * yw:stride * yw + k] * W[yc, xc, k, k])
    """
    
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias, device="cuda"):
        self.in_planes = in_planes
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        
        fan_in = in_planes * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        self.W = (torch.randn(planes, in_planes, kernel_size, kernel_size) * std).to(self.device)
        
        self.W.requires_grad = True
        if bias == False:
            self.b = None
        else:
            self.b = torch.zeros(planes).to(self.device)
            self.b.requires_grad = True
        
    def parameters(self):
        return [self.W, self.b] if self.b is not None else [self.W]
        
    def forward(self, X):
        batch_size = X.size(0)
        in_h, in_w = X.size(-2), X.size(-1)
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        X_padded = torch.nn.functional.pad(X, (self.padding, self.padding, self.padding, self.padding))
        
        # tensor(Batch_size, in_planes * kernel_size * kernel_size, L)
        # L = num of filters, out_h * out_w
        X_unfold = torch.nn.functional.unfold(
            X_padded,
            kernel_size=self.kernel_size,
            padding=0,
            stride=self.stride
        )
        
        W_flatten = self.W.view(self.planes, -1)
        
        output = W_flatten @ X_unfold
        
        output = output.view(batch_size, self.planes, out_h, out_w)
        
        if self.b is not None:
            output += self.b.view(1, -1, 1, 1)
            
        return output
        """
        output = torch.zeros((batch_size, self.planes, out_h, out_w), device=self.device)
        
        for b in range(batch_size):
            for oc in range(self.planes):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        window = X_padded[b, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]

                        temp = torch.sum(window * self.W[oc])
                        
                        if self.b is not None:
                            temp += self.b[oc]
                            
                        output[b, oc, oh, ow] = temp
        
        return output
        """
        
        
        
        
    
class BatchNorm2d:
    def __init__(self, planes):
        self.planes = planes
        
    def forward(self):
        pass