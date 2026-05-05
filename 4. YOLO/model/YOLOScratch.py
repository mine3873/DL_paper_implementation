import numpy as np
import torch.nn as nn
import torch

"""
after conv operation:
out_h = 1 + (in_h + 2p - kernel_size) / stride
out_w = 1 + (in_w + 2p - kernel_size) / stride
"""

"""
list: (kernel_size, out_channel, stride, padding)
'M': Maxpool Layer
tuple: [list, ..., list, iteration]
"""
ARCHITECTURE_CONFIG = [
    (7, 64, 2, 3), 'M',
    (3, 192, 1, 1), 'M',
    (1, 128, 1, 0), (3, 256, 1, 1), (1,256, 1, 0), (3, 512, 1, 1), 'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], (1, 512, 1, 0), (3, 1024, 1, 1), 'M',
    [(1,512, 1, 0), (3, 1024, 1, 1), 2], (3, 1024, 1, 1), (3, 1024, 2, 1), 
    (3, 1024, 1, 1), (3, 1024, 1, 1)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, config, **kwargs):
        super(CNNBlock, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.leakyReLU = nn.LeakyReLU(self.config.leakyRelu_w)

    def forward(self, X):
        return self.leakyReLU(self.bn(self.conv(X)))

class YOLOScratch(nn.Module):
    def __init__(self, config, in_channel=3, **kwargs):
        super(YOLOScratch, self).__init__()
        self.config = config
        self.in_channel = in_channel
        self.architecture = ARCHITECTURE_CONFIG
        self.conv_layers = self.create_conv_layers(self.architecture)
        self.fc_layers = self.create_fc_layers()
        
        
    def create_conv_layers(self, architecture):
        layers = []
        in_channel = self.in_channel
        
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(
                    in_channel=in_channel,
                    out_channel=x[1],
                    config=self.config,
                    kernel_size=x[0],
                    stride=x[2],
                    padding=x[3]
                )]
                in_channel = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                iteration = x[-1]
                for _ in range(iteration):
                    for conv in x[:-1]:
                        layers += [CNNBlock(
                            in_channel=in_channel,
                            out_channel=conv[1],
                            config=self.config,
                            kernel_size=conv[0],
                            stride=conv[2],
                            padding=conv[3]
                        )]
                        in_channel = conv[1]
        
        return nn.Sequential(*layers)
    
    def create_fc_layers(self):
        S = self.config.S
        B = self.config.B
        C = self.config.C
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(self.config.leakyRelu_w),
            nn.Dropout(self.config.dropout),
            nn.Linear(4096, S * S * (B * 5 + C))
        )
    
    def load_darknet_weights(self, weights_path):
        """
        darknet:
        [BN bias, BN weight, BN mean, BN var]
        """
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        
        ptr = 0
        for m in self.modules():
            if isinstance(m, CNNBlock):
                conv = m.conv
                bn = m.bn
                
                num_b = bn.bias.numel()
                num_w = conv.weight.numel()
                
                if ptr + (num_b * 4) + num_w > len(weights):
                    break
                
                bn_bias = torch.from_numpy(weights[ptr : ptr + num_b])
                bn.bias.data.copy_(bn_bias)
                ptr += num_b
                
                bn_weight = torch.from_numpy(weights[ptr : ptr + num_b])
                bn.weight.data.copy_(bn_weight)
                ptr += num_b
                
                bn_mean = torch.from_numpy(weights[ptr : ptr + num_b])
                bn.running_mean.data.copy_(bn_mean)
                ptr += num_b
                
                bn_var = torch.from_numpy(weights[ptr : ptr + num_b])
                bn.running_var.data.copy_(bn_var)
                ptr += num_b
                
                
                conv_weight = torch.from_numpy(weights[ptr : ptr + num_w])
                conv.weight.data.copy_(conv_weight.view_as(conv.weight))
                ptr += num_w
            
            if ptr >= len(weights):
                break
        
    
    def forward(self, X):
        S = self.config.S
        B = self.config.B
        C = self.config.C
        X = self.conv_layers(X)
        return self.fc_layers(X).reshape(-1, S, S, B * 5 + C)
        