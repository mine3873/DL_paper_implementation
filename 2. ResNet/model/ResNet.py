import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    ((H + 2p - f_h + s) / s), ((W + 2p - f_w + s) / s)
    """
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.expansion = 1
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, num_layers, config):
        super(ResNet, self).__init__()
        self.config = config
        self.in_planes = config.num_channel[1]
        
        self.conv1 = nn.Conv2d(
            self.config.num_channel[0],
            self.config.num_channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
            )
        self.bn1 = nn.BatchNorm2d(self.config.num_channel[1])
        
        self.layer1 = self.make_layer(
            ResidualBlock,
            self.config.num_channel[1],
            num_layers,
            stride=1
            )
        self.layer2 = self.make_layer(
            ResidualBlock,
            self.config.num_channel[2],
            num_layers,
            stride=2
            )
        self.layer3 = self.make_layer(
            ResidualBlock,
            self.config.num_channel[3],
            num_layers,
            stride=2
            )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.config.num_channel[3], len(config.classes))        
        
        
    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        # (batch_size, 16, 32, 32)
        
        out = self.layer1(out)
        # (batch_size, 16, 32, 32)
        
        out = self.layer2(out)
        # (batch_size, 32, 16, 16)
        
        out = self.layer3(out)
        # (batch_size, 64, 8, 8)
        
        out = self.avg_pool(out)
        # (batch_size, 64, 1, 1)
        
        out = torch.flatten(out, 1)
        # (batch_size, 64)
        
        out = self.fc(out)
        # (batch_size, num_classes)
        
        return out
    