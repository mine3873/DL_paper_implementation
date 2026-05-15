import torch
import torch.nn as nn

class ConvLayers(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, model_name: str = "resnet"):
        super(ConvLayers, self).__init__()
        
        self.model_name = model_name
        
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch)
        
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)
        
        self.relu = nn.ReLU()
        
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=out_ch)
            )
        
    def forward(self, x):
        h = self.bn1(self.conv1(x))
        h = self.bn2(self.conv2(h))
        
        if self.model_name == "resnet":
            h += self.shortcut(x)
        
        return self.relu(h)
        
class Net(nn.Module):
    def __init__(self, layer_num, filters, class_n, model_name: str = "resnet",):
        super(Net, self).__init__()
        
        assert layer_num > 0
        self.layer_num = layer_num
        
        assert model_name == "resnet" or model_name == "plainnet"
        self.model_name = model_name
        
        assert isinstance(filters, (list, tuple))
        self.head = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, padding=1, bias=False)
        
        self.conv_layers1 = self._make_layers(
            in_ch=filters[1], out_ch=filters[1])
        
        self.conv_layers2 = self._make_layers(
            in_ch=filters[1], out_ch=filters[2], stride=2)
        
        self.conv_layers3 = self._make_layers(
            in_ch=filters[2], out_ch=filters[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.tail = nn.Linear(in_features=filters[3], out_features=class_n)
        
        
          
    def _make_layers(self, in_ch, out_ch, stride=1):
        strides = [stride] + [1] * (self.layer_num - 1)
        layers = []
        
        for s in strides:
            layers.append(ConvLayers(in_ch, out_ch, stride=s, model_name=self.model_name))
            in_ch = out_ch
        
        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        # x (B, C, H, W)
        h = self.head(x)
        
        h = self.conv_layers1(h)
        h = self.conv_layers2(h)
        h = self.conv_layers3(h)
        
        h = self.avgpool(h)
        h = torch.flatten(h, start_dim=1)
        h = self.tail(h)
        
        return h
        
        


    
