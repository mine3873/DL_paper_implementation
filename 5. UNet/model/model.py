import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch,):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch)
        
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        
        return h


class UNet(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024), class_n=2):
        super(UNet, self).__init__()
        self.chs = chs
        self.class_n = class_n
        downs, ups = self._make_layers_()
        
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)
        self.tail = nn.Conv2d(in_channels=self.chs[1], out_channels=self.class_n ,kernel_size=1)
        
        
    def _make_layers_(self):
        downs = []
        
        chs_len = len(self.chs)
        
        assert self.chs is not None and isinstance(self.chs, (list, tuple))
        
        for i in range(chs_len - 1):
            downs.append(ConvBlock(
                in_ch=self.chs[i], out_ch=self.chs[i+1]))
            
            if self.chs[i+1] != self.chs[-1]:
                downs.append(nn.MaxPool2d(kernel_size=2, stride=2)) 
        
        ups = []
        
        for i in reversed(range(1, chs_len - 1)):
            ups.append(
                nn.ConvTranspose2d(
                    in_channels=self.chs[i+1], out_channels=self.chs[i],
                    kernel_size=2, stride=2, bias=False
                    ))
            
            ups.append(ConvBlock(in_ch=self.chs[i+1], out_ch=self.chs[i]))
        
        return downs, ups
        
    def forward(self, x):
        outputs = []
        h = x
        for block in self.downs:
            h = block(h)
            if isinstance(block, ConvBlock) and h.size(1) != 1024:
                outputs.append(h)
            
        
        for block in self.ups:
            if isinstance(block, nn.ConvTranspose2d):
                h = block(h)
            elif isinstance(block, ConvBlock):
                prev_out = outputs.pop()
                
                cur_size = h.size(-1)
                prev_size = prev_out.size(-1)
                size_diff = (prev_size - cur_size) // 2
                
                prev_out = prev_out[
                    :, :,
                    size_diff : size_diff + cur_size,
                    size_diff : size_diff + cur_size]
                h = block(torch.concat([prev_out, h], dim=1))
            else:
                NotImplementedError()    
        
        return self.tail(h)
    
    
if __name__ == "__main__":
    x = torch.rand((1, 1, 572, 572))
    
    model = UNet()
    
    output = model(x)
    
    print(output.shape)
