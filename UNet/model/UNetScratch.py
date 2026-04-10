import torch
import torch.nn as nn

UNetArchitecture = [
    ()
]

class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0, bias=False)
        
        
    def forward(self, X):
        return self.relu(self.conv2(self.relu(self.conv1(X))))
        
        
        
class UNetScratch(nn.Module):
    """
    Upsampling (Transposed Convolution):
    H_out = (H_in - 1) * stride
          - 2p + kernel_size + out_padding
          
    when kernel_size = 2, stride = 2, padding = 0
    2H = (H - 1) * 2 - 0 + 2 + out_padding
       = 2H -2 + 2 + out_padding
    
    out_padding = 0
    
    
    """
    def __init__(self, config):
        super(UNetScratch, self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        layers = self.makeLayers()
        
        self.conv_layers = nn.ModuleList(layers[0])
        self.up_conv_layers = nn.ModuleList(layers[1])
        self.expa_layers = nn.ModuleList(layers[2])
        
        self.dropout = nn.Dropout2d(p=config.dropout, inplace=False)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
        self._initialize_weights()
        
        
    def makeLayers(self):
        conv_layers = []
        in_channel, out_channel = 1, 64
        for i in range(5):
            conv_layers += [Block(in_channel, out_channel)]
            if i < 4:
                in_channel = out_channel
                out_channel = in_channel * 2
        
        up_conv_layers = []
        expa_layers = []
        
        for _ in range(4):
            in_channel = out_channel
            out_channel = in_channel // 2
            up_conv_layers += [nn.ConvTranspose2d(
                in_channels=in_channel, out_channels=out_channel,
                kernel_size=2, stride=2, padding=0, output_padding=0, bias=False
            )]
            expa_layers += [Block(in_channel, out_channel)]
            
        return conv_layers, up_conv_layers, expa_layers
            
    def forward(self, X):
        conv_outputs = []
        for i, conv in enumerate(self.conv_layers):
            X = conv(X)
            if i < 4:
                conv_outputs += [X]
                X = self.maxPool(X)
        
        X = self.dropout(X)
        
        for up_conv, expa in zip(self.up_conv_layers, self.expa_layers):
            X = up_conv(X)
            conv_output = conv_outputs.pop(-1)
            X_h, X_w = X.size(-2), X.size(-1)
            conv_h, conv_w = conv_output.size(-2), conv_output.size(-1)
            h_diff = (conv_h - X_h) // 2
            w_diff = (conv_w - X_w) // 2
            cropped_conv = conv_output[:, :, h_diff:h_diff + X_h, w_diff:w_diff + X_w]
            
            X = torch.concat((cropped_conv, X), dim=1)
            X = expa(X)
        
        return self.final_conv(X)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        
            
if __name__ == "__main__":  
    pass
        
            
