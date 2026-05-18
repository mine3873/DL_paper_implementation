import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, d_z, chs=[1024, 512, 256, 128, 3], ds_name="LSUN"):
        super(Generator, self).__init__()
        self.d_z = d_z
        
        assert isinstance(chs, list)
        chs.insert(0, d_z)
        
        self.layers = nn.ModuleList([])
        
        
        if ds_name=="LSUN":
            kernel_size = 4
        elif ds_name=="MNIST":
            kernel_size = 7
        for i in range(len(chs) - 1):
            if i > 0:
                self.layers.append(nn.BatchNorm2d(num_features=chs[i]))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.ConvTranspose2d(
                in_channels=chs[i], out_channels=chs[i+1],
                kernel_size=(4 if i != 0 else kernel_size), padding=int(i != 0), stride=(int(i != 0) + 1), bias=False
                ))

        self.tanh = nn.Tanh()
        
    def forward(self, x):
        assert x is not None and x.size(-1) == self.d_z
        
        if len(x.shape) < 4:
            while True:
                x = x.unsqueeze(-1)
                if len(x.shape) == 4:
                    break
        h = x
        
        for block in self.layers:
            h = block(h)
            
        h = self.tanh(h)
                 
        return h
        
class Discriminator(nn.Module):
    def __init__(self, d_z, chs: list =[3, 128, 256, 512, 1024], ds_name="LSUN", leakySlope=0.2):
        super(Discriminator, self).__init__()
        self.d_z = d_z
        
        assert isinstance(chs, list)
        chs.append(1)
        self.layers = nn.ModuleList([])
        
        if ds_name=="LSUN":
            kernel_size = 4
        elif ds_name=="MNIST":
            kernel_size = 7
        
        for i in range(len(chs) - 1):
            self.layers.append(nn.Conv2d(
                in_channels=chs[i], out_channels=chs[i+1],
                kernel_size=(4 if i != len(chs) - 2 else kernel_size), padding=int(i != len(chs) - 2), stride=(int(i != len(chs) - 2) + 1), bias=False
            ))
            if i > 0 and i < len(chs) - 2:
                self.layers.append(nn.BatchNorm2d(num_features=chs[i + 1]))
                self.layers.append(nn.LeakyReLU(negative_slope=leakySlope))
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        assert x is not None      
        
        h = x
        for block in self.layers:
            h = block(h)
        
        h = h.view(x.size(0), -1)
        
        return self.sigmoid(h)
        
        
if __name__ == "__main__":
    B = 32
    d_z = 100    
    
    def test_G():
        z = torch.randn([B, d_z])
        model = Generator(d_z=d_z, chs=list(reversed([3, 128, 256, 512, 1024])),ds_name="LSUN")
        
        output = model(z)
        
        print(output.shape)
    
    def test_D():
        x = torch.randn((B, 3, 64, 64))
        model = Discriminator(d_z=d_z, chs=[3, 128, 256, 512, 1024],ds_name="LSUN")
        
        output = model(x)
        
        print(output.shape)
        
    test_D()
    