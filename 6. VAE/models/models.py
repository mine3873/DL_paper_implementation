import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_z, chs, leaky=True, leakySlope=0.2, ds_name="LSUN"):
        super(Encoder, self).__init__()
        
        self.d_z = d_z
        self.ds_name = ds_name
        
        if ds_name=="LSUN":
            kernel_size = 4
        elif ds_name=="MNIST":
            kernel_size = 7
        
        self.blocks = nn.ModuleList([])
        
        # [3, 128, 256, 512, 1024]
        for i in range(len(chs) - 1):
            self.blocks.append(nn.Conv2d(
                in_channels=chs[i], out_channels=chs[i+1], kernel_size=4, padding=1, stride=2, bias=False
            ))
            if i > 0:
                self.blocks.append(nn.BatchNorm2d(num_features=chs[i+1]))
            
            self.blocks.append(nn.LeakyReLU(leakySlope) if leaky else nn.ReLU())
        
        d_fc_input = chs[-1] * kernel_size * kernel_size
        
        self.to_mu = nn.Linear(in_features=d_fc_input, out_features=d_z)
        self.to_logvar = nn.Linear(in_features=d_fc_input, out_features=d_z)
        
    def forward(self, x):
        h = x
        
        for block in self.blocks:
            h = block(h)
        
        h = h.view(h.size(0), -1)
        
        return self.to_mu(h), self.to_logvar(h)
        
class Decoder(nn.Module):
    def __init__(self, d_z, chs, ds_name="LSUN"):
        super(Decoder, self).__init__()
        self.d_z = d_z
        self.ds_name = ds_name
        chs.insert(0, d_z)
        
        if ds_name=="LSUN":
            kernel_size = 4
        elif ds_name=="MNIST":
            kernel_size = 7
            
        self.blocks = nn.ModuleList([])
        
        # [1024, 512, 256, 128, 3]
        for i in range(len(chs) - 1):
            if i > 0:
                self.blocks.append(nn.BatchNorm2d(num_features=chs[i]))
                self.blocks.append(nn.ReLU())

            self.blocks.append(nn.ConvTranspose2d(
                in_channels=chs[i], out_channels=chs[i+1],
                kernel_size=(4 if i != 0 else kernel_size), padding=int(i != 0), stride=(int(i != 0) + 1), bias=False
                ))
        
        if ds_name=="MNIST":
            self.tail = nn.Sigmoid()
        elif ds_name=="LSUN":
            self.tail = nn.Tanh()
        

    def forward(self, x):
        h = x.view(x.size(0), -1, 1, 1)
        for block in self.blocks:
            h = block(h)
        
        h = self.tail(h)
         
        return h       