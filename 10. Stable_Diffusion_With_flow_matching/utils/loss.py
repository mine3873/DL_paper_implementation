import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class ReconLoss(nn.Module):
    def __init__(self, perceptual_weight: float=1.0):
        super(ReconLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        
        self.vgg_feature1 = nn.Sequential(*vgg[:4])
        self.vgg_feature2 = nn.Sequential(*vgg[4:9])
        self.vgg_feature3 = nn.Sequential(*vgg[9:16])
        self.vgg_feature4 = nn.Sequential(*vgg[16:23])
        self.vgg_feature5 = nn.Sequential(*vgg[23:30])
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std
    
    def forward(self, x, recon_x):
        recon_norm = self._normalize(recon_x)
        x_norm = self._normalize(x)
        
        perceptual_loss = 0.0
        
        h_recon = self.vgg_feature1(recon_norm)
        h_x = self.vgg_feature1(x_norm)
        perceptual_loss += self.mse(h_recon, h_x)
        
        h_recon = self.vgg_feature2(h_recon)
        h_x = self.vgg_feature2(h_x)
        perceptual_loss += self.mse(h_recon, h_x)
        
        h_recon = self.vgg_feature3(h_recon)
        h_x = self.vgg_feature3(h_x)
        perceptual_loss += self.mse(h_recon, h_x)
        
        h_recon = self.vgg_feature4(h_recon)
        h_x = self.vgg_feature4(h_x)
        perceptual_loss += self.mse(h_recon, h_x)
        
        h_recon = self.vgg_feature5(h_recon)
        h_x = self.vgg_feature5(h_x)
        perceptual_loss += self.mse(h_recon, h_x)
        
        pixel_loss = self.l1(x, recon_x)
        
        return pixel_loss + self.perceptual_weight * perceptual_loss

class RegularizationLoss(nn.Module):
    def __init__(self, regularizer: str='KL'):
        super(RegularizationLoss, self).__init__()
        self.regularizer = regularizer
        
    def forward(self, mu, logvar):
        
        if self.regularizer == 'KL':
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        else:
            return None

class GanLoss(nn.Module):
    def __init__(self):
        super(GanLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, disc_x=None, disc_recon_x=None, for_AE: bool = True):
        if for_AE:
            assert disc_recon_x is not None
            return self.bce(disc_recon_x, torch.ones_like(disc_recon_x))
        else:
            loss_real = self.bce(disc_x, torch.ones_like(disc_x))
            loss_fake = self.bce(disc_recon_x, torch.zeros_like(disc_recon_x))
            
            return 0.5 * (loss_real + loss_fake)
        
class AutoEncoderLoss(nn.Module):
    def __init__(self, ):
        super(AutoEncoderLoss, self).__init__()
        
        self.recon = ReconLoss(perceptual_weight=1.0)
        self.reg = RegularizationLoss(regularizer='KL')
        self.gan = GanLoss()
        
    def forward(self, x, recon_x, mu=None, logvar=None, disc=None, for_AE: bool = True):
        gan_loss = self.gan(disc(x), disc(recon_x), for_AE)
        if for_AE == False:
            return gan_loss
        
        recon_loss = self.recon(x, recon_x)
        reg_loss = self.reg(mu, logvar)
        
        return recon_loss, gan_loss, reg_loss
    
class LDM_With_FM_Loss(nn.Module):
    def __init__(self,):
        super(LDM_With_FM_Loss, self).__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, model_output, x0, x1, sigma_min):
        target = x1 - (1 - sigma_min) * x0
        
        loss = self.criterion(model_output, target)
        
        return loss
