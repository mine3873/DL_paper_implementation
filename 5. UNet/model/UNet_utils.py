import os
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

class ISBIDataset(Dataset):
    def __init__(self, data_root_dir, train: str = "train", transform=None):
        self.data_root_dir = data_root_dir
        self.transfrom = transform
        
        assert train == "train" or train=="test"
        self.image_dir = f"{data_root_dir}/{train}/imgs"
        self.mask_dir = f"{data_root_dir}/{train}/labels"
        
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        
        self.weight_maps = []
        for m_name in self.masks:
            m_path = os.path.join(self.mask_dir, m_name)
            m_array = np.array(Image.open(m_path).convert("L"))
            
            labeled_m = label(m_array)
            w_map = create_weight_map(labeled_m)
            self.weight_maps.append(w_map)
            
        
    def __len__(self):
        assert self.images != None
        return len(self.images)
    
    def __getitem__(self, i):
        img_path = os.path.join(self.image_dir, self.images[i])
        mask_path = os.path.join(self.mask_dir, self.masks[i])
        
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        w_map = self.weight_maps[i]
        
        if self.transfrom is not None:
            augmented = self.transfrom(image=image, mask=mask, weight_map=w_map)
            image = augmented['image']
            mask = augmented['mask']
            w_map= augmented['weight_map']
            
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        w_map = torch.from_numpy(w_map).float()
        
        return image, mask, w_map
        
def create_weight_map(mask, w0=10, sigma=5):
    wc = np.zeros_like(mask, dtype=np.float32)
    
    cells = np.unique(mask) 
    cells = cells[cells > 0] # (N, )
    
    distance = []
    for cell_id in cells:
        dist = distance_transform_edt(mask != cell_id) # (H, W)
        distance.append(dist)
        
    distance = np.sort(np.stack(distance, axis=0), axis=0) # (N, H, W)
    
    d1 = distance[0]
    d2 = distance[1]
    
    return wc + w0 * np.exp(-((d1 + d2)**2) / (2 * (sigma**2)))

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        
    def forward(self, model_output, target, weight_map):
        cur_size = model_output.size(-1)
        
        target = F.center_crop(target, [cur_size, cur_size])
        weight_map = F.center_crop(weight_map, [cur_size, cur_size])
        
        target = target.squeeze(1).long()
        
        loss = self.criterion(model_output, target)
        
        loss = loss * (weight_map + 1.0)
        
        return loss.mean()
    
    
        