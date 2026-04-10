import os
import glob
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from torch.utils.data import Dataset
from torchvision import tv_tensors

class UNetDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None, w0=10, sigma=5):
        self.img_files = sorted(glob.glob(os.path.join(root_dir, mode, "imgs", "*.png")))
        self.label_files = sorted(glob.glob(os.path.join(root_dir, mode, "labels", "*.png")))
        
        self.transform = transform
        self.w0 = w0
        self.sigma = sigma
        
    def generate_weight_map(self, mask):
        w_c = np.zeros_like(mask, dtype=np.float32)
        w_c[mask == 0] = 1.0
        w_c[mask == 1] = 2.0
        
        labeled_mask, n_labels = label(mask)
        
        if n_labels < 2:
            return w_c.astype(np.float32)
        
        all_distances = []
        for i in range(1, n_labels + 1):
            dist = distance_transform_edt(labeled_mask != i)
            all_distances.append(dist)
            
        all_distances = np.stack(all_distances, axis=0)
        all_distances = np.sort(all_distances, axis=0)
        
        d1 = all_distances[0]
        d2 = all_distances[1]
        
        dist_term = -((d1 + d2) ** 2) / (2 * (self.sigma ** 2))
        dist_term = np.clip(dist_term, -50, 0)
        
        w = w_c + self.w0 * np.exp(dist_term)
        
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
        return w.astype(np.float32)
        
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, i):
        image = Image.open(self.img_files[i]).convert("L")
        label = Image.open(self.label_files[i]).convert("L")
        
        mask_np = (np.array(label) > 128).astype(np.uint8)
        weight_np = self.generate_weight_map(mask_np)
        
        weight = Image.fromarray(weight_np, mode='F')
        
        image = tv_tensors.Image(image)
        label = tv_tensors.Mask(mask_np) 
        weight = tv_tensors.Mask(weight_np[None, ...])
        
        if self.transform:
            image, label, weight = self.transform(image, label, weight)
        
        label = (label > 0.5).long().squeeze(0)
        weight = weight.float().squeeze(0)
        
        return image, label, weight