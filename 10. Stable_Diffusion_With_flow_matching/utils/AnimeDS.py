import os
from PIL import Image
from torch.utils.data import Dataset

class AnimeFaceDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transfrom = transform
        self.img_files = [
            f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_files[idx])
        img = Image.open(img_name).convert('RGB')
        
        if self.transfrom:
            img = self.transfrom(img)
            
        return img, 0
        