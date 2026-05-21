from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import random
import os

class CLIPDataset(Dataset):
    def __init__(self, csv_file, img_dir, config, transform=None, tokenizer=None, mode='train'):
        df = pd.read_csv(csv_file, sep='|')
        df.columns = [col.strip() for col in df.columns]
        df['comment'] = df['comment'].fillna("unknown")
        #self.df.iloc[:, 2] = self.df.iloc[:, 2].fillna("unknown")
        
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        
        self.img_names = df['image_name'].unique().tolist()
        self.captions_dict = df.groupby('image_name')['comment'].apply(list).to_dict()
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        
        captions = self.captions_dict[img_name]
        if self.mode == 'train':
            caption = str(random.choice(captions)).strip()
        else:
            caption = str(captions[0]).strip()
        
        tokenized = self.tokenizer(
            caption, padding='max_length',
            max_length=self.config.max_len, truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids'].squeeze(0)
        
        eos_idx = (input_ids != self.tokenizer.pad_token_id).sum().item() - 1
        if eos_idx < 0: eos_idx = 0
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, input_ids, eos_idx, caption