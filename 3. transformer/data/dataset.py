import torch
from torch.utils.data import Dataset

class TranslationDataSet(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.sp = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        ko_text, en_text = self.data[i]
        
        src_tokens = [self.sp.bos_id()] + self.sp.encode_as_ids(ko_text)+ [self.sp.eos_id()]
        tgt_tokens = [self.sp.bos_id()] + self.sp.encode_as_ids(en_text)+ [self.sp.eos_id()]
        
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)
        