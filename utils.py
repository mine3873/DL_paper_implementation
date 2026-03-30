from torch.nn.utils.rnn import pad_sequence
import torch
import os
import sentencepiece as spm

def get_lr_lambda(d_model, warmup_steps):
    """
    - return:
    lrate = (d_model ** -0.5) * min(step_num**-0.5, step_num * (warmup_steps ** -1.5))
    """
    return lambda step: (d_model ** -0.5) * min(
        max(1, step) ** -0.5, 
        max(1, step) * (warmup_steps ** -1.5)
    )
    
    
def get_collate_fn(pad_idx):
    """
    add <pad> to sentences, except the longest sentence.
    
    - return:
    tensor (Batch, seq_len)
    """
    def collate_fn(batch):
        src_batch = []
        tgt_batch = []
        
        for item in batch:
            src_item = item[0] if torch.is_tensor(item[0]) else torch.tensor(item[0])
            tgt_item = item[1] if torch.is_tensor(item[1]) else torch.tensor(item[1])
            
            src_batch.append(src_item)
            tgt_batch.append(tgt_item)

        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

        return src_padded, tgt_padded
        
    return collate_fn

def create_masks(src, tgt, pad_idx, device):
    """
    src_mask: tensor(Batch, 1, 1, seq_len_src)
    tgt_mask: tensor(Batch, 1, seq_len_tgt, seq_len_tgt)
    """
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    
    #lower-trianglar 
    tgt_len = tgt.size(1)
    lt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    
    tgt_mask = tgt_mask & lt_mask
    
    return src_mask, tgt_mask

def load_data(file_path, prefix):
    ko_path = os.path.join(file_path, f"{prefix}.ko")
    en_path = os.path.join(file_path, f"{prefix}.en")
    
    with open(ko_path, 'r', encoding='utf-8') as f:
        ko = f.readlines()

    with open(en_path, 'r', encoding='utf-8') as f:
        en = f.readlines()

    return list(zip(ko, en))

def train_tokenizer(input_files, model_prefix='tokenizer', vocab_size=32000):
    """
    input_files : list(train.ko, train.en)
    """
    train_args = (
        f"--input={','.join(input_files)} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage=1.0 "
        f"--model_type=bpe "
        f"--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 "
        f"--add_dummy_prefix=true "
        f"--normalization_rule_name=identity"
    )
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(train_args)
    print(f"success to train tokenizer.")
    
    
def load_tokenizer(model_path='tokenizer.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
    