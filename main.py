import wandb
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import ResNet50_Weights
from transformers import get_cosine_schedule_with_warmup
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader, Subset
from CLIPConfig import CLIPConfig
from CLIPDataset import CLIPDataset
from CLIPScratch import CLIPScratch, BasicBlock, BottleNeckBlock
from CLIPLoss import CLIPLoss
from trainer import CLIPTrainer
from typing import Literal
import random

# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE = 96
BATCH_SIZE_TEST = 64
EPOCHS = 30

PATIENCE = 5

# img encoder 
LAYERS = (3, 4, 6, 3)
IMG_ENC_OUT_DIM = 2048
BLOCK = BottleNeckBlock

# text encoder
MAX_LEN = 77

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6

DROPOUT = 0.1

TEXT_ENC_OUT_DIM = 512

# CLIP 
D_E: int = 512

# optimizer 
LR = 0.0002
BETA1 = 0.9
BETA2 = 0.98
WEIGHT_DECAY = 0.1

# scheduler 
WARMUP_STEPS_RATE = 0.1
# criterion 



CSV_FILE_PATH = 'data/flickr30k_images/results.csv'
IMG_DIR = 'data/flickr30k_images/flickr30k_images'
TRAIN_RATE = 0.9

# ==================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def setup():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std) 
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std) 
    ])
    
    imageNet_val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    
    config = CLIPConfig(
        batch_size=BATCH_SIZE, batch_size_test=BATCH_SIZE_TEST, epochs=EPOCHS, patience=PATIENCE,
        
        layers=LAYERS, img_enc_out_dim=IMG_ENC_OUT_DIM,
        block=BLOCK,
        
        vocab_size=tokenizer.vocab_size,
        max_len=MAX_LEN, 
        
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, text_enc_out_dim=TEXT_ENC_OUT_DIM,
        d_e=D_E,
        
        pad_idx=tokenizer.pad_token_id,
        
        lr=LR, beta1=BETA1, beta2=BETA2, weight_decay=WEIGHT_DECAY,
    )
    
    
    full_dataset_train = CLIPDataset(
        csv_file=CSV_FILE_PATH, img_dir=IMG_DIR, config=config, transform=train_transform, tokenizer=tokenizer, mode='train'
        )
    full_dataset_val = CLIPDataset(
        csv_file=CSV_FILE_PATH, img_dir=IMG_DIR, config=config, transform=val_transform, tokenizer=tokenizer, mode='test'
        )
    
    total_size = len(full_dataset_train)
    indices = list(range(total_size))
    
    train_size = int(total_size * TRAIN_RATE)
    
    random.seed(77) 
    random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = Subset(dataset=full_dataset_train, indices=train_idx)
    val_dataset = Subset(dataset=full_dataset_val, indices=val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)
    
    imageNet_dataset = datasets.ImageFolder(root='data/imagenet-val', transform=imageNet_val_transform)
    imageNet_val_loader = DataLoader(imageNet_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=4)
    
    
    model = CLIPScratch(config).to(config.device)
    model.load_pretrained_all()
    
    return config, model, train_loader, val_loader, tokenizer, imageNet_val_loader

def load_state_dict(model, optimizer, scheduler):
    checkpoint = torch.load("CLIP__epoch_15.pth", map_location=config.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    

def train(config, model, train_loader, val_loader, tokenizer):
    wandb.init(
        project="CLIP-Scratch", 
        name=f"train-with-batch_size-{config.batch_size}",
        config=config.__dict__ if hasattr(config, '__dict__') else config,
        group="Architectures"
        )
    
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("avg_epoch_loss", summary="min")
    
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
        )
    """
    
    params = [
        {'params': model.img_encoder.parameters(), 'lr': config.lr * 0.1},
        {'params': model.text_encoder.parameters(), 'lr': config.lr * 0.1},
        
        {'params': model.Wi.parameters()}, 
        {'params': model.Wt.parameters()},
        {'params': [model.t]}
    ]
    optimizer = torch.optim.AdamW(
        params, 
        lr=config.lr, 
        betas=(config.beta1, config.beta2), 
        weight_decay=config.weight_decay
    )
    
    total_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(total_training_steps * WARMUP_STEPS_RATE)

    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps
    )
    
    criterion = CLIPLoss()
    
    trainer = CLIPTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    wandb.finish()


def test(
    config, model, tokenizer, test_loader,
    dataset: Literal['imagenet', 'flickr30k_images'] = 'imagenet',
    test_mode: Literal['zero-shot', 'retrieval'] = 'zero-shot'
    ):
    
    allowed_datasets = ['imagenet', 'flickr30k_images']
    allowed_modes = ['zero-shot', 'retrieval']
    
    assert dataset in allowed_datasets 
    assert test_mode in allowed_modes
    
    checkpoint = torch.load("CLIP_epoch_best_val.pth", map_location=config.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    wandb.init(
        project="CLIP-Scratch", 
        name=f"test-{test_mode}-{dataset}",
        config=config.__dict__ if hasattr(config, '__dict__') else config
        )
    
    trainer = CLIPTrainer(
        config, model, tokenizer=tokenizer
    )
    
    if dataset == 'flickr30k_images':
        if test_mode == 'zero-shot':
            class_names = [
                'person', 'dog', 'cat', 'bicycle', 'car', 
                'tree', 'water', 'building', 'shirt', 'guitar'
            ]
            
            trainer.zero_shot_test(class_names, test_loader)
        elif test_mode == 'retrieval':
            trainer.retrieval_test(test_loader)
    elif dataset == 'imagenet':
        if test_mode == 'zero-shot':
            weights = ResNet50_Weights.DEFAULT
            categories = weights.meta["categories"]
            wnids = sorted(test_loader.dataset.classes)
            mapping = {wnid: name for wnid, name in zip(wnids, categories)}
            ordered_class_names = [mapping[wnid] for wnid in test_loader.dataset.classes]
            
            trainer.imageNet_zero_shot_test(test_loader, ordered_class_names)
    
    
    
    wandb.finish()
    
if __name__ == "__main__":
    config, model, train_loader, val_loader, tokenizer, imageNet_val_loader = setup()
    
    #train(config, model, train_loader, val_loader, tokenizer)
    test(
        config,
        model,
        tokenizer,
        val_loader,
        dataset='flickr30k_images',
        test_mode='retrieval'
        )
