import torch
from model.UNetConfig import UNetConfig
from data.UNetDatset import UNetDataset
from model.UNetScratch import UNetScratch
from model.trainer import UnetTrainer
from model.UNetLoss import UNetLoss
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 1
BATCH_SIZE_VAL = 1

EPOCHS = 300

IMAGE_SIZE = 572

LR = 0.001
MOMENTUM = 0.99
DROPOUT = 0.1
WEIGHT_DECAY = 0.0005

EPS = 1e-6
LR_MIN = 1e-6

# tarnsfrom parameters
TRANSFROM_ALPHA = 0.5
TRANSFROM_SIGMA = 5.0

# Weight map parameters
WEIGHT_W0 = 10.0
WEIGHT_SIGMA = 5.0

DATA_ROOT_PATH = 'UNet/data/ISBI_2012'
# ==================================


def setup():
    config = UNetConfig(
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        epochs=EPOCHS,
        image_size=IMAGE_SIZE,
        momentum=MOMENTUM,
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
        eps=EPS,
        lr=LR,
        lr_min=LR_MIN,
        transfrom_alpha=TRANSFROM_ALPHA,
        transfrom_sigma=TRANSFROM_SIGMA,
        weight_w0=WEIGHT_W0,
        weight_sigma=WEIGHT_SIGMA,
        
        
    )
    
    train_transform = v2.Compose([
        v2.Resize((config.image_size, config.image_size)),
        v2.RandomRotation(degrees=(0, 360), interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),   # ..
        v2.ElasticTransform(alpha=config.transfrom_alpha, sigma=config.transfrom_sigma),# ..
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),        # added extra data augmentation
        v2.ColorJitter(brightness=0.2, contrast=0.2),                                   # ..
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        ])
    
    val_transform = v2.Compose([
        v2.Resize((config.image_size, config.image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    train_dataset = UNetDataset(
        root_dir=DATA_ROOT_PATH,
        mode="train",
        transform=train_transform,
        w0=config.weight_w0,
        sigma=config.weight_sigma
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=2, 
        pin_memory=True 
    )
    
    val_dataset = UNetDataset(
        root_dir=DATA_ROOT_PATH,
        mode="test",
        transform=val_transform,
        w0=config.weight_w0,
        sigma=config.weight_sigma
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_val,
        shuffle=True,
        num_workers=2, 
        pin_memory=True 
    )
    
    model = UNetScratch(config).to(config.device)
    
    return config, model, train_loader, val_loader

def train(config, model, train_loader, val_loader):
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        momentum=config.momentum,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    criterion = UNetLoss(eps=config.eps)
    
    total_steps = config.epochs * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=total_steps,
        eta_min=config.lr_min
    )
    
    trainer = UnetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )
    
    print(f"start training...")
    trainer.train()
    
    save_plot_history(trainer.history, config)
    
def save_plot_history(history, config):
    with open("training_history.json", "w") as f:
        model_info = {
            "config": vars(config),
            "history": history
        }
        json.dump(model_info, f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title(f'Train & Val Loss (BS: {config.batch_size_train})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['lr'], 'g-s', label='Learning Rate')
    plt.title(f'Learning Rate (BS: {config.batch_size_train})')
    plt.xlabel('Epochs')
    plt.ylabel('Lr')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(f"loss_augm_bs{config.batch_size_train}.png")
    plt.show()

def test(config, model, val_loader):
    model.load_state_dict(torch.load(f"UNet/outputs/UNet_model_data_augm_best.pth", weights_only=True))
    
    trainer = UnetTrainer(
        model=model,
        config=config,
        val_loader=val_loader
    )
    
    trainer.test(num_images=4)

def check_Dice_loss(config, model, val_loader):
    model.load_state_dict(torch.load(f"UNet/outputs/UNet_model_data_augm_best.pth", weights_only=True))
    
    criterion = UNetLoss(eps=config.eps)
    
    trainer = UnetTrainer(
        model=model,
        config=config,
        val_loader=val_loader,
        criterion=criterion
    )
    
    _, dice_loss = trainer.validate()
    print(f"Dice loss : {dice_loss:.4f}")


if __name__ == "__main__":
    """
    images, labels, weights = next(iter(train_loader))
    print(f"Image shape: {images.shape}")  # (B, 1, H, W)
    print(f"Label shape: {labels.shape}")  # (B, H, W)
    print(f"Weight shape: {weights.shape}") # (B, H, W)
    """
    config, model, train_loader, val_loader = setup()
    #train(config, model, train_loader, val_loader)
    #test(config, model, val_loader)
    check_Dice_loss(config, model, val_loader)
    