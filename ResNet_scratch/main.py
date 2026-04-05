import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from model.ResNetConfig import ResNetConfig
from model.ResNet import ResNet
from model.trainer import ResNetTrainer
import matplotlib.pyplot as plt
import numpy as np
import json


# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 16
NUM_WORKERS = 2
NUM_LAYERS = 5
EPOCHS = 46
MONENTUM = 0.9
WEIGHT_DECAY = 0.0001
GAMMA = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# ==================================


def setup():
    config = ResNetConfig(
        mean=MEAN,
        std=STD,
        classes=CLASSES,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        num_workers=NUM_WORKERS,
        num_layers=NUM_LAYERS,
        momentum=MONENTUM,
        weight_decay=WEIGHT_DECAY,
        gamma=GAMMA,
        epochs=EPOCHS,
        
    )
    
    """
    in paper:
     We follow the simple data augmen- tation in [24] for training:
     4 pixels are padded on each side, 
     and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
     For testing, we only evaluate the single view of the original 32×32 image.
    """
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            config.mean,
            config.std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            config.mean,
            config.std)
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='.', train=True, download=True, transform=train_transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='.', train=True, download=True, transform=test_transform
        )
    test_dataset = torchvision.datasets.CIFAR10(
        root='.', train=False, download=True, transform=test_transform
    )
    
    indices = np.arange(50000)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[:45000], indices[45000:]
    
    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_train, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size_val, shuffle=False, num_workers=NUM_WORKERS)
    
    model = ResNet(
        num_layers=config.num_layers,
        config=config
    )
    model = model.to(config.device)
    
    return config, model, train_loader, val_loader, test_loader
    
def train(config, model, train_loader, val_loader):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss()
    
    """
    in paper: 
    We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations, 
    and terminate training at 64k iterations, which is determined on a 45k/5k train/val split.
    ...
    ...
    ->
    num_train_data / batch_size => 45,000 / 32 = 1,406.25 = 1,406
    
    step1: 32,000 / 1,406 ~ 22.75
    step2: 48,000 / 1,406 ~ 34.14
    terminate: 64,000 / 1,406 ~ 45.51
    => 
    milestones = [23, 34]
    Epochs = 46
    """
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[23, 34],
        gamma=config.gamma
    )
    
    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        config=config
    )
    
    print(f"start training...")
    trainer.train()
    
    save_plot_history(trainer.history, config)
    
def save_plot_history(history, config):
    with open(f"training_history_n{config.num_layers}.json", "w") as f:
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
    plt.title(f'Train & Val Loss (BS: {config.batch_size_train} N:{config.num_layers})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['lr'], 'g-s', label='Learning Rate')
    plt.title(f'Learning Rate (BS: {config.batch_size_train} N:{config.num_layers})')
    plt.xlabel('Epochs')
    plt.ylabel('Lr')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(f"loss_bs{config.batch_size_train}_ep{config.epochs}_n{config.num_layers}.png")
    plt.show()

def test(config, model, test_loader):
    n_layers = [3, 5, 7]
    for n in n_layers:
        config.num_layers = n
        
        model = ResNet(
            num_layers=config.num_layers,
            config=config
        )
        model = model.to(config.device)
        
        model.load_state_dict(torch.load(f"ResNet_scratch/outputs/ResNet_model_best_n{n}.pth", weights_only=True))

        trainer = ResNetTrainer(model=model, config=config, test_loader=test_loader)
    
        trainer.test(num_images=20)
    
    
    

if __name__ == "__main__":
    config, model, train_loader, val_loader, test_loader = setup()
    
    """
    images: tensor(batch_size, 3, 32, 32) = tensor(batch_size, C, H, W)
    C : Channels
    H, W: Height, Weight
    """
    #train(config, model, train_loader, val_loader)
    test(config, model, test_loader)
    
    