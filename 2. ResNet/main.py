import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.models import Net
from models.config import ResNetConfig
from models.trainer import ResnetTrainer
from utils.utils import initialize_weights


DATASET = "cifar10"
MODEL_NAME = "resnet"
# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL = 64


NUM_LAYERS = 3

EPOCHS = 60

LR = 0.1

MONENTUM = 0.9
WEIGHT_DECAY = 0.0001
GAMMA = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# ==================================


def setup():
    config = ResNetConfig(
        batch_size_train=BATCH_SIZE_TRAIN, batch_size_val=BATCH_SIZE_VAL,
        epochs=EPOCHS, 
    )
    
    if DATASET == "cifar10":
        """
        img : (3, 32, 32)
        """
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, transform=train_transform, download=False
        )
        val_dataset = datasets.CIFAR10(
            root='./data', train=False, transform=val_transform, download=False
        )
        
    else:
        raise NotImplementedError()
        
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL)
    
    
    return config, train_loader, val_loader
    
def train(config, train_loader, val_loader):
    
    for n in [3, 9]:
        for model_name in ['resnet', 'plainnet']:
            wandb.init(
                project="ResNet", 
                name=f"train-{model_name}-n{n}",
                config=config.__dict__ if hasattr(config, '__dict__') else config,
                reinit=True
                )
            
            model = Net(
                layer_num=n, filters=(3, 16, 32, 64),
                class_n = len(CLASSES), model_name=model_name
                ).to(config.device)
            
            initialize_weights(model, name="kaiming_normal")
            
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=LR,
                momentum=MONENTUM,
                weight_decay=WEIGHT_DECAY
            )
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[30, 45],
                gamma=GAMMA
            )
            
            trainer = ResnetTrainer(
                config=config, model=model, model_name=model_name, num_layers=n,
                train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, scheduler=scheduler, optimizer=optimizer,
                log_period=100, wandb_log_period=10
            )
            
            trainer.train()
    
            wandb.finish()
    

if __name__ == "__main__":
    config, train_loader, val_loader = setup()
    
    train(config, train_loader, val_loader)
    

    