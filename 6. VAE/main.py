import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from models.VAELoss import VAELoss
from models.VAEConfig import VAEConfig
from models.VAE_Scratch import VAEScratch, weights_init
from models.trainer import VAETrainer
import matplotlib.pyplot as plt
import json


# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE = 128

EPOCHS = 50

LR = 0.0002
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
WEIGHT_DECAY = 0.01

LOSS_BETA = 0.0

Z_DIM = 100
# ==================================

def setup():
    config = VAEConfig(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR, adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        z_dim=Z_DIM, loss_beta=LOSS_BETA,
    )
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(root='./data/celeba', transform=transform)
    
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    fixed_noise = torch.randn(64, config.z_dim, 1, 1, device=config.device)
    
    model = VAEScratch(z_dim=config.z_dim)
    
    model.apply(weights_init)
    
    model.to(config.device)
    
    return config, train_loader, fixed_noise, model

def train(config, train_loader, fixed_noise, model):
    
    criterion = VAELoss(beta=config.loss_beta)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    trainer = VAETrainer(
        model=model,
        config=config,
        train_laoder=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        fixed_noise=fixed_noise
    )
    
    trainer.train()
    
    save_plot_history(trainer.history, config)
    
def save_plot_history(history, config):
    with open("training_history.json", "w") as f:
        model_info = {
            "config": vars(config),
            "history": history
        }
        json.dump(model_info, f)
        
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-o', label='Loss')
    plt.title(f'Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['lr'], 'g-s', label='Learning Rate')
    plt.title(f'Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Lr')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(f"loss.png")
    plt.show()


if __name__ == "__main__":
    config, train_loader, fixed_noise, model = setup()
    train(config, train_loader, fixed_noise, model)
    #test(config, models)