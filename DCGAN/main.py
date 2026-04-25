import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from DCGAN_scrath import Generator, Discriminator, weights_init
from DCGANConfig import DCGANConfig
from trainer import DCGANTrainer
import matplotlib.pyplot as plt
import json

# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 128

EPOCHS = 20

LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
WEIGHT_DECAY = 0.01

DATA_ROOT_PATH = 'UNet/data/ISBI_2012'
# ==================================


def setup():
    config = DCGANConfig(
        batch_size_train=BATCH_SIZE_TRAIN,
        epochs=EPOCHS,
        lr=LR, beta1=BETA1, weight_decay=WEIGHT_DECAY,
    )
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(root='./data/celeba', transform=transform)
    
    train_loader = DataLoader(dataset, batch_size=config.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
    fixed_noise = torch.randn(64, 100, 1, 1, device=config.device)
    
    
    model_G = Generator()
    model_D = Discriminator()
    
    model_G.apply(weights_init)
    model_D.apply(weights_init)
    
    model_G.to(config.device)
    model_D.to(config.device)
    
    models = (model_G, model_D)
    
    
    
    return config, train_loader, fixed_noise, models
    
def train(config, train_loader, fixed_noise, models):
    model_G, model_D = models
    
    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer_G = torch.optim.AdamW(model_G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    optimizer_D = torch.optim.AdamW(model_D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=config.epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=config.epochs)
    
    optimizers = (optimizer_G, optimizer_D)
    schedulers = (scheduler_G, scheduler_D)
    
    trainer = DCGANTrainer(
        models=models,
        config=config,
        train_laoder=train_loader,
        criterion=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
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
        
    epochs = range(1, len(history['loss_D']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss_D'], 'b-o', label='Loss_D')
    plt.plot(epochs, history['loss_G'], 'r-o', label='Loss_G')
    plt.title(f'D & G Loss')
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
    config, train_loader, fixed_noise, models = setup()
    train(config, train_loader, fixed_noise, models)