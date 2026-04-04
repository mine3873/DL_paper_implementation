from model.config import MLPConfig
from model.Trainer import MLPTrainer
from model.Scheduler import CosineAnnealing
from model.Optimizer import AdamW
from model.criterion import CrossEntropy
from model.MultiPerceptronLayer import MLP
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import json

# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 16
EPOCHS = 10
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-6
WEIGHT_DECAY = 0.1
LR_MIN = 1e-6
# ==================================

def setUp():
    config = MLPConfig(
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        epochs=EPOCHS,
        beta1=BETA1,
        beta2=BETA2,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        lr_min=LR_MIN
    )
    
    model = MLP()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='.', train=True, download=False, transform=transform)
    val_dataset = datasets.MNIST(root='.', train=False, download=False, transform=transform)
    
    val_size = len(val_dataset) // 2
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size_val, shuffle=True)
    
    return config, model, train_loader, val_loader, test_loader

def train(config, model, train_loader, val_loader):
    
    criterion = CrossEntropy()
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=0.001,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    
    total_steps = config.epochs * len(train_loader)
    scheduler = CosineAnnealing(
        optimizer=optimizer,
        T_max=total_steps,
        lr_min=config.lr_min
    )
    
    trainer = MLPTrainer(
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
    
    plt.savefig(f"loss_bs{config.batch_size_train}_ep{config.epochs}.png")
    plt.show()

def test(config, model, test_loader):
    checkpoint = torch.load("mlp_model_best.pth")
    with torch.no_grad():
        model.W1.copy_(checkpoint['W1'])
        model.b1.copy_(checkpoint['b1'])
        model.W2.copy_(checkpoint['W2'])
        model.b2.copy_(checkpoint['b2'])
    trainer = MLPTrainer(
        model=model,
        config=config,
        test_loader=test_loader
    )
    trainer.test(num_images=20)
    
    

if __name__ == "__main__":
    config, model, train_loader, val_loader, test_loader = setUp()
    #train(config, model, train_loader, val_loader)
    #test(config, model, test_loader)
