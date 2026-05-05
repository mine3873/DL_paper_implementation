from data.utils_dataset import YOLODataset
from model.YOLOConfig import YOLOConfig
from model.YOLOScratch import YOLOScratch
from model.loss import YOLOLoss
from model.trainer import YOLOTrainer
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

# ==================================
# PARAMETERS & PATHS
# ==================================
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 16

WARMUP_EPOCHS = 1
EPOCHS = 135

LAMBDA_COORD = 5
LAMBDA_NOOBJ = .5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DROPOUT = .5
LEAKYRELU_W = 0.1
EPS = 1e-6

DATA_ROOT_PATH = 'VOCdevkit'
YEARS = ('2007','2012')
CLASSES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    )

IMGAE_SIZE = 448
S = 7
B = 2
C = 20

LR = 1e-2

PATH_DARKNET_WEIGHTS = 'extraction.weights'

# ==================================

def setup():
    config = YOLOConfig(
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        warmup_epochs=WARMUP_EPOCHS,
        epochs=EPOCHS,
        lr=LR,
        lambda_coord=LAMBDA_COORD,
        lambda_noobj=LAMBDA_NOOBJ,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        leakyRelu_w=LEAKYRELU_W,
        eps=EPS,
        data_root_path=DATA_ROOT_PATH,
        years=YEARS,
        classes=CLASSES,
        image_size=IMGAE_SIZE,
        S=S, B=B, C=C, 
    )
    
    train_dataset = YOLODataset(config, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
    )
    
    val_dataset = YOLODataset(config, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_val,
        shuffle=True
    )
    
    model = YOLOScratch(
        config=config,
    )
    model.to(config.device)
    
    """
    data : imgs, labels
    
    imgs : tensor(batch_size, Channel, Height, Width)
    labels : tensor(batch_size, S, S, B * 5 + Class)
    """
    
    return config, model, train_loader, val_loader
    
def train(config, model, train_loader, val_loader):
    
    model.load_darknet_weights(PATH_DARKNET_WEIGHTS)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    criterion = YOLOLoss(
        config=config
    )
    
    step_per_epochs = len(train_loader)
    
    warmup_steps = step_per_epochs * config.warmup_epochs
    
    scheduler_warmpup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    scheduler_main = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[75 * step_per_epochs, 105 * step_per_epochs], 
        gamma=0.1
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmpup, scheduler_main],
        milestones=[warmup_steps]
    )
    
    trainer = YOLOTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion
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
    plt.ylim(0, 200)
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
    
    plt.savefig(f"loss_bs{config.batch_size_train}.png")
    plt.show()
    

def test(model, config, test_loader):
    criterion = YOLOLoss(
        config=config
    )
    
    model.load_state_dict(torch.load("YOLO_model_best.pth", weights_only=True))
    trainer = YOLOTrainer(
        model=model, config=config, test_loader=test_loader, criterion=criterion
        )
    
    trainer.test(num_images=10, thresh=0.04, iou_thresh=0.3)


    
if __name__ == "__main__":
    config, model, train_loader, val_loader = setup()   
    #train(config, model, train_loader, val_loader)
    #test(model, config, val_loader)