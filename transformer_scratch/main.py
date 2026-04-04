from model.config import TransformerConfig
from model.trainer import TransformerTrainer
from model.transformer_scratch import Transformer_scatch
from model.utils import get_lr_lambda, TransformerCollate, load_data, load_tokenizer
from torch.utils.data import DataLoader
from data.dataset import TranslationDataSet
import matplotlib.pyplot as plt
import json
import torch

# ==================================
# PARAMETERS & PATHS
# ==================================
data_file_path = f'data/AIhub'
tokenizer_path = f'tokenizer/AIhub/tokenizer.model'
BATCH_SIZE = 32
BATCH_SIZE_VAL = 16
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1
EPOCHS = 10
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-6
# ==================================

def setUp():
    tokenizer = load_tokenizer(tokenizer_path)
    pad_idx = tokenizer.pad_id()
    
    config = TransformerConfig(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=pad_idx,
        batch_size=BATCH_SIZE,
        batch_size_val=BATCH_SIZE_VAL,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPS,
        label_smoothing=LABEL_SMOOTHING
        )
    
    model = Transformer_scatch(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layer=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout
    ).to(device=config.device)
    
    return model, tokenizer, config

def train(model, tokenizer, config):
    train_data = TranslationDataSet(
        data=load_data(data_file_path, 'train'),
        tokenizer=tokenizer
    )
    val_data = TranslationDataSet(
        data=load_data(data_file_path, 'val'),
        tokenizer=tokenizer
    )
    
    collate_fn = TransformerCollate(pad_idx=config.pad_idx)
    
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size_val,
        collate_fn=collate_fn
        )

    # AdamW
    # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
    # s_t = beta2 * s_{t-1} + (1-beta2) * (g_T ** 2)
    # m_t = m_t / (1 - (beta1 ** t))
    # s_t = s_t / (1 - (beta2 ** t))
    # theta_{t+1} = theta_t - lr * (m_t / (math.sqrt(s_t) + epsilon) + lambda * theta_t)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr = 1.0,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=1e-2 #lambda
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda= get_lr_lambda(config.d_model, config.warmup_steps, factor=0.3)
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, ignore_index=config.pad_idx)

    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config
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
    plt.title(f'Train & Val Loss (BS: {config.batch_size}, WS: {config.warmup_steps})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['lr'], 'g-s', label='Learning Rate')
    plt.title(f'Learning Rate (BS: {config.batch_size}, WS: {config.warmup_steps})')
    plt.xlabel('Epochs')
    plt.ylabel('Lr')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(f"loss_bs{config.batch_size}_ep{config.epochs}.png")
    plt.show()



def test(model, tokenizer, config):
    model.load_state_dict(torch.load("outputs/transformer_model_best.pth", weights_only=True))
    
    trainer = TransformerTrainer(model=model, config=config)
    
    while True:
        input_text = input("input Korean sentence (quit : q): ")
        
        if input_text.lower() == 'q':
            break
        
        result = trainer.translate(sentence=input_text, tokenizer=tokenizer)
        
        print(f"original : {input_text}")
        print(f"result   : {result}\n")

"""
def bleu_test(model, tokenizer, config):
    val_data = TranslationDataSet(
        data=load_data(data_file_path, 'val'),
        tokenizer=tokenizer
    )
    collate_fn = TransformerCollate(pad_idx=config.pad_idx)
    
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size_val,
        collate_fn=collate_fn
        )
    
    trainer = TransformerTrainer(model=model, config=config)
    
    bleu_score = trainer.evaluate_bleu(val_loader, tokenizer)
"""
        
        
if __name__ == "__main__":
    model, tokenizer, config = setUp()
    
    #train(model, tokenizer, config)
    test(model, tokenizer, config)
    #bleu_test(model, tokenizer, config)