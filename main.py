from config import TransformerConfig
from trainer import TransformerTrainer
from transformer_scratch import Transformer_scatch
from utils import get_lr_lambda, get_collate_fn, load_data, load_tokenizer
from torch.utils.data import DataLoader
from dataset import TranslationDataSet
import torch


# ==================================
# PARAMETERS & PATHS
# ==================================
data_file_path = f'data/AIhub'
tokenizer_path = f'tokenizer/AIhub/tokenizer.model'
BATCH_SIZE = 32
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1
EPOCHS = 10
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
# ==================================

def main():
    tokenizer = load_tokenizer(tokenizer_path)
    pad_idx = tokenizer.pad_id()

    train_data = TranslationDataSet(
        data=load_data(data_file_path, 'train'),
        tokenizer=tokenizer
    )
    val_data = TranslationDataSet(
        data=load_data(data_file_path, 'val'),
        tokenizer=tokenizer
    )

    config = TransformerConfig(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=pad_idx,
        batch_size=BATCH_SIZE,
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

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(pad_idx=config.pad_idx)
        )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        collate_fn=get_collate_fn(pad_idx=config.pad_idx)
        )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = 1.0,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda= get_lr_lambda(config.d_model, config.warmup_steps)
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

if __name__ == "__main__":
    main()