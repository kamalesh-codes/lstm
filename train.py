import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from utils import PoetryDataset
from model import PoetryLSTM
from tqdm import tqdm
import os
import math

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def validate(model, loader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            
            # Accuracy
            preds = torch.argmax(output, dim=-1)
            correct = (preds == y).sum().item()
            total_acc += correct
            total_samples += y.numel()
            
    avg_loss = total_loss / len(loader)
    accuracy = total_acc / total_samples
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity

def train():
    # Hyperparameters
    FILE_PATH = 'data/input.txt'
    SEQ_LENGTH = 100
    BATCH_SIZE = 64 
    EMBED_SIZE = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    LEARNING_RATE = 0.002
    EPOCHS = 10
    VAL_SPLIT = 0.1 # 10% for validation
    
    local_rank = setup()
    device = torch.device(f'cuda:{local_rank}')

    # Data Loading and Splitting
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    split_idx = int(len(full_text) * (1 - VAL_SPLIT))
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]
    
    # Use train_text to build vocabulary to avoid leakage
    train_dataset = PoetryDataset(train_text, SEQ_LENGTH)
    vocab_size = train_dataset.vocab_size
    # Ensure val_dataset uses the same mapping as train_dataset
    val_dataset = PoetryDataset(val_text, SEQ_LENGTH)
    val_dataset.char2int = train_dataset.char2int
    val_dataset.int2char = train_dataset.int2char
    val_dataset.data = torch.tensor([train_dataset.char2int.get(ch, 0) for ch in val_text], dtype=torch.long)
    val_dataset.vocab_size = vocab_size

    # Loaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Model
    model = PoetryLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Metric tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_ppl': []
    }

    model.train()
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        
        # Training Progress Bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=(local_rank != 0))
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x)
            
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            if local_rank == 0:
                pbar.set_postfix({'loss': f"{loss_val:.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Evaluation
        val_loss, val_acc, val_ppl = validate(model, val_loader, criterion, device, vocab_size)
        
        if local_rank == 0:
            print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val PPL: {val_ppl:.2f}")
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_ppl'].append(val_ppl)
        
        model.train()

    # Save model and plots on master process
    if local_rank == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'chars': train_dataset.chars,
            'char2int': train_dataset.char2int,
            'int2char': train_dataset.int2char,
            'vocab_size': vocab_size,
            'embed_size': EMBED_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS
        }, 'poetry_lstm.pth')
        
        # Plotting
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.plot(history['train_loss'], label='Train Loss', color='tab:blue', marker='o')
        ax1.plot(history['val_loss'], label='Val Loss', color='tab:cyan', marker='x')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:red')
        ax2.plot(history['val_acc'], label='Val Acc', color='tab:red', marker='s')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')
        
        plt.title('Training and Validation Metrics')
        plt.savefig('training_metrics.png')
        print("Metrics plot saved as training_metrics.png")

    cleanup()

if __name__ == "__main__":
    train()
