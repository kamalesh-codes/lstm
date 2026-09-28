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
import hydra
import urllib.request
from omegaconf import DictConfig, OmegaConf

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def ensure_data(data_path):
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Downloading Tiny Shakespeare dataset...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, data_path)
            print(f"Successfully downloaded dataset to {data_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e

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
            
            preds = torch.argmax(output, dim=-1)
            correct = (preds == y).sum().item()
            total_acc += correct
            total_samples += y.numel()
            
    avg_loss = total_loss / len(loader)
    accuracy = total_acc / total_samples
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth'))

def load_checkpoint(checkpoint_path, model, optimizer, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    
    return epoch, history

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    
    data_path = os.path.join(orig_cwd, cfg.paths.data_path)
    plot_dir = os.path.join(orig_cwd, cfg.paths.plots_dir)
    os.makedirs(plot_dir, exist_ok=True)
    plot_save_path = os.path.join(plot_dir, cfg.paths.plot_save_path)
    checkpoint_dir = os.path.join(orig_cwd, cfg.paths.checkpoint_dir)

    # 1. Ensure data is available (only Rank 0 downloads)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        ensure_data(data_path)

    # 2. Setup distributed environment
    local_rank = setup()
    # Crucial: Block all processes until the data download is complete
    dist.barrier() 
    device = torch.device(f'cuda:{local_rank}')

    # 3. Load data and compute vocab (Must be identical across ranks)
    with open(data_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    split_idx = int(len(full_text) * (1 - cfg.training.val_split))
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]
    
    train_dataset = PoetryDataset(train_text, cfg.model.seq_length)
    vocab_size = train_dataset.vocab_size
    
    if local_rank == 0:
        print(f"Vocab size detected: {vocab_size}")

    val_dataset = PoetryDataset(val_text, cfg.model.seq_length)
    val_dataset.char2int = train_dataset.char2int
    val_dataset.int2char = train_dataset.int2char
    val_dataset.data = torch.tensor([train_dataset.char2int.get(ch, 0) for ch in val_text], dtype=torch.long)
    val_dataset.vocab_size = vocab_size

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, drop_last=False)

    # 4. Model Initialization
    model = PoetryLSTM(
        vocab_size, 
        cfg.model.embed_size, 
        cfg.model.hidden_size, 
        cfg.model.num_layers
    ).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_ppl': []
    }
    start_epoch = 0
    best_val_loss = float('inf')

    if cfg.training.resume:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            start_epoch, history = load_checkpoint(checkpoint_path, model, optimizer, device)
            if local_rank == 0:
                print(f"Resumed from epoch {start_epoch}")
        else:
            if local_rank == 0:
                print("Resume requested but no checkpoint found. Starting from scratch.")

    model.train()
    for epoch in range(start_epoch, cfg.training.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]", disable=(local_rank != 0))
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
        val_loss, val_acc, val_ppl = validate(model, val_loader, criterion, device, vocab_size)
        
        if local_rank == 0:
            print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val PPL: {val_ppl:.2f}")
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_ppl'].append(val_ppl)
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % cfg.training.checkpoint_interval == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'best_val_loss': best_val_loss,
                }, is_best, checkpoint_dir)

        model.train()

    if local_rank == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'chars': train_dataset.chars,
            'char2int': train_dataset.char2int,
            'int2char': train_dataset.int2char,
            'vocab_size': vocab_size,
            'embed_size': cfg.model.embed_size,
            'hidden_size': cfg.model.hidden_size,
            'num_layers': cfg.model.num_layers
        }, os.path.join(orig_cwd, cfg.paths.model_save_path))
        
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
        plt.savefig(plot_save_path)
        print(f"Metrics plot saved as {plot_save_path}")

    cleanup()

if __name__ == "__main__":
    train()
