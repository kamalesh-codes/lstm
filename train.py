import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from utils import PoetryDataset
from model import PoetryLSTM
import os

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

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
    
    local_rank = setup()
    device = torch.device(f'cuda:{local_rank}')

    # Data
    dataset = PoetryDataset(FILE_PATH, SEQ_LENGTH)
    vocab_size = dataset.vocab_size
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)

    # Model
    model = PoetryLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Metric tracking
    train_losses = []

    model.train()
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        total_loss = 0
        
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x)
            
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            
            if local_rank == 0:
                train_losses.append(loss_val)
            
            if local_rank == 0 and (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(loader)}], Loss: {loss_val:.4f}")
        
        avg_loss = total_loss / len(loader)
        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

    # Save model and plot on master process
    if local_rank == 0:
        # Save Weights
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'chars': dataset.chars,
            'char2int': dataset.char2int,
            'int2char': dataset.int2char,
            'vocab_size': vocab_size,
            'embed_size': EMBED_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS
        }, 'poetry_lstm.pth')
        print("Model saved to poetry_lstm.pth")

        # Save Metrics Plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss over Time')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        print("Loss plot saved as training_loss.png")

    cleanup()

if __name__ == "__main__":
    train()
