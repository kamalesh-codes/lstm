import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from utils import PoetryDataset
from model import PoetryLSTM
import os

def setup():
    # Initialize the process group for DDP
    # nccl is the recommended backend for NVIDIA GPUs
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
    BATCH_SIZE = 64 # This is batch size PER GPU
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
    
    # DistributedSampler ensures each GPU sees a unique slice of the data
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)

    # Model
    model = PoetryLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        # Required for DistributedSampler to shuffle differently every epoch
        sampler.set_epoch(epoch)
        total_loss = 0
        
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x)
            
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Only print from the master process to avoid log spam
            if local_rank == 0 and (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

    # Save model only on the master process (rank 0)
    if local_rank == 0:
        # Use model.module to save the original PoetryLSTM without the DDP wrapper
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

    cleanup()

if __name__ == "__main__":
    train()
