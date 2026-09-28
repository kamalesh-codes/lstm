import torch
from torch.utils.data import Dataset, DataLoader

class PoetryDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        
        # Unique characters in the dataset
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        
        # Mappings
        self.char2int = {ch: i for i, ch in enumerate(self.chars)}
        self.int2char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Convert text to integers
        self.data = torch.tensor([self.char2int[ch] for ch in self.text], dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence of length seq_length
        # Target sequence is the same but shifted by one character
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y

def get_dataloader(file_path, seq_length, batch_size):
    dataset = PoetryDataset(file_path, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, dataset
