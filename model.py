import torch
import torch.nn as nn

class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(PoetryLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        embeds = self.embedding(x) # (batch, seq_len, embed_size)
        
        # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Project to vocab size: (batch, seq_len, vocab_size)
        out = self.fc(lstm_out)
        
        return out, hidden
