import torch
import torch.nn as nn
import random
from model import PoetryLSTM

def sample(probs, temperature=1.0):
    # Scale probabilities by temperature
    probs = torch.pow(probs, 1.0 / temperature)
    probs = probs / torch.sum(probs)
    
    # Sample from the distribution
    char_idx = torch.multinomial(probs, 1).item()
    return char_idx

def generate(seed_text, gen_length=500, temperature=1.0):
    # Load model and vocab
    checkpoint = torch.load('poetry_lstm.pth')
    chars = checkpoint['chars']
    char2int = checkpoint['char2int']
    int2char = checkpoint['int2char']
    vocab_size = checkpoint['vocab_size']
    embed_size = checkpoint['embed_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    
    model = PoetryLSTM(vocab_size, embed_size, hidden_size, num_layers)
    # Handle DataParallel prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Encode seed
    input_seq = torch.tensor([char2int[ch] for ch in seed_text], dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = seed_text
    
    with torch.no_grad():
        # Initial forward pass to get the hidden state for the seed
        output, hidden = model(input_seq)
        
        # The last character's prediction
        last_char_logits = output[0, -1, :]
        probs = torch.softmax(last_char_logits, dim=-1)
        
        for _ in range(gen_length):
            # Sample next character
            next_char_idx = sample(probs, temperature)
            next_char = int2char[next_char_idx]
            generated_text += next_char
            
            # Update input for next step (just the last character)
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
            
            # Forward pass with previous hidden state
            output, hidden = model(input_seq, hidden)
            
            # Get probabilities for the next character
            last_char_logits = output[0, -1, :]
            probs = torch.softmax(last_char_logits, dim=-1)
            
    return generated_text

if __name__ == "__main__":
    import sys
    seed = sys.argv[1] if len(sys.argv) > 1 else "FIRST CITIZEN"
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    print(f"Generating with seed: '{seed}' and temperature: {temp}...")
    print("-" * 30)
    print(generate(seed, temperature=temp))
    print("-" * 30)
