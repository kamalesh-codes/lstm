import torch
import torch.nn as nn
import random
from model import PoetryLSTM
import os

def select_device():
    """
    Determines the best available device for inference.
    Fallback order: Multiple GPUs (selects 0) -> Single GPU -> CPU.
    """
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            print(f"🚀 Multi-GPU environment detected ({count} GPUs). Using cuda:0 for optimal sequential inference.")
        else:
            print(f"✅ Single GPU detected. Using cuda:0.")
        return torch.device('cuda:0')
    
    print("⚠️ No GPUs found. Falling back to CPU.")
    return torch.device('cpu')

def sample(probs, temperature=1.0):
    # Scale probabilities by temperature
    probs = torch.pow(probs, 1.0 / temperature)
    probs = probs / torch.sum(probs)
    
    # Sample from the distribution
    char_idx = torch.multinomial(probs, 1).item()
    return char_idx

def generate(seed_text, gen_length=500, temperature=1.0):
    # 1. Device Selection
    device = select_device()
    
    # 2. Load Checkpoint
    # map_location='cpu' is critical for loading GPU models on CPU machines
    if not os.path.exists('poetry_lstm.pth'):
        raise FileNotFoundError("Model weights not found! Please train the model first.")
        
    checkpoint = torch.load('poetry_lstm.pth', map_location=device)
    
    chars = checkpoint['chars']
    char2int = checkpoint['char2int']
    int2char = checkpoint['int2char']
    vocab_size = checkpoint['vocab_size']
    embed_size = checkpoint['embed_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    
    # 3. Model Initialization
    model = PoetryLSTM(vocab_size, embed_size, hidden_size, num_layers)
    
    # Handle DataParallel 'module.' prefix if it exists
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    # 4. Prepare Seed
    # Convert seed characters to integers; use 0 (or most common) for unknown characters
    input_seq = torch.tensor(
        [char2int.get(ch, 0) for ch in seed_text], 
        dtype=torch.long
    ).unsqueeze(0).to(device)
    
    generated_text = seed_text
    
    # 5. Generation Loop
    with torch.no_grad():
        # Initial pass to prime the LSTM with the seed sequence
        output, hidden = model(input_seq)
        
        # Start predicting from the last character of the seed
        last_char_logits = output[0, -1, :]
        probs = torch.softmax(last_char_logits, dim=-1)
        
        for _ in range(gen_length):
            # Sample next character
            next_char_idx = sample(probs, temperature)
            next_char = int2char[next_char_idx]
            generated_text += next_char
            
            # Only feed the newly generated character back into the model
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
            
            # Forward pass using the state (hidden) from the previous step
            output, hidden = model(input_seq, hidden)
            
            # Get probabilities for the next character
            last_char_logits = output[0, -1, :]
            probs = torch.softmax(last_char_logits, dim=-1)
            
    return generated_text

if __name__ == "__main__":
    import sys
    # CLI Arguments: [seed] [temperature] [length]
    seed = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    length = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    print(f"Generating with seed: '{seed}', temperature: {temp}, length: {length}")
    print("-" * 50)
    try:
        result = generate(seed, gen_length=length, temperature=temp)
        print(result)
    except Exception as e:
        print(f"Error during generation: {e}")
    print("-" * 50)
