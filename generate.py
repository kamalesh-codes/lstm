import torch
import torch.nn as nn
import random
import yaml
import urllib.request
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
    
    # 2. Load Config and Vocab (Fallbacks for slim checkpoints)
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    ensure_data('data/input.txt')
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        char2int = {ch: i for i, ch in enumerate(chars)}
        int2char = {i: ch for i, ch in enumerate(chars)}
        vocab_size = len(chars)

    embed_size = config['model']['embed_size']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    
    # 3. Load Checkpoint
    if not os.path.exists('model_best.pth'):
        raise FileNotFoundError("Model weights not found! Please train the model first.")
        
    checkpoint = torch.load('model_best.pth', map_location=device)
    
    # Override with checkpoint values if available
    chars = checkpoint.get('chars', chars)
    char2int = checkpoint.get('char2int', char2int)
    int2char = checkpoint.get('int2char', int2char)
    vocab_size = checkpoint.get('vocab_size', vocab_size)
    embed_size = checkpoint.get('embed_size', embed_size)
    hidden_size = checkpoint.get('hidden_size', hidden_size)
    num_layers = checkpoint.get('num_layers', num_layers)
    
    # 4. Model Initialization
    model = PoetryLSTM(vocab_size, embed_size, hidden_size, num_layers)
    
    # Handle DataParallel 'module.' prefix if it exists
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    # 5. Prepare Seed
    input_seq = torch.tensor(
        [char2int.get(ch, 0) for ch in seed_text], 
        dtype=torch.long
    ).unsqueeze(0).to(device)
    
    generated_text = seed_text
    
    # 6. Generation Loop
    with torch.no_grad():
        output, hidden = model(input_seq)
        last_char_logits = output[0, -1, :]
        probs = torch.softmax(last_char_logits, dim=-1)
        
        for _ in range(gen_length):
            next_char_idx = sample(probs, temperature)
            next_char = int2char[next_char_idx]
            generated_text += next_char
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
            output, hidden = model(input_seq, hidden)
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
