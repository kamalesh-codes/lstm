# ✍️ LSTM Poetry Generator

A high-capacity, character-level language model built with PyTorch that learns to mimic the style of a given text (e.g., Shakespeare) and generate new, original poetry.

## 🚀 Features
- **Architecture**: Deep LSTM with embedding layers.
- **Multi-GPU Scaling**: Implemented with `DistributedDataParallel` (DDP) for maximum throughput on NVIDIA T4 GPUs.
- **Configuration**: Fully managed by **Hydra**, allowing hyperparameter tuning directly from the command line.
- **Metrics**: Tracks Training/Validation Loss, Accuracy, and Perplexity (PPL).
- **Visualization**: Generates detailed training curves in the `plots/` directory.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:kamalesh-codes/lstm.git
   cd lstm
   ```

2. Install dependencies:
   ```bash
   pip install torch matplotlib hydra-core tqdm
   ```

## 🏋️ Training

The model uses `torchrun` to launch distributed training across multiple GPUs.

### Basic Training
To start training with the default high-capacity settings:
```bash
torchrun --nproc_per_node=2 train.py
```

### Customizing Training (Hydra Overrides)
You can override any setting in `conf/config.yaml` via the command line:
```bash
torchrun --nproc_per_node=2 train.py \
    training.epochs=20 \
    training.batch_size=256 \
    model.hidden_size=512 \
    training.learning_rate=0.001
```

## 📊 Evaluation

### Does it use a separate test set?
The model uses a **Validation Set** (default: 10% of the input text). 

**The evaluation process works as follows:**
1. **Split**: The dataset is split into Training (90%) and Validation (10%) sets.
2. **Per-Epoch Eval**: After every training epoch, the model is switched to `.eval()` mode.
3. **Metrics**: It calculates the following on the unseen validation data:
   - **Validation Loss**: Generalization error.
   - **Accuracy**: Percentage of correctly predicted next characters.
   - **Perplexity (PPL)**: A measure of how well the probability distribution predicts the sample.
4. **Visuals**: A final plot is saved to `plots/training_metrics.png` showing the convergence of both training and validation metrics.

## ✍️ Generating Text

Once training is complete and `poetry_lstm.pth` is saved, use the generation script:

```bash
python3 generate.py "ROMEO:" 0.7 500
```

- **Seed**: The first argument is the starting text (e.g., `"ROMEO:"`).
- **Temperature**: The second argument (e.g., `0.7`) controls randomness. 
  - **Lower (< 0.7)**: More confident, predictable, and coherent.
  - **Higher (> 1.0)**: More creative, diverse, but potentially chaotic.
- **Length**: The third argument (e.g., `500`) specifies how many characters to generate.
## 📂 Project Structure
- `train.py`: DDP training pipeline with Hydra integration.
- `model.py`: LSTM model architecture.
- `utils.py`: Dataset and preprocessing utilities.
- `generate.py`: Inference script for text generation.
- `conf/`: Hydra configuration files.
- `plots/`: Saved training metric graphs.
- `data/`: Input text files.
