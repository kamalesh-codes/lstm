# LSTM Poetry Generator

A high-capacity, character-level language model implemented in PyTorch. This model learns the statistical properties of a provided text dataset to generate new text in a similar style.

## Features

- Architecture: Deep LSTM with embedding layers for sequence modeling.
- Multi-GPU Scaling: DistributedDataParallel (DDP) implementation for optimized training on NVIDIA T4 GPUs.
- Configuration: Managed by Hydra, allowing comprehensive hyperparameter overrides via command line.
- Metrics: Tracks Training Loss, Validation Loss, Validation Accuracy, and Perplexity.
- Visualization: Generates convergence curves for training and validation metrics.
- Checkpointing: Support for periodic state saving and training resumption.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:kamalesh-codes/lstm.git
   cd lstm
   ```

2. Install dependencies:
   ```bash
   pip install torch matplotlib hydra-core tqdm
   ```

## Training

The model is executed using `torchrun` to enable distributed training across available GPUs.

### Standard Training
To initiate training with the default high-capacity configuration:
```bash
torchrun --nproc_per_node=2 train.py
```

### Resuming from Checkpoint
To resume training from the last saved checkpoint (preserving epoch count, optimizer state, and metric history):
```bash
torchrun --nproc_per_node=2 train.py training.resume=true
```

### Hyperparameter Overrides
Hydra allows for dynamic configuration changes at runtime:
```bash
torchrun --nproc_per_node=2 train.py \
    training.epochs=30 \
    training.batch_size=256 \
    model.hidden_size=512 \
    training.learning_rate=0.001
```

## Evaluation

The model utilizes a validation split (default: 10%) to monitor generalization.

1. Split: The dataset is divided into training and validation sets.
2. Epoch Evaluation: After each training epoch, the model is evaluated on the unseen validation set.
3. Metrics: The following are computed:
   - Validation Loss: Measures the generalization error.
   - Accuracy: The percentage of correctly predicted subsequent characters.
   - Perplexity (PPL): An exponential measure of the cross-entropy loss, representing model uncertainty.
4. Visuals: Convergence plots are saved to the `plots/` directory upon completion.

## Inference

Once the training process is complete and the model weights are saved as `poetry_lstm.pth`, text can be generated using the following command:

```bash
python3 generate.py "Seed Text:" 0.7 500
```

### Arguments:
- Seed: The initial text sequence to prime the model.
- Temperature: Controls the randomness of the output. 
  - Values < 0.7: Result in more coherent and predictable text.
  - Values > 1.0: Result in more diverse and creative text.
- Length: The total number of characters to be generated.

## Project Structure

- `train.py`: Distributed training pipeline with checkpointing and Hydra integration.
- `model.py`: LSTM network architecture.
- `utils.py`: Dataset handling and preprocessing.
- `generate.py`: Inference script for text generation.
- `conf/`: YAML configuration files.
- `plots/`: Saved metric visualizations.
- `checkpoints/`: Saved model states for resumption.
- `data/`: Input text datasets.
