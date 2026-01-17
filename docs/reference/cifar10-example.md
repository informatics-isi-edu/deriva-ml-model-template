# CIFAR-10 Example

This template includes a complete example for training a CNN on the CIFAR-10 dataset, demonstrating the full DerivaML workflow.

## Overview

The example includes:
- Data loading script to populate a catalog
- CNN model with configurable architecture
- Multiple configuration variants
- Full provenance tracking

## Loading CIFAR-10 Data

First, load CIFAR-10 images into a Deriva catalog:

```bash
uv run load-cifar10 --host <hostname> --catalog_id <catalog_id> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--host` | Deriva server hostname (e.g., `www.eye-ai.org`) |
| `--catalog_id` | Catalog ID to load data into |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_images` | 100 | Number of images to load |
| `--domain_schema` | `cifar10` | Domain schema name |
| `--working_dir` | | Working directory for temp files |
| `--batch_size` | 50 | Batch size for uploads |
| `--train_only` | | Only load training images |
| `--test_only` | | Only load test images |

### Example

```bash
# Load 500 images
uv run load-cifar10 --host dev.eye-ai.org --catalog_id 5 --num_images 500

# Load only training images
uv run load-cifar10 --host localhost --catalog_id 45 --num_images 1000 --train_only
```

## Model Architecture

The CNN (`src/models/cifar10_cnn.py`) uses a simple 2-layer architecture:

```
Input (3x32x32)
    │
    ▼
Conv2d(3, 32) + ReLU + MaxPool2d
    │ (32x16x16)
    ▼
Conv2d(32, 64) + ReLU + MaxPool2d
    │ (64x8x8)
    ▼
Flatten (64*8*8 = 4096)
    │
    ▼
Linear(4096, 128) + ReLU + Dropout
    │
    ▼
Linear(128, 10)
    │
    ▼
Output (10 classes)
```

**Expected accuracy**: ~60-70% with default parameters.

## Configuration

All model parameters are configurable via Hydra:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conv1_channels` | 32 | First conv layer output channels |
| `conv2_channels` | 64 | Second conv layer output channels |
| `hidden_size` | 128 | Hidden layer size |
| `dropout_rate` | 0.0 | Dropout probability |
| `learning_rate` | 1e-3 | Optimizer learning rate |
| `epochs` | 10 | Training epochs |
| `batch_size` | 64 | Training batch size |
| `weight_decay` | 0.0 | L2 regularization |
| `label_column` | "Label" | Column name for class labels |

## Available Configurations

| Name | Description |
|------|-------------|
| `cifar10_default` | Standard configuration (10 epochs) |
| `cifar10_quick` | Fast testing (3 epochs, large batch) |
| `cifar10_large` | Bigger model (64/128 channels, 256 hidden) |
| `cifar10_regularized` | With dropout and weight decay |
| `cifar10_fast_lr` | Higher learning rate (1e-2) |
| `cifar10_slow_lr` | Lower learning rate (1e-4, 30 epochs) |
| `cifar10_extended` | Best accuracy (50 epochs, larger model, regularization) |

## Running the Model

### Basic Usage

```bash
# Default configuration
uv run src/deriva_run.py model_config=cifar10_default

# Quick test
uv run src/deriva_run.py model_config=cifar10_quick

# Extended training
uv run src/deriva_run.py model_config=cifar10_extended
```

### Override Parameters

```bash
# Change learning rate
uv run src/deriva_run.py model_config=cifar10_default model_config.learning_rate=0.01

# Change epochs and batch size
uv run src/deriva_run.py model_config=cifar10_default model_config.epochs=20 model_config.batch_size=128
```

## Data Pipeline

The model uses DerivaML's `restructure_assets()` to organize images:

```
working_dir/cifar10_data/
├─ training/
│  ├─ airplane/
│  │  ├─ image001.png
│  │  └─ image002.png
│  ├─ automobile/
│  ├─ bird/
│  └─ ...
└─ testing/
   ├─ airplane/
   └─ ...
```

This structure is compatible with `torchvision.datasets.ImageFolder`.

### Dataset Requirements

Your datasets need:
- Images in an `Image` asset table
- A `Label` column with class names
- Dataset types "Training" and "Testing" to separate splits

## Model Outputs

The model produces:

| Output | Asset Type | Description |
|--------|------------|-------------|
| `model.pt` | Model | Trained model weights |
| `metrics.json` | Execution_Metadata | Training/validation metrics |
| `training_curves.png` | Image | Loss and accuracy plots |

## Extending the Example

### Custom Architecture

Modify `src/models/cifar10_cnn.py`:

```python
class MyCNN(nn.Module):
    def __init__(self, ...):
        # Your architecture
        ...
```

### New Configuration Variants

Add to `src/configs/cifar10_cnn.py`:

```python
model_store(
    CIFAR10Config,
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=512,
    name="cifar10_deep",
)
```

### Different Dataset

1. Create a data loader for your dataset
2. Modify the restructure call in `cifar10_cnn.py`
3. Update configurations as needed
