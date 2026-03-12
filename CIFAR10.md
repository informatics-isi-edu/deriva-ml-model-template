# CIFAR-10 Example

The template includes a complete CIFAR-10 CNN example demonstrating the full DerivaML workflow: data loading, model training, hyperparameter sweeps, and ROC analysis.

## Quick Start

```bash
# Load data into catalog
uv run load-cifar10 --host <hostname> --catalog_id <id> --num_images 500

# Train the model
uv run deriva-ml-run +experiment=cifar10_quick

# Run a learning rate sweep
uv run deriva-ml-run +multirun=lr_sweep

# Analyze results with ROC curves
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb
```

## Dataset Types

The CIFAR-10 example includes multiple dataset configurations:

| Type | Use Case |
|------|----------|
| `cifar10_small_labeled_split` | Quick experiments with evaluation (recommended) |
| `cifar10_labeled_split` | Full experiments with evaluation |
| `cifar10_small_split` | Quick training without evaluation |
| `cifar10_split` | Full training without evaluation |

**Important:** For ROC analysis or accuracy metrics, use the **labeled** datasets. The unlabeled datasets have test images without ground truth labels.

## Model

The CIFAR-10 example uses a 2-layer CNN (`src/models/cifar10_cnn.py`) for image classification.

**Architecture:** Conv2d(3, C1) → ReLU → MaxPool → Conv2d(C1, C2) → ReLU → MaxPool → Linear(C2×8×8, hidden) → ReLU → Linear(hidden, 10)

**Data flow:** Downloads dataset as BDBag → `restructure_assets()` creates ImageFolder layout → torchvision DataLoader → training/evaluation → saves weights + prediction CSV as execution assets.

### Model Configs

Defined in `src/configs/cifar10_cnn.py`:

| Config | Channels | Hidden | Epochs | LR | Notes |
|--------|----------|--------|--------|------|-------|
| `default_model` | 32→64 | 128 | 10 | 1e-3 | Standard training |
| `cifar10_quick` | 32→64 | 128 | 3 | 1e-3 | Fast validation |
| `cifar10_large` | 64→128 | 256 | 20 | 1e-3 | More capacity |
| `cifar10_regularized` | 32→64 | 128 | 20 | 1e-3 | Dropout 0.25, weight decay 1e-4 |
| `cifar10_fast_lr` | 32→64 | 128 | 15 | 1e-2 | Fast convergence |
| `cifar10_slow_lr` | 32→64 | 128 | 30 | 1e-4 | Stable convergence |
| `cifar10_extended` | 64→128 | 256 | 50 | 1e-3 | Best accuracy, full regularization |
| `cifar10_test_only` | 32→64 | 128 | — | — | Load weights, evaluate only |

### Experiment Presets

Defined in `src/configs/experiments.py`:

| Experiment | Model Config | Dataset | Purpose |
|-----------|-------------|---------|---------|
| `cifar10_quick` | quick | small labeled split | Fast pipeline validation |
| `cifar10_default` | default | small training | Standard training |
| `cifar10_extended` | extended | small labeled split | Best accuracy on small set |
| `cifar10_quick_full` | quick | full labeled split | Baseline on full data |
| `cifar10_extended_full` | extended | full labeled split | Production run |
| `cifar10_test_only` | test_only | small labeled testing | Evaluate pretrained weights |

### ROC Analysis Notebook

The `notebooks/roc_analysis.ipynb` notebook compares model predictions across experiments by generating ROC curves. Configured via `src/configs/roc_analysis.py`. Takes asset RIDs (prediction CSVs) as input.

See the [full CIFAR-10 documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/cifar10-example/) for additional details.
