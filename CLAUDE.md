# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DerivaML Model Template - a template for creating ML models integrated with DerivaML. This specific instance implements a CIFAR-10 CNN classifier with 7 model variants and an ROC analysis notebook.

## Common Commands

```bash
# Environment setup
uv sync                                    # Initialize environment
uv sync --group=jupyter                   # Add Jupyter support
uv sync --group=pytorch                   # Add PyTorch support

# Running models
uv run deriva-ml-run +experiment=cifar10_quick       # Quick training (3 epochs)
uv run deriva-ml-run +experiment=cifar10_extended     # Extended training (50 epochs)
uv run deriva-ml-run +multirun=quick_vs_extended      # Compare quick vs extended
uv run deriva-ml-run dry_run=true                     # Dry run (no catalog writes)
uv run deriva-ml-run --info                           # Show available configs

# Notebook execution
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb assets=my_assets

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Testing
uv run pytest

# Version management (bump-version is provided by deriva-ml)
uv run bump-version major|minor|patch

# Data loading
uv run load-cifar10 --host <hostname> --catalog_id <id> --num_images 500
```

## Models

### CIFAR-10 2-Layer CNN (`src/models/cifar10_cnn.py`)

A PyTorch convolutional neural network for CIFAR-10 image classification.

**Architecture:** Conv2d(3, C1) → ReLU → MaxPool → Conv2d(C1, C2) → ReLU → MaxPool → Linear(C2×8×8, hidden) → ReLU → Linear(hidden, 10)

**Model configs** (in `src/configs/cifar10_cnn.py`):

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

**Experiments** (in `src/configs/experiments.py`):

| Experiment | Model Config | Dataset | Purpose |
|-----------|-------------|---------|---------|
| `cifar10_quick` | quick | small labeled split | Fast pipeline validation |
| `cifar10_default` | default | small training | Standard training |
| `cifar10_extended` | extended | small labeled split | Best accuracy on small set |
| `cifar10_quick_full` | quick | full labeled split | Baseline on full data |
| `cifar10_extended_full` | extended | full labeled split | Production run |
| `cifar10_test_only` | test_only | small labeled testing | Evaluate pretrained weights |

**Data flow:** Downloads dataset as BDBag → `restructure_assets()` creates ImageFolder layout → torchvision DataLoader → training/evaluation → saves weights + prediction CSV as execution assets.

### ROC Analysis Notebook (`notebooks/roc_analysis.ipynb`)

Compares model predictions across experiments by generating ROC curves. Configured via `src/configs/roc_analysis.py`. Takes asset RIDs (prediction CSVs) as input.

## Source Layout

- `src/configs/` — Hydra-zen configuration (Python, no YAML)
  - `cifar10_cnn.py` — Model configs (architectures, hyperparameters)
  - `experiments.py` — Experiment configs (model + dataset combinations)
  - `multiruns.py` — Multirun sweep configs (parameter combinations)
  - `roc_analysis.py` — ROC notebook asset configs
  - `dev/` — Per-environment Deriva connection overrides
- `src/models/` — Model implementations (cifar10_cnn.py)
- `src/scripts/` — Data loading (load_cifar10.py)
- `notebooks/` — Analysis notebooks (roc_analysis.ipynb)

## Key Rules

- **Commit before running** — DerivaML tracks git commit hash for provenance
- **Use labeled datasets for evaluation** — `cifar10_small_labeled_split` or `cifar10_labeled_split` (unlabeled splits have no test ground truth)
- **`Execution_Asset`** for model outputs (weights, predictions, plots); `Execution_Metadata` is auto-managed
- **Test with `dry_run=true`** before production runs
- **`--config` on `deriva-ml-run-notebook` does NOT override `run_notebook()` config name** — use positional Hydra overrides instead (e.g., `assets=my_assets_prod`)

## Catalog Environments

| Config Name | Hostname | Usage |
|------------|----------|-------|
| `default_deriva` | localhost | Local development and testing |

Production configs use a `*_prod` suffix convention. Add alternate configs in `src/configs/dev/`.
