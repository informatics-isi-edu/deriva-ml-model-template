# End-to-End Test Plan: DerivaML Model Template

This plan validates the complete DerivaML workflow from catalog creation through model training, hyperparameter sweeps, and analysis.

## Overview

**Objective:** Create a new CIFAR-10 catalog, run all multirun experiments plus test-only mode, and validate results with ROC analysis notebooks.

**Estimated Duration:** Several hours (depends on hardware and dataset size)

---

## Phase 1: Catalog Creation and Setup

### Step 1.1: Create New CIFAR-10 Catalog and Load Data

Use the `load-cifar10` script with the `--create-catalog` option to create a fresh catalog and load data in one step:

```bash
uv run load-cifar10 --hostname localhost --create-catalog cifar10 --num-images 1000 --show-urls
```

This command:
1. Creates a new DerivaML catalog with project name `cifar10`
2. Sets up the domain schema with Image table, Image_Class vocabulary, and Image_Classification feature
3. Downloads CIFAR-10 from Kaggle
4. Uploads 1,000 images (500 training + 500 testing)
5. Creates all dataset hierarchies
6. Shows Chaise URLs for easy access

**Expected Output:** New catalog ID and summary of created datasets with RIDs.

**What this creates:**
- `Image` asset table with 1,000 32×32 RGB images
- `Image_Class` vocabulary (10 CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- `Image_Classification` feature linking images to their ground truth labels
- Dataset hierarchy:
  - `Complete` - All images with labels
  - `Split` (train/test 80/20)
  - `Small_Split` (subset for quick testing)
  - `Labeled_Split` (both partitions have ground truth for ROC analysis)
  - `Small_Labeled_Split` (small version with labels)

### Step 1.2: Update Configuration

Edit `src/configs/deriva.py` to point to the new catalog:

```python
from hydra_zen import builds, store
from deriva_ml.config import DerivaMLConfig

# Store default connection pointing to new catalog
store(
    builds(DerivaMLConfig, hostname="localhost", catalog_id=<NEW_CATALOG_ID>),
    group="deriva_ml",
    name="default_deriva",
)
```

### Step 1.3: Validate Configuration

Before running experiments, validate that all RIDs in the configuration exist:

```bash
# Using MCP tool
mcp__deriva-ml__validate_rids(
    dataset_rids=["<DATASET_RIDS_FROM_CONFIGS>"],
    dataset_versions={"<RID>": "<VERSION>"},
)
```

**Expected Output:** `is_valid: true` with no errors. Warnings for missing descriptions are OK but should be addressed.

### Step 1.4: Commit Changes

**CRITICAL:** DerivaML tracks code provenance. All code must be committed before running models.

```bash
git add -A
git commit -m "Configure for e2e test catalog <CATALOG_ID>"
```

---

## Phase 2: Run All Multirun Experiments

### Multirun Configuration Summary

| Multirun Name | Description | Experiments/Parameters | Expected Runs |
|---------------|-------------|------------------------|---------------|
| `quick_vs_extended` | Compare training intensity | cifar10_quick (3 epochs) vs cifar10_extended (50 epochs) | 2 |
| `quick_vs_extended_full` | Full dataset comparison | Same configs on 10k images | 2 |
| `lr_sweep` | Learning rate optimization | LR: [0.0001, 0.001, 0.01, 0.1] | 4 |
| `epoch_sweep` | Training duration analysis | Epochs: [5, 10, 25, 50] | 4 |
| `lr_batch_grid` | Hyperparameter grid search | LR × Batch: [0.001, 0.01] × [64, 128] | 4 |

### Step 2.1: Quick vs Extended (Small Dataset)

**Purpose:** Validate training pipeline with fast runs before committing to longer experiments.

```bash
uv run deriva-ml-run +multirun=quick_vs_extended
```

**Multirun Description:**
> ## CIFAR-10 CNN Multi-Experiment Comparison
>
> **Objective:** Compare model performance across two training configurations to evaluate
> the trade-off between training speed and model accuracy.
>
> | Experiment | Epochs | Architecture | Regularization | Dataset |
> |------------|--------|--------------|----------------|---------|
> | `cifar10_quick` | 3 | 32→64 channels, 128 hidden | None | Small Split (1,000 images) |
> | `cifar10_extended` | 50 | 64→128 channels, 256 hidden | Dropout 0.25, Weight Decay 1e-4 | Small Split (1,000 images) |

**Expected Outputs per run:**
- Model weights (`model_weights.pt`)
- Prediction probabilities (`prediction_probabilities.csv`)
- Test set predictions recorded to `Image_Classification` feature

### Step 2.2: Learning Rate Sweep

**Purpose:** Identify optimal learning rate for the architecture.

```bash
uv run deriva-ml-run +multirun=lr_sweep
```

**Multirun Description:**
> ## Learning Rate Hyperparameter Sweep
>
> **Objective:** Identify the optimal learning rate for the CIFAR-10 CNN architecture
> by comparing training dynamics across a range of values.
>
> | Learning Rate | Expected Behavior |
> |--------------|-------------------|
> | 0.0001 | Slow convergence, may underfit in limited epochs |
> | 0.001 | Standard rate, good balance of speed and stability |
> | 0.01 | Fast convergence, risk of overshooting minima |
> | 0.1 | Aggressive, likely unstable training |

**What to watch for:**
- Training loss curve shape
- Presence of training instability (loss spikes at high LR)
- Generalization gap (train vs test accuracy)

### Step 2.3: Epoch Sweep

**Purpose:** Analyze how training duration affects performance and identify overfitting onset.

```bash
uv run deriva-ml-run +multirun=epoch_sweep
```

**Multirun Description:**
> ## Training Duration (Epochs) Sweep
>
> **Objective:** Analyze how training duration affects model performance and identify
> the point of diminishing returns or overfitting onset.
>
> | Epochs | Expected Behavior |
> |--------|-------------------|
> | 5 | Underfitting, model still learning basic features |
> | 10 | Reasonable performance, may still be improving |
> | 25 | Good performance, watch for overfitting signs |
> | 50 | Extended training, likely overfitting on small dataset |

**What to watch for:**
- Training vs test accuracy divergence
- Optimal early stopping point

### Step 2.4: Learning Rate × Batch Size Grid

**Purpose:** Find optimal combination of learning rate and batch size.

```bash
uv run deriva-ml-run +multirun=lr_batch_grid
```

**Multirun Description:**
> ## Learning Rate and Batch Size Grid Search
>
> **Objective:** Find optimal combination of learning rate and batch size.
>
> | Parameter | Values |
> |-----------|--------|
> | Learning Rate | 0.001, 0.01 |
> | Batch Size | 64, 128 |
>
> **Total runs:** 4 (2 × 2 grid)

**What to watch for:**
- Smaller batch sizes may need lower learning rates
- Larger batch sizes can tolerate higher learning rates

### Step 2.5: Full Dataset Comparison (Optional)

**Purpose:** Production-quality validation on complete dataset.

```bash
uv run deriva-ml-run +multirun=quick_vs_extended_full
```

**Note:** This requires loading the full 10,000 image dataset and takes significantly longer.

---

## Phase 3: Test-Only Mode (Inference)

### Step 3.1: Update Asset Configuration

After running experiments, update `src/configs/assets.py` to reference the new model weights:

```python
# Find the RID of model weights from quick_vs_extended run
# Use MCP: mcp__deriva-ml__find_assets(pattern="model_weights", limit=10)

store(
    AssetConfig(
        rids=["<WEIGHTS_RID>"],
        description="Pre-trained CIFAR-10 CNN weights from e2e test quick training run (3 epochs, 32→64 channels). Use for inference-only experiments to validate model loading and prediction pipeline.",
    ),
    group="assets",
    name="e2e_test_weights",
)
```

### Step 3.2: Run Test-Only Experiment

```bash
uv run deriva-ml-run +experiment=cifar10_test_only assets=e2e_test_weights
```

**What this does:**
- Loads pre-trained weights (no training)
- Runs inference on test set
- Records predictions to catalog
- Saves prediction probabilities

---

## Phase 4: Update ROC Analysis Configurations

### Step 4.1: Gather Prediction Probability Asset RIDs

Use MCP tools to find all prediction probability files:

```python
# Connect to catalog
mcp__deriva-ml__connect_catalog(hostname="localhost", catalog_id="<CATALOG_ID>")

# Find prediction probability files from each multirun
mcp__deriva-ml__find_assets(pattern="prediction_probabilities", limit=20)
```

### Step 4.2: Update assets.py

Add asset configurations for ROC analysis. **Always include descriptions** explaining the source and purpose of each asset configuration:

```python
# src/configs/assets.py

# =============================================================================
# Quick vs Extended Comparison Assets
# =============================================================================

store(
    AssetConfig(
        rids=["<QUICK_PROBS_RID>", "<EXTENDED_PROBS_RID>"],
        description="""Prediction probability files from quick vs extended training comparison.

Contains outputs from two experiments on the small labeled split (1,000 images):
- cifar10_quick: 3 epochs, 32→64 channels, batch 128, no regularization
- cifar10_extended: 50 epochs, 64→128 channels, batch 64, dropout 0.25, weight decay 1e-4

Use with roc_analysis notebook to compare ROC curves and evaluate the accuracy
trade-off between fast validation runs and production training.""",
    ),
    group="assets",
    name="roc_quick_vs_extended",
)

# =============================================================================
# Learning Rate Sweep Assets
# =============================================================================

store(
    AssetConfig(
        rids=[
            "<LR_0001_PROBS_RID>",
            "<LR_001_PROBS_RID>",
            "<LR_01_PROBS_RID>",
            "<LR_1_PROBS_RID>",
        ],
        description="""Prediction probability files from learning rate hyperparameter sweep.

Compares four learning rates on the small labeled split (1,000 images):
- lr=0.0001: Conservative, slow convergence expected
- lr=0.001: Standard baseline (default)
- lr=0.01: Aggressive, may show faster convergence or instability
- lr=0.1: Very aggressive, likely unstable training

All runs use: 10 epochs, 32→64 channels, batch 128. Use with roc_analysis
notebook to identify optimal learning rate based on AUC scores.""",
    ),
    group="assets",
    name="roc_lr_sweep",
)

# =============================================================================
# Epoch Sweep Assets
# =============================================================================

store(
    AssetConfig(
        rids=[
            "<EPOCH_5_PROBS_RID>",
            "<EPOCH_10_PROBS_RID>",
            "<EPOCH_25_PROBS_RID>",
            "<EPOCH_50_PROBS_RID>",
        ],
        description="""Prediction probability files from training duration (epochs) sweep.

Compares four epoch counts on the small labeled split (1,000 images):
- 5 epochs: Early stopping point, likely underfitting
- 10 epochs: Moderate training, reasonable baseline
- 25 epochs: Extended training, near convergence expected
- 50 epochs: Full training, watch for overfitting on small dataset

All runs use: extended architecture (64→128 channels), dropout 0.25, weight
decay 1e-4. Use with roc_analysis notebook to identify optimal training
duration and detect overfitting onset.""",
    ),
    group="assets",
    name="roc_epoch_sweep",
)

# =============================================================================
# LR × Batch Size Grid Search Assets
# =============================================================================

store(
    AssetConfig(
        rids=[
            "<LR001_B64_PROBS_RID>",
            "<LR001_B128_PROBS_RID>",
            "<LR01_B64_PROBS_RID>",
            "<LR01_B128_PROBS_RID>",
        ],
        description="""Prediction probability files from learning rate × batch size grid search.

Explores interaction between learning rate and batch size (2×2 grid):
- lr=0.001, batch=64: Conservative LR with smaller batches
- lr=0.001, batch=128: Conservative LR with larger batches
- lr=0.01, batch=64: Aggressive LR with smaller batches
- lr=0.01, batch=128: Aggressive LR with larger batches

All runs use: 10 epochs, 32→64 channels, small labeled split (1,000 images).
Use with roc_analysis notebook to find optimal hyperparameter combination
and understand LR/batch interaction effects.""",
    ),
    group="assets",
    name="roc_lr_batch_grid",
)
```

### Step 4.3: Commit Configuration Updates

```bash
git add -A
git commit -m "Add ROC analysis asset configurations for e2e test"
```

---

## Phase 5: Run ROC Analysis Notebooks

### Step 5.1: Quick vs Extended Analysis

```bash
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --config roc_quick_vs_extended
```

**ROC Analysis Configuration:**
> **roc_quick_vs_extended**: ROC analysis comparing quick (3 epoch) vs extended (50 epoch) training on small dataset (1,000 images).
>
> - `show_per_class`: True - Plot individual ROC curves for each CIFAR-10 class
> - `confidence_threshold`: 0.0 - Include all predictions

### Step 5.2: Learning Rate Sweep Analysis

```bash
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --config roc_lr_sweep
```

**ROC Analysis Configuration:**
> **roc_lr_sweep**: ROC analysis comparing learning rate sweep (0.0001, 0.001, 0.01, 0.1).
>
> Expected to show:
> - Very low LR (0.0001): Underfitting, lower AUC
> - Optimal LR (0.001): Best AUC
> - High LR (0.01): May show slight degradation
> - Very high LR (0.1): Poor performance due to training instability

### Step 5.3: Epoch Sweep Analysis

```bash
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --config roc_epoch_sweep
```

**ROC Analysis Configuration:**
> **roc_epoch_sweep**: ROC analysis comparing epoch sweep (5, 10, 25, 50 epochs).
>
> Expected to show:
> - 5 epochs: Underfitting, lower AUC
> - 10 epochs: Improved performance
> - 25 epochs: Near-optimal on small dataset
> - 50 epochs: May show overfitting (AUC plateau or decline)

### Step 5.4: Grid Search Analysis

```bash
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --config roc_lr_batch_grid
```

**ROC Analysis Configuration:**
> **roc_lr_batch_grid**: ROC analysis comparing LR × batch size grid (2×2).
>
> Shows interaction effects between learning rate and batch size.

---

## Phase 6: Validation Checklist

### Catalog Validation (via MCP)

```python
# Check dataset hierarchy
mcp__deriva-ml__list_datasets()

# Verify executions were recorded
mcp__deriva-ml__list_executions(limit=20)

# Check features were populated
mcp__deriva-ml__list_feature_values(feature_name="Image_Classification", limit=10)

# Verify provenance
mcp__deriva-ml__list_dataset_executions(rid="<DATASET_RID>")
```

### Expected Execution Count

| Phase | Expected Executions |
|-------|---------------------|
| Data loading | 1 (load-cifar10 creates datasets) |
| quick_vs_extended | 3 (1 parent + 2 children) |
| lr_sweep | 5 (1 parent + 4 children) |
| epoch_sweep | 5 (1 parent + 4 children) |
| lr_batch_grid | 5 (1 parent + 4 children) |
| test_only | 1 |
| ROC analysis (×4) | 4 |
| **Total** | ~24 executions |

### Artifacts Checklist

For each training run:
- [ ] Model weights uploaded (`model_weights.pt`)
- [ ] Prediction probabilities saved (`prediction_probabilities.csv`)
- [ ] Test predictions recorded to `Image_Classification` feature
- [ ] Execution status: `Complete`

For each notebook run:
- [ ] ROC curve outputs generated
- [ ] Execution recorded with notebook reference

---

## Troubleshooting

For general DerivaML troubleshooting (connection issues, code provenance, dataset versioning, labeled vs unlabeled splits), see the DerivaML MCP server instructions.

### Project-Specific Issues

1. **CUDA out of memory**
   - Reduce batch size: `model_config.batch_size=32`

---

## Summary

This end-to-end test validates:
1. ✅ Catalog creation and data loading via MCP
2. ✅ Hydra-zen configuration system
3. ✅ All 5 multirun configurations
4. ✅ Test-only (inference) mode
5. ✅ ROC analysis notebooks with all configurations
6. ✅ Full provenance tracking across all experiments
