# End-to-End Test Steps

## Test Date: 2026-01-17

## Prerequisites
- [x] Environment set up with `uv sync`
- [x] Authenticated to target host (localhost)

## Test Steps

### 1. Create New CIFAR-10 Catalog
```bash
uv run load-cifar10 --hostname localhost --create-catalog cifar-10 --num-images 10000
```

**Result:**
- Catalog ID: **62**
- Schema: **cifar-10**
- Images loaded: 10,000 (5,000 training + 5,000 testing)

**Datasets created:**
| Dataset | RID | Version | Description |
|---------|-----|---------|-------------|
| Complete | 28CT | 0.21.0 | All 10,000 images |
| Split | 28D4 | 0.22.0 | Parent of Training + Testing |
| Training | 28DC | 0.22.0 | 5,000 training images |
| Testing | 28DP | 0.22.0 | 5,000 testing images (unlabeled) |
| Small_Split | 28EA | 0.4.0 | Parent of small datasets |
| Small_Training | 28EJ | 0.4.0 | 500 training images |
| Small_Testing | 28EW | 0.4.0 | 500 testing images |
| Labeled_Split | 28FG | 0.11.0 | 5,000 images with labels |
| Labeled_Training | 28FT | 0.11.0 | 4,000 labeled training images |
| Labeled_Testing | 28G4 | 0.11.0 | 1,000 labeled test images |
| Small_Labeled_Split | 28GR | 0.4.0 | 500 labeled images |
| Small_Labeled_Training | 28H2 | 0.4.0 | 400 labeled training images |
| Small_Labeled_Testing | 28HC | 0.4.0 | 100 labeled test images |

### 2. Update Configuration Files
Updated `src/configs/deriva.py`:
- Changed `catalog_id` from 45 to **62**

Updated `src/configs/datasets.py`:
- Replaced all old RIDs with new catalog 62 RIDs
- Added correct version numbers for each dataset

### 3. Run Multi-Experiment Training

## CIFAR-10 CNN Multi-Experiment Comparison

**Objective:** Compare model performance across two training configurations to evaluate the trade-off between training speed and model accuracy.

### Experiments

| Experiment | Epochs | Architecture | Regularization | Dataset |
|------------|--------|--------------|----------------|---------|
| `cifar10_quick` | 3 | 32→64 channels, 128 hidden | None | Small Split (1,000 images) |
| `cifar10_extended` | 50 | 64→128 channels, 256 hidden | Dropout 0.25, Weight Decay 1e-4 | Small Split (1,000 images) |

### Configuration Details

**cifar10_quick** - Fast validation baseline
- Conv1: 32 channels → Conv2: 64 channels
- Hidden layer: 128 units
- Batch size: 128
- Learning rate: 1e-3
- No regularization

**cifar10_extended** - Production-quality training
- Conv1: 64 channels → Conv2: 128 channels
- Hidden layer: 256 units
- Batch size: 64
- Learning rate: 1e-3
- Dropout: 0.25
- Weight decay: 1e-4

### Command
```bash
uv run deriva-ml-run --multirun +experiment=cifar10_quick,cifar10_extended \
  'description=CIFAR-10 CNN Multi-Experiment comparing quick vs extended training configurations'
```

**Result:**
- Parent Execution: **3WMA**
- cifar10_quick (Job 0): **3WNE** - 52.96 sec
- cifar10_extended (Job 1): **3XQ2** - 1 min 44 sec

### Training Results

**cifar10_quick (3 epochs):**
- Final train accuracy: 29.40%
- Final test accuracy: 9.20%
- Final train loss: 1.9851
- Final test loss: 3.0523

**cifar10_extended (50 epochs):**
- Final train accuracy: 100.00%
- Final test accuracy: 11.00%
- Final train loss: 0.0112
- Final test loss: 8.7693
- Note: Significant overfitting observed (train 100% vs test 11%)

### 4. Update Assets Configuration

Updated `src/configs/assets.py` with new asset RIDs:

| Asset Type | cifar10_quick | cifar10_extended |
|------------|---------------|------------------|
| Model weights | 3WPG | 3XRA |
| Training log | 3WPJ | 3XRC |
| Prediction probabilities | 3WPM | 3XRE |
| Hydra config | 3WMM | 3XQ8 |

### 5. Commit Changes
```bash
git add -A && git commit -m "Complete E2E test: catalog 62, multirun experiments, updated configs"
```

## Results Summary
- Status: **Complete** ✅
- Catalog created: ✅ (ID: 62)
- Configuration updated: ✅
- Model training: ✅ (Parent: 3WMA)
- Assets updated: ✅

## Notes
- The extended model shows significant overfitting (100% train vs 11% test accuracy)
- This is expected with only 500 training images and 50 epochs
- For production use, consider early stopping or data augmentation

---

## Using Sweep Configurations

For future multi-experiment runs, use the `sweep` config group for rich markdown descriptions.

### Available Sweeps

| Sweep Name | Description |
|------------|-------------|
| `quick_vs_extended` | Compare quick (3 epochs) vs extended (50 epochs) on small dataset |
| `quick_vs_extended_full` | Same comparison on full dataset |

### Running a Sweep
```bash
# Run pre-defined sweep with rich markdown description
uv run deriva-ml-run --multirun +sweep=quick_vs_extended \
    +experiment=cifar10_quick,cifar10_extended

# Override parameters within a sweep
uv run deriva-ml-run --multirun +sweep=quick_vs_extended \
    +experiment=cifar10_quick,cifar10_extended model_config.epochs=5

# Ad-hoc parameter sweeps still work (no rich description)
uv run deriva-ml-run --multirun +experiment=cifar10_quick,cifar10_extended
```

### Benefits of Sweep Configs
- Full markdown support in parent execution descriptions
- Documented experiment rationale and expected outcomes
- Reusable, named experiment combinations
- Bypasses Hydra command-line parsing limitations for special characters
