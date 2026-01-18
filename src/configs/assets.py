"""Asset Configuration.

This module defines asset configurations for execution (model weights, checkpoints, etc.).

Configuration Group: assets
---------------------------
Assets are additional files needed beyond the dataset. They are specified as
lists of Resource IDs (RIDs) and automatically downloaded when the execution
is initialized.

Typical assets include:
- Pre-trained model weights
- Model checkpoints
- Configuration files
- Reference data files

REQUIRED: A configuration named "default_asset" must be defined.
This is used as the default (typically an empty list) when no override is specified.

Example usage:
    # Use default (no assets)
    uv run src/deriva_run.py

    # Use specific assets
    uv run src/deriva_run.py assets=multirun_quick_weights

Configuration Format:
    my_assets = ["3RA", "3R8"]
    asset_store(my_assets, name="my_asset_config")
"""

from hydra_zen import store

# ---------------------------------------------------------------------------
# Asset Configurations
# ---------------------------------------------------------------------------
# Define asset sets as lists of RID strings. Each RID references a file
# in the Deriva catalog that will be downloaded for the execution.

assets = []

# =============================================================================
# Catalog 65: CIFAR-10 Multi-Experiment Assets (localhost, schema: cifar10)
# =============================================================================

# -----------------------------------------------------------------------------
# quick_vs_extended multirun (parent: 3WJA)
# -----------------------------------------------------------------------------
# Child executions:
#   - cifar10_quick (3WKE): 3 epochs, 32→64 channels, batch 128
#   - cifar10_extended (3XN2): 50 epochs, 64→128 channels, dropout 0.25, weight decay 1e-4

# Model weights from multirun comparison
multirun_quick_weights = ["3WMG"]  # cifar10_cnn_weights.pt from cifar10_quick
multirun_extended_weights = ["3XPA"]  # cifar10_cnn_weights.pt from cifar10_extended
multirun_comparison_weights = ["3WMG", "3XPA"]  # Both model weights

# Training logs
multirun_quick_log = ["3WMJ"]  # training_log.txt from cifar10_quick
multirun_extended_log = ["3XPC"]  # training_log.txt from cifar10_extended

# Prediction probability files (for ROC analysis)
multirun_quick_probabilities = ["3WMM"]  # prediction_probabilities.csv from cifar10_quick
multirun_extended_probabilities = ["3XPE"]  # prediction_probabilities.csv from cifar10_extended
multirun_comparison_probabilities = ["3WMM", "3XPE"]  # Both probability files

# Complete asset sets (weights + hydra config)
multirun_quick_assets = ["3WMG", "3WJM"]  # weights + hydra config
multirun_extended_assets = ["3XPA", "3XN8"]  # weights + hydra config

# -----------------------------------------------------------------------------
# quick_vs_extended_full multirun (parent: 3YQM) - Full dataset (10,000 images)
# -----------------------------------------------------------------------------
# Child executions:
#   - cifar10_quick_full (3YRR): 3 epochs, 32→64 channels, batch 128
#   - cifar10_extended_full (48KM): 50 epochs, 64→128 channels, dropout 0.25, weight decay 1e-4

# Model weights from full dataset comparison
full_quick_weights = ["3YST"]  # cifar10_cnn_weights.pt from cifar10_quick_full
full_extended_weights = ["48MW"]  # cifar10_cnn_weights.pt from cifar10_extended_full
full_comparison_weights = ["3YST", "48MW"]  # Both model weights

# Training logs
full_quick_log = ["3YSW"]  # training_log.txt from cifar10_quick_full
full_extended_log = ["48MY"]  # training_log.txt from cifar10_extended_full

# Prediction probability files (for ROC analysis)
full_quick_probabilities = ["3YSY"]  # prediction_probabilities.csv from cifar10_quick_full
full_extended_probabilities = ["48N0"]  # prediction_probabilities.csv from cifar10_extended_full
full_comparison_probabilities = ["3YSY", "48N0"]  # Both probability files

# -----------------------------------------------------------------------------
# lr_sweep multirun (parent: 4JFE) - Learning rate hyperparameter sweep
# -----------------------------------------------------------------------------
# Child executions:
#   - lr=0.0001 (4JGJ): 10 epochs, 32→64 channels, batch 128
#   - lr=0.001 (4KJ6): 10 epochs, 32→64 channels, batch 128
#   - lr=0.01 (4MM0): 10 epochs, 32→64 channels, batch 128
#   - lr=0.1 (4NNT): 10 epochs, 32→64 channels, batch 128

# Model weights from lr sweep
lr_sweep_0001_weights = ["4JHM"]  # cifar10_cnn_weights.pt, lr=0.0001
lr_sweep_001_weights = ["4KKE"]  # cifar10_cnn_weights.pt, lr=0.001
lr_sweep_01_weights = ["4MN8"]  # cifar10_cnn_weights.pt, lr=0.01
lr_sweep_1_weights = ["4NQ2"]  # cifar10_cnn_weights.pt, lr=0.1
lr_sweep_all_weights = ["4JHM", "4KKE", "4MN8", "4NQ2"]  # All lr sweep weights

# Prediction probabilities from lr sweep
lr_sweep_0001_probabilities = ["4JHR"]  # prediction_probabilities.csv, lr=0.0001
lr_sweep_001_probabilities = ["4KKJ"]  # prediction_probabilities.csv, lr=0.001
lr_sweep_01_probabilities = ["4MNC"]  # prediction_probabilities.csv, lr=0.01
lr_sweep_1_probabilities = ["4NQ6"]  # prediction_probabilities.csv, lr=0.1
lr_sweep_all_probabilities = ["4JHR", "4KKJ", "4MNC", "4NQ6"]  # All lr sweep probabilities

# -----------------------------------------------------------------------------
# epoch_sweep multirun (parent: 4PRC) - Training duration sweep
# -----------------------------------------------------------------------------
# Child executions:
#   - epochs=5 (4PSG): 64→128 channels, 256 hidden, dropout 0.25, weight decay 1e-4
#   - epochs=10 (4QV4): 64→128 channels, 256 hidden, dropout 0.25, weight decay 1e-4
#   - epochs=25 (4RWY): 64→128 channels, 256 hidden, dropout 0.25, weight decay 1e-4
#   - epochs=50 (4SYR): 64→128 channels, 256 hidden, dropout 0.25, weight decay 1e-4

# Model weights from epoch sweep
epoch_sweep_5_weights = ["4PTJ"]  # cifar10_cnn_weights.pt, epochs=5
epoch_sweep_10_weights = ["4QWC"]  # cifar10_cnn_weights.pt, epochs=10
epoch_sweep_25_weights = ["4RY6"]  # cifar10_cnn_weights.pt, epochs=25
epoch_sweep_50_weights = ["4T00"]  # cifar10_cnn_weights.pt, epochs=50
epoch_sweep_all_weights = ["4PTJ", "4QWC", "4RY6", "4T00"]  # All epoch sweep weights

# Prediction probabilities from epoch sweep
epoch_sweep_5_probabilities = ["4PTP"]  # prediction_probabilities.csv, epochs=5
epoch_sweep_10_probabilities = ["4QWG"]  # prediction_probabilities.csv, epochs=10
epoch_sweep_25_probabilities = ["4RYA"]  # prediction_probabilities.csv, epochs=25
epoch_sweep_50_probabilities = ["4T04"]  # prediction_probabilities.csv, epochs=50
epoch_sweep_all_probabilities = ["4PTP", "4QWG", "4RYA", "4T04"]  # All epoch sweep probabilities

# -----------------------------------------------------------------------------
# lr_batch_grid multirun (parent: 4V1A) - Learning rate and batch size grid search
# -----------------------------------------------------------------------------
# Child executions (2x2 grid):
#   - lr=0.001, batch=64 (4V2E): 10 epochs, 32→64 channels
#   - lr=0.001, batch=128 (4W42): 10 epochs, 32→64 channels
#   - lr=0.01, batch=64 (4X5W): 10 epochs, 32→64 channels
#   - lr=0.01, batch=128 (4Y7P): 10 epochs, 32→64 channels

# Model weights from lr_batch_grid
grid_lr001_batch64_weights = ["4V3G"]  # cifar10_cnn_weights.pt, lr=0.001, batch=64
grid_lr001_batch128_weights = ["4W5A"]  # cifar10_cnn_weights.pt, lr=0.001, batch=128
grid_lr01_batch64_weights = ["4X74"]  # cifar10_cnn_weights.pt, lr=0.01, batch=64
grid_lr01_batch128_weights = ["4Y8Y"]  # cifar10_cnn_weights.pt, lr=0.01, batch=128
grid_all_weights = ["4V3G", "4W5A", "4X74", "4Y8Y"]  # All grid search weights

# Prediction probabilities from lr_batch_grid
grid_lr001_batch64_probabilities = ["4V3M"]  # prediction_probabilities.csv, lr=0.001, batch=64
grid_lr001_batch128_probabilities = ["4W5E"]  # prediction_probabilities.csv, lr=0.001, batch=128
grid_lr01_batch64_probabilities = ["4X78"]  # prediction_probabilities.csv, lr=0.01, batch=64
grid_lr01_batch128_probabilities = ["4Y92"]  # prediction_probabilities.csv, lr=0.01, batch=128
grid_all_probabilities = ["4V3M", "4W5E", "4X78", "4Y92"]  # All grid search probabilities

# -----------------------------------------------------------------------------
# cifar10_test_only experiment (execution: 4ZA8) - Evaluation only
# -----------------------------------------------------------------------------
# Uses multirun_quick_weights (3WMG) as input, runs evaluation on test set

# Evaluation results
test_only_evaluation_results = ["4ZBG"]  # evaluation_results.txt

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "assets" must match the parameter name in run_model()

asset_store = store(group="assets")

# REQUIRED: default_asset - used when no assets are specified (typically empty)
asset_store(assets, name="default_asset")

# quick_vs_extended model weights
asset_store(multirun_quick_weights, name="multirun_quick_weights")
asset_store(multirun_extended_weights, name="multirun_extended_weights")
asset_store(multirun_comparison_weights, name="multirun_comparison_weights")

# quick_vs_extended training logs
asset_store(multirun_quick_log, name="multirun_quick_log")
asset_store(multirun_extended_log, name="multirun_extended_log")

# quick_vs_extended prediction probabilities (for ROC analysis)
asset_store(multirun_quick_probabilities, name="multirun_quick_probabilities")
asset_store(multirun_extended_probabilities, name="multirun_extended_probabilities")
asset_store(multirun_comparison_probabilities, name="multirun_comparison_probabilities")

# quick_vs_extended complete asset sets
asset_store(multirun_quick_assets, name="multirun_quick_assets")
asset_store(multirun_extended_assets, name="multirun_extended_assets")

# quick_vs_extended_full model weights
asset_store(full_quick_weights, name="full_quick_weights")
asset_store(full_extended_weights, name="full_extended_weights")
asset_store(full_comparison_weights, name="full_comparison_weights")

# quick_vs_extended_full training logs
asset_store(full_quick_log, name="full_quick_log")
asset_store(full_extended_log, name="full_extended_log")

# quick_vs_extended_full prediction probabilities
asset_store(full_quick_probabilities, name="full_quick_probabilities")
asset_store(full_extended_probabilities, name="full_extended_probabilities")
asset_store(full_comparison_probabilities, name="full_comparison_probabilities")

# lr_sweep model weights
asset_store(lr_sweep_0001_weights, name="lr_sweep_0001_weights")
asset_store(lr_sweep_001_weights, name="lr_sweep_001_weights")
asset_store(lr_sweep_01_weights, name="lr_sweep_01_weights")
asset_store(lr_sweep_1_weights, name="lr_sweep_1_weights")
asset_store(lr_sweep_all_weights, name="lr_sweep_all_weights")

# lr_sweep prediction probabilities
asset_store(lr_sweep_0001_probabilities, name="lr_sweep_0001_probabilities")
asset_store(lr_sweep_001_probabilities, name="lr_sweep_001_probabilities")
asset_store(lr_sweep_01_probabilities, name="lr_sweep_01_probabilities")
asset_store(lr_sweep_1_probabilities, name="lr_sweep_1_probabilities")
asset_store(lr_sweep_all_probabilities, name="lr_sweep_all_probabilities")

# epoch_sweep model weights
asset_store(epoch_sweep_5_weights, name="epoch_sweep_5_weights")
asset_store(epoch_sweep_10_weights, name="epoch_sweep_10_weights")
asset_store(epoch_sweep_25_weights, name="epoch_sweep_25_weights")
asset_store(epoch_sweep_50_weights, name="epoch_sweep_50_weights")
asset_store(epoch_sweep_all_weights, name="epoch_sweep_all_weights")

# epoch_sweep prediction probabilities
asset_store(epoch_sweep_5_probabilities, name="epoch_sweep_5_probabilities")
asset_store(epoch_sweep_10_probabilities, name="epoch_sweep_10_probabilities")
asset_store(epoch_sweep_25_probabilities, name="epoch_sweep_25_probabilities")
asset_store(epoch_sweep_50_probabilities, name="epoch_sweep_50_probabilities")
asset_store(epoch_sweep_all_probabilities, name="epoch_sweep_all_probabilities")

# lr_batch_grid model weights
asset_store(grid_lr001_batch64_weights, name="grid_lr001_batch64_weights")
asset_store(grid_lr001_batch128_weights, name="grid_lr001_batch128_weights")
asset_store(grid_lr01_batch64_weights, name="grid_lr01_batch64_weights")
asset_store(grid_lr01_batch128_weights, name="grid_lr01_batch128_weights")
asset_store(grid_all_weights, name="grid_all_weights")

# lr_batch_grid prediction probabilities
asset_store(grid_lr001_batch64_probabilities, name="grid_lr001_batch64_probabilities")
asset_store(grid_lr001_batch128_probabilities, name="grid_lr001_batch128_probabilities")
asset_store(grid_lr01_batch64_probabilities, name="grid_lr01_batch64_probabilities")
asset_store(grid_lr01_batch128_probabilities, name="grid_lr01_batch128_probabilities")
asset_store(grid_all_probabilities, name="grid_all_probabilities")

# cifar10_test_only evaluation results
asset_store(test_only_evaluation_results, name="test_only_evaluation_results")

# =============================================================================
# ROC Analysis Asset Configurations
# =============================================================================
# Named configurations for ROC analysis notebook

# Quick vs Extended comparison (small dataset)
asset_store(multirun_comparison_probabilities, name="roc_quick_vs_extended")

# Quick vs Extended comparison (full dataset)
asset_store(full_comparison_probabilities, name="roc_full_quick_vs_extended")

# Learning rate sweep comparison
asset_store(lr_sweep_all_probabilities, name="roc_lr_sweep")

# Epoch sweep comparison
asset_store(epoch_sweep_all_probabilities, name="roc_epoch_sweep")

# LR x Batch grid search comparison
asset_store(grid_all_probabilities, name="roc_lr_batch_grid")
