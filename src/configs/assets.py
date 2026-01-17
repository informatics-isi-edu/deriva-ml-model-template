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
# Catalog 62: CIFAR-10 Multi-Experiment Assets (localhost, schema: cifar-10)
# =============================================================================
# Parent execution: 3WMA (multirun sweep)
# Child executions:
#   - cifar10_quick (3WNE): 3 epochs, 32→64 channels, batch 128
#   - cifar10_extended (3XQ2): 50 epochs, 64→128 channels, dropout 0.25, weight decay 1e-4

# Model weights from multirun comparison
multirun_quick_weights = ["3WPG"]  # cifar10_cnn_weights.pt from cifar10_quick
multirun_extended_weights = ["3XRA"]  # cifar10_cnn_weights.pt from cifar10_extended
multirun_comparison_weights = ["3WPG", "3XRA"]  # Both model weights

# Training logs
multirun_quick_log = ["3WPJ"]  # training_log.txt from cifar10_quick
multirun_extended_log = ["3XRC"]  # training_log.txt from cifar10_extended

# Prediction probability files (for ROC analysis)
multirun_quick_probabilities = ["3WPM"]  # prediction_probabilities.csv from cifar10_quick
multirun_extended_probabilities = ["3XRE"]  # prediction_probabilities.csv from cifar10_extended
multirun_comparison_probabilities = ["3WPM", "3XRE"]  # Both probability files

# Complete asset sets (weights + config)
multirun_quick_assets = ["3WPG", "3WMM"]  # weights + hydra config
multirun_extended_assets = ["3XRA", "3XQ8"]  # weights + hydra config

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "assets" must match the parameter name in run_model()

asset_store = store(group="assets")

# REQUIRED: default_asset - used when no assets are specified (typically empty)
asset_store(assets, name="default_asset")

# Model weights
asset_store(multirun_quick_weights, name="multirun_quick_weights")
asset_store(multirun_extended_weights, name="multirun_extended_weights")
asset_store(multirun_comparison_weights, name="multirun_comparison_weights")

# Training logs
asset_store(multirun_quick_log, name="multirun_quick_log")
asset_store(multirun_extended_log, name="multirun_extended_log")

# Prediction probabilities (for ROC analysis)
asset_store(multirun_quick_probabilities, name="multirun_quick_probabilities")
asset_store(multirun_extended_probabilities, name="multirun_extended_probabilities")
asset_store(multirun_comparison_probabilities, name="multirun_comparison_probabilities")

# Complete asset sets
asset_store(multirun_quick_assets, name="multirun_quick_assets")
asset_store(multirun_extended_assets, name="multirun_extended_assets")
