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
    uv run deriva-ml-run

    # Use specific assets
    uv run deriva-ml-run assets=multirun_quick_weights

Configuration Format:
    asset_store(
        with_description(
            ["RID1", "RID2"],
            "Description of what these assets are for",
        ),
        name="my_asset_config",
    )
"""

from hydra_zen import store
from deriva_ml.execution import with_description

# ---------------------------------------------------------------------------
# Asset Store
# ---------------------------------------------------------------------------
asset_store = store(group="assets")

# REQUIRED: default_asset - used when no assets are specified (typically empty)
# Note: Using plain list (not with_description) because this is used as a merge base
# in notebook configs. with_description creates a DictConfig which can't merge with
# the ListConfig default in BaseConfig.
asset_store(
    [],
    name="default_asset",
)

# Alias for clarity in notebook configs
asset_store(
    [],
    name="no_assets",
)

# =============================================================================
# Catalog 2: CIFAR-10 E2E Test Assets (localhost, schema: cifar10)
# =============================================================================

# -----------------------------------------------------------------------------
# quick_vs_extended multirun (parent: JY2)
# -----------------------------------------------------------------------------
# Compares quick training (3 epochs) vs extended training (50 epochs)
# on small labeled dataset (500 images)

# Prediction probabilities for ROC analysis
# Note: Using plain list for notebook config compatibility
asset_store(
    ["K0C", "K96"],
    name="roc_quick_vs_extended",
)

# -----------------------------------------------------------------------------
# lr_sweep multirun (parent: KHC) - Learning rate hyperparameter sweep
# -----------------------------------------------------------------------------
# Tests learning rates: 0.0001, 0.001, 0.01, 0.1
# All use: 10 epochs, 32->64 channels, batch 128

# Prediction probabilities for ROC analysis
# KKP: lr=0.0001 (exec KJG), KWG: lr=0.001 (exec KV4),
# M5A: lr=0.01 (exec M3Y), ME4: lr=0.1 (exec MCR)
asset_store(
    ["KKP", "KWG", "M5A", "ME4"],
    name="roc_lr_sweep",
)

# -----------------------------------------------------------------------------
# epoch_sweep multirun (parent: MPA) - Training duration sweep
# -----------------------------------------------------------------------------
# Tests epochs: 5, 10, 25, 50
# All use: 64->128 channels, 256 hidden, dropout 0.25, weight decay 1e-4

# Prediction probabilities for ROC analysis
# MRM: 5 epochs (exec MQE), N1E: 10 epochs (exec N02),
# NA8: 25 epochs (exec N8W), NK2: 50 epochs (exec NHP)
asset_store(
    ["MRM", "N1E", "NA8", "NK2"],
    name="roc_epoch_sweep",
)

# -----------------------------------------------------------------------------
# lr_batch_grid multirun (parent: NV8) - Learning rate and batch size grid search
# -----------------------------------------------------------------------------
# 2x2 grid: lr in [0.001, 0.01], batch_size in [64, 128]
# All use: 10 epochs, 32->64 channels

# Prediction probabilities for ROC analysis
# NXJ: lr=0.001/batch=64 (exec NWC), P6C: lr=0.001/batch=128 (exec P50),
# PF6: lr=0.01/batch=64 (exec PDT), PR0: lr=0.01/batch=128 (exec PPM)
asset_store(
    ["NXJ", "P6C", "PF6", "PR0"],
    name="roc_lr_batch_grid",
)

# -----------------------------------------------------------------------------
# Test-only mode assets
# -----------------------------------------------------------------------------
# For running inference-only with pre-trained weights

asset_store(
    with_description(
        ["K08"],
        "Pre-trained weights for test-only mode (from cifar10_quick, execution JZ6). "
        "Use with cifar10_test_only experiment for inference.",
    ),
    name="e2e_test_weights",
)
