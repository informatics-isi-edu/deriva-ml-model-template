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
asset_store(
    with_description([], "No assets - empty default configuration"),
    name="default_asset",
)

# =============================================================================
# Catalog 67: CIFAR-10 E2E Test Assets (localhost, schema: cifar10_e2e_test)
# =============================================================================

# -----------------------------------------------------------------------------
# quick_vs_extended multirun (parent: K00) - Small dataset comparison
# -----------------------------------------------------------------------------
# Compares quick training (3 epochs) vs extended training (50 epochs)
# on small labeled dataset (500 images)

asset_store(
    with_description(
        ["K24"],
        "Model weights (cifar10_cnn_weights.pt) from cifar10_quick: "
        "3 epochs, 32→64 channels, batch 128. Source: execution K12.",
    ),
    name="multirun_quick_weights",
)

asset_store(
    with_description(
        ["K4G"],
        "Model weights (cifar10_cnn_weights.pt) from cifar10_extended: "
        "50 epochs, 64→128 channels, dropout 0.25, weight decay 1e-4. Source: execution K38.",
    ),
    name="multirun_extended_weights",
)

asset_store(
    with_description(
        ["K24", "K4G"],
        "Both model weights from quick_vs_extended multirun for comparison analysis.",
    ),
    name="multirun_comparison_weights",
)

# -----------------------------------------------------------------------------
# lr_sweep multirun (parent: K6C) - Learning rate hyperparameter sweep
# -----------------------------------------------------------------------------
# Tests learning rates: 0.0001, 0.001, 0.01, 0.1
# All use: 10 epochs, 32→64 channels, batch 128

asset_store(
    with_description(
        ["K8J"],
        "Model weights from lr=0.0001 experiment. Source: execution K7G.",
    ),
    name="lr_sweep_0001_weights",
)

asset_store(
    with_description(
        ["KAY"],
        "Model weights from lr=0.001 experiment (default lr). Source: execution K9P.",
    ),
    name="lr_sweep_001_weights",
)

asset_store(
    with_description(
        ["KDA"],
        "Model weights from lr=0.01 experiment. Source: execution KC2.",
    ),
    name="lr_sweep_01_weights",
)

asset_store(
    with_description(
        ["KFP"],
        "Model weights from lr=0.1 experiment (high lr). Source: execution KEE.",
    ),
    name="lr_sweep_1_weights",
)

asset_store(
    with_description(
        ["K8J", "KAY", "KDA", "KFP"],
        "All model weights from learning rate sweep (lr=0.0001, 0.001, 0.01, 0.1). "
        "Use for comparing effect of learning rate on model performance.",
    ),
    name="lr_sweep_all_weights",
)

# -----------------------------------------------------------------------------
# epoch_sweep multirun (parent: KHJ) - Training duration sweep
# -----------------------------------------------------------------------------
# Tests epochs: 5, 10, 25, 50
# All use: 64→128 channels, 256 hidden, dropout 0.25, weight decay 1e-4

asset_store(
    with_description(
        ["KKR"],
        "Model weights from epochs=5 experiment. Source: execution KJP.",
    ),
    name="epoch_sweep_5_weights",
)

asset_store(
    with_description(
        ["KP4"],
        "Model weights from epochs=10 experiment. Source: execution KMW.",
    ),
    name="epoch_sweep_10_weights",
)

asset_store(
    with_description(
        ["KRG"],
        "Model weights from epochs=25 experiment. Source: execution KQ8.",
    ),
    name="epoch_sweep_25_weights",
)

asset_store(
    with_description(
        ["KTW"],
        "Model weights from epochs=50 experiment. Source: execution KSM.",
    ),
    name="epoch_sweep_50_weights",
)

asset_store(
    with_description(
        ["KKR", "KP4", "KRG", "KTW"],
        "All model weights from epoch sweep (5, 10, 25, 50 epochs). "
        "Use for comparing effect of training duration on model quality.",
    ),
    name="epoch_sweep_all_weights",
)

# -----------------------------------------------------------------------------
# lr_batch_grid multirun (parent: KWR) - Learning rate and batch size grid search
# -----------------------------------------------------------------------------
# 2x2 grid: lr in [0.001, 0.01], batch_size in [64, 128]
# All use: 10 epochs, 32→64 channels

asset_store(
    with_description(
        ["KYY"],
        "Model weights from lr=0.001, batch=64 experiment. Source: execution KXW.",
    ),
    name="grid_lr001_batch64_weights",
)

asset_store(
    with_description(
        ["M1A"],
        "Model weights from lr=0.001, batch=128 experiment. Source: execution M02.",
    ),
    name="grid_lr001_batch128_weights",
)

asset_store(
    with_description(
        ["M3P"],
        "Model weights from lr=0.01, batch=64 experiment. Source: execution M2E.",
    ),
    name="grid_lr01_batch64_weights",
)

asset_store(
    with_description(
        ["M62"],
        "Model weights from lr=0.01, batch=128 experiment. Source: execution M4T.",
    ),
    name="grid_lr01_batch128_weights",
)

asset_store(
    with_description(
        ["KYY", "M1A", "M3P", "M62"],
        "All model weights from LR x batch size grid search (2x2 grid). "
        "Use for analyzing interaction between learning rate and batch size.",
    ),
    name="grid_all_weights",
)

# -----------------------------------------------------------------------------
# Test-only mode assets
# -----------------------------------------------------------------------------
# For running inference-only with pre-trained weights

asset_store(
    with_description(
        ["K24"],
        "Pre-trained weights for test-only mode (from cifar10_quick). "
        "Use with cifar10_test_only experiment for inference.",
    ),
    name="e2e_test_weights",
)
