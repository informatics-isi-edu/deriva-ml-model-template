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
# Catalog 2: CIFAR-10 E2E Test Assets (localhost, schema: cifar10_e2e_test)
# =============================================================================

# -----------------------------------------------------------------------------
# quick_vs_extended multirun (parent: JY4)
# -----------------------------------------------------------------------------
# Compares quick training (3 epochs) vs extended training (50 epochs)
# on small labeled dataset (500 images)

asset_store(
    with_description(
        ["K0A"],
        "Model weights (cifar10_cnn_weights.pt) from cifar10_quick: "
        "3 epochs, 32->64 channels, batch 128. Source: execution JZ8.",
    ),
    name="multirun_quick_weights",
)

asset_store(
    with_description(
        ["K94"],
        "Model weights (cifar10_cnn_weights.pt) from cifar10_extended: "
        "50 epochs, 64->128 channels, dropout 0.25, weight decay 1e-4. Source: execution K7W.",
    ),
    name="multirun_extended_weights",
)

asset_store(
    with_description(
        ["K0A", "K94"],
        "Both model weights from quick_vs_extended multirun for comparison analysis.",
    ),
    name="multirun_comparison_weights",
)

# Prediction probabilities for ROC analysis
# Note: Using plain list for notebook config compatibility. Description is in the
# notebook config or can be accessed via roc_quick_vs_extended_described.
asset_store(
    ["K0E", "K98"],
    name="roc_quick_vs_extended",
)
# With description for non-notebook uses
asset_store(
    with_description(
        ["K0E", "K98"],
        "Prediction probability files from quick_vs_extended multirun. "
        "Contains class probabilities for 100 labeled test images from each run.",
    ),
    name="roc_quick_vs_extended_described",
)

# -----------------------------------------------------------------------------
# lr_sweep multirun (parent: KHE) - Learning rate hyperparameter sweep
# -----------------------------------------------------------------------------
# Tests learning rates: 0.0001, 0.001, 0.01, 0.1
# All use: 10 epochs, 32->64 channels, batch 128

asset_store(
    with_description(
        ["KKM"],
        "Model weights from lr=0.0001 experiment. Source: execution KJJ.",
    ),
    name="lr_sweep_0001_weights",
)

asset_store(
    with_description(
        ["KWE"],
        "Model weights from lr=0.001 experiment (default lr). Source: execution KV6.",
    ),
    name="lr_sweep_001_weights",
)

asset_store(
    with_description(
        ["M58"],
        "Model weights from lr=0.01 experiment. Source: execution M40.",
    ),
    name="lr_sweep_01_weights",
)

asset_store(
    with_description(
        ["ME2"],
        "Model weights from lr=0.1 experiment (high lr). Source: execution MCT.",
    ),
    name="lr_sweep_1_weights",
)

asset_store(
    with_description(
        ["KKM", "KWE", "M58", "ME2"],
        "All model weights from learning rate sweep (lr=0.0001, 0.001, 0.01, 0.1). "
        "Use for comparing effect of learning rate on model performance.",
    ),
    name="lr_sweep_all_weights",
)

# Prediction probabilities for ROC analysis
# Note: Using plain list for notebook config compatibility
asset_store(
    ["KKR", "KWJ", "M5C", "ME6"],
    name="roc_lr_sweep",
)
asset_store(
    with_description(
        ["KKR", "KWJ", "M5C", "ME6"],
        "Prediction probability files from lr_sweep multirun. "
        "lr=0.0001 (KKR), lr=0.001 (KWJ), lr=0.01 (M5C), lr=0.1 (ME6).",
    ),
    name="roc_lr_sweep_described",
)

# -----------------------------------------------------------------------------
# epoch_sweep multirun (parent: MPC) - Training duration sweep
# -----------------------------------------------------------------------------
# Tests epochs: 5, 10, 25, 50
# All use: 64->128 channels, 256 hidden, dropout 0.25, weight decay 1e-4

asset_store(
    with_description(
        ["MRJ"],
        "Model weights from epochs=5 experiment. Source: execution MQG.",
    ),
    name="epoch_sweep_5_weights",
)

asset_store(
    with_description(
        ["N1C"],
        "Model weights from epochs=10 experiment. Source: execution N04.",
    ),
    name="epoch_sweep_10_weights",
)

asset_store(
    with_description(
        ["NA6"],
        "Model weights from epochs=25 experiment. Source: execution N8Y.",
    ),
    name="epoch_sweep_25_weights",
)

asset_store(
    with_description(
        ["NK0"],
        "Model weights from epochs=50 experiment. Source: execution NHR.",
    ),
    name="epoch_sweep_50_weights",
)

asset_store(
    with_description(
        ["MRJ", "N1C", "NA6", "NK0"],
        "All model weights from epoch sweep (5, 10, 25, 50 epochs). "
        "Use for comparing effect of training duration on model quality.",
    ),
    name="epoch_sweep_all_weights",
)

# Prediction probabilities for ROC analysis
# Note: Using plain list for notebook config compatibility
asset_store(
    ["MRP", "N1G", "NAA", "NK4"],
    name="roc_epoch_sweep",
)
asset_store(
    with_description(
        ["MRP", "N1G", "NAA", "NK4"],
        "Prediction probability files from epoch_sweep multirun. "
        "5 epochs (MRP), 10 epochs (N1G), 25 epochs (NAA), 50 epochs (NK4).",
    ),
    name="roc_epoch_sweep_described",
)

# -----------------------------------------------------------------------------
# lr_batch_grid multirun (parent: NVA) - Learning rate and batch size grid search
# -----------------------------------------------------------------------------
# 2x2 grid: lr in [0.001, 0.01], batch_size in [64, 128]
# All use: 10 epochs, 32->64 channels

asset_store(
    with_description(
        ["NXG"],
        "Model weights from lr=0.001, batch=64 experiment. Source: execution NWE.",
    ),
    name="grid_lr001_batch64_weights",
)

asset_store(
    with_description(
        ["P6A"],
        "Model weights from lr=0.001, batch=128 experiment. Source: execution P52.",
    ),
    name="grid_lr001_batch128_weights",
)

asset_store(
    with_description(
        ["PF4"],
        "Model weights from lr=0.01, batch=64 experiment. Source: execution PDW.",
    ),
    name="grid_lr01_batch64_weights",
)

asset_store(
    with_description(
        ["PQY"],
        "Model weights from lr=0.01, batch=128 experiment. Source: execution PPP.",
    ),
    name="grid_lr01_batch128_weights",
)

asset_store(
    with_description(
        ["NXG", "P6A", "PF4", "PQY"],
        "All model weights from LR x batch size grid search (2x2 grid). "
        "Use for analyzing interaction between learning rate and batch size.",
    ),
    name="grid_all_weights",
)

# Prediction probabilities for ROC analysis
# Note: Using plain list for notebook config compatibility
asset_store(
    ["NXM", "P6E", "PF8", "PR2"],
    name="roc_lr_batch_grid",
)
asset_store(
    with_description(
        ["NXM", "P6E", "PF8", "PR2"],
        "Prediction probability files from lr_batch_grid multirun. "
        "lr=0.001/batch=64 (NXM), lr=0.001/batch=128 (P6E), "
        "lr=0.01/batch=64 (PF8), lr=0.01/batch=128 (PR2).",
    ),
    name="roc_lr_batch_grid_described",
)

# -----------------------------------------------------------------------------
# Test-only mode assets
# -----------------------------------------------------------------------------
# For running inference-only with pre-trained weights

asset_store(
    with_description(
        ["K0A"],
        "Pre-trained weights for test-only mode (from cifar10_quick). "
        "Use with cifar10_test_only experiment for inference.",
    ),
    name="e2e_test_weights",
)
