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
    uv run deriva-ml-run assets=roc_quick_vs_extended

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
# Catalog 3: CIFAR-10 Assets (localhost, schema: cifar10)
# =============================================================================

# -----------------------------------------------------------------------------
# quick_vs_extended multirun (parent: 3WPM)
# -----------------------------------------------------------------------------
# Compares quick training (3 epochs) vs extended training (50 epochs)
# on small labeled dataset (500 images)

# Prediction probabilities for ROC analysis
# 3WRT: cifar10_quick (exec 3WQM), 3X1G: cifar10_extended (exec 3X08)
asset_store(
    ["3WRT", "3X1G"],
    name="roc_quick_vs_extended",
)

# Pre-trained weights from cifar10_quick (execution 3WQM)
asset_store(
    with_description(
        ["3WRP"],
        "Pre-trained weights from cifar10_quick (execution 3WQM, 3 epochs). "
        "Use with cifar10_test_only experiment for inference.",
    ),
    name="quick_weights",
)
