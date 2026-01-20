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
# NOTE: Asset configurations will be populated after running multirun experiments.
# Run the experiments first, then use MCP tools to find asset RIDs:
#   mcp__deriva-ml__find_assets(pattern="model_weights", limit=20)
#   mcp__deriva-ml__find_assets(pattern="prediction_probabilities", limit=20)

# -----------------------------------------------------------------------------
# Placeholder configurations - to be populated after Phase 2 experiments
# -----------------------------------------------------------------------------

# ROC analysis asset placeholders (prediction probability files)
# These will be updated in Phase 4 after multirun experiments complete

asset_store(
    [],
    name="roc_quick_vs_extended",
)

asset_store(
    [],
    name="roc_lr_sweep",
)

asset_store(
    [],
    name="roc_epoch_sweep",
)

asset_store(
    [],
    name="roc_lr_batch_grid",
)

# Test-only mode placeholder
asset_store(
    [],
    name="e2e_test_weights",
)
