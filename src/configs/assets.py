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
# Catalog 2: CIFAR-10 E2E Test Assets (localhost, schema: cifar10_e2e_test)
# =============================================================================
# Asset configurations will be populated after running experiments in Phase 2.
# Each multirun experiment generates model weights and prediction probability
# files that can be referenced here for subsequent analysis.
#
# After running experiments, update this file with the generated asset RIDs
# for use in test-only mode and ROC analysis notebooks.
