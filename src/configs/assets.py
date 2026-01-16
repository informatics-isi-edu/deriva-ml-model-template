"""
Asset Configuration Module
==========================

Defines configurations for execution assets (model weights, checkpoints, etc.).

In DerivaML, assets are additional files needed beyond the dataset. They are
specified as Resource IDs (RIDs) and automatically downloaded when the
execution is initialized.

Typical assets include:
- Pre-trained model weights
- Model checkpoints
- Configuration files
- Reference data files

Configuration Format
--------------------
Assets are specified as a simple list of RID strings::

    my_assets = ["3RA", "3R8"]
    asset_store(my_assets, name="my_asset_config")

The RIDs reference files stored in the Deriva catalog's asset table.
"""

from hydra_zen import store

# ---------------------------------------------------------------------------
# Asset Configurations
# ---------------------------------------------------------------------------
# Define asset sets as lists of RID strings. Each RID references a file
# in the Deriva catalog that will be downloaded for the execution.

assets = []

# Model weights from multirun comparison (catalog 45)
# Parent execution: 3JQ8
# - cifar10_quick (3JRC): 3 epochs, 32→64 channels
# - cifar10_extended (3KT0): 50 epochs, 64→128 channels
multirun_quick_weights = ["3JSE"]
multirun_extended_weights = ["3KV8"]
multirun_comparison_weights = ["3JSE", "3KV8"]

# CIFAR-10 CNN model weights and training log from small dataset experiment (catalog 45)
# Uses weights from multirun cifar10_quick experiment
cifar10_small_experiment_weights = ["3JSE"]
cifar10_small_experiment_log = ["3JSG"]
cifar10_small_experiment_assets = ["3JSE", "3JSG"]

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "assets" must match the parameter name in run_model()

asset_store = store(group="assets")
asset_store(assets, name="default_asset")
asset_store(multirun_quick_weights, name="multirun_quick_weights")
asset_store(multirun_extended_weights, name="multirun_extended_weights")
asset_store(multirun_comparison_weights, name="multirun_comparison_weights")
asset_store(cifar10_small_experiment_weights, name="cifar10_small_experiment_weights")
asset_store(cifar10_small_experiment_log, name="cifar10_small_experiment_log")
asset_store(cifar10_small_experiment_assets, name="cifar10_small_experiment_assets")
