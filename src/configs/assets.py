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

# CIFAR-10 CNN model weights trained on small dataset (1,000 images, 3 epochs)
cifar10_cnn_weights = ["3NKR"]

# CIFAR-10 CNN model weights from execution 3NMW (quick model, small training dataset)
cifar10_quick_weights = ["3NP0"]

# Model weights from quick and extended experiments on small dataset
# Execution 3NQ4 (cifar10_quick) and 3NSE (cifar10_extended)
my_experiment_assets = ["3NRA", "3NTJ"]

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "assets" must match the parameter name in run_model()

asset_store = store(group="assets")
asset_store(assets, name="default_asset")
asset_store(cifar10_cnn_weights, name="cifar10_cnn_weights")
asset_store(cifar10_quick_weights, name="cifar10_quick_weights")
asset_store(my_experiment_assets, name="my_experiment_assets")
