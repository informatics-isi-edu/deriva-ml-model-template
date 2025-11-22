"""This module defines configurations for the model assets.

In DerivaML a model is typically stored in a file asset that is passed to an execution.
"""
from hydra_zen import store

# Define two model assets by providing the RID of the asset.
assets_test1 = ["3QG", "3QJ"]
assets_test2 = ["3QM", "3QP"]

# Store the configurations in hydra-zen store.
# Note that the name of the group in the store needs to match the name of the argument in the main function
# that will be instantiated to the configuration value.
# These asset lists will be used in the execution configuration.
asset_store = store(group="assets")
asset_store(assets_test1, name="model1")
asset_store(assets_test2, name="model2")
