"""This module defines configurations for the model assets.

In DerivaML all of the additional files that may be needed beyond the dataset are  stored in a set of
execution asset files which are specified as part of an ExecutionConfiguration.
DerivaML will automatically download all of the specified assets as part of initializing the execution.

Typically, the execution assets will at least contain the file with the model weights in it.
"""
from hydra_zen import store
from deriva_ml.execution import AssetRIDConfig

# Define two model assets by providing the RID of the asset.
assets_1 = [AssetRIDConfig("3RA"), AssetRIDConfig("3R8")]  # RID for hyperparamets json file.....
assets_2 = [AssetRIDConfig("3R6"), AssetRIDConfig("3R4")]

# Store the configurations in hydra-zen store.
# Note that the name of the group in the store needs to match the name of the argument in the main function
# that will be instantiated to the configuration value.

asset_store = store(group="assets")
asset_store(assets_1, items=assets_1, name="default_asset")
asset_store(assets_2, items=assets_2, name="weights_2")
