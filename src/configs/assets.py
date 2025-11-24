"""This module defines configurations for the model assets.

In DerivaML all of the additional files that may be needed beyond the dataset are  stored in a set of
execution asset files which are specified as part of an ExecutionConfiguration.  DerivaML will automatically download
all of the specified assets as part of initializing the execution.

Typically, the execution assets will at least contain the file with the model weights in it.
"""
from hydra_zen import store, just

from deriva_ml import RID
# Define two model assets by providing the RID of the asset.
assets_1 = ["3RA", "3R8"]
assets_2 = ["3R6", "3R4"]

# Store the configurations in hydra-zen store.
# Note that the name of the group in the store needs to match the name of the argument in the main function
# that will be instantiated to the configuration value.

asset_store = store(group="assets")
asset_store(just(assets_1), name="weights_1")
asset_store(just(assets_2), name="weights_2")
