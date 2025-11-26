"""
This module defines configurations for dataset collections that can be used in different model runs.
"""
from hydra_zen import store
from deriva_ml.dataset import  DatasetSpecConfig

# Configure a list of datasets by specifying the RID and version of each dataset that goes into the collection.
datasets_test1 = [DatasetSpecConfig(rid="2-7K8W", version="4.6.0")]
datasets_test2 = [DatasetSpecConfig(rid="2-7KA2", version="2.6.0")]
datasets_test3 = []

# Create three configurations and store them into hydra-zen store.
# Note that the name of the group has to match the name of the argument in the main function that will be
# instantiated to the configuration value.
datasets_store = store(group="datasets")
datasets_store(datasets_test1, name="test1")
datasets_store(datasets_test2, name="test2")
datasets_store(datasets_test3, name="test3")
