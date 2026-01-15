"""
This module defines configurations for dataset collections that can be used in different model runs.
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Configure a list of datasets by specifying the RID and version of each dataset that goes into the collection.

# CIFAR-10 Full datasets (localhost catalog 25)
datasets_training = [DatasetSpecConfig(rid="3XT", version="0.4.0")]  # Training set with 50,000 images
datasets_testing = [DatasetSpecConfig(rid="3Y2", version="0.4.0")]  # Testing set
datasets_split = [DatasetSpecConfig(rid="3XJ", version="0.4.0")]  # Split dataset (nested training + testing)
datasets_complete = [DatasetSpecConfig(rid="3XA", version="0.3.0")]  # Complete dataset

# CIFAR-10 Small datasets for quick testing
datasets_small_training = [DatasetSpecConfig(rid="3YW", version="0.4.0")]  # Small training set with 500 images
datasets_small_testing = [DatasetSpecConfig(rid="3Z4", version="0.4.0")]  # Small testing set with 500 images
datasets_small_split = [DatasetSpecConfig(rid="3YM", version="0.4.0")]  # Small split dataset with 1,000 images

# Create configurations and store them into hydra-zen store.
# Note that the name of the group has to match the name of the argument in the main function that will be
# instantiated tFo the configuration value.
datasets_store = store(group="datasets")

# Full datasets
datasets_store(datasets_training, name="cifar10_training")
datasets_store(datasets_testing, name="cifar10_testing")
datasets_store(datasets_split, name="cifar10_split")
datasets_store(datasets_complete, name="cifar10_complete")

# Small datasets for quick testing
datasets_store(datasets_small_training, name="cifar10_small_training")
datasets_store(datasets_small_testing, name="cifar10_small_testing")
datasets_store(datasets_small_split, name="cifar10_small_split")

# Default
datasets_store(datasets_split, name="default_dataset")

