"""
This module defines configurations for dataset collections that can be used in different model runs.
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Configure a list of datasets by specifying the RID and version of each dataset that goes into the collection.

# =============================================================================
# Catalog 37: CIFAR-10 with 10,000 images (localhost)
# =============================================================================

# Full datasets
datasets_training = [DatasetSpecConfig(rid="28AW", version="0.22.0")]  # Training set with 5,000 images
datasets_testing = [DatasetSpecConfig(rid="28B4", version="0.22.0")]  # Testing set with 5,000 images
datasets_split = [DatasetSpecConfig(rid="28AM", version="0.22.0")]  # Split dataset (nested training + testing)
datasets_complete = [DatasetSpecConfig(rid="28AC", version="0.21.0")]  # Complete dataset with 10,000 images

# Small datasets for quick testing
datasets_small_training = [DatasetSpecConfig(rid="28BY", version="0.4.0")]  # Small training set with 500 images
datasets_small_testing = [DatasetSpecConfig(rid="28C6", version="0.4.0")]  # Small testing set with 500 images
datasets_small_split = [DatasetSpecConfig(rid="28BP", version="0.4.0")]  # Small split dataset with 1,000 images

# =============================================================================
# Store configurations in hydra-zen
# =============================================================================

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

# Default dataset
datasets_store(datasets_split, name="default_dataset")
