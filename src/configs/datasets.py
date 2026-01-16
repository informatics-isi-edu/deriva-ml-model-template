"""Dataset Configuration.

This module defines dataset configurations for model training and evaluation.

Configuration Group: datasets
-----------------------------
Datasets are specified as lists of DatasetSpecConfig objects, where each object
identifies a dataset by its RID and optionally a version. Multiple datasets can
be combined into a single configuration for training on multiple data sources.

REQUIRED: A configuration named "default_dataset" must be defined.
This is used as the default dataset when no override is specified.

Example usage:
    # Use default dataset
    uv run src/deriva_run.py

    # Use a specific dataset
    uv run src/deriva_run.py datasets=cifar10_training

    # Combine multiple datasets
    datasets_combined = [
        DatasetSpecConfig(rid="ABC1"),
        DatasetSpecConfig(rid="ABC2"),
    ]
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# ---------------------------------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------------------------------
# Configure a list of datasets by specifying the RID and version of each
# dataset that goes into the collection. The group name "datasets" must
# match the parameter name in BaseConfig.

# =============================================================================
# Catalog 45: CIFAR-10 with 10,000 images (localhost)
# =============================================================================

# Full datasets
datasets_training = [DatasetSpecConfig(rid="28D4", version="0.22.0")]  # Training set with 5,000 images
datasets_testing = [DatasetSpecConfig(rid="28DC", version="0.22.0")]  # Testing set with 5,000 images
datasets_split = [DatasetSpecConfig(rid="28CW", version="0.22.0")]  # Split dataset (nested training + testing)
datasets_complete = [DatasetSpecConfig(rid="28CM", version="0.21.0")]  # Complete dataset with 10,000 images

# Small datasets for quick testing
datasets_small_training = [DatasetSpecConfig(rid="28E6", version="0.4.0")]  # Small training set with 500 images
datasets_small_testing = [DatasetSpecConfig(rid="28EE", version="0.4.0")]  # Small testing set with 500 images
datasets_small_split = [DatasetSpecConfig(rid="28DY", version="0.4.0")]  # Small split dataset with 1,000 images

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

# REQUIRED: default_dataset - used when no dataset is specified
datasets_store(datasets_split, name="default_dataset")
