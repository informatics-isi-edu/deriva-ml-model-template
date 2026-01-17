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
        DatasetSpecConfig(rid="ABC1", version="1.0.0"),
        DatasetSpecConfig(rid="ABC2", version="1.0.0"),
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
# Catalog 62: CIFAR-10 with 10,000 images (localhost, schema: cifar-10)
# =============================================================================

# Full datasets
datasets_complete = [DatasetSpecConfig(rid="28CT", version="0.22.0")]  # Complete dataset with 10,000 images
datasets_split = [DatasetSpecConfig(rid="28D4", version="0.22.0")]  # Split dataset (nested training + testing)
datasets_training = [DatasetSpecConfig(rid="28DC", version="0.22.0")]  # Training set with 5,000 images
datasets_testing = [DatasetSpecConfig(rid="28DP", version="0.22.0")]  # Testing set with 5,000 images (unlabeled)

# Small datasets for quick testing
datasets_small_split = [DatasetSpecConfig(rid="28EA", version="0.22.0")]  # Small split dataset with 1,000 images
datasets_small_training = [DatasetSpecConfig(rid="28EJ", version="0.22.0")]  # Small training set with 500 images
datasets_small_testing = [DatasetSpecConfig(rid="28EW", version="0.22.0")]  # Small testing set with 500 images

# Labeled split dataset - created from training images only (all have ground truth)
# This enables ROC analysis since both train and test partitions have labels
datasets_labeled_split = [DatasetSpecConfig(rid="28FG", version="0.22.0")]  # 5000 images: 4000 train + 1000 test
datasets_labeled_training = [DatasetSpecConfig(rid="28FT", version="0.22.0")]  # 4000 labeled training images
datasets_labeled_testing = [DatasetSpecConfig(rid="28G4", version="0.22.0")]  # 1000 labeled test images (with ground truth!)

# Small labeled datasets
datasets_small_labeled_split = [DatasetSpecConfig(rid="28GR", version="0.22.0")]  # 500 images: 400 train + 100 test
datasets_small_labeled_training = [DatasetSpecConfig(rid="28H2", version="0.22.0")]  # 400 labeled training images
datasets_small_labeled_testing = [DatasetSpecConfig(rid="28HC", version="0.22.0")]  # 100 labeled test images

# =============================================================================
# Store configurations in hydra-zen
# =============================================================================

datasets_store = store(group="datasets")

# Full datasets
datasets_store(datasets_complete, name="cifar10_complete")
datasets_store(datasets_split, name="cifar10_split")
datasets_store(datasets_training, name="cifar10_training")
datasets_store(datasets_testing, name="cifar10_testing")

# Small datasets for quick testing
datasets_store(datasets_small_split, name="cifar10_small_split")
datasets_store(datasets_small_training, name="cifar10_small_training")
datasets_store(datasets_small_testing, name="cifar10_small_testing")

# Labeled split dataset (all images have ground truth - suitable for ROC analysis)
datasets_store(datasets_labeled_split, name="cifar10_labeled_split")
datasets_store(datasets_labeled_training, name="cifar10_labeled_training")
datasets_store(datasets_labeled_testing, name="cifar10_labeled_testing")

# Small labeled datasets
datasets_store(datasets_small_labeled_split, name="cifar10_small_labeled_split")
datasets_store(datasets_small_labeled_training, name="cifar10_small_labeled_training")
datasets_store(datasets_small_labeled_testing, name="cifar10_small_labeled_testing")

# REQUIRED: default_dataset - used when no dataset is specified
datasets_store(datasets_split, name="default_dataset")
