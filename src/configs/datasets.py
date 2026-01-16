"""
This module defines configurations for dataset collections that can be used in different model runs.
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Configure a list of datasets by specifying the RID and version of each dataset that goes into the collection.

# =============================================================================
# Catalog 37: CIFAR-10 with 10,000 images (localhost)
# =============================================================================

# Full datasets (catalog 37)
datasets_37_training = [DatasetSpecConfig(rid="28AW", version="0.22.0")]  # Training set with 5,000 images
datasets_37_testing = [DatasetSpecConfig(rid="28B4", version="0.22.0")]  # Testing set with 5,000 images
datasets_37_split = [DatasetSpecConfig(rid="28AM", version="0.22.0")]  # Split dataset (nested training + testing)
datasets_37_complete = [DatasetSpecConfig(rid="28AC", version="0.21.0")]  # Complete dataset with 10,000 images

# Small datasets for quick testing (catalog 37)
datasets_37_small_training = [DatasetSpecConfig(rid="28BY", version="0.4.0")]  # Small training set with 500 images
datasets_37_small_testing = [DatasetSpecConfig(rid="28C6", version="0.4.0")]  # Small testing set with 500 images
datasets_37_small_split = [DatasetSpecConfig(rid="28BP", version="0.4.0")]  # Small split dataset with 1,000 images

# =============================================================================
# Catalog 25: CIFAR-10 with ~60,000 images (localhost)
# =============================================================================

# Full datasets (catalog 25)
datasets_25_training = [DatasetSpecConfig(rid="3XT", version="0.5.0")]  # Training set with 50,000 images
datasets_25_testing = [DatasetSpecConfig(rid="3Y2", version="0.5.0")]  # Testing set with 10,000 images
datasets_25_split = [DatasetSpecConfig(rid="3XJ", version="0.5.0")]  # Split dataset (nested training + testing)
datasets_25_complete = [DatasetSpecConfig(rid="3XA", version="0.3.0")]  # Complete dataset with 60,000 images

# Small datasets for quick testing (catalog 25)
datasets_25_small_training = [DatasetSpecConfig(rid="3YW", version="0.5.0")]  # Small training set with 500 images
datasets_25_small_testing = [DatasetSpecConfig(rid="3Z4", version="0.5.0")]  # Small testing set with 500 images
datasets_25_small_split = [DatasetSpecConfig(rid="3YM", version="0.5.0")]  # Small split dataset with 1,000 images

# =============================================================================
# Store configurations in hydra-zen
# =============================================================================

datasets_store = store(group="datasets")

# Catalog 37 datasets (10k images) - use these with deriva_ml.catalog_id=37
datasets_store(datasets_37_training, name="cifar10_37_training")
datasets_store(datasets_37_testing, name="cifar10_37_testing")
datasets_store(datasets_37_split, name="cifar10_37_split")
datasets_store(datasets_37_complete, name="cifar10_37_complete")
datasets_store(datasets_37_small_training, name="cifar10_37_small_training")
datasets_store(datasets_37_small_testing, name="cifar10_37_small_testing")
datasets_store(datasets_37_small_split, name="cifar10_37_small_split")

# Catalog 25 datasets (60k images) - use these with deriva_ml.catalog_id=25
datasets_store(datasets_25_training, name="cifar10_25_training")
datasets_store(datasets_25_testing, name="cifar10_25_testing")
datasets_store(datasets_25_split, name="cifar10_25_split")
datasets_store(datasets_25_complete, name="cifar10_25_complete")
datasets_store(datasets_25_small_training, name="cifar10_25_small_training")
datasets_store(datasets_25_small_testing, name="cifar10_25_small_testing")
datasets_store(datasets_25_small_split, name="cifar10_25_small_split")

# Aliases for backward compatibility (point to catalog 37 as default)
datasets_store(datasets_37_training, name="cifar10_training")
datasets_store(datasets_37_testing, name="cifar10_testing")
datasets_store(datasets_37_split, name="cifar10_split")
datasets_store(datasets_37_complete, name="cifar10_complete")
datasets_store(datasets_37_small_training, name="cifar10_small_training")
datasets_store(datasets_37_small_testing, name="cifar10_small_testing")
datasets_store(datasets_37_small_split, name="cifar10_small_split")

# Default dataset
datasets_store(datasets_37_split, name="default_dataset")
