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
from deriva_ml.execution import with_description

# ---------------------------------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------------------------------
# Configure a list of datasets by specifying the RID and version of each
# dataset that goes into the collection. The group name "datasets" must
# match the parameter name in BaseConfig.

# =============================================================================
# Catalog 2: CIFAR-10 E2E Test with 1,000 images (localhost, schema: cifar10_e2e_test)
# =============================================================================

datasets_store = store(group="datasets")

# Special config for notebooks that don't need datasets (e.g., ROC analysis)
datasets_store(
    [],
    name="no_datasets",
)

# -----------------------------------------------------------------------------
# Full datasets (1,000 images total in this catalog)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AWP", version="0.3.0")],
        "Complete CIFAR-10 dataset with all 1,000 images (500 training + 500 testing). "
        "Use for full-scale experiments.",
    ),
    name="cifar10_complete",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AX0", version="0.4.0")],
        "Split dataset containing nested training (500) and testing (500) subsets. "
        "Testing images are unlabeled. Use for standard train/test workflows.",
    ),
    name="cifar10_split",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AX8", version="0.4.0")],
        "Training partition with 500 labeled CIFAR-10 images. "
        "All images have ground truth classifications.",
    ),
    name="cifar10_training",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AXJ", version="0.4.0")],
        "Testing partition with 500 CIFAR-10 images. "
        "These images are unlabeled (no ground truth) for blind evaluation.",
    ),
    name="cifar10_testing",
)

# -----------------------------------------------------------------------------
# Small datasets for quick testing (1,000 images total)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AY6", version="0.4.0")],
        "Small split dataset with 1,000 images (500 training + 500 testing). "
        "Use for quick iteration and debugging.",
    ),
    name="cifar10_small_split",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AYE", version="0.4.0")],
        "Small training set with 500 labeled images. "
        "Use for rapid prototyping and testing model code.",
    ),
    name="cifar10_small_training",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AYR", version="0.4.0")],
        "Small testing set with 500 unlabeled images. "
        "Use for quick inference testing.",
    ),
    name="cifar10_small_testing",
)

# -----------------------------------------------------------------------------
# Labeled split datasets (for ROC analysis)
# Created from training images only - ALL have ground truth labels
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AZC", version="0.4.0")],
        "Labeled split dataset with 500 images (400 train + 100 test). "
        "BOTH partitions have ground truth labels, enabling ROC curve analysis "
        "and proper evaluation metrics on the test set.",
    ),
    name="cifar10_labeled_split",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="AZP", version="0.4.0")],
        "Labeled training partition with 400 images. "
        "All images have ground truth classifications.",
    ),
    name="cifar10_labeled_training",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B00", version="0.4.0")],
        "Labeled testing partition with 100 images WITH ground truth. "
        "Use for evaluation when you need metrics like accuracy, ROC curves, etc.",
    ),
    name="cifar10_labeled_testing",
)

# -----------------------------------------------------------------------------
# Small labeled datasets (500 images total, all labeled)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B0M", version="0.4.0")],
        "Small labeled split with 500 images (400 train + 100 test). "
        "Both partitions have labels. Use for quick testing with evaluation metrics.",
    ),
    name="cifar10_small_labeled_split",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B0Y", version="0.4.0")],
        "Small labeled training set with 400 images. "
        "For rapid prototyping with labeled data.",
    ),
    name="cifar10_small_labeled_training",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B18", version="0.4.0")],
        "Small labeled testing set with 100 images WITH ground truth. "
        "For quick evaluation testing with metrics.",
    ),
    name="cifar10_small_labeled_testing",
)

# -----------------------------------------------------------------------------
# REQUIRED: default_dataset - used when no dataset is specified
# -----------------------------------------------------------------------------
# Note: Using plain list for notebook config compatibility (with_description creates
# a DictConfig which can't merge with BaseConfig's ListConfig defaults)

datasets_store(
    [DatasetSpecConfig(rid="AX0", version="0.4.0")],
    name="default_dataset",
)
