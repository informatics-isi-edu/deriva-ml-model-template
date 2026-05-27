"""Dataset Configurations.

This module declares the *names* of dataset groups the experiments and
notebooks reference. Each entry is intentionally **empty by default** — the
checked-in template ships without RIDs because RIDs are catalog-specific and
would be stale on any fresh clone.

After running ``load-cifar10`` against your own catalog, fill in the RIDs and
versions printed by the loader (or read them with
``ml.find_datasets()``). Two recommended patterns:

1. **Edit this file in place.** Replace each empty list with a
   ``DatasetSpecConfig(rid=..., version=...)``. Wrap with
   ``with_description(..., "...")`` if you want a description that appears in
   ``deriva-ml-run --info``.

2. **Add a per-environment override.** Create
   ``src/configs/dev/datasets_<env>.py`` registering ``<name>_<env>`` configs
   in the same ``datasets`` group, then select on the CLI:
   ``deriva-ml-run datasets=cifar10_small_labeled_split_<env>`` (see
   ``dev/datasets_localhost.py`` for a worked example).

The empty defaults pass config validation but will fail at execution time
("Dataset '' not found") — which is the desired behavior: a fresh clone must
not silently run against someone else's RIDs.

Configuration Group: ``datasets``
"""

from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig  # noqa: F401  (re-exported for users editing this file)
from deriva_ml.execution import with_description  # noqa: F401

datasets_store = store(group="datasets")

# -----------------------------------------------------------------------------
# Empty placeholders — fill in for your catalog before running.
# -----------------------------------------------------------------------------
# Example (after running load-cifar10):
#
#   datasets_store(
#       with_description(
#           [DatasetSpecConfig(rid="28FA", version="0.21.0")],
#           "Complete CIFAR-10 dataset (10,000 images).",
#       ),
#       name="cifar10_complete",
#   )

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="JZ8", version="0.1.0.post1.dev3")],
        "Complete CIFAR-10 dataset (1500 images).",
    ),
    name="cifar10_complete",
)

# -----------------------------------------------------------------------------
# Original Toronto split — train images from training source, test from test_batch.
#
# Use this family when you want the "standard" CIFAR-10 split where training
# and testing come from distinct Toronto source batches.
# -----------------------------------------------------------------------------
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="JZJ", version="0.1.0.post1.dev1")],
        "CIFAR-10 split: training + testing partitions from Toronto sources.",
    ),
    name="cifar10_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="JZT", version="0.1.0.post1.dev2")],
        "CIFAR-10 training partition (750 images, labeled).",
    ),
    name="cifar10_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="K04", version="0.1.0.post1.dev2")],
        "CIFAR-10 testing partition (750 images, labeled).",
    ),
    name="cifar10_testing",
)

# Small Toronto-source variants (smaller stratified sub-samples).
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="K0M", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 split (1000 stratified images).",
    ),
    name="cifar10_small_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="K0W", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 training partition (500 stratified images).",
    ),
    name="cifar10_small_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="K16", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 testing partition (500 stratified images).",
    ),
    name="cifar10_small_testing",
)

# -----------------------------------------------------------------------------
# Training-derived labeled split — stratified 80/20 from training images only.
# Test_batch images are NOT used here. Created by split_dataset(seed=42).
# Use for cross-validation / ROC analysis where the test_batch must stay unseen.
# -----------------------------------------------------------------------------
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="TX0", version="0.1.0.post1.dev1")],
        "CIFAR-10 labeled split: stratified 80/20 from training images, seed=42.",
    ),
    name="cifar10_labeled_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="TX8", version="0.1.0.post1.dev2")],
        "Training subset (600 samples) stratified from training images, seed=42.",
    ),
    name="cifar10_labeled_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="TXJ", version="0.1.0.post1.dev1")],
        "Testing subset (150 samples) stratified from training images, seed=42.",
    ),
    name="cifar10_labeled_testing",
)

# Small labeled-split variant (400/100).
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WD2", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 labeled split: stratified 400/100 from training, seed=123.",
    ),
    name="cifar10_small_labeled_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WDA", version="0.1.0.post1.dev1")],
        "Small labeled training subset (400 samples), seed=123.",
    ),
    name="cifar10_small_labeled_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WDM", version="0.1.0.post1.dev1")],
        "Small labeled testing subset (100 samples), seed=123.",
    ),
    name="cifar10_small_labeled_testing",
)

# -----------------------------------------------------------------------------
# Validation dataset — held-out evaluation for the Dataset_Type=Validation lane.
# Created by the Curator (XEM, 100 stratified images from K04, seed=2026).
# Use ALONGSIDE a training/split dataset (e.g. cifar10_labeled_split) to
# exercise the cifar10_cnn runner's Validation dispatch lane (D01).
# Caveat (tk-003): XEM overlaps with K04. For strict held-out, train on
# TX0 (stratified from JZT only) so XEM is fully unseen.
# -----------------------------------------------------------------------------
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="XEM", version="0.1.0.post1.dev1")],
        "CIFAR-10 Validation set: 100 stratified images from K04, seed=2026.",
    ),
    name="cifar10_validation",
)

# Composite config: TX0 training/testing split + XEM validation bag.
# Drives the Validation dispatch lane in cifar10_cnn.
datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="TX0", version="0.1.0.post1.dev1"),
            DatasetSpecConfig(rid="XEM", version="0.1.0.post1.dev1"),
        ],
        "CIFAR-10 labeled split (TX0) + Validation bag (XEM) for dispatch lane.",
    ),
    name="cifar10_labeled_split_with_validation",
)

# -----------------------------------------------------------------------------
# Special-case configs (always empty by design)
# -----------------------------------------------------------------------------

# Notebooks (e.g., ROC analysis) that consume asset RIDs, not datasets.
datasets_store([], name="no_datasets")

# Script-only experiments that manage their own data.
datasets_store([], name="none")

# REQUIRED: ``default_dataset`` is used when no dataset override is specified.
# Pin to the labeled small split — typical e2e default.
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WD2", version="0.1.0.post1.dev1")],
        "Default = cifar10_small_labeled_split (seed=123, 500-image stratified split).",
    ),
    name="default_dataset",
)
