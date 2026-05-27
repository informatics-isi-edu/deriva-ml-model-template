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

# Toronto-source split.
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

# Training-derived labeled split (stratified 80/20 from training images, seed=42).
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

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WD2", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 labeled split: stratified 400/100 from training, seed=42.",
    ),
    name="cifar10_small_labeled_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WDA", version="0.1.0.post1.dev1")],
        "Small labeled training subset (400 samples), seed=42.",
    ),
    name="cifar10_small_labeled_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WDM", version="0.1.0.post1.dev1")],
        "Small labeled testing subset (100 samples), seed=42.",
    ),
    name="cifar10_small_labeled_testing",
)

# -----------------------------------------------------------------------------
# Special-case configs (always empty by design)
# -----------------------------------------------------------------------------

datasets_store([], name="no_datasets")
datasets_store([], name="none")

# REQUIRED: ``default_dataset`` — pin to the small labeled split for fast runs.
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="WD2", version="0.1.0.post1.dev1")],
        "Default = cifar10_small_labeled_split (seed=42, 500-image stratified split).",
    ),
    name="default_dataset",
)
