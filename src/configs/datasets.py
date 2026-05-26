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
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# -----------------------------------------------------------------------------
# E2E test catalog 18 (e2e-test-20260526) — 500-image bootstrap on 2026-05-26.
# RIDs captured from `ml.find_datasets()` after load-cifar10 phases
# images + datasets. Versions are dev (.post1.dev1) — release via
# `ml.lookup_dataset(rid).release(minor=True)` if downstream wants pinned.
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="96E", version="0.1.0.post1.dev1")],
        "Complete CIFAR-10 dataset (500 images).",
    ),
    name="cifar10_complete",
)

# Toronto-source split — training images and test_batch from distinct sources.
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="96R", version="0.1.0.post1.dev1")],
        "CIFAR-10 split: training + testing partitions from Toronto sources.",
    ),
    name="cifar10_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="970", version="0.1.0.post1.dev1")],
        "CIFAR-10 training partition (250 images, labeled).",
    ),
    name="cifar10_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="97A", version="0.1.0.post1.dev1")],
        "CIFAR-10 testing partition (250 images, labeled).",
    ),
    name="cifar10_testing",
)

# Small variants — fewer images per partition for quick-running tests.
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="97T", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 split (fewer images per partition).",
    ),
    name="cifar10_small_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="982", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 training partition.",
    ),
    name="cifar10_small_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="98C", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 testing partition.",
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
        [DatasetSpecConfig(rid="C7Y", version="0.1.0.post1.dev1")],
        "CIFAR-10 labeled split: stratified 80/20 from training images, seed=42.",
    ),
    name="cifar10_labeled_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="C86", version="0.1.0.post1.dev1")],
        "Training subset (200 samples) stratified from training images, seed=42.",
    ),
    name="cifar10_labeled_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="C8G", version="0.1.0.post1.dev1")],
        "Testing subset (50 samples) stratified from training images, seed=42.",
    ),
    name="cifar10_labeled_testing",
)

# Small labeled split variant (seed=123).
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="CRR", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 labeled split: stratified from training, seed=123.",
    ),
    name="cifar10_small_labeled_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="CS0", version="0.1.0.post1.dev1")],
        "Small labeled training subset (200 samples), seed=123.",
    ),
    name="cifar10_small_labeled_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="CSA", version="0.1.0.post1.dev1")],
        "Small labeled testing subset (50 samples), seed=123.",
    ),
    name="cifar10_small_labeled_testing",
)

# -----------------------------------------------------------------------------
# Special-case configs (always empty by design)
# -----------------------------------------------------------------------------

# Notebooks (e.g., ROC analysis) that consume asset RIDs, not datasets.
datasets_store([], name="no_datasets")

# Script-only experiments that manage their own data.
datasets_store([], name="none")

# REQUIRED: ``default_dataset`` is used when no dataset override is specified.
# Pin to the labeled small split — what a typical e2e run wants.
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="CRR", version="0.1.0.post1.dev1")],
        "Default = cifar10_small_labeled_split (seed=123, 250-image stratified split).",
    ),
    name="default_dataset",
)
