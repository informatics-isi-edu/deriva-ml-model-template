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

# [E2E-DROP] Wiring for catalog 27 (e2e-test-20260528). The block below will be
# reverted at wrap-up; main keeps the empty placeholders.
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="M0M", version="0.1.0.post1.dev3")],
        "Complete CIFAR-10 dataset (1,100 labeled: 550 train + 550 test).",
    ),
    name="cifar10_complete",
)

# -----------------------------------------------------------------------------
# Original Toronto split — 50K training / 10K test_batch
#
# Training images and test images come from *different* Toronto source batches.
# Use this family when you want to train on the full 50K training set and
# evaluate against the official 10K test_batch (the "standard" CIFAR-10 split).
# Both halves carry ground-truth labels in the Toronto distribution.
# -----------------------------------------------------------------------------
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="M0Y", version="0.1.0.post1.dev1")],
        "CIFAR-10 Toronto split: 550 training + 550 test images.",
    ),
    name="cifar10_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="M16", version="0.1.0.post1.dev2")],
        "CIFAR-10 Toronto training partition (550 labeled images).",
    ),
    name="cifar10_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="M1G", version="0.1.0.post1.dev2")],
        "CIFAR-10 Toronto testing partition (550 labeled images).",
    ),
    name="cifar10_testing",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="M20", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 Toronto split: stratified 500/500 for quick testing.",
    ),
    name="cifar10_small_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="M28", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 Toronto training set (500 stratified).",
    ),
    name="cifar10_small_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="M2J", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 Toronto testing set (500 stratified).",
    ),
    name="cifar10_small_testing",
)

# [E2E-DROP] Toronto train+test pair bundled for Modeler arc. Per Curator
# tk-002 the TCC / VAP "labeled split" families leak across train/test
# because split_dataset partitioned feature rows (not images) on top of
# tk-001's loader-retry double-tagging. The Toronto family (M16 training
# x M1G testing) is leakage-free by construction (different Toronto source
# pools) with 55/class on each side. Using it as a single dataset group so
# the model harness sees one Training bag and one Testing bag in
# execution.datasets and the per-epoch / final-epoch test_acc numbers are
# trustworthy for the Analyst.
datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="M16", version="0.1.0.post1.dev2"),
            DatasetSpecConfig(rid="M1G", version="0.1.0.post1.dev2"),
        ],
        "CIFAR-10 Toronto leakage-free pair: M16 training (550) + M1G testing (550).",
    ),
    name="cifar10_toronto_pair",
)

# -----------------------------------------------------------------------------
# Training-derived holdout split — 80/20 (or 400/100) of training images only
#
# Both the training and testing partitions are drawn from the 50K Toronto
# training images; the test_batch images are *not* used here. Created by
# split_dataset() with a fixed seed (42). Use this family for cross-validation
# workflows, ROC analysis, or experiments where the test_batch must stay
# unseen for final evaluation.
# -----------------------------------------------------------------------------
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="TCC", version="0.1.0.post1.dev1")],
        "CIFAR-10 labeled holdout split: stratified 80/20 from training (440/110, seed=42).",
    ),
    name="cifar10_labeled_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="TCM", version="0.1.0.post1.dev1")],
        "Training subset (440) of cifar10_training (stratified by class, seed=42).",
    ),
    name="cifar10_labeled_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="TCY", version="0.1.0.post1.dev1")],
        "Testing subset (110) of cifar10_training (stratified by class, seed=42).",
    ),
    name="cifar10_labeled_testing",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="VAP", version="0.1.0.post1.dev1")],
        "Small CIFAR-10 labeled split: stratified 400/100 from training (seed=42).",
    ),
    name="cifar10_small_labeled_split",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="VAY", version="0.1.0.post1.dev1")],
        "Training subset (400) of cifar10_training (stratified, seed=42).",
    ),
    name="cifar10_small_labeled_training",
)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="VB8", version="0.1.0.post1.dev1")],
        "Testing subset (100) of cifar10_training (stratified, seed=42).",
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
# Set to the small labeled split — small enough for fast iteration, labeled
# on both halves for evaluation work.
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="VAP", version="0.1.0.post1.dev1")],
        "Default dataset: cifar10_small_labeled_split (VAP).",
    ),
    name="default_dataset",
)
