"""Localhost catalog 1407 dataset RIDs (cifar10_e2e schema, 200 images).

Loaded by `load-cifar10 --hostname localhost --create-catalog cifar10_e2e --num-images 200`.

These configs register names with a `_localhost` suffix that point at the
RIDs in catalog 1407. Select one at the CLI:

    uv run deriva-ml-run --host localhost --catalog 1407 \\
        +experiment=cifar10_quick datasets=cifar10_small_labeled_split_localhost \\
        dry_run=true

The default `datasets.py` configs still point at the seeded demo catalog (6)
RIDs (28DM, 28HJ, etc.) which do not exist in this catalog.
"""

from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# -----------------------------------------------------------------------------
# Full datasets (200 images total, 100 train + 100 test)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="60A", version="0.2.0")],
        "Complete CIFAR-10 dataset on localhost catalog 1407 (200 images).",
    ),
    name="cifar10_complete_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="60M", version="0.4.0")],
        "Split dataset on localhost 1407 (100 train + 100 test, test unlabeled).",
    ),
    name="cifar10_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="60W", version="0.4.0")],
        "Training partition on localhost 1407 (100 labeled images).",
    ),
    name="cifar10_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="616", version="0.4.0")],
        "Testing partition on localhost 1407 (100 unlabeled images).",
    ),
    name="cifar10_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small datasets (alias of full at this scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="61T", version="0.4.0")],
        "Small split on localhost 1407.",
    ),
    name="cifar10_small_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="622", version="0.4.0")],
        "Small training set on localhost 1407.",
    ),
    name="cifar10_small_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="62C", version="0.4.0")],
        "Small testing set on localhost 1407.",
    ),
    name="cifar10_small_testing_localhost",
)

# -----------------------------------------------------------------------------
# Labeled split datasets (both partitions labeled, full catalog scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="7AG", version="0.4.0")],
        "Labeled split on localhost 1407 (both partitions labeled).",
    ),
    name="cifar10_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="7AR", version="0.4.0")],
        "Labeled training partition on localhost 1407.",
    ),
    name="cifar10_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="7B2", version="0.4.0")],
        "Labeled testing partition on localhost 1407.",
    ),
    name="cifar10_labeled_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small labeled datasets (alias of full labeled at this scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="7KE", version="0.4.0")],
        "Small labeled split on localhost 1407.",
    ),
    name="cifar10_small_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="7KP", version="0.4.0")],
        "Small labeled training set on localhost 1407.",
    ),
    name="cifar10_small_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="7M0", version="0.4.0")],
        "Small labeled testing set on localhost 1407.",
    ),
    name="cifar10_small_labeled_testing_localhost",
)
