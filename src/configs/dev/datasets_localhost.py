"""Localhost catalog 1337 dataset RIDs (cifar10_test schema, 10K images).

Loaded by `load-cifar10 --hostname localhost --create-catalog cifar10_test --num-images 10000`.

These configs register names with a `_localhost` suffix that point at the
RIDs in catalog 1337. Select one at the CLI:

    uv run deriva-ml-run --host localhost --catalog 1337 \\
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
# Full datasets (10,000 images total, 5,000 train + 5,000 test)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2W4J", version="0.21.0")],
        "Complete CIFAR-10 dataset on localhost catalog 1337 (10,000 images).",
    ),
    name="cifar10_complete_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2W4W", version="0.22.0")],
        "Split dataset on localhost 1337 (5,000 train + 5,000 test, test unlabeled).",
    ),
    name="cifar10_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2W54", version="0.22.0")],
        "Training partition on localhost 1337 (5,000 labeled images).",
    ),
    name="cifar10_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2W5E", version="0.22.0")],
        "Testing partition on localhost 1337 (5,000 unlabeled images).",
    ),
    name="cifar10_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small datasets (1,000 images total, 500 train + 500 test)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2W62", version="0.4.0")],
        "Small split on localhost 1337 (500 train + 500 test).",
    ),
    name="cifar10_small_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2W6A", version="0.4.0")],
        "Small training set on localhost 1337 (500 labeled images).",
    ),
    name="cifar10_small_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2W6M", version="0.4.0")],
        "Small testing set on localhost 1337 (500 unlabeled images).",
    ),
    name="cifar10_small_testing_localhost",
)

# -----------------------------------------------------------------------------
# Labeled split datasets (4,000 train + 1,000 test, all labeled)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="45EA", version="0.12.0")],
        "Labeled split on localhost 1337 (4,000 train + 1,000 test, both labeled).",
    ),
    name="cifar10_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="45EJ", version="0.12.0")],
        "Labeled training partition on localhost 1337 (4,000 images).",
    ),
    name="cifar10_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="45EW", version="0.12.0")],
        "Labeled testing partition on localhost 1337 (1,000 labeled images).",
    ),
    name="cifar10_labeled_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small labeled datasets (500 images total, all labeled)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="4FB0", version="0.4.0")],
        "Small labeled split on localhost 1337 (400 train + 100 test, all labeled).",
    ),
    name="cifar10_small_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="4FB8", version="0.4.0")],
        "Small labeled training set on localhost 1337 (400 labeled images).",
    ),
    name="cifar10_small_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="4FBJ", version="0.4.0")],
        "Small labeled testing set on localhost 1337 (100 labeled images).",
    ),
    name="cifar10_small_labeled_testing_localhost",
)
