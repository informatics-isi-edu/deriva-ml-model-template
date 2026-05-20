"""Localhost catalog 46 dataset RIDs (e2e-test-20260519d schema, 500 images).

Loaded by `load-cifar10 --hostname localhost --create-catalog e2e-test-20260519d --num-images 500`.

These configs register names with a `_localhost` suffix that point at the
RIDs in catalog 46. Select one at the CLI:

    uv run deriva-ml-run --host localhost --catalog 46 \\
        +experiment=cifar10_quick datasets=cifar10_small_labeled_split_localhost \\
        dry_run=true

The default `datasets.py` configs still point at the seeded demo catalog
RIDs which do not exist in this catalog.
"""

from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# -----------------------------------------------------------------------------
# Full datasets (500 images total, 250 train + 250 test)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="850", version="0.1.0.post1.dev1")],
        "Complete CIFAR-10 dataset on localhost catalog 46 (500 images).",
    ),
    name="cifar10_complete_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="85A", version="0.1.0.post1.dev1")],
        "Split dataset on localhost 46 (250 train + 250 test).",
    ),
    name="cifar10_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="85J", version="0.1.0.post1.dev1")],
        "Training partition on localhost 46 (250 labeled images).",
    ),
    name="cifar10_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="85W", version="0.1.0.post1.dev1")],
        "Testing partition on localhost 46 (250 labeled images).",
    ),
    name="cifar10_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small datasets (alias of full at this scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="86C", version="0.1.0.post1.dev1")],
        "Small split on localhost 46.",
    ),
    name="cifar10_small_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="86M", version="0.1.0.post1.dev1")],
        "Small training set on localhost 46.",
    ),
    name="cifar10_small_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="86Y", version="0.1.0.post1.dev1")],
        "Small testing set on localhost 46.",
    ),
    name="cifar10_small_testing_localhost",
)

# -----------------------------------------------------------------------------
# Labeled split datasets (both partitions labeled, full catalog scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B6C", version="0.1.0.post1.dev1")],
        "Labeled split on localhost 46 (both partitions labeled).",
    ),
    name="cifar10_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B6M", version="0.1.0.post1.dev1")],
        "Labeled training partition on localhost 46.",
    ),
    name="cifar10_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B6Y", version="0.1.0.post1.dev1")],
        "Labeled testing partition on localhost 46.",
    ),
    name="cifar10_labeled_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small labeled datasets (alias of full labeled at this scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="BR0", version="0.1.0.post1.dev1")],
        "Small labeled split on localhost 46.",
    ),
    name="cifar10_small_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="BR8", version="0.1.0.post1.dev1")],
        "Small labeled training set on localhost 46.",
    ),
    name="cifar10_small_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="BRJ", version="0.1.0.post1.dev1")],
        "Small labeled testing set on localhost 46.",
    ),
    name="cifar10_small_labeled_testing_localhost",
)
