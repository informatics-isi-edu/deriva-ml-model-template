"""Localhost catalog 36 dataset RIDs (e2e-test-20260519b schema, 500 images).

Loaded by `load-cifar10 --hostname localhost --create-catalog e2e-test-20260519b --num-images 500`.

These configs register names with a `_localhost` suffix that point at the
RIDs in catalog 36. Select one at the CLI:

    uv run deriva-ml-run --host localhost --catalog 36 \\
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
        [DatasetSpecConfig(rid="84W", version="0.1.0.post1.dev1")],
        "Complete CIFAR-10 dataset on localhost catalog 36 (500 images).",
    ),
    name="cifar10_complete_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="856", version="0.1.0.post1.dev1")],
        "Split dataset on localhost 36 (250 train + 250 test).",
    ),
    name="cifar10_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="85E", version="0.1.0.post1.dev1")],
        "Training partition on localhost 36 (250 labeled images).",
    ),
    name="cifar10_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="85R", version="0.1.0.post1.dev1")],
        "Testing partition on localhost 36 (250 labeled images).",
    ),
    name="cifar10_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small datasets (alias of full at this scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="868", version="0.1.0.post1.dev1")],
        "Small split on localhost 36.",
    ),
    name="cifar10_small_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="86G", version="0.1.0.post1.dev1")],
        "Small training set on localhost 36.",
    ),
    name="cifar10_small_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="86T", version="0.1.0.post1.dev1")],
        "Small testing set on localhost 36.",
    ),
    name="cifar10_small_testing_localhost",
)

# -----------------------------------------------------------------------------
# Labeled split datasets (both partitions labeled, full catalog scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B68", version="0.1.0.post1.dev1")],
        "Labeled split on localhost 36 (both partitions labeled).",
    ),
    name="cifar10_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B6G", version="0.1.0.post1.dev1")],
        "Labeled training partition on localhost 36.",
    ),
    name="cifar10_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="B6T", version="0.1.0.post1.dev1")],
        "Labeled testing partition on localhost 36.",
    ),
    name="cifar10_labeled_testing_localhost",
)

# -----------------------------------------------------------------------------
# Small labeled datasets (alias of full labeled at this scale)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="BQW", version="0.1.0.post1.dev1")],
        "Small labeled split on localhost 36.",
    ),
    name="cifar10_small_labeled_split_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="BR4", version="0.1.0.post1.dev1")],
        "Small labeled training set on localhost 36.",
    ),
    name="cifar10_small_labeled_training_localhost",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="BRE", version="0.1.0.post1.dev1")],
        "Small labeled testing set on localhost 36.",
    ),
    name="cifar10_small_labeled_testing_localhost",
)
