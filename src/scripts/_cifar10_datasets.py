"""CIFAR-10 Stage 3: create the dataset hierarchy.

This module demonstrates the **dataset-hierarchy pattern**: how
to query the catalog for existing assets, partition them by some
attribute (here: filename prefix), and assemble a nested dataset
structure with derived holdout splits — all inside one Execution.
Copy this module when you need to turn a set of uploaded assets
into a structured dataset hierarchy with train/test splits.

This module is the datasets layer. Given a catalog with the
schema set up and some Image asset rows uploaded (Stages 1 and
2 complete), it creates:

    - ``Complete`` (Labeled) — all images.
    - ``Split`` — parent of Training and Testing.
    - ``Training`` (Labeled) — train-prefix images.
    - ``Testing`` (Labeled) — test-prefix images.
    - ``Small_Split`` — parent of Small_Training and Small_Testing.
    - ``Small_Training`` (Labeled) — 500 random train-prefix images.
    - ``Small_Testing`` (Labeled) — 500 random test-prefix images.
    - ``Labeled_Split`` (and Training/Testing children) — 80/20
      split of training images via ``split_dataset()``.
    - ``Small_Labeled_Split`` (and Training/Testing children) —
      400/100 split for small-scale work.

Stage 3 reads back ``Image`` rows from the catalog and uses
each filename's ``train_`` or ``test_`` prefix to decide which
dataset each image belongs to. No in-memory state from Stage 2
is needed.

Public API:
    - ``create_dataset_hierarchy(ml, batch_size=500)`` — does
      all the work in one Execution.
    - ``run_datasets_phase(ml, batch_size=500)`` — orchestrator
      alias for symmetry with run_schema_phase / run_assets_phase.
"""

from __future__ import annotations

import logging
import random

from deriva_ml import DerivaML
from deriva_ml.dataset.split import split_dataset
from deriva_ml.execution import ExecutionConfiguration

logger = logging.getLogger(__name__)

SMALL_TRAIN_SIZE = 500
SMALL_TEST_SIZE = 500


def create_dataset_hierarchy(ml: DerivaML, batch_size: int = 500) -> dict[str, str]:
    """Create the full CIFAR-10 dataset hierarchy.

    Queries the catalog for all ``Image`` asset rows, splits them
    by filename prefix (``train_`` vs ``test_``), creates the
    parent and child dataset rows, assigns members in batches,
    and finally creates the labeled-split families via
    ``split_dataset()``.

    All work happens inside one Execution for clean provenance.

    Args:
        ml: Connected DerivaML instance.
        batch_size: Batch size for ``add_dataset_members`` calls.

    Returns:
        Mapping of dataset name to its RID. Keys include
        ``complete``, ``split``, ``training``, ``testing``,
        ``small_split``, ``small_training``, ``small_testing``,
        ``labeled_split``, ``labeled_training``,
        ``labeled_testing``, ``small_labeled_split``,
        ``small_labeled_training``, ``small_labeled_testing``.

    Example:
        >>> datasets = create_dataset_hierarchy(ml)
        >>> datasets["training"]
        'X-12345-NXYZ'
    """
    assets = ml.list_assets("Image")
    logger.info(f"Found {len(assets)} Image assets to organize")

    train_rids = [a.asset_rid for a in assets if a.filename.startswith("train_")]
    test_rids = [a.asset_rid for a in assets if a.filename.startswith("test_")]
    all_rids = train_rids + test_rids
    logger.info(f"  Train: {len(train_rids)}, Test: {len(test_rids)}")

    workflow = ml.create_workflow(
        name="CIFAR-10 Dataset Hierarchy",
        workflow_type="CIFAR_Data_Load",
        description="Create CIFAR-10 dataset hierarchy from uploaded images",
    )
    config = ExecutionConfiguration(workflow=workflow)

    datasets: dict[str, str] = {}

    with ml.create_execution(config) as exe:
        logger.info(f"  Datasets execution RID: {exe.execution_rid}")

        # Parent + child datasets
        complete = exe.create_dataset(
            description="Complete CIFAR-10 dataset with all labeled images",
            dataset_types=["Complete", "Labeled"],
        )
        datasets["complete"] = complete.dataset_rid

        split = exe.create_dataset(
            description="CIFAR-10 dataset split into training and testing subsets",
            dataset_types=["Split"],
        )
        datasets["split"] = split.dataset_rid

        training = exe.create_dataset(
            description="CIFAR-10 training set with 50,000 labeled images",
            dataset_types=["Training", "Labeled"],
        )
        datasets["training"] = training.dataset_rid

        testing = exe.create_dataset(
            description="CIFAR-10 testing set with 10,000 labeled images",
            dataset_types=["Testing", "Labeled"],
        )
        datasets["testing"] = testing.dataset_rid

        split.add_dataset_members(
            [training.dataset_rid, testing.dataset_rid], validate=False
        )

        small_split = exe.create_dataset(
            description="Small CIFAR-10 dataset split with 1,000 randomly selected images for testing",
            dataset_types=["Split"],
        )
        datasets["small_split"] = small_split.dataset_rid

        small_training = exe.create_dataset(
            description="Small CIFAR-10 training set with 500 labeled images for quick testing",
            dataset_types=["Training", "Labeled"],
        )
        datasets["small_training"] = small_training.dataset_rid

        small_testing = exe.create_dataset(
            description="Small CIFAR-10 testing set with 500 labeled images for quick testing",
            dataset_types=["Testing", "Labeled"],
        )
        datasets["small_testing"] = small_testing.dataset_rid

        small_split.add_dataset_members(
            [small_training.dataset_rid, small_testing.dataset_rid], validate=False
        )

    exe.upload_execution_outputs(clean_folder=True)

    # Member assignment runs against the catalog directly
    # (the Execution above has already been committed)
    logger.info("Assigning Image RIDs to datasets...")

    def _batched_add(ds_rid: str, rids: list[str], label: str) -> None:
        ds = ml.lookup_dataset(ds_rid)
        added = 0
        for i in range(0, len(rids), batch_size):
            batch = rids[i : i + batch_size]
            ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
        logger.info(f"  {label}: added {added}/{len(rids)} images")

    if all_rids:
        _batched_add(datasets["complete"], all_rids, "Complete")
    if train_rids:
        _batched_add(datasets["training"], train_rids, "Training")
    if test_rids:
        _batched_add(datasets["testing"], test_rids, "Testing")

    # Small splits — random sample if enough; else use all
    if train_rids:
        sample = (
            random.sample(train_rids, SMALL_TRAIN_SIZE)
            if len(train_rids) >= SMALL_TRAIN_SIZE
            else train_rids
        )
        _batched_add(datasets["small_training"], sample, "Small_Training")
    if test_rids:
        sample = (
            random.sample(test_rids, SMALL_TEST_SIZE)
            if len(test_rids) >= SMALL_TEST_SIZE
            else test_rids
        )
        _batched_add(datasets["small_testing"], sample, "Small_Testing")

    # Labeled splits derived from Training
    if train_rids:
        logger.info("Creating Labeled_Split (80/20 of training)...")
        labeled = split_dataset(
            ml,
            datasets["training"],
            test_size=0.2,
            seed=42,
            training_types=["Labeled"],
            testing_types=["Labeled"],
            element_table="Image",
            split_description="CIFAR-10 labeled split: 80/20 from training images",
        )
        datasets["labeled_split"] = labeled.split.rid
        datasets["labeled_training"] = labeled.training.rid
        datasets["labeled_testing"] = labeled.testing.rid

        logger.info("Creating Small_Labeled_Split...")
        if len(train_rids) >= 500:
            small_labeled = split_dataset(
                ml,
                datasets["training"],
                test_size=100,
                train_size=400,
                seed=42,
                training_types=["Labeled"],
                testing_types=["Labeled"],
                element_table="Image",
                split_description="Small CIFAR-10 labeled split: 400/100 from training",
            )
        else:
            small_labeled = split_dataset(
                ml,
                datasets["training"],
                test_size=0.2,
                seed=123,
                training_types=["Labeled"],
                testing_types=["Labeled"],
                element_table="Image",
                split_description="Small CIFAR-10 labeled split from training",
            )
        datasets["small_labeled_split"] = small_labeled.split.rid
        datasets["small_labeled_training"] = small_labeled.training.rid
        datasets["small_labeled_testing"] = small_labeled.testing.rid

    return datasets


def run_datasets_phase(ml: DerivaML, batch_size: int = 500) -> dict[str, str]:
    """Stage 3 orchestrator alias.

    Args:
        ml: Connected DerivaML instance.
        batch_size: Batch size for dataset-member additions.

    Returns:
        Mapping of dataset name to RID (see create_dataset_hierarchy).

    Example:
        >>> rids = run_datasets_phase(ml)
        >>> rids["training"]
        'X-12345-NXYZ'
    """
    return create_dataset_hierarchy(ml, batch_size=batch_size)
