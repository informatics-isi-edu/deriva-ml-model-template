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
    - ``Small_Training`` (Labeled) — up to ``SMALL_TRAIN_SIZE``
      stratified train-prefix images (capped at the actual pool).
    - ``Small_Testing`` (Labeled) — up to ``SMALL_TEST_SIZE``
      stratified test-prefix images (capped at the actual pool).
    - ``Labeled_Split`` (and Training/Testing children) — 80/20
      split of training images via ``split_dataset()``.
    - ``Small_Labeled_Split`` (and Training/Testing children) —
      fixed 400/100 split when the training pool is >=500, else
      an 80/20 fallback for small-scale work.

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

from scripts._cifar10_assets import class_from_filename

logger = logging.getLogger(__name__)

SMALL_TRAIN_SIZE = 500
SMALL_TEST_SIZE = 500

# Column to stratify by for class-balanced splits — matches the
# Image_Classification feature populated in stage 2b.
STRATIFY_COLUMN = "Execution_Image_Image_Classification.Image_Class"


def _build_dataset_descriptions(
    train_count: int,
    test_count: int,
    small_train_count: int,
    small_test_count: int,
) -> dict[str, str]:
    """Build dataset descriptions parameterized by actual member counts.

    The CIFAR-10 loader's ``--num-images`` flag controls how many
    Toronto source images get pulled into the catalog, so the resulting
    dataset member counts depend on the run, not on the Toronto
    defaults (50,000 / 10,000). Hard-coding the Toronto numbers in the
    descriptions produced misleading metadata at any
    ``--num-images < default`` (see e2e finding curator/03 from
    2026-05-26).

    This helper centralises the description text so it can be
    unit-tested without a live catalog round-trip.

    Args:
        train_count: Actual count of images in the ``Training`` dataset.
        test_count: Actual count of images in the ``Testing`` dataset.
        small_train_count: Actual count of images in ``Small_Training``
            (``min(SMALL_TRAIN_SIZE, train_count)``).
        small_test_count: Actual count of images in ``Small_Testing``
            (``min(SMALL_TEST_SIZE, test_count)``).

    Returns:
        Mapping with one entry per Toronto-family dataset key. Keys are
        ``complete``, ``split``, ``training``, ``testing``,
        ``small_split``, ``small_training``, and ``small_testing``.

    Example:
        >>> d = _build_dataset_descriptions(
        ...     train_count=250, test_count=250,
        ...     small_train_count=250, small_test_count=250,
        ... )
        >>> "250 labeled images" in d["training"]
        True
    """
    total = train_count + test_count
    small_total = small_train_count + small_test_count
    return {
        "complete": (
            f"Complete CIFAR-10 dataset: {total:,} labeled images "
            f"({train_count:,} train + {test_count:,} test)."
        ),
        "split": (
            "CIFAR-10 dataset split into training and testing subsets "
            f"({train_count:,} / {test_count:,} images)."
        ),
        "training": (f"CIFAR-10 training partition: {train_count:,} labeled images."),
        "testing": (f"CIFAR-10 testing partition: {test_count:,} labeled images."),
        "small_split": (
            f"Small CIFAR-10 dataset split: {small_total:,} stratified "
            f"images for quick testing "
            f"({small_train_count:,} / {small_test_count:,} train/test)."
        ),
        "small_training": (
            f"Small CIFAR-10 training set: {small_train_count:,} "
            "stratified images for quick testing."
        ),
        "small_testing": (
            f"Small CIFAR-10 testing set: {small_test_count:,} "
            "stratified images for quick testing."
        ),
    }


def _labeled_split_description(train_count: int) -> str:
    """Describe the 80/20 stratified labeled split of the training set.

    Args:
        train_count: Total number of images in the ``Training`` dataset.

    Returns:
        A description that reports the resulting child partition sizes
        (80/20 of ``train_count``, rounded down/up to keep the total).

    Example:
        >>> _labeled_split_description(250)
        'CIFAR-10 labeled split: stratified 80/20 from training images (200 / 50, seed=42).'
    """
    test_n = train_count // 5
    train_n = train_count - test_n
    return (
        "CIFAR-10 labeled split: stratified 80/20 from training images "
        f"({train_n:,} / {test_n:,}, seed=42)."
    )


def _small_labeled_split_description(train_count: int) -> str:
    """Describe the Small_Labeled_Split of the training set.

    At ``train_count >= 500`` the loader uses a fixed 400/100 stratified
    split; otherwise it falls back to an 80/20 fraction.

    Args:
        train_count: Total number of images in the ``Training`` dataset.

    Returns:
        A description that reports the resulting partition sizes.

    Example:
        >>> _small_labeled_split_description(600)
        'Small CIFAR-10 labeled split: stratified 400/100 from training (seed=42).'
        >>> _small_labeled_split_description(250)
        'Small CIFAR-10 labeled split: stratified 80/20 from training (200 / 50, seed=123).'
    """
    if train_count >= 500:
        return (
            "Small CIFAR-10 labeled split: stratified 400/100 from training (seed=42)."
        )
    test_n = train_count // 5
    train_n = train_count - test_n
    return (
        "Small CIFAR-10 labeled split: stratified 80/20 from training "
        f"({train_n:,} / {test_n:,}, seed=123)."
    )


def stratified_sample_rids(
    rids: list[str],
    classes: list[str | None],
    sample_size: int,
    seed: int,
) -> list[str]:
    """Pick a class-balanced sample of asset RIDs.

    Mirrors :func:`scripts._cifar10_assets.stratified_sample_by_class`
    but operates over a flat ``(rid, class)`` pairing — used for the
    ``Small_Training`` / ``Small_Testing`` random samples that previously
    called :func:`random.sample` on the full RID list (which left them
    skewed toward whichever classes came first in the catalog query).

    Args:
        rids: All candidate RIDs.
        classes: Parallel list of class names. ``None`` entries are
            treated as unknown-class and excluded from the result.
        sample_size: Number of RIDs to return. If ``>= len(rids)`` the
            full known-class set is returned.
        seed: Seed used for per-class and final shuffles.

    Returns:
        A list of ``sample_size`` RIDs with roughly equal class
        representation when ``sample_size >= len(unique classes)``.

    Example:
        >>> rids = [f"R{i}" for i in range(6)]
        >>> classes = ["a", "b", "a", "b", "a", "b"]
        >>> sample = stratified_sample_rids(rids, classes, 4, seed=1)
        >>> sum(1 for r, c in zip(rids, classes) if r in sample and c == "a")
        2
    """
    if sample_size <= 0:
        return []
    by_class: dict[str, list[str]] = {}
    for rid, cls in zip(rids, classes):
        if cls is None:
            continue
        by_class.setdefault(cls, []).append(rid)

    total_known = sum(len(v) for v in by_class.values())
    if sample_size >= total_known:
        rng = random.Random(seed)
        flat = [r for v in by_class.values() for r in v]
        rng.shuffle(flat)
        return flat

    num_classes = len(by_class)
    if num_classes == 0:
        return []
    if sample_size < num_classes:
        logger.warning(
            "Stratified RID sample requested for %d items but %d classes "
            "are available; result will be class-biased.",
            sample_size,
            num_classes,
        )

    class_rng = random.Random(seed)
    sorted_classes = sorted(by_class.keys())
    shuffled_by_class: dict[str, list[str]] = {}
    for cls in sorted_classes:
        bucket = list(by_class[cls])
        class_rng.shuffle(bucket)
        shuffled_by_class[cls] = bucket

    base_quota = sample_size // num_classes
    remainder = sample_size % num_classes
    order_rng = random.Random(seed + 1)
    class_order = list(sorted_classes)
    order_rng.shuffle(class_order)
    extras = set(class_order[:remainder])

    picked: list[str] = []
    for cls in sorted_classes:
        quota = base_quota + (1 if cls in extras else 0)
        picked.extend(shuffled_by_class[cls][:quota])

    if len(picked) < sample_size:
        already = set(picked)
        leftover: list[str] = []
        for cls in sorted_classes:
            quota = base_quota + (1 if cls in extras else 0)
            leftover.extend(shuffled_by_class[cls][quota:])
        leftover_rng = random.Random(seed + 2)
        leftover_rng.shuffle(leftover)
        for rid in leftover:
            if len(picked) >= sample_size:
                break
            if rid in already:
                continue
            picked.append(rid)

    final_rng = random.Random(seed + 3)
    final_rng.shuffle(picked)
    return picked


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
    train_classes = [
        class_from_filename(a.filename)
        for a in assets
        if a.filename.startswith("train_")
    ]
    test_classes = [
        class_from_filename(a.filename)
        for a in assets
        if a.filename.startswith("test_")
    ]
    all_rids = train_rids + test_rids
    logger.info(f"  Train: {len(train_rids)}, Test: {len(test_rids)}")

    workflow = ml.create_workflow(
        name="CIFAR-10 Dataset Hierarchy",
        workflow_type="CIFAR_Data_Load",
        description="Create CIFAR-10 dataset hierarchy from uploaded images",
    )
    config = ExecutionConfiguration(workflow=workflow)

    # Descriptions reflect the actual catalog state for this run (see
    # finding curator/03, 2026-05-26): hard-coding the Toronto defaults
    # (50,000 / 10,000) lied at any --num-images < default.
    small_train_count = min(SMALL_TRAIN_SIZE, len(train_rids))
    small_test_count = min(SMALL_TEST_SIZE, len(test_rids))
    descriptions = _build_dataset_descriptions(
        train_count=len(train_rids),
        test_count=len(test_rids),
        small_train_count=small_train_count,
        small_test_count=small_test_count,
    )

    datasets: dict[str, str] = {}

    with ml.create_execution(config) as exe:
        logger.info(f"  Datasets execution RID: {exe.execution_rid}")

        # Parent + child datasets
        complete = exe.create_dataset(
            description=descriptions["complete"],
            dataset_types=["Complete", "Labeled"],
        )
        datasets["complete"] = complete.dataset_rid

        split = exe.create_dataset(
            description=descriptions["split"],
            dataset_types=["Split"],
        )
        datasets["split"] = split.dataset_rid

        training = exe.create_dataset(
            description=descriptions["training"],
            dataset_types=["Training", "Labeled"],
        )
        datasets["training"] = training.dataset_rid

        testing = exe.create_dataset(
            description=descriptions["testing"],
            dataset_types=["Testing", "Labeled"],
        )
        datasets["testing"] = testing.dataset_rid

        split.add_dataset_members(
            [training.dataset_rid, testing.dataset_rid], validate=False
        )

        small_split = exe.create_dataset(
            description=descriptions["small_split"],
            dataset_types=["Split"],
        )
        datasets["small_split"] = small_split.dataset_rid

        small_training = exe.create_dataset(
            description=descriptions["small_training"],
            dataset_types=["Training", "Labeled"],
        )
        datasets["small_training"] = small_training.dataset_rid

        small_testing = exe.create_dataset(
            description=descriptions["small_testing"],
            dataset_types=["Testing", "Labeled"],
        )
        datasets["small_testing"] = small_testing.dataset_rid

        small_split.add_dataset_members(
            [small_training.dataset_rid, small_testing.dataset_rid], validate=False
        )

    exe.commit_output_assets(clean_folder=True)

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

    # Small splits — stratified sample by class if enough; else use all.
    # The previous random.sample() left bird/ship-only partitions when
    # the source was class-skewed (#13). Stratified sampling keeps each
    # class proportionally represented at any sample size >= 10.
    if train_rids:
        if len(train_rids) >= SMALL_TRAIN_SIZE:
            sample = stratified_sample_rids(
                train_rids, train_classes, SMALL_TRAIN_SIZE, seed=42
            )
        else:
            sample = train_rids
        _batched_add(datasets["small_training"], sample, "Small_Training")
    if test_rids:
        if len(test_rids) >= SMALL_TEST_SIZE:
            sample = stratified_sample_rids(
                test_rids, test_classes, SMALL_TEST_SIZE, seed=43
            )
        else:
            sample = test_rids
        _batched_add(datasets["small_testing"], sample, "Small_Testing")

    # Labeled splits derived from Training. Stratify by the
    # Image_Classification feature so each child partition keeps a
    # balanced class distribution — without this, the 400/100 small
    # split inherited the source class skew and ended up bird/ship-only
    # at --num-images 500 (#13).
    #
    # ``split_dataset`` runs inside an Execution the caller has already
    # opened: deriva-ml never invents workflow provenance, so this
    # script (which is the caller) is responsible for registering the
    # workflow and opening an execution that says "the bytes in this
    # script decided to do these splits." We reuse a single workflow
    # and a single execution across both labeled splits so the lineage
    # is coherent (one Execution row, two Split datasets nested under
    # the source Training dataset).
    if train_rids:
        split_workflow = ml.create_workflow(
            name="CIFAR-10 Labeled Split",
            workflow_type="Dataset_Split",
            description=(
                "Stratified labeled splits from the CIFAR-10 training set; "
                "produces Labeled_Split and Small_Labeled_Split."
            ),
        )
        with ml.create_execution(
            ExecutionConfiguration(
                workflow=split_workflow,
                description="Create Labeled_Split and Small_Labeled_Split from the training set",
            )
        ) as split_exe:
            logger.info(
                "Creating Labeled_Split (80/20 of training) in execution %s...",
                split_exe.execution_rid,
            )
            labeled = split_dataset(
                ml,
                datasets["training"],
                split_exe,
                test_size=0.2,
                seed=42,
                stratify_by_column=STRATIFY_COLUMN,
                training_types=["Labeled"],
                testing_types=["Labeled"],
                element_table="Image",
                include_tables=["Image", "Execution_Image_Image_Classification"],
                row_per="Execution_Image_Image_Classification",
                split_description=_labeled_split_description(len(train_rids)),
            )
            datasets["labeled_split"] = labeled.split.rid
            datasets["labeled_training"] = labeled.training.rid
            datasets["labeled_testing"] = labeled.testing.rid

            logger.info("Creating Small_Labeled_Split...")
            if len(train_rids) >= 500:
                small_labeled = split_dataset(
                    ml,
                    datasets["training"],
                    split_exe,
                    test_size=100,
                    train_size=400,
                    seed=42,
                    stratify_by_column=STRATIFY_COLUMN,
                    training_types=["Labeled"],
                    testing_types=["Labeled"],
                    element_table="Image",
                    include_tables=["Image", "Execution_Image_Image_Classification"],
                    row_per="Execution_Image_Image_Classification",
                    split_description=_small_labeled_split_description(len(train_rids)),
                )
            else:
                small_labeled = split_dataset(
                    ml,
                    datasets["training"],
                    split_exe,
                    test_size=0.2,
                    seed=123,
                    stratify_by_column=STRATIFY_COLUMN,
                    training_types=["Labeled"],
                    testing_types=["Labeled"],
                    element_table="Image",
                    include_tables=["Image", "Execution_Image_Image_Classification"],
                    row_per="Execution_Image_Image_Classification",
                    split_description=_small_labeled_split_description(len(train_rids)),
                )
            datasets["small_labeled_split"] = small_labeled.split.rid
            datasets["small_labeled_training"] = small_labeled.training.rid
            datasets["small_labeled_testing"] = small_labeled.testing.rid

        split_exe.commit_output_assets(clean_folder=True)

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
