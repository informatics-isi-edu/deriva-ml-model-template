"""Curator: carve a balanced Validation slice from the K04 testing partition.

Rationale
---------
The Phase-0 bootstrap left the catalog with Training and Testing labeled
partitions, but no dedicated Validation set. The cifar10_cnn runner (post
PR #29) dispatches a Validation lane when a Dataset_Type=Validation member
is present, so downstream model development would otherwise either:

  - peek at K04 (Testing) for early stopping / model selection, which
    contaminates the held-out evaluation set, or
  - carve a Validation slice ad-hoc inside each training run, which
    doesn't reproduce across runs.

This script produces a stable, balanced 150-image Validation set drawn
from K04, leaving K04 itself untouched as the canonical held-out test
partition.

Strategy
--------
- Pool: K04 Image members (750 total, 75 per CIFAR-10 class).
- Selection: 15 images per class via ``random.Random(20260527).shuffle``
  on sorted RIDs, so the slice is reproducible across re-runs.
- Created via ``exe.create_dataset`` with types ``["Validation",
  "Labeled"]`` and provenance back to a Dataset_Management workflow.

The new dataset's RID is printed at the end so it can be wired into
``src/configs/datasets.py``.

Usage
-----
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/curator_create_validation.py

Example
-------
    $ DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/curator_create_validation.py
    Selected 150 image RIDs (15/class) from K04
    Validation dataset RID: K2P
"""

from __future__ import annotations

import random
from collections import defaultdict

from deriva_ml import DerivaML, ExecutionConfiguration

HOSTNAME = "localhost"
CATALOG_ID = "95"
SOURCE_DATASET_RID = "K04"  # Testing partition: 750 images, 75/class
PER_CLASS = 15  # → 150 total Validation images
SEED = 20260527  # date of this curator run; deterministic shuffle


def main() -> str:
    """Create the Validation dataset and return its RID.

    Returns:
        The RID of the newly-created Validation dataset.
    """
    ml = DerivaML(hostname=HOSTNAME, catalog_id=CATALOG_ID)

    # Source pool: K04 image members
    k04 = ml.lookup_dataset(SOURCE_DATASET_RID)
    members = k04.list_dataset_members()
    k04_rids = {m["RID"] for m in members.get("Image", [])}

    # Class lookup via Image_Classification feature
    pb = ml.pathBuilder()
    feat_table = pb.schemas["e2e-test-20260527c"].tables[
        "Execution_Image_Image_Classification"
    ]
    feats = list(feat_table.entities().fetch())
    img_to_class = {
        f["Image"]: f["Image_Class"] for f in feats if f["Image"] in k04_rids
    }

    # Stratified deterministic selection
    by_class: dict[str, list[str]] = defaultdict(list)
    for img, cls in img_to_class.items():
        by_class[cls].append(img)

    rng = random.Random(SEED)
    selected: list[str] = []
    for cls in sorted(by_class.keys()):
        pool = sorted(by_class[cls])
        rng.shuffle(pool)
        selected.extend(pool[:PER_CLASS])

    print(
        f"Selected {len(selected)} image RIDs "
        f"({PER_CLASS}/class across {len(by_class)} classes) from {SOURCE_DATASET_RID}"
    )

    # Provenance: workflow + execution
    workflow = ml.create_workflow(
        name="Curator Validation Carve-out",
        workflow_type="Dataset_Management",
        description=(
            "Carve a balanced 150-image Validation slice from K04 "
            "(Testing partition). 15 per CIFAR-10 class, "
            f"deterministic via Random({SEED}).shuffle on sorted RIDs."
        ),
    )
    config = ExecutionConfiguration(
        workflow=workflow,
        description=(
            "Create cifar10_validation_150 dataset for downstream model "
            "selection / early stopping without contaminating K04."
        ),
    )

    with ml.create_execution(config) as exe:
        print(f"  Execution RID: {exe.execution_rid}")
        validation = exe.create_dataset(
            description=(
                "CIFAR-10 Validation slice: 150 stratified labeled images "
                "(15 per class) drawn from K04 (testing partition). "
                f"Reproducible via Random({SEED}).shuffle on sorted RIDs. "
                "Intended for early stopping / model selection during "
                "training; downstream consumers should still hold K04 out "
                "as the canonical test partition."
            ),
            dataset_types=["Validation", "Labeled"],
        )
        validation.add_dataset_members({"Image": selected}, validate=False)

    exe.commit_output_assets(clean_folder=True)

    print(f"Validation dataset RID: {validation.dataset_rid}")
    return validation.dataset_rid


if __name__ == "__main__":
    main()
