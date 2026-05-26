"""Curator: create curated dataset variants on top of the bootstrap.

Run this once per fresh e2e catalog after ``load-cifar10`` has populated
the standard 13 datasets. It creates two additional dataset rows that
serve downstream personas:

1. ``cifar10_validation_from_test`` (Validation, Labeled) — wraps the
   250 images from the existing Toronto-test partition (``97A``) with
   the ``Validation`` ``Dataset_Type`` so the Developer's cifar10_cnn
   runner (which expects ``Validation``-typed bags per pending task
   D01) has a clean, held-out evaluation set distinct from the
   in-pool C8G / CSA test subsets. Same image RIDs as ``97A`` — this
   is a *semantic relabeling*, not a re-sample.

2. ``cifar10_balanced_demo`` (Testing, Labeled) — a tiny 50-image,
   5-per-class balanced sample from the Complete set (``96E``). Two
   downstream uses: (a) a sub-minute smoke-test set the Developer can
   run sweeps against without spending real GPU time, and (b) a
   guaranteed-every-cell-populated set for the Analyst's confusion
   matrix and per-class ROC slices on the 500-image bootstrap.

Both datasets are created inside a single Curation Execution whose
provenance roots in this script. Member assignment is class-aware
(stratified) for the demo set; the validation set is a straight copy
of ``97A``'s image members.

Usage:

    cd /path/to/deriva-ml-model-template-e2e
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/curator_create_datasets.py

The DIRTY override is only needed while iterating during the e2e run;
production curation should commit the script first.

The script prints the new dataset RIDs at the end. Wire them into
``src/configs/datasets.py`` so downstream experiments can pin them.
"""

from __future__ import annotations

import logging
import random
import sys
from collections import Counter
from pathlib import Path

# scripts/ is not a package; src/ is. Make src/ importable so we can
# reuse class_from_filename and the stratification helper.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("curator")

HOSTNAME = "localhost"
CATALOG_ID = "18"

# RIDs of the bootstrap datasets the curator builds on top of.
COMPLETE_RID = "96E"   # 500 images, fully labeled
TORONTO_TEST_RID = "97A"  # 250 images held out from training

DEMO_PER_CLASS = 5  # 5 per class × 10 classes = 50 images


def stratified_pick(
    rid_to_class: dict[str, str], per_class: int, seed: int
) -> list[str]:
    """Return ``per_class`` RIDs from each class, deterministically.

    Args:
        rid_to_class: Map from image RID to its class label.
        per_class: How many to draw from each class.
        seed: Reproducibility seed.

    Returns:
        Flat list of RIDs, sorted by class then by RID for stable
        ordering (the deterministic shuffle drives selection inside
        each class, not the final order).

    Raises:
        ValueError: If any class has fewer than ``per_class`` RIDs
            available.
    """
    by_class: dict[str, list[str]] = {}
    for rid, cls in rid_to_class.items():
        by_class.setdefault(cls, []).append(rid)

    picked: list[str] = []
    rng = random.Random(seed)
    for cls in sorted(by_class):
        bucket = sorted(by_class[cls])
        if len(bucket) < per_class:
            raise ValueError(
                f"class {cls!r} has only {len(bucket)} candidates, need {per_class}"
            )
        rng.shuffle(bucket)
        picked.extend(bucket[:per_class])
    picked.sort()
    return picked


def fetch_image_class_map(ml: DerivaML) -> dict[str, str]:
    """Build {Image RID -> Image_Class} from the ground-truth feature.

    Reads every ``Execution_Image_Image_Classification`` row and uses
    the most recent label per image (in practice there's only one
    producing execution, the bootstrap, so the "most recent" rule is
    a no-op here).

    Returns:
        Dict from Image RID to its CIFAR-10 class string.
    """
    features = ml.find_features("Image")
    ic = next(f for f in features if f.feature_name == "Image_Classification")
    pb = ml.catalog.getPathBuilder()
    schema = ic.feature_table.schema.name
    table = ic.feature_table.name
    rows = list(pb.schemas[schema].tables[table].entities().fetch())
    return {r["Image"]: r["Image_Class"] for r in rows}


def main() -> None:
    log.info("Connecting to catalog %s/%s", HOSTNAME, CATALOG_ID)
    ml = DerivaML(HOSTNAME, CATALOG_ID)

    # --- Validation set: same members as 97A, retyped as Validation. ---
    val_src = ml.lookup_dataset(TORONTO_TEST_RID)
    val_members = [m["RID"] for m in val_src.list_dataset_members().get("Image", [])]
    if not val_members:
        raise RuntimeError(
            f"source dataset {TORONTO_TEST_RID} has no Image members"
        )
    log.info("Validation source %s has %d image members", TORONTO_TEST_RID, len(val_members))

    # --- Balanced demo: 5-per-class from 96E. ---
    complete_src = ml.lookup_dataset(COMPLETE_RID)
    complete_members = [
        m["RID"] for m in complete_src.list_dataset_members().get("Image", [])
    ]
    log.info("Complete source %s has %d image members", COMPLETE_RID, len(complete_members))

    cls_map = fetch_image_class_map(ml)
    complete_class = {rid: cls_map[rid] for rid in complete_members if rid in cls_map}
    log.info("Class distribution in source: %s", dict(sorted(Counter(complete_class.values()).items())))

    demo_rids = stratified_pick(complete_class, per_class=DEMO_PER_CLASS, seed=2026_05_26)
    demo_dist = Counter(complete_class[r] for r in demo_rids)
    log.info("Demo sample: %d images, distribution: %s", len(demo_rids), dict(sorted(demo_dist.items())))
    assert all(c == DEMO_PER_CLASS for c in demo_dist.values()), "demo set is not balanced"

    # --- Open a curation Execution for provenance and create the rows. ---
    workflow = ml.create_workflow(
        name="CIFAR-10 Curator Variants",
        workflow_type="Dataset_Split",
        description=(
            "Curator-added dataset variants on top of the load-cifar10 bootstrap: "
            "a Validation-typed wrapper around 97A, and a 5-per-class balanced "
            "demo subset of 96E."
        ),
    )
    config = ExecutionConfiguration(
        workflow=workflow,
        description="Create curator dataset variants (validation, balanced demo)",
    )

    created: dict[str, str] = {}
    with ml.create_execution(config) as exe:
        log.info("Curation execution RID: %s", exe.execution_rid)

        val_ds = exe.create_dataset(
            description=(
                "Validation set (250 images) wrapping cifar10_testing (97A) "
                "with Dataset_Type=Validation so cifar10_cnn-style runners can "
                "consume it as a held-out evaluator distinct from in-pool "
                "C8G/CSA test subsets. Image RIDs identical to 97A."
            ),
            dataset_types=["Validation", "Labeled"],
        )
        created["validation_from_test"] = val_ds.dataset_rid

        demo_ds = exe.create_dataset(
            description=(
                f"Balanced demo subset (5 per class × 10 classes = 50 images) "
                f"hand-picked from cifar10_complete (96E) with seed=20260526 "
                "for reproducibility. Use for sub-minute smoke runs and for "
                "evaluation slices that need every confusion-matrix cell "
                "guaranteed populated."
            ),
            dataset_types=["Testing", "Labeled"],
        )
        created["balanced_demo"] = demo_ds.dataset_rid

    exe.commit_output_assets(clean_folder=True)

    # Member assignment runs after the Execution commits (mirrors the
    # bootstrap script's pattern).
    log.info("Assigning members to validation set %s...", created["validation_from_test"])
    ml.lookup_dataset(created["validation_from_test"]).add_dataset_members(
        {"Image": val_members}, validate=False
    )
    log.info("Assigning members to balanced demo set %s...", created["balanced_demo"])
    ml.lookup_dataset(created["balanced_demo"]).add_dataset_members(
        {"Image": demo_rids}, validate=False
    )

    log.info("=" * 70)
    log.info("Curator datasets created:")
    for name, rid in created.items():
        ds = ml.lookup_dataset(rid)
        members = ds.list_dataset_members()
        n_imgs = len(members.get("Image", []))
        log.info(
            "  %-22s  RID=%s  version=%s  images=%d  types=%s",
            name,
            rid,
            ds.current_version,
            n_imgs,
            ds.dataset_types,
        )
    log.info("=" * 70)


if __name__ == "__main__":
    main()
