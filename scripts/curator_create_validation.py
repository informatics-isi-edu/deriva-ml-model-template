"""Create a curated Validation dataset for catalog 93.

The 2026-05-27 bootstrap shipped 13 datasets covering Complete,
Training, Testing, and Split types, but no ``Validation`` dataset.
The cifar10_cnn runner dispatches on ``Dataset_Type`` ("D01" lineage
from the 2026-05-26 run): a Validation-typed bag is consumed as
held-out evaluation rather than training. Without a Validation
dataset in the catalog, the dispatch lane has no data to feed.

This script creates a 100-image stratified-by-class Validation
dataset drawn from K04 (the 750-image testing partition that no
existing Training subset consumes from), using seed=2026 to keep
it reproducible and distinct from the seed=42/43 Toronto small-split
and seed=42 TX*/WD* sub-splits.

Example:
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/curator_create_validation.py
"""

from __future__ import annotations

import random
import warnings
from collections import Counter, defaultdict

from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

K04_TESTING_RID = "K04"  # CIFAR-10 testing partition (750 images, fully labeled)
VALIDATION_SIZE = 100
PER_CLASS = 10  # 100 / 10 classes
SEED = 2026


def _stratified_pick(
    rid_class: dict[str, str],
    per_class: int,
    seed: int,
) -> list[str]:
    """Pick ``per_class`` RIDs from each class, deterministically.

    Args:
        rid_class: Map of Image RID to Image_Class label.
        per_class: Number of RIDs to draw per class.
        seed: Seed for deterministic sampling.

    Returns:
        Flat list of selected RIDs, ordered class-by-class.

    Raises:
        ValueError: If any class has fewer than ``per_class`` images.
    """
    by_class: dict[str, list[str]] = defaultdict(list)
    for rid, cls in rid_class.items():
        by_class[cls].append(rid)
    rng = random.Random(seed)
    picked: list[str] = []
    for cls in sorted(by_class):
        bucket = by_class[cls]
        if len(bucket) < per_class:
            raise ValueError(
                f"Class {cls!r} has {len(bucket)} images; need {per_class}."
            )
        rng_local = random.Random(seed + hash(cls) % 1000)
        rng_local.shuffle(bucket)
        picked.extend(bucket[:per_class])
    return picked


def main() -> None:
    """Create the Validation dataset under a curator-tagged execution."""
    ml = DerivaML(hostname="localhost", catalog_id="93")

    # Pull K04 members + their classes via the Image_Classification feature.
    k04 = ml.lookup_dataset(K04_TESTING_RID)
    members = k04.list_dataset_members()
    image_rids = [r["RID"] for r in members.get("Image", [])]
    print(f"K04 testing partition has {len(image_rids)} Image members")

    feature_vals = list(ml.feature_values("Image", "Image_Classification"))
    label_map: dict[str, str] = {}
    for fv in feature_vals:
        label_map[fv.Image] = fv.Image_Class
    rid_class = {rid: label_map[rid] for rid in image_rids if rid in label_map}
    missing = [rid for rid in image_rids if rid not in label_map]
    if missing:
        raise RuntimeError(
            f"{len(missing)} K04 Image RIDs have no Image_Classification label"
        )
    dist = Counter(rid_class.values())
    print(f"K04 class distribution: {dict(sorted(dist.items()))}")

    picked = _stratified_pick(rid_class, PER_CLASS, SEED)
    picked_dist = Counter(rid_class[r] for r in picked)
    print(f"Picked {len(picked)} images; distribution: {dict(sorted(picked_dist.items()))}")
    assert len(picked) == VALIDATION_SIZE
    assert all(c == PER_CLASS for c in picked_dist.values())

    # Register / dedupe workflow and open an execution for provenance.
    workflow = ml.create_workflow(
        name="Curator Validation Subset (cat 93)",
        workflow_type="CIFAR_Data_Load",
        description=(
            "Curator-created stratified Validation subset (10/class = 100) "
            "drawn from K04 testing partition, seed=2026. "
            "Distinct from the Toronto small-split (seed=42/43) and "
            "TX*/WD* sub-splits (seed=42). E2E run 2026-05-27."
        ),
    )
    config = ExecutionConfiguration(workflow=workflow)

    with ml.create_execution(config) as exe:
        print(f"Curator execution RID: {exe.execution_rid}")
        validation = exe.create_dataset(
            description=(
                "Curator-created Validation subset: 100 stratified images "
                "(10/class) drawn from K04 (testing partition), seed=2026. "
                "Created 2026-05-27 to give the cifar10_cnn Validation "
                "dispatch lane real data to feed."
            ),
            dataset_types=["Validation", "Labeled"],
        )
        print(f"Validation dataset RID: {validation.dataset_rid}")

    exe.commit_output_assets(clean_folder=True)

    # Membership add (post-execution against catalog directly).
    ds = ml.lookup_dataset(validation.dataset_rid)
    ds.add_dataset_members({"Image": picked}, validate=False)
    print(f"Added {len(picked)} Image members to {validation.dataset_rid}")
    print(f"New version: {ds.current_version}")

    # Verify membership.
    after = ds.list_dataset_members()
    after_count = len(after.get("Image", []))
    print(f"Post-write member count: {after_count}")
    assert after_count == VALIDATION_SIZE, (
        f"Expected {VALIDATION_SIZE} members, got {after_count}"
    )


if __name__ == "__main__":
    main()
