"""Curator verification of the CIFAR-10 split hierarchy on catalog 69.

Proves — with set arithmetic on the actual Image member RIDs — that the
canonical CIFAR-10 splits in the e2e-test-20260605 catalog mean what their
names and descriptions imply:

- Canonical Training (KE0) and Testing (KEA) are disjoint and together
  equal the Complete dataset (H8M).
- The training-derived labeled split (RQW/RR6) draws *only* from the
  canonical training partition (KE0), never from the test partition (KEA).
- Each partition is class-balanced (uniform across the 10 CIFAR-10 classes).

This is a read-only check. It performs no catalog writes.

Run:
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/curator_verify_splits.py
"""

from __future__ import annotations

import logging
from collections import Counter

from deriva.core import ErmrestCatalog, get_credential

HOSTNAME = "localhost"
CATALOG_ID = "69"
DOMAIN_SCHEMA = "e2e-test-20260605"

# Datasets under test (RIDs from the 2026-06-05 bootstrap; see tacit-knowledge tk-001).
COMPLETE = "H8M"
ORPHAN = "F2J"
TRAIN = "KE0"
TEST = "KEA"
SMALL_LABELED_TRAIN = "RQW"
SMALL_LABELED_TEST = "RR6"
LABELED_TRAIN = "QMA"
LABELED_TEST = "QMM"


def image_rids(catalog: ErmrestCatalog, dataset_rid: str) -> set[str]:
    """Return the set of Image RIDs that belong to a dataset.

    Reads the ``Dataset_Image`` association table directly (one ERMrest
    request) rather than downloading a bag — cheap and exact.

    Args:
        catalog: An open ErmrestCatalog handle.
        dataset_rid: The RID of the dataset whose Image members to fetch.

    Returns:
        The set of Image RIDs associated with ``dataset_rid``.

    Example:
        >>> rids = image_rids(catalog, "KE0")  # doctest: +SKIP
        >>> len(rids)  # doctest: +SKIP
        550
    """
    pb = catalog.getPathBuilder()
    di = pb.schemas[DOMAIN_SCHEMA].tables["Dataset_Image"]
    rows = di.filter(di.column_definitions["Dataset"] == dataset_rid).attributes(
        di.column_definitions["Image"]
    )
    return {r["Image"] for r in rows}


def class_of_images(catalog: ErmrestCatalog) -> dict[str, str]:
    """Map each Image RID to its ground-truth class Name.

    Fetches every ``Image_Classification`` feature row and keeps only the
    ground-truth ones (null Confidence; predictions carry a Confidence score,
    the loader's ground truth does not). Filtering happens client-side so the
    check stays correct once prediction rows land in the same table.

    Args:
        catalog: An open ErmrestCatalog handle.

    Returns:
        A dict mapping Image RID -> ground-truth class Name.

    Example:
        >>> labels = class_of_images(catalog)  # doctest: +SKIP
        >>> labels["488"]  # doctest: +SKIP
        'bird'
    """
    pb = catalog.getPathBuilder()
    feat = pb.schemas[DOMAIN_SCHEMA].tables["Execution_Image_Image_Classification"]
    rows = feat.attributes(
        feat.column_definitions["Image"],
        feat.column_definitions["Image_Class"],
        feat.column_definitions["Confidence"],
    )
    # Ground truth = null Confidence (predictions carry a Confidence score).
    return {r["Image"]: r["Image_Class"] for r in rows if r["Confidence"] is None}


def balance(rids: set[str], labels: dict[str, str]) -> Counter:
    """Return per-class counts for a set of Image RIDs.

    Args:
        rids: A set of Image RIDs.
        labels: Image RID -> class Name mapping from ``class_of_images``.

    Returns:
        A Counter of class Name -> count over ``rids``.

    Example:
        >>> balance({"a", "b"}, {"a": "cat", "b": "cat"})
        Counter({'cat': 2})
    """
    return Counter(labels[r] for r in rids if r in labels)


def main() -> None:
    """Run all split-integrity checks and print a pass/fail report."""
    credential = get_credential(HOSTNAME)
    catalog = ErmrestCatalog("https", HOSTNAME, CATALOG_ID, credentials=credential)
    logging.getLogger("deriva").setLevel(logging.ERROR)

    complete = image_rids(catalog, COMPLETE)
    orphan = image_rids(catalog, ORPHAN)
    train = image_rids(catalog, TRAIN)
    test = image_rids(catalog, TEST)
    s_ltrain = image_rids(catalog, SMALL_LABELED_TRAIN)
    s_ltest = image_rids(catalog, SMALL_LABELED_TEST)
    ltrain = image_rids(catalog, LABELED_TRAIN)
    ltest = image_rids(catalog, LABELED_TEST)
    labels = class_of_images(catalog)

    print(f"Complete H8M:                {len(complete)} images")
    print(f"Orphan   F2J:                {len(orphan)} images")
    print(f"Training KE0:                {len(train)} images")
    print(f"Testing  KEA:                {len(test)} images")
    print(f"Labeled train QMA:           {len(ltrain)} images")
    print(f"Labeled test  QMM:           {len(ltest)} images")
    print(f"Small labeled train RQW:     {len(s_ltrain)} images")
    print(f"Small labeled test  RR6:     {len(s_ltest)} images")
    print(f"Ground-truth labels:         {len(labels)} images")
    print()

    checks: list[tuple[str, bool]] = []

    # 1. Canonical train/test are disjoint and partition the Complete set.
    checks.append(("KE0 ∩ KEA == ∅ (train/test disjoint)", len(train & test) == 0))
    checks.append(
        ("KE0 ∪ KEA == H8M (train+test == complete)", (train | test) == complete)
    )
    checks.append(("KE0 ⊆ H8M", train <= complete))
    checks.append(("KEA ⊆ H8M", test <= complete))

    # 2. Orphan F2J holds the same image set as H8M (it is a full duplicate, not empty).
    checks.append(
        ("F2J image set == H8M image set (full duplicate)", orphan == complete)
    )

    # 3. Labeled split draws only from training (KE0), never from test (KEA).
    checks.append(("QMA ⊆ KE0 (labeled train from canonical train)", ltrain <= train))
    checks.append(("QMM ⊆ KE0 (labeled holdout from canonical train)", ltest <= train))
    checks.append(
        ("QMA ∩ KEA == ∅ (no test leakage into labeled train)", len(ltrain & test) == 0)
    )
    checks.append(
        (
            "QMM ∩ KEA == ∅ (no test leakage into labeled holdout)",
            len(ltest & test) == 0,
        )
    )
    checks.append(
        ("QMA ∩ QMM == ∅ (labeled train/holdout disjoint)", len(ltrain & ltest) == 0)
    )

    # 4. Small labeled split is a strict subset of the labeled split partitions.
    checks.append(
        ("RQW ⊆ KE0 (small labeled train from canonical train)", s_ltrain <= train)
    )
    checks.append(
        ("RR6 ⊆ KE0 (small labeled holdout from canonical train)", s_ltest <= train)
    )
    checks.append(
        ("RQW ∩ KEA == ∅ (no test leakage, small train)", len(s_ltrain & test) == 0)
    )
    checks.append(
        ("RR6 ∩ KEA == ∅ (no test leakage, small holdout)", len(s_ltest & test) == 0)
    )
    checks.append(
        ("RQW ∩ RR6 == ∅ (small train/holdout disjoint)", len(s_ltrain & s_ltest) == 0)
    )

    # 5. Class balance — uniform across 10 classes for each partition we evaluate on.
    for name, rids in [
        ("KE0", train),
        ("KEA", test),
        ("RQW", s_ltrain),
        ("RR6", s_ltest),
    ]:
        b = balance(rids, labels)
        uniform = len(b) == 10 and len(set(b.values())) == 1
        checks.append(
            (
                f"{name} class balance uniform across 10 classes ({dict(sorted(b.items()))})",
                uniform,
            )
        )

    print("CHECKS")
    print("------")
    all_pass = True
    for desc, ok in checks:
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{flag}] {desc}")

    print()
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
