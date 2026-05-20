"""Smoke tests for src/scripts/_cifar10_datasets.py.

Stage 3 needs a live Deriva catalog for its dataset-creation work,
so the orchestrator-level tests are sparse — end-to-end behavior
is exercised in the load-cifar10 smoke test in Task A13. The
pure RID-level stratified-sampling helper is tested directly here.
"""

from __future__ import annotations

from collections import Counter


def _make_rid_corpus(per_class: int) -> tuple[list[str], list[str]]:
    """Build (rids, classes) with per_class items in each of 10 classes."""
    classes_names = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    rids: list[str] = []
    classes: list[str] = []
    counter = 0
    for cls in classes_names:
        for _ in range(per_class):
            rids.append(f"R-{counter:06d}")
            classes.append(cls)
            counter += 1
    return rids, classes


def test_module_exposes_expected_api():
    from scripts._cifar10_datasets import (
        create_dataset_hierarchy,
        run_datasets_phase,
        stratified_sample_rids,
    )

    for fn in (create_dataset_hierarchy, run_datasets_phase, stratified_sample_rids):
        assert callable(fn)


def test_stratified_rid_sample_balances_partition():
    """Feed a class-balanced 100-RID set, sample 50: 5 per class."""
    from scripts._cifar10_datasets import stratified_sample_rids

    rids, classes = _make_rid_corpus(per_class=10)
    sample = stratified_sample_rids(rids, classes, sample_size=50, seed=42)
    rid_to_class = dict(zip(rids, classes))

    assert len(sample) == 50
    counts = Counter(rid_to_class[r] for r in sample)
    assert all(n == 5 for n in counts.values()), counts


def test_stratified_rid_sample_handles_imbalanced_source():
    """Skewed source (most images bird/ship): result still spreads per quota."""
    from scripts._cifar10_datasets import stratified_sample_rids

    classes = ["bird"] * 200 + ["ship"] * 50 + ["airplane"] * 5 + ["truck"] * 5
    rids = [f"R-{i:06d}" for i in range(len(classes))]
    sample = stratified_sample_rids(rids, classes, sample_size=12, seed=42)

    rid_to_class = dict(zip(rids, classes))
    counts = Counter(rid_to_class[r] for r in sample)
    # Base quota 3 per class (4 classes), remainder 0. So each class
    # contributes exactly 3 — not 6 bird-skewed copies.
    assert len(sample) == 12
    assert all(n == 3 for n in counts.values()), counts


def test_stratified_rid_sample_ignores_none_class_entries():
    from scripts._cifar10_datasets import stratified_sample_rids

    rids = ["A", "B", "C", "D"]
    classes = ["x", None, "y", None]
    sample = stratified_sample_rids(rids, classes, sample_size=2, seed=42)

    assert set(sample) == {"A", "C"}


def test_stratified_rid_sample_empty_inputs_return_empty():
    from scripts._cifar10_datasets import stratified_sample_rids

    assert stratified_sample_rids([], [], sample_size=5, seed=42) == []
    assert stratified_sample_rids(["A"], ["x"], sample_size=0, seed=42) == []
